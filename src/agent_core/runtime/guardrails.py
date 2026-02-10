# src/agent_core/runtime/guardrails.py
from typing import Optional
from ..schemas.tool import ToolCall, ToolResult


class Guardrails:
    def __init__(self):
        self.last_action_sig = None
        self.repeat_count = 0

    def _signature(self, action: ToolCall) -> str:
        return f"{action.name}:{action.args}"

    def track(self, action: ToolCall):
        sig = self._signature(action)
        if sig == self.last_action_sig:
            self.repeat_count += 1
        else:
            self.last_action_sig = sig
            self.repeat_count = 0

    def intervene(
        self,
        action: ToolCall,
        last_result: Optional[ToolResult],
        observation: str,
    ) -> Optional[ToolCall]:
        """
        Guardrails may override the next action,
        but NEVER generate domain content.
        """

        # Case 1: Missing file detected → must create file
        if last_result and last_result.error:
            err = last_result.error.lower()
            if "no such file or directory" in err:
                return ToolCall(
                    name="file_write",
                    args={
                        "path": self._infer_missing_path(err),
                        "content": "__LLM_GENERATE_SAMPLE__",
                    },
                )

        # Case 2: Repetition breaker
        if self.repeat_count >= 2:
            if action.name == "shell_exec":
                return ToolCall(
                    name="file_write",
                    args={
                        "path": self._infer_path_from_action(action),
                        "content": "__LLM_GENERATE_SAMPLE__",
                    },
                )

        return None

    def _infer_missing_path(self, err: str) -> str:
        # minimal heuristic; no domain semantics
        # example: "cannot access 'data.csv'"
        import re
        m = re.search(r"'([^']+)'", err)
        return m.group(1) if m else "unknown_file.txt"

    def _infer_path_from_action(self, action: ToolCall) -> str:
        # best-effort; fallback safe default
        cmd = str(action.args.get("cmd", ""))
        import re
        m = re.search(r"(\S+\.\w+)", cmd)
        return m.group(1) if m else "generated_file.txt"
