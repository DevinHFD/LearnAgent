from __future__ import annotations

import os
from typing import Optional, List

from src.agent_core.bench.tasks import get_task_library
from src.agent_core.specs.task_spec import TaskSpec
from src.agent_core.verify.verifier import verify
from src.agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolResult, ToolCall
from src.agent_core.runtime.executor import execute_tool
from src.agent_core.llm.robust_action import robust_next_action


MAX_STEPS = 30


def to_spec(task_text: str) -> TaskSpec:
    spec = TaskSpec(task=task_text)
    spec.allowed_tools = ["shell_exec", "python_exec", "file_write", "pip_install"]
    spec.stdout_is_number = True
    return spec


def reset(files: list[str]):
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def run_one(task_text: str, required_files: Optional[List[str]] = None):
    spec = to_spec(task_text)
    if required_files:
        spec.required_files = required_files

    reset(spec.required_files or [])

    rm = RunManager()
    ctx = rm.start(tag="agent_day11_robust_action")
    rm.save_text(ctx, "task.txt", task_text)

    last: Optional[ToolResult] = None
    hint = "Start."

    for step in range(1, MAX_STEPS + 1):
        obs = (
            f"Task:\n{task_text}\n\n"
            f"Allowed tools: {spec.allowed_tools}\n"
            f"Hint: {hint}\n\n"
            "Return ONE JSON tool call only.\n"
        )
        rm.save_text(ctx, f"step_{step:02d}_obs.txt", obs)

        action = robust_next_action(
            rm=rm,
            ctx=ctx,
            task=task_text,
            observation=obs,
            rules=None,
            allowed_tools=spec.allowed_tools,
            step=step,
        )
        rm.save_json(ctx, f"step_{step:02d}_action_final.json", action.model_dump())

        result = execute_tool(action, task=task_text)
        rm.save_json(ctx, f"step_{step:02d}_result.json", result.model_dump())
        last = result

        v = verify(spec, last, check_stdout=True)
        rm.save_json(ctx, f"step_{step:02d}_verify.json", {"ok": v.ok, "hint": v.hint, "messages": v.messages, "gaps": v.gaps})
        if v.ok:
            rm.save_text(ctx, "final.txt", "DONE")
            print(f"[Day11] OK run_id={ctx.run_id}")
            return True
        hint = v.hint + v.messages[0] if v.messages else v.hint

    rm.save_text(ctx, "final.txt", "FAILED")
    print(f"[Day11] FAILED run_id={ctx.run_id}")
    return False


if __name__ == "__main__":
    bt = get_task_library()[0]
    run_one(bt.task, required_files=getattr(bt, "required_files", None))