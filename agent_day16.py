from __future__ import annotations

import json

from src.agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolCall
from src.agent_core.tools.registry_v2 import ToolRegistryV2, ToolSpec
from src.agent_core.tool_executor_v2 import execute_tool_v2

# reuse your existing tool funcs
from src.agent_core.tools import shell_exec, python_exec, file_write, pip_install  # adjust if your import differs


def build_registry() -> ToolRegistryV2:
    reg = ToolRegistryV2()
    reg.register(ToolSpec(
        name="shell_exec",
        description="Run a shell command in project environment.",
        args_schema={"type":"object","required":["cmd"],"properties":{"cmd":{"type":"string"}}},
        fn=lambda a: shell_exec(a),
        safety_notes="Avoid destructive commands (rm -rf, etc.)."
    ))
    reg.register(ToolSpec(
        name="python_exec",
        description="Execute Python code and return stdout/stderr.",
        args_schema={"type":"object","required":["code"],"properties":{"code":{"type":"string"}}},
        fn=lambda a: python_exec(a),
    ))
    reg.register(ToolSpec(
        name="file_write",
        description="Write content to a file on disk (relative path).",
        args_schema={"type":"object","required":["path","content"],"properties":{"path":{"type":"string"},"content":{"type":"string"}}},
        fn=lambda a: file_write(a),
    ))
    reg.register(ToolSpec(
        name="pip_install",
        description="Install Python packages into current venv.",
        args_schema={"type":"object","required":["packages"],"properties":{"packages":{"type":"string"}}},
        fn=lambda a: pip_install(a),
    ))
    return reg


if __name__ == "__main__":
    rm = RunManager()
    ctx = rm.start(tag="agent_day16_tool_protocol_v2")

    reg = build_registry()
    rm.save_text(ctx, "tools_prompt.txt", reg.to_prompt())

    # quick sanity checks
    calls = [
        ToolCall(name="shell_exec", args={"cmd": "pwd"}),
        ToolCall(name="python_exec", args={"code": "print(sum(range(1,11)))"}),
        ToolCall(name="file_write", args={"path": "day16_ok.txt", "content": "hello"}),
    ]

    outs = []
    for i, c in enumerate(calls, 1):
        r = execute_tool_v2(reg, c)
        rm.save_json(ctx, f"call_{i}.json", c.model_dump())
        rm.save_json(ctx, f"result_{i}.json", r.model_dump())
        outs.append(r.ok)

    print(json.dumps({"ok_all": all(outs), "run_id": ctx.run_id}, ensure_ascii=False))