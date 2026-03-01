from __future__ import annotations

from typing import Optional

from .schemas.tool import ToolCall, ToolResult
from .tools.validate_args import validate_args
from .tools.registry_v2 import ToolRegistryV2


def execute_tool_v2(reg: ToolRegistryV2, call: ToolCall, task: Optional[str] = None) -> ToolResult:
    spec = reg.get(call.name)
    if spec is None:
        return ToolResult(name=call.name, ok=False, output="", error="Unknown tool")

    ok_args, msg = validate_args(spec.args_schema, call.args or {})
    if not ok_args:
        return ToolResult(name=call.name, ok=False, output="", error=f"Bad args: {msg}")

    try:
        out = spec.fn(call.args or {})
        return ToolResult(name=call.name, ok=True, output=out, error=None)
    except Exception as e:
        return ToolResult(name=call.name, ok=False, output="", error=str(e))