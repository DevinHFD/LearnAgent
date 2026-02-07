from ..schemas.tool import ToolCall, ToolResult
from ..tools import TOOLS

def execute_tool(call: ToolCall) -> ToolResult:
    fn = TOOLS.get(call.name)
    if fn is None:
        return ToolResult(name=call.name, ok=False, output="", error="Unknown tool")

    try:
        out = fn(call.args)
        return ToolResult(name=call.name, ok=True, output=out)
    except Exception as e:
        return ToolResult(name=call.name, ok=False, output="", error=str(e))
