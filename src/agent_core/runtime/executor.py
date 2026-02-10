from ..schemas.tool import ToolCall, ToolResult
from ..tools import TOOLS
import traceback

def execute_tool(call: ToolCall, task: str | None = None) -> ToolResult:
    args = dict(call.args)

    # expand LLM-generated sample placeholder
    if call.name == "file_write" and args.get("content") == "__LLM_GENERATE_SAMPLE__":
        if not task:
            raise ValueError("file_write requested __LLM_GENERATE_SAMPLE__ but task is None")
        from ..llm.sample_generator import generate_sample
        path = args.get("path", "generated_file.txt")
        args["content"] = generate_sample(task, path)

    fn = TOOLS.get(call.name)
    if fn is None:
        return ToolResult(
            name=call.name,
            ok=False,
            output="",
            error=f"Unknown tool: {call.name}. Available: {sorted(TOOLS.keys())}"
        )

    try:
        out = fn(args)
        return ToolResult(name=call.name, ok=True, output=out, error=None)
    except Exception:
        return ToolResult(name=call.name, ok=False, output="", error=traceback.format_exc())
