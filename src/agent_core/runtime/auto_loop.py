from ..llm.code_writer import write_code
from ..runtime.executor import execute_tool
from ..schemas.tool import ToolCall

MAX_ITERS = 6

def auto_code_loop(task: str) -> str:
    feedback = None

    for i in range(MAX_ITERS):
        code_block = write_code(task, feedback)

        call = ToolCall(
            name="python_exec",
            args={"code": code_block.code}
        )

        result = execute_tool(call)

        if result.ok:
            return result.output

        feedback = result.error

    raise RuntimeError("Auto code loop failed after max iterations")
