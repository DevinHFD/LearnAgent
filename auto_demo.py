from agent_core.runtime.run_manager import RunManager
from agent_core.llm.code_writer import write_code_with_trace
from agent_core.runtime.executor import execute_tool
from agent_core.schemas.tool import ToolCall

MAX_ITERS = 6

def auto_code_loop_with_logging(task: str) -> str:
    rm = RunManager()
    ctx = rm.start(tag="auto_code_loop")

    rm.save_text(ctx, "task.txt", task)

    feedback = None

    for i in range(1, MAX_ITERS + 1):
        # 1) LLM writes code (structured)
        code_block, trace = write_code_with_trace(task, feedback)

        rm.save_text(ctx, f"iter_{i:02d}_llm_input.txt", trace["user_msg"])
        rm.save_text(ctx, f"iter_{i:02d}_llm_raw.json", trace["raw"])
        if trace["fixed"] is not None:
            rm.save_text(ctx, f"iter_{i:02d}_llm_fixed.json", trace["fixed"])

        rm.save_text(ctx, f"iter_{i:02d}_code.py", code_block.code)


        # 2) Execute tool
        call = ToolCall(name="python_exec", args={"code": code_block.code})
        result = execute_tool(call)

        rm.save_json(ctx, f"iter_{i:02d}_result.json", result.model_dump())

        # 3) Check success / update feedback
        if result.ok:
            rm.save_text(ctx, "final_output.txt", result.output)
            print("run_id:", ctx.run_id)
            return result.output

        feedback = result.error or "Unknown error"

    # fail after max iters
    rm.save_error(ctx, f"Failed after {MAX_ITERS} iterations. Last error: {feedback}")
    print("run_id:", ctx.run_id)
    raise RuntimeError("Auto code loop failed after max iterations")


if __name__ == "__main__":
    out = auto_code_loop_with_logging("用 matplotlib 画 y=x^2 的图并保存为 plot.png（确保保存成功）")
    print("Final:", out)

