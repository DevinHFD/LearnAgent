import re
from agent_core.runtime.run_manager import RunManager
from agent_core.llm.code_writer import write_code_with_trace
from agent_core.runtime.executor import execute_tool
from agent_core.schemas.tool import ToolCall
from agent_core.runtime.verifier import require_file


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

        # check and install the missing packages 
        missing = None
        if result.error:
            m = re.search(r"No module named '([^']+)'", result.error)
            if m:
                missing = m.group(1)

        if missing:
            rm.save_text(ctx, f"iter_{i:02d}_auto_action.txt", f"Detected missing module: {missing}. Installing...")
            install_call = ToolCall(name="pip_install", args={"packages": [missing]})
            install_result = execute_tool(install_call)
            rm.save_json(ctx, f"iter_{i:02d}_pip_install.json", install_result.model_dump())

            if install_result.ok:
                # 安装成功后，继续下一轮（让 LLM 代码再执行一次）
                feedback = f"Installed missing package: {missing}. Re-run the code."
                continue
            else:
                feedback = f"Failed to install {missing}: {install_result.error}"
        # 3) Check success / update feedback
        if result.ok:
            ok2, msg2 = require_file("plot.png")
            rm.save_text(ctx, f"iter_{i:02d}_verify.txt", msg2)

            if ok2:
                rm.save_text(ctx, "final_output.txt", "plot.png generated successfully")
                print("run_id:", ctx.run_id)
                return "plot.png generated successfully"

            # 执行成功但目标未达成
            feedback = msg2
        else:
            feedback = result.error or "Unknown error"

    # fail after max iters
    rm.save_error(ctx, f"Failed after {MAX_ITERS} iterations. Last error: {feedback}")
    print("run_id:", ctx.run_id)
    raise RuntimeError("Auto code loop failed after max iterations")


if __name__ == "__main__":
    out = auto_code_loop_with_logging("用 matplotlib 画 y=x^2 的图并保存为 plot.png（确保保存成功）")
    print("Final:", out)

