from agent_core.runtime.run_manager import RunManager
from agent_core.runtime.executor import execute_tool
from agent_core.llm.action_router import next_action
from agent_core.runtime.verifier import require_file
from agent_core.schemas.tool import ToolCall

MAX_STEPS = 12

def agent_loop(task: str):
    rm = RunManager()
    ctx = rm.start(tag="agent_day2")

    rm.save_text(ctx, "task.txt", task)

    observation = "Task started."

    for step in range(1, MAX_STEPS + 1):
        rm.save_text(ctx, f"step_{step:02d}_observation.txt", observation)

        action = next_action(task, observation)
        rm.save_json(ctx, f"step_{step:02d}_action.json", action.model_dump())

        result = execute_tool(action)
        rm.save_json(ctx, f"step_{step:02d}_result.json", result.model_dump())

        if result.ok:
            observation = f"SUCCESS:\n{result.output}"
        else:
            observation = f"ERROR:\n{result.error}"

        # goal check: if plot.png created, stop
        ok2, msg2 = require_file("plot.png")
        rm.save_text(ctx, f"step_{step:02d}_verify.txt", msg2)

        if ok2:
            rm.save_text(ctx, "final.txt", "plot.png generated successfully")
            print("run_id:", ctx.run_id)
            print("DONE")
            return

    raise RuntimeError("Agent failed to complete task")

if __name__ == "__main__":
    agent_loop("Generate a plot of y = x^2 and save it as plot.png")
