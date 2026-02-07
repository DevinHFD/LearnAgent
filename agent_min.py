from agent_core.llm.retry import generate_plan
from agent_core.llm.tool_router import route_to_tool
from agent_core.runtime.executor import execute_tool
from agent_core.runtime.run_manager import RunManager

rm = RunManager()
ctx = rm.start(tag="agent_min")

task = "用python计算 1 到 100 的和，并输出结果"
rm.save_text(ctx, "task.txt", task)

plan = generate_plan(f"把任务拆成步骤：{task}", ctx=ctx)
rm.save_json(ctx, "plan.json", plan.model_dump())

# 只做一步：直接让模型给出工具调用（demo）
tool_call = route_to_tool("写出python代码来计算 1 到 100 的和，并用 python_exec 执行。")
rm.save_json(ctx, "tool_call.json", tool_call.model_dump())

result = execute_tool(tool_call)
rm.save_json(ctx, "tool_result.json", result.model_dump())

print("run:", ctx.run_id)
print("tool_call:", tool_call)
print("result:", result)
