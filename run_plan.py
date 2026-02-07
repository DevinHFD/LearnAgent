from agent_core.runtime.run_manager import RunManager
from agent_core.llm.retry import generate_plan

rm = RunManager()
ctx = rm.start(tag="plan")

plan = generate_plan("设计一个动物实验，比较A组和B组的主要终点", ctx=ctx)
print("run_id:", ctx.run_id)
print(plan)
