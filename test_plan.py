from agent_core.llm.retry import generate_plan

plan = generate_plan("设计一个动物实验，比较A组和B组的主要终点")
print(plan)
