from agent_core.llm.client import LLMClient

c = LLMClient()
print("Model:", c.model)

out = c.chat([{"role":"user","content":"只回答：OK"}], temperature=0)
print("LLM says:", out)
