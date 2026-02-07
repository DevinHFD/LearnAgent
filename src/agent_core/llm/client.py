from openai import OpenAI
from .settings import OPENAI_API_BASE, OPENAI_API_KEY, CHAT_MODEL, DEFAULT_TEMP

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY
        )
        self.model = CHAT_MODEL

    def chat(self, messages, temperature=None):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or DEFAULT_TEMP,
        )
        return r.choices[0].message.content
