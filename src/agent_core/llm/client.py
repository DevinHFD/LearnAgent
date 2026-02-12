from openai import OpenAI
from .settings import OPENAI_API_BASE, OPENAI_API_KEY, CHAT_MODEL, DEFAULT_TEMP

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY
        )
        self.model = CHAT_MODEL

    def chat(self, messages, temperature=0, tools=None, tool_choice=None, response_format=None):
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        r = self.client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""
