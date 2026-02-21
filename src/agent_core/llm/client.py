from openai import OpenAI
from .settings import OPENAI_API_BASE, OPENAI_API_KEY, CHAT_MODEL, DEFAULT_TEMP
import json

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
        msg = r.choices[0].message

        # 1) normal content
        if msg.content and msg.content.strip():
            return msg.content

        # 2) tool_calls -> convert to ToolCall JSON directly
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            tc = tcs[0]
            fn = tc.function
            tool_name = fn.name  # e.g. "shell_exec" / "file_write" / "python_exec"
            args_raw = fn.arguments  # usually JSON string

            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except Exception:
                    # still return something parseable
                    args = {"_raw": args_raw}
            else:
                args = args_raw

            # IMPORTANT: return ToolCall JSON (no extra wrapper!)
            return json.dumps({"name": tool_name, "args": args}, ensure_ascii=False)

        return ""

