import json
from tenacity import retry, stop_after_attempt, wait_fixed
from ..schemas.tool import ToolCall
from .client import LLMClient

client = LLMClient()

SYSTEM = (
    "You are a tool router. Return ONLY valid JSON.\n"
    "Choose a tool and provide args.\n"
    f"Schema: {ToolCall.model_json_schema()}\n"
    "Tool description:\n"
    "- python_exec: execute python code. args: {code: string}\n"
)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def route_to_tool(user_task: str) -> ToolCall:
    raw = client.chat(
        [{"role":"system","content":SYSTEM},
         {"role":"user","content":user_task}],
        temperature=0,
    )
    try:
        return ToolCall.model_validate(json.loads(raw))
    except Exception:
        fix = client.chat(
            [{"role":"system","content":"Fix the JSON. Return ONLY corrected JSON."},
             {"role":"user","content":raw}],
            temperature=0,
        )
        return ToolCall.model_validate(json.loads(fix))
