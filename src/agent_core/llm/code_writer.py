import json
from tenacity import retry, stop_after_attempt, wait_fixed
from ..schemas.code import CodeBlock
from .client import LLMClient
import json
from pydantic import ValidationError

client = LLMClient()

SYSTEM = (
    "Write ONLY valid Python code as JSON.\n"
    f"Schema: {CodeBlock.model_json_schema()}\n"
    "The code should solve the task."
)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def write_code(task: str, feedback: str | None = None) -> CodeBlock:
    msg = task if feedback is None else f"{task}\nPrevious error:\n{feedback}"

    raw = client.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": msg}],
        temperature=0,
    )

    try:
        return CodeBlock.model_validate(json.loads(raw))
    except Exception:
        fix = client.chat(
            [{"role": "system", "content": "Fix JSON. Return ONLY corrected JSON."},
             {"role": "user", "content": raw}],
            temperature=0,
        )
        return CodeBlock.model_validate(json.loads(fix))

def write_code_with_trace(task: str, feedback: str | None = None) -> tuple[CodeBlock, dict]:
    """
    Returns (CodeBlock, trace)
    trace contains: user_msg, raw, fixed(optional)
    """
    msg = task if feedback is None else f"{task}\nPrevious error:\n{feedback}"

    raw = client.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": msg}],
        temperature=0,
    )

    trace = {"user_msg": msg, "raw": raw, "fixed": None}

    try:
        cb = CodeBlock.model_validate(json.loads(raw))
        return cb, trace
    except Exception:
        fix = client.chat(
            [{"role": "system", "content": "Fix JSON. Return ONLY corrected JSON."},
             {"role": "user", "content": raw}],
            temperature=0,
        )
        trace["fixed"] = fix
        cb = CodeBlock.model_validate(json.loads(fix))
        return cb, trace

