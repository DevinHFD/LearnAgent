import json
from tenacity import retry, stop_after_attempt, wait_fixed
from ..schemas.tool import ToolCall
from .client import LLMClient

client = LLMClient()

SYSTEM = (
    "You are an autonomous agent controller. Choose the next tool call.\n"
    "Return ONLY valid JSON that matches the schema.\n"
    f"Schema: {ToolCall.model_json_schema()}\n\n"
    "Available tools:\n"
    "- python_exec: run python code. args: {code: string}\n"
    "- pip_install: install python packages. args: {packages: string or list[string]}\n"
    "- file_write: write files. args: {path: string, content: string}\n"
    "- shell_exec: run shell commands (ls, cat, pwd, etc). args: {cmd: string}\n\n"
    "Guidelines:\n"
    "- Use pip_install when you see ModuleNotFoundError\n"
    "- Use shell_exec to inspect files or directories\n"
    "- Use file_write to create artifacts\n"
    "- Prefer simple actions\n"
    "- When using python_exec, ALWAYS print the final answer (a single number) to stdout.\n"
    "Python code rules:\n"
    "- Always output multi-line Python with \\n newlines.\n"
    "- Never write try/except/with/if using semicolons on one line.\n"
    "- Prefer csv module over pandas unless explicitly needed.\n\n"
)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def next_action(task: str, observation: str, rules: list[str] | None = None) -> ToolCall:
    policy = "" if not rules else "\nLEARNED RULES:\n" + "\n".join(rules)

    raw = client.chat(
        [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"TASK:\n{task}\n\nOBSERVATION:\n{observation}{policy}"}],
        temperature=0,
    )

    try:
        return ToolCall.model_validate(json.loads(raw))
    except Exception:
        fix = client.chat(
            [{"role": "system", "content": "Fix JSON. Return ONLY corrected JSON."},
             {"role": "user", "content": raw}],
            temperature=0,
        )
        return ToolCall.model_validate(json.loads(fix))
