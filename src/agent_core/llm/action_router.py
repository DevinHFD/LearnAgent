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

FALLBACK = ToolCall(name="shell_exec", args={"cmd": "pwd && ls"})

def _parse_toolcall(s: str) -> ToolCall:
    return ToolCall.model_validate(json.loads(s))

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def next_action(task: str, observation: str, rules: list[str] | None = None) -> ToolCall:
    policy = "" if not rules else "\nLEARNED RULES:\n" + "\n".join(rules)

    raw = client.chat(
        [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"TASK:\n{task}\n\nOBSERVATION:\n{observation}{policy}"}],
        temperature=0,
    )
     # 1) direct parse
    try:
        return _parse_toolcall(raw)
    except Exception:
        pass

    # 2) ask model to fix json
    fix = client.chat(
        [{"role": "system", "content": "Fix the JSON. Return ONLY corrected JSON."},
         {"role": "user", "content": raw}],
        temperature=0,
    )

    # 3) parse fix, else fallback
    try:
        if fix and fix.strip():
            return _parse_toolcall(fix)
    except Exception:
        pass

    # 4) last resort fallback: safe, informative, progress-making
    return FALLBACK
