import subprocess
import sys
from typing import Dict, Any

def _wrap_code(code: str) -> str:
    code = code.strip()

    # 如果已经包含 print 或者多行/语句，直接执行
    if "print(" in code or "\n" in code or ";" in code:
        return code

    # 否则把“单表达式”包成 print(expr)
    return f"print({code})"

def python_exec(args: Dict[str, Any]) -> str:
    """
    Execute python code in a subprocess using current interpreter (venv).
    Args expects: {"code": "..."}.
    If code is a single expression without print, auto-wrap with print().
    """
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("python_exec requires args['code'] as a non-empty string")

    code_to_run = _wrap_code(code)

    proc = subprocess.run(
        [sys.executable, "-c", code_to_run],
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    if proc.returncode != 0:
        raise RuntimeError(err or f"python exited with code {proc.returncode}")
    return out
