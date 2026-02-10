from pathlib import Path
import re

def require_file(path: str) -> tuple[bool, str]:
    p = Path(path)
    if p.exists() and p.is_file() and p.stat().st_size > 0:
        return True, f"OK: file exists: {path} (size={p.stat().st_size} bytes)"
    return False, f"FAIL: file not found or empty: {path}"

def mean_output_ok(stdout: str) -> tuple[bool, str]:
    if stdout is None:
        return False, "FAIL: no stdout"
    s = stdout.strip()
    if re.search(r"\b20(\.0+)?\b", s) or re.search(r"\b\d+(\.\d+)?\b", s):
        return True, f"OK: got numeric output: {s}"
    return False, f"FAIL: stdout not numeric: {s}"