from pathlib import Path

def require_file(path: str) -> tuple[bool, str]:
    p = Path(path)
    if p.exists() and p.is_file() and p.stat().st_size > 0:
        return True, f"OK: file exists: {path} (size={p.stat().st_size} bytes)"
    return False, f"FAIL: file not found or empty: {path}"
