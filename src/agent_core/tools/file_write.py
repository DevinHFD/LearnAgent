from pathlib import Path
from typing import Dict, Any

def file_write(args: Dict[str, Any]) -> str:
    """
    Write text to a file.
    Args expects: {"path": "relative/or/absolute", "content": "text"}
    Default: relative to current working directory.
    """
    path = args.get("path")
    content = args.get("content")

    if not isinstance(path, str) or not path.strip():
        raise ValueError("file_write requires args['path'] as non-empty string")
    if not isinstance(content, str):
        raise ValueError("file_write requires args['content'] as string")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {str(p)}"
