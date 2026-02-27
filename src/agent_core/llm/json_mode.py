from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..llm.client import LLMClient


_REPAIR_SYS = (
    "You are a strict JSON repair tool.\n"
    "Input may be empty, partial, wrapped, or contain extra text.\n"
    "Return ONLY valid JSON object, no markdown, no commentary.\n"
)

_REPAIR_USER = """Fix the following into a STRICT JSON object with keys:
- name: one of ["shell_exec","python_exec","file_write","pip_install"]
- args: object

Rules:
- Do NOT wrap with tool_call.
- Do NOT nest another tool call inside args.
- Keep only required fields.

BAD:
{bad}
"""


def repair_to_toolcall_json(raw: str) -> Dict[str, Any]:
    client = LLMClient()
    fixed = client.chat(
        messages=[
            {"role": "system", "content": _REPAIR_SYS},
            {"role": "user", "content": _REPAIR_USER.format(bad=raw or "")},
        ],
        temperature=0,
    )
    return json.loads(fixed)


def try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None