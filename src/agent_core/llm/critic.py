from __future__ import annotations

import json
from typing import Any, Dict, List

from .client import LLMClient


SYS = (
    "You are a Critic for an agent system.\n"
    "Given task, last action/result, and verifier gaps, propose a short corrective instruction.\n"
    "Return STRICT JSON: {\"instruction\": \"...\"}\n"
)

USER = """Task:
{task}

Last action:
{action}

Last result:
{result}

Verifier gaps:
{gaps}

Verifier hint:
{hint}

Write a single corrective instruction that tells the executor exactly what to do next.
"""


def critique(task: str, action: Dict[str, Any], result: Dict[str, Any], gaps: Dict[str, Any], hint: str) -> str:
    client = LLMClient()
    raw = client.chat(
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER.format(task=task, action=json.dumps(action, ensure_ascii=False),
                                                   result=json.dumps(result, ensure_ascii=False),
                                                   gaps=json.dumps(gaps, ensure_ascii=False), hint=hint)},
        ],
        temperature=0,
    )
    obj = json.loads(raw)
    return str(obj.get("instruction", "")).strip()