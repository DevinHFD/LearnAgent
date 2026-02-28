from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..llm.client import LLMClient
from ..llm.normalize import normalize_toolcall_obj, ALLOWED


@dataclass
class Candidate:
    obj: Dict[str, Any]
    score: float
    reason: str


SYS = (
    "You propose tool calls.\n"
    "Return STRICT JSON array of tool calls, each {\"name\":...,\"args\":...}.\n"
    "No wrappers, no nesting.\n"
    "Names limited to: shell_exec, python_exec, file_write, pip_install.\n"
)

USER = """Task:
{task}

Observation:
{obs}

Return {k} candidates as a JSON array.
Make candidates diverse.
"""


def propose_candidates(task: str, obs: str, k: int = 4) -> List[Dict[str, Any]]:
    client = LLMClient()
    raw = client.chat(
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER.format(task=task, obs=obs, k=k)},
        ],
        temperature=0.7,
    )
    arr = json.loads(raw)
    out = []
    if isinstance(arr, list):
        for x in arr:
            if isinstance(x, dict):
                x = normalize_toolcall_obj(x)
                if x.get("name") in ALLOWED and isinstance(x.get("args"), dict):
                    out.append(x)
    return out[:k]


def score_by_gaps(gaps: Dict[str, Any]) -> float:
    """
    Higher is better (fewer gaps).
    """
    missing_files = len(gaps.get("missing_files", []) or [])
    missing_cols = len((gaps.get("csv_missing_columns", {}) or {}).keys())
    rows_needed = len((gaps.get("csv_rows_needed", {}) or {}).keys())
    stdout_err = 1 if gaps.get("stdout_error") else 0

    # weight artifacts > stdout (stdout only relevant in compute)
    return -(5 * missing_files + 3 * missing_cols + 2 * rows_needed + 1 * stdout_err)