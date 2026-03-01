# src/agent_core/search/beam.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from ..llm.client import LLMClient
from ..llm.normalize import normalize_toolcall_obj, ALLOWED

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# ①  Prompt – ask for *strict* JSON again (helps for the repair step)
# ----------------------------------------------------------------------
SYS = (
    "You are a tool‑call generator. Return **strict JSON** – a plain array of "
    "objects with the shape `{\"name\":...,\"args\":...}`. "
    "Do NOT wrap the JSON in markdown, do NOT add any explanation, and do NOT "
    "output anything else. "
    "If you cannot produce a valid JSON array, output an empty array [] instead."
)

USER = """Task:
{task}

Observation:
{obs}

Return exactly {k} candidate tool calls as a JSON **array**.
Make the candidates diverse.
"""


# ----------------------------------------------------------------------
# ②  Helper – pull the first [...] block out of a noisy string
# ----------------------------------------------------------------------
def _extract_first_json_array(text: str) -> str:
    """
    Strip markdown fences, ignore surrounding text and return the substring
    that starts with the first '[' and ends with the last ']'.
    """
    # Remove possible markdown fences  ```json   ...   ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array delimiters found")
    return text[start : end + 1]


# ----------------------------------------------------------------------
# ③  Defensive parsing – try to recover even broken objects
# ----------------------------------------------------------------------
def _split_top_level_objects(arr_txt: str) -> List[str]:
    """
    Given a string that *looks* like a JSON array, split it into a list of the
    raw text of each top‑level object (without the surrounding commas).
    It works by counting braces while iterating.
    """
    objects = []
    brace_depth = 0
    current = []
    for ch in arr_txt:
        if ch == "{" and brace_depth == 0:
            # start of a new object
            current = ["{"]
            brace_depth = 1
            continue
        if brace_depth:
            current.append(ch)
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    objects.append("".join(current))
    return objects


def _clean_object(txt: str) -> str:
    """
    Very light‑weight “best‑effort” cleanup for a single JSON object string:
      * Escape unescaped double quotes inside string literals.
      * If the object ends before the closing '}' add it.
    The function **does not** guarantee a valid JSON – it only tries to
    repair the most common LLM slip (missing escape or missing final brace).
    """
    # 1️⃣ ensure the outer braces are balanced – if missing, add a '}'
    if txt.count("{") > txt.count("}"):
        txt = txt + "}"
    # 2️⃣ Try to escape stray double‑quotes that appear after a ':' but are
    #    not already preceded by a backslash.
    #    (covers cases like:  "command": "cat ... "INSERT ...")
    def _escape_inside_quotes(match):
        # match groups:   1 = key part up to colon, 2 = the raw value (may contain quotes)
        key, raw_val = match.groups()
        # Escape any double quote that is not already escaped
        escaped = re.sub(r'(?<!\\)"', r'\\"', raw_val)
        return f'{key}"{escaped}"'

    txt = re.sub(r'("command"\s*:\s*)"(.*?)(?<!\\)"(?=[\s,}])', _escape_inside_quotes, txt, flags=re.DOTALL)
    return txt


def _try_defensive_parse(raw_arr_txt: str) -> List[Dict[str, Any]]:
    """
    Returns a list of candidate dicts, possibly empty.
    """
    cleaned_objects: List[Dict[str, Any]] = []
    for obj_txt in _split_top_level_objects(raw_arr_txt):
        obj_txt = _clean_object(obj_txt)
        try:
            obj = json.loads(obj_txt)
            cleaned_objects.append(obj)
        except json.JSONDecodeError:
            # Give up on this single object, but continue with the rest
            log.debug("Failed to parse recovered object: %s", obj_txt)
    return cleaned_objects


# ----------------------------------------------------------------------
# ④  LLM‑as‑repair fallback
# ----------------------------------------------------------------------
def _repair_with_llm(broken_raw: str) -> List[Dict[str, Any]]:
    """
    Sends the broken output back to the LLM with a short “please fix the JSON”
    instruction and tries to parse the reply.
    """
    client = LLMClient()
    repair_prompt = (
        "The previous response was not valid JSON. "
        "Please return *only* a valid JSON array containing the same objects. "
        "Do not add any explanation or markdown."
        f"\n\nBroken output:\n{broken_raw}"
    )
    repaired = client.chat(
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": repair_prompt},
        ],
        temperature=0.0,          # deterministic repair
    )
    try:
        arr_txt = _extract_first_json_array(repaired)
        return json.loads(arr_txt)          # should succeed now
    except Exception as exc:                # pragma: no cover – just safety net
        log.warning(
            "LLM repair failed. raw repair reply: %s. error: %s",
            repr(repaired),
            exc,
        )
        return []


# ----------------------------------------------------------------------
# ⑤  Main entry – propose_candidates
# ----------------------------------------------------------------------
def propose_candidates(task: str, obs: str, k: int = 4) -> List[Dict[str, Any]]:
    client = LLMClient()
    raw = client.chat(
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": USER.format(task=task, obs=obs, k=k)},
        ],
        temperature=0.7,
    )

    # ---- 1️⃣ Try the “happy path” (already clean) ----
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        # ---- 2️⃣ Extract the outer [...] and try defensive parsing ----
        try:
            arr_txt = _extract_first_json_array(raw)
            arr = _try_defensive_parse(arr_txt)
        except Exception as exc:        # any problem during extraction/defense
            log.debug("Defensive parse failed (%s). Trying LLM repair.", exc)
            arr = []                     # will be replaced by repair step

    # ---- 3️⃣ If we still have nothing, ask the LLM to fix it ----
    if not arr:
        arr = _repair_with_llm(raw)

    # ---- 4️⃣ Normalisation & filtering -------------------------------------------------
    out: List[Dict[str, Any]] = []
    if isinstance(arr, list):
        for entry in arr:
            if not isinstance(entry, dict):
                continue
            entry = normalize_toolcall_obj(entry)
            if entry.get("name") in ALLOWED and isinstance(entry.get("args"), dict):
                out.append(entry)

    return out[:k]


# ----------------------------------------------------------------------
# ⑥  Scoring unchanged (kept for completeness)
# ----------------------------------------------------------------------
def score_by_gaps(gaps: Dict[str, Any]) -> float:
    """Higher score = fewer gaps."""
    missing_files = len(gaps.get("missing_files", []) or [])
    missing_cols = len((gaps.get("csv_missing_columns", {}) or {}).keys())
    rows_needed = len((gaps.get("csv_rows_needed", {}) or {}).keys())
    stdout_err = 1 if gaps.get("stdout_error") else 0
    return -(5 * missing_files + 3 * missing_cols + 2 * rows_needed + 1 * stdout_err)
