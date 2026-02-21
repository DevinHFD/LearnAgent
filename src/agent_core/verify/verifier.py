from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..specs.task_spec import TaskSpec
from ..schemas.tool import ToolResult


@dataclass
class VerifyResult:
    ok: bool
    messages: List[str] = field(default_factory=list)
    hint: str = ""


def _ok(msg: str) -> Tuple[bool, str]:
    return True, msg


def _fail(msg: str) -> Tuple[bool, str]:
    return False, msg


def check_file_exists(path: str) -> Tuple[bool, str]:
    if os.path.exists(path):
        return _ok(f"OK: file exists: {path} (size={os.path.getsize(path)} bytes)")
    return _fail(f"Missing required file: {path}")


def check_stdout_is_number(text: str) -> Tuple[bool, str]:
    s = (text or "").strip()
    if not s:
        return _fail("stdout empty; expected a number")
    try:
        float(s)
        return _ok(f"OK: stdout is number: {s}")
    except Exception:
        return _fail(f"stdout not a number: {s!r}")


def check_stdout_exact(text: str, expected: str) -> Tuple[bool, str]:
    s = (text or "").strip()
    if s == expected.strip():
        return _ok(f"OK: stdout matches expected exactly: {expected!r}")
    return _fail(f"stdout mismatch: got {s!r}, expected {expected!r}")


def check_csv_has_columns(path: str, required: List[str]) -> Tuple[bool, str]:
    if not os.path.exists(path):
        return _fail(f"CSV missing for schema check: {path}")
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return _fail(f"CSV is empty (no header): {path}")
    header_set = {h.strip() for h in header if h is not None}
    missing = [c for c in required if c not in header_set]
    if missing:
        return _fail(f"CSV {path} missing required columns: {missing}. header={header}")
    return _ok(f"OK: CSV {path} contains required columns: {required}")


def check_csv_min_rows(path: str, n: int) -> Tuple[bool, str]:
    if not os.path.exists(path):
        return _fail(f"CSV missing for row count check: {path}")
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # header
        except StopIteration:
            return _fail(f"CSV is empty (no header): {path}")
        rows = [r for r in reader if r and any(str(x).strip() for x in r)]
    if len(rows) < n:
        return _fail(f"CSV {path} has too few data rows: {len(rows)} < {n}")
    return _ok(f"OK: CSV {path} has >= {n} data rows ({len(rows)})")


def verify(spec: TaskSpec, last: Optional[ToolResult]) -> VerifyResult:
    msgs: List[str] = []

    # 1) required files
    for f in spec.required_files:
        ok, msg = check_file_exists(f)
        msgs.append(msg)
        if not ok:
            return VerifyResult(
                ok=False,
                messages=msgs,
                hint=f"Create missing file {f!r} ON DISK using file_write, then continue.",
            )

    # 2) CSV schema constraints
    for path, cols in spec.csv_required_columns.items():
        ok, msg = check_csv_has_columns(path, cols)
        msgs.append(msg)
        if not ok:
            return VerifyResult(
                ok=False,
                messages=msgs,
                hint=f"Fix {path!r} header using file_write so it contains columns {cols}.",
            )

    # 3) CSV row count constraints
    for path, n in spec.csv_min_rows.items():
        ok, msg = check_csv_min_rows(path, n)
        msgs.append(msg)
        if not ok:
            return VerifyResult(
                ok=False,
                messages=msgs,
                hint=f"Add at least {n} data rows to {path!r} using file_write, then continue.",
            )

    # 4) stdout constraints
    stdout = ""
    if last is not None and getattr(last, "output", None) is not None:
        stdout = str(last.output)

    if spec.stdout_exact is not None:
        ok, msg = check_stdout_exact(stdout, spec.stdout_exact)
        msgs.append(msg)
        if not ok:
            return VerifyResult(
                ok=False,
                messages=msgs,
                hint=f"Recompute and print EXACT stdout {spec.stdout_exact!r} using python_exec.",
            )

    if spec.stdout_is_number:
        ok, msg = check_stdout_is_number(stdout)
        msgs.append(msg)
        if not ok:
            return VerifyResult(
                ok=False,
                messages=msgs,
                hint="Compute the final numeric answer and print ONLY the number to stdout using python_exec.",
            )

    return VerifyResult(ok=True, messages=msgs, hint="DONE")