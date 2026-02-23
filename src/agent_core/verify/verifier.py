from __future__ import annotations
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..specs.task_spec import TaskSpec
from ..schemas.tool import ToolResult


@dataclass
class VerifyResult:
    ok: bool
    messages: List[str] = field(default_factory=list)
    hint: str = ""
    gaps: Dict[str, object] = field(default_factory=dict)


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


def _init_gaps() -> Dict[str, object]:
    return {
        "missing_files": [],            # list[str]
        "csv_missing_columns": {},      # dict[file, list[cols]]
        "csv_rows_needed": {},          # dict[file, int]
        "stdout_error": None,           # str|None
    }


def verify(spec: TaskSpec, last: Optional[ToolResult], check_stdout: bool = True) -> VerifyResult:
    """
    If check_stdout=False: validate ONLY artifacts (files/csv schema/rows) and return structured gaps.
    If check_stdout=True: validate artifacts + stdout constraints.
    """
    msgs: List[str] = []
    gaps: Dict[str, object] = _init_gaps()

    # 1) required files
    for f in spec.required_files:
        ok, msg = check_file_exists(f)
        msgs.append(msg)
        if not ok:
            gaps["missing_files"].append(f)

    # 2) CSV schema constraints
    for path, cols in spec.csv_required_columns.items():
        ok, msg = check_csv_has_columns(path, cols)
        msgs.append(msg)
        if not ok:
            gaps["csv_missing_columns"][path] = cols

    # 3) CSV row count constraints
    for path, n in spec.csv_min_rows.items():
        ok, msg = check_csv_min_rows(path, n)
        msgs.append(msg)
        if not ok:
            gaps["csv_rows_needed"][path] = n

    artifacts_ok = (
        len(gaps["missing_files"]) == 0
        and len(gaps["csv_missing_columns"]) == 0
        and len(gaps["csv_rows_needed"]) == 0
    )

    if not check_stdout:
        hint = "ARTIFACTS_OK" if artifacts_ok else "Fix artifacts based on gaps."
        return VerifyResult(ok=artifacts_ok, messages=msgs, hint=hint, gaps=gaps)

    # If artifacts fail, stop early (stdout checks meaningless)
    if not artifacts_ok:
        return VerifyResult(
            ok=False,
            messages=msgs,
            hint="Artifacts not satisfied. Create/fix files ON DISK using file_write based on gaps.",
            gaps=gaps,
        )

    # 4) stdout constraints
    stdout = ""
    if last is not None and getattr(last, "output", None) is not None:
        stdout = str(last.output)

    if spec.stdout_exact is not None:
        ok, msg = check_stdout_exact(stdout, spec.stdout_exact)
        msgs.append(msg)
        if not ok:
            gaps["stdout_error"] = msg
            return VerifyResult(
                ok=False,
                messages=msgs,
                hint=f"Recompute and print EXACT stdout {spec.stdout_exact!r} using python_exec.",
                gaps=gaps,
            )

    if spec.stdout_is_number:
        ok, msg = check_stdout_is_number(stdout)
        msgs.append(msg)
        if not ok:
            gaps["stdout_error"] = msg
            return VerifyResult(
                ok=False,
                messages=msgs,
                hint="Compute the final numeric answer and print ONLY the number to stdout using python_exec.",
                gaps=gaps,
            )

    return VerifyResult(ok=True, messages=msgs, hint="DONE", gaps=gaps)


def verify_artifacts_only(spec: TaskSpec) -> VerifyResult:
    return verify(spec, last=None, check_stdout=False)