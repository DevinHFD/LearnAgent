from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class BenchTask:
    task_id: str
    task: str
    tags: List[str]
    # Optional expected stdout exact string (after strip)
    expected_stdout: Optional[str] = None
    # Optional required files
    required_files: Optional[List[str]] = None
    # Optional csv schema constraints: file -> required columns
    csv_required_columns: Optional[Dict[str, List[str]]] = None
    # Optional csv minimum rows: file -> min rows (excluding header)
    csv_min_rows: Optional[Dict[str, int]] = None


def get_task_library() -> List[BenchTask]:
    return [
        BenchTask(
            task_id="users_events_v1",
            task=(
                "Create users.csv and events.csv on disk (must use file_write). "
                "users.csv must have column user_id with 3 users: 1,2,3. "
                "events.csv must have columns event_id,user_id with two events: (1,1) and (2,3). "
                "Then print the number of unique users who have at least one event."
            ),
            tags=["csv", "file_write", "verifier_first"],
            expected_stdout="2",
            required_files=["users.csv", "events.csv"],
            csv_required_columns={"users.csv": ["user_id"], "events.csv": ["event_id", "user_id"]},
            csv_min_rows={"users.csv": 3, "events.csv": 2},
        ),
        BenchTask(
            task_id="mean_csv_v1",
            task=(
                "Read data.csv and compute the mean of column 'value'. "
                "If the file does not exist, CREATE data.csv ON DISK using file_write "
                "with header 'value' and three rows 10,20,30. "
                "Then print ONLY the mean number."
            ),
            tags=["csv", "mean", "file_write"],
            expected_stdout="20.0",
            required_files=["data.csv"],
            csv_required_columns={"data.csv": ["value"]},
            csv_min_rows={"data.csv": 3},
        ),
        BenchTask(
            task_id="report_md_v1",
            task=(
                "Create a report.md ON DISK using file_write. "
                "The report must contain a title line starting with '# ' and a bullet list with at least 3 items. "
                "Then print ONLY the number 3 to stdout."
            ),
            tags=["markdown", "file_write"],
            expected_stdout="3",
            required_files=["report.md"],
        ),
    ]