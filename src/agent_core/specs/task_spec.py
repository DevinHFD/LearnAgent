from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskSpec:
    """
    Verifier-first task specification.

    - required_files: files that MUST exist on disk to consider task complete
    - csv_required_columns: mapping of csv file -> required column names
    - csv_min_rows: mapping of csv file -> minimum data rows (excluding header)
    - stdout_is_number: stdout must be parseable as number (int/float)
    - stdout_exact: if provided, stdout must equal this exact string after strip
    """
    task: str

    required_files: List[str] = field(default_factory=list)
    csv_required_columns: Dict[str, List[str]] = field(default_factory=dict)
    csv_min_rows: Dict[str, int] = field(default_factory=dict)

    stdout_is_number: bool = False
    stdout_exact: Optional[str] = None

    # optional: tools whitelist for this task
    allowed_tools: List[str] = field(default_factory=lambda: ["shell_exec", "python_exec", "file_write", "pip_install"])