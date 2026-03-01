from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any

from ..bench.tasks import get_task_library, BenchTask
from ..learning.rule_store import RuleStore


@dataclass
class CurriculumConfig:
    episodes_per_task: int = 3
    strategies: List[str] = None

    def __post_init__(self):
        if self.strategies is None:
            self.strategies = ["baseline", "beam", "critic"]


def write_markdown_report(path: str, summary: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# LearnAgent Curriculum Report\n"]
    lines.append("## Summary\n")
    lines.append("```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))