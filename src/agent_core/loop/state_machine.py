from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class Phase:
    name: str
    allowed_tools: List[str]
    instruction: str


class StateMachine:
    def __init__(self, phases: List[Phase]):
        self.phases: Dict[str, Phase] = {p.name: p for p in phases}

    def get(self, name: str) -> Phase:
        return self.phases[name]