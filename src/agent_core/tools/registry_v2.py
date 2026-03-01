from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List


@dataclass
class ToolSpec:
    name: str
    description: str
    args_schema: Dict[str, Any]  # JSON-schema-like
    fn: Callable[[Dict[str, Any]], Any]
    safety_notes: str = ""


class ToolRegistryV2:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def to_prompt(self) -> str:
        # Short, structured tool card list for LLM
        lines = []
        for name in self.names():
            t = self._tools[name]
            lines.append(f"- {t.name}: {t.description}")
            lines.append(f"  args_schema: {t.args_schema}")
            if t.safety_notes:
                lines.append(f"  safety: {t.safety_notes}")
        return "\n".join(lines)