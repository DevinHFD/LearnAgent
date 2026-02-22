from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class RuleStore:
    path: str = "memory/compiled_rules.json"
    rules: List[str] = field(default_factory=list)

    def load(self) -> List[str]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.rules = list(obj.get("rules", []))
        return self.rules

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"rules": self.rules}, f, ensure_ascii=False, indent=2)

    def add(self, new_rules: List[str]):
        s = set(self.rules)
        for r in new_rules:
            r = (r or "").strip()
            if r and r not in s:
                self.rules.append(r)
                s.add(r)