from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ArmStats:
    n: int = 0
    total_reward: float = 0.0

    @property
    def mean(self) -> float:
        return self.total_reward / self.n if self.n > 0 else 0.0


@dataclass
class UCB1:
    arms: Dict[str, ArmStats] = field(default_factory=dict)
    c: float = 1.0

    def add_arm(self, name: str):
        if name not in self.arms:
            self.arms[name] = ArmStats()

    def select(self) -> str:
        # ensure exploration
        for k, st in self.arms.items():
            if st.n == 0:
                return k

        t = sum(st.n for st in self.arms.values())
        best_k = None
        best_ucb = -1e9
        for k, st in self.arms.items():
            ucb = st.mean + self.c * math.sqrt(2.0 * math.log(t) / st.n)
            if ucb > best_ucb:
                best_ucb = ucb
                best_k = k
        assert best_k is not None
        return best_k

    def update(self, name: str, reward: float):
        st = self.arms[name]
        st.n += 1
        st.total_reward += float(reward)