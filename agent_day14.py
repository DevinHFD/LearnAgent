from __future__ import annotations

import json
from typing import Dict, Any

from src.agent_core.bench.tasks import get_task_library, BenchTask
from src.agent_core.learning.bandit import UCB1
from src.agent_core.learning.rule_store import RuleStore

# Reuse Day12/Day13 runners as "strategies"
from agent_day12 import run as run_beam
from agent_day13 import run as run_critic
from agent_day11 import run_one as run_baseline


def reward(ok: bool) -> float:
    return 1.0 if ok else 0.0


if __name__ == "__main__":
    tasks = get_task_library()
    store = RuleStore()
    _ = store.load()

    bandit = UCB1(c=0.7)
    bandit.add_arm("baseline")
    bandit.add_arm("beam")
    bandit.add_arm("critic")

    results: Dict[str, Any] = {"runs": []}

    # run 9 episodes across tasks (round-robin)
    for i in range(9):
        bt: BenchTask = tasks[i % len(tasks)]
        arm = bandit.select()

        if arm == "baseline":
            ok = run_baseline(bt.task, required_files=getattr(bt, "required_files", None))
        elif arm == "beam":
            ok = run_beam(bt)
        else:
            ok = run_critic(bt)

        r = reward(ok)
        bandit.update(arm, r)

        results["runs"].append({"i": i, "task_id": bt.task_id, "strategy": arm, "ok": ok, "reward": r})
        results["bandit"] = {k: {"n": st.n, "mean": st.mean} for k, st in bandit.arms.items()}

        print(f"[Day14] ep={i} task={bt.task_id} arm={arm} ok={ok} bandit={results['bandit']}")

    with open("eval_day14_bandit.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Wrote eval_day14_bandit.json")