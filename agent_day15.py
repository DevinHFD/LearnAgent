from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Any

from src.agent_core.bench.tasks import get_task_library, BenchTask
from src.agent_core.learning.curriculum import CurriculumConfig, write_markdown_report
from src.agent_core.learning.rule_store import RuleStore

from agent_day11 import run_one as run_baseline
from agent_day12 import run as run_beam
from agent_day13 import run as run_critic
from agent_day7 import load_success_histories, mine_rules  # 用你 Day7 的规则挖掘


def run_strategy(bt: BenchTask, strategy: str) -> bool:
    if strategy == "baseline":
        return run_baseline(bt.task, required_files=getattr(bt, "required_files", None))
    if strategy == "beam":
        return run_beam(bt)
    if strategy == "critic":
        return run_critic(bt)
    raise ValueError(strategy)


if __name__ == "__main__":
    cfg = CurriculumConfig(episodes_per_task=3, strategies=["baseline", "beam", "critic"])
    tasks = get_task_library()

    results: Dict[str, Any] = {
        "config": cfg.__dict__,
        "tasks": [],
        "by_strategy": {},
    }

    by_strategy = defaultdict(lambda: {"ok": 0, "n": 0})

    for bt in tasks:
        bt_res = {"task_id": bt.task_id, "runs": []}
        for strat in cfg.strategies:
            for i in range(cfg.episodes_per_task):
                ok = run_strategy(bt, strat)
                bt_res["runs"].append({"strategy": strat, "i": i, "ok": ok})
                by_strategy[strat]["n"] += 1
                by_strategy[strat]["ok"] += (1 if ok else 0)
        results["tasks"].append(bt_res)

    results["by_strategy"] = {
        k: {"success_rate": (v["ok"] / v["n"] if v["n"] else 0.0), "n": v["n"]}
        for k, v in by_strategy.items()
    }

    with open("eval_day15_curriculum.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- 自动更新 compiled rules（从成功 episode 挖掘）----
    eps = load_success_histories()
    new_rules = mine_rules(eps) if eps else []
    store = RuleStore()
    store.load()
    store.add(new_rules)
    store.save()

    results["mined_rules_added"] = new_rules

    # Markdown report
    write_markdown_report("docs/day15_report.md", results)

    print(json.dumps(results["by_strategy"], ensure_ascii=False, indent=2))
    print("Wrote eval_day15_curriculum.json")
    print("Wrote docs/day15_report.md")
    print(f"Updated compiled rules: {store.path}")