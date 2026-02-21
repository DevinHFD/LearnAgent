import json
from statistics import mean, median
from pathlib import Path

from agent_day3 import run_episode, compile_rules   # reuse your Day3 engine
from agent_core.llm.reflection import reflect
from agent_core.memory.episodic import EpisodicMemory
from agent_core.eval.metrics import compute_metrics
from agent_core.eval.scoring import score_run
import os

def reset_env():
    for f in ["users.csv", "events.csv"]:
        if os.path.exists(f):
            os.remove(f)
N = 1

def run_strategy(task: str, rules, tag: str = "agent_day4", meta: dict | None = None) -> dict:
    reset_env()
    try:
        hist, ok, run_id = run_episode(task, rules=rules, tag=tag, meta=meta)
        memory.save_episode(task, hist, meta=meta)
        m = compute_metrics(hist)
        s = score_run(m, ok)
        return {"ok": ok, "run_id": run_id, "metrics": m, "score": s}
    except Exception as e:
        return {
            "ok": False,
            "run_id": None,
            "metrics": {"steps": 0, "tool_counts": {}, "errors": 1, "repeats": 0, "guardrail_overrides": 0},
            "score": -999.0,
            "error": repr(e),
        }

if __name__ == "__main__":
    memory = EpisodicMemory()

    task = "Read users.csv and events.csv, then print the number of unique users who have at least one event. If either file is missing, CREATE the missing file(s) ON DISK using file_write (do not use in-memory samples),then read them from disk to compute the answer."

    # strategy 1: no rules
    results = {"task": task, "strategies": {}}

    print("Running strategy: no_rules")
    r1 = [run_strategy(task, rules=None, tag=f"day4_no_rules_{i}",meta={"strategy": "no_rules", "trial": i}) for i in range(N)]
    results["strategies"]["no_rules"] = r1

    # reflection from memory so far (optional)
    episodes = memory.load_all()
    try:
        learned = reflect(episodes) if episodes else {"rules": []}
    except Exception as e:
        print("Reflection failed, fallback to empty rules:", repr(e))
        learned = {"rules": []}

    # strategy 2: compiled base rules
    base_rules = compile_rules(task, learned.get("rules", []))
    print("Running strategy: compiled_rules")
    r2 = [run_strategy(task, rules=base_rules, tag=f"day4_compiled_rules_{i}",meta={"strategy": "compiled_rules", "trial": i}) for i in range(N)]
    results["strategies"]["compiled_rules"] = r2

    # summarize
    summary = {}
    for k, runs in results["strategies"].items():
        scores = [x["score"] for x in runs]
        oks = [x["ok"] for x in runs]
        summary[k] = {
            "mean_score": mean(scores),
            "median_score": median(scores),
            "success_rate": sum(oks) / len(oks),
        }

    results["summary"] = summary

    Path("eval_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote eval_results.json")
