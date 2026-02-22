from __future__ import annotations

import glob
import json
import os
from typing import List, Dict, Any

from src.agent_core.learning.rule_store import RuleStore
from src.agent_core.llm.client import LLMClient  # 你已有的 client
# 如果你的 client 类名不同，把这一行改掉即可


PROMPT_SYSTEM = (
    "You are a rule miner. Extract GENERAL, reusable rules from successful agent episodes.\n"
    "Return STRICT JSON: {\"rules\": [\"...\"]}.\n"
    "Rules must be actionable, short, and not tied to specific file names unless generic (e.g., 'CSV').\n"
    "Do not include more than 8 rules."
)


def load_success_histories(runs_dir: str = "runs") -> List[Dict[str, Any]]:
    episodes = []
    for run in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        final_path = os.path.join(run, "final.txt")
        hist_path = os.path.join(run, "history.json")
        if not os.path.exists(final_path) or not os.path.exists(hist_path):
            continue
        with open(final_path, "r", encoding="utf-8") as f:
            final = f.read().strip()
        if "DONE" not in final:
            continue
        with open(hist_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        episodes.append({"run": os.path.basename(run), "history": obj.get("history", [])})
    return episodes[-20:]  # 最近20条成功记录


def mine_rules(episodes: List[Dict[str, Any]]) -> List[str]:
    client = LLMClient()
    payload = json.dumps({"episodes": episodes}, ensure_ascii=False)[:12000]

    raw = client.chat(
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=0,
    )
    try:
        obj = json.loads(raw)
        rules = obj.get("rules", [])
        return [str(r).strip() for r in rules if str(r).strip()]
    except Exception:
        return []


if __name__ == "__main__":
    eps = load_success_histories()
    if not eps:
        print("[Day7] No successful histories found. Run Day6/Day5 first.")
        raise SystemExit(1)

    rules = mine_rules(eps)
    store = RuleStore()
    store.load()
    store.add(rules)
    store.save()

    print("[Day7] mined rules:")
    for r in rules:
        print("-", r)
    print(f"[Day7] compiled rules saved to {store.path}")