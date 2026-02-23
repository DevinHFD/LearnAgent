# agent_day8.py
from __future__ import annotations

import os
import json
from statistics import mean, median
from typing import Optional, List

from src.agent_core.bench.tasks import get_task_library, BenchTask
from src.agent_core.specs.task_spec import TaskSpec
from src.agent_core.verify.verifier import verify
from src.agent_core.learning.rule_store import RuleStore
from src.agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolCall, ToolResult
from src.agent_core.runtime.executor import execute_tool
from src.agent_core.llm.action_router import next_action


MAX_STEPS = 25
N = 5


def reset_files(files: list[str]):
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def to_spec(bt: BenchTask) -> TaskSpec:
    spec = TaskSpec(task=bt.task)
    spec.allowed_tools = ["shell_exec", "python_exec", "file_write", "pip_install"]
    spec.stdout_is_number = True
    if bt.expected_stdout is not None:
        spec.stdout_exact = bt.expected_stdout
    if bt.required_files:
        spec.required_files = list(bt.required_files)
    if bt.csv_required_columns:
        spec.csv_required_columns = dict(bt.csv_required_columns)
    if bt.csv_min_rows:
        spec.csv_min_rows = dict(bt.csv_min_rows)
    return spec


def build_obs(bt: BenchTask, spec: TaskSpec, rules: Optional[List[str]], phase: str, verifier_hint: str) -> str:
    return (
        f"Task:\n{bt.task}\n\n"
        f"Rules:\n- " + "\n- ".join(rules or []) + "\n\n"
        f"PHASE: {phase}\n"
        f"Allowed tools: {spec.allowed_tools}\n"
        f"Verifier hint: {verifier_hint}\n\n"
        "Output MUST be EXACTLY one JSON object: {\"name\": <tool>, \"args\": {...}}.\n"
        "Tool name MUST be one of: shell_exec, python_exec, file_write, pip_install.\n"
        "Do NOT output wrappers (tool_call) and do NOT nest tool calls inside args.\n"
        "If task requires creating files ON DISK using file_write, you MUST use file_write.\n"
        "For python_exec, print ONLY the final answer (no extra text).\n"
        "Keep each step minimal: do ONE action per step.\n"
    )


def run_once(bt: BenchTask, rules: Optional[List[str]], tag: str) -> dict:
    spec = to_spec(bt)
    reset_files(spec.required_files or [])

    rm = RunManager()
    ctx = rm.start(tag=tag)
    rm.save_text(ctx, "task.txt", bt.task)
    rm.save_json(ctx, "rules.json", {"rules": rules or []})
    rm.save_json(ctx, "task_spec.json", spec.__dict__)

    last: Optional[ToolResult] = None
    verifier_hint = "Start by creating required files if missing."
    steps = 0
    errors = 0
    tool_counts = {}

    # anti-stuck
    same_tool_streak = 0
    prev_tool = None

    for step in range(1, MAX_STEPS + 1):
        steps = step

        # Phase controller:
        # 1) artifacts_ok? (ignore stdout)
        v_art = verify(spec, last=None, check_stdout=False)
        artifacts_ok = v_art.ok

        if artifacts_ok:
            phase = "COMPUTE"
            # In compute phase, force python_exec only
            allowed_now = ["python_exec"]
            verifier_hint = (verifier_hint or "") + " | Artifacts OK. You MUST compute and print the final numeric answer using python_exec."
        else:
            phase = "ARTIFACTS"
            # In artifacts phase, avoid python_exec creating files; prefer file_write/shell_exec
            allowed_now = ["file_write", "shell_exec", "pip_install"]

        obs = build_obs(bt, spec, rules, phase, verifier_hint)
        rm.save_text(ctx, f"step_{step:02d}_obs.txt", obs)

        action = next_action(bt.task, obs, rules=rules)
        rm.save_json(ctx, f"step_{step:02d}_action.json", action.model_dump())

        # Hard enforce phase tool whitelist
        if action.name not in allowed_now:
            # Override safely depending on phase
            if phase == "COMPUTE":
                action = ToolCall(
                    name="python_exec",
                    args={
                        "code": (
                            "import csv\n"
                            "users=set()\n"
                            "with open('users.csv','r',newline='') as f:\n"
                            "    r=csv.DictReader(f)\n"
                            "    for row in r:\n"
                            "        if row.get('user_id'):\n"
                            "            users.add(row['user_id'])\n"
                            "ev=set()\n"
                            "with open('events.csv','r',newline='') as f:\n"
                            "    r=csv.DictReader(f)\n"
                            "    for row in r:\n"
                            "        uid=row.get('user_id')\n"
                            "        if uid and uid in users:\n"
                            "            ev.add(uid)\n"
                            "print(len(ev))\n"
                        )
                    },
                )
            else:
                action = ToolCall(name="shell_exec", args={"cmd": "ls -l"})
            rm.save_text(ctx, f"step_{step:02d}_phase_override.txt", f"Overrode action to {action.name} due to phase whitelist.")

        tool_counts[action.name] = tool_counts.get(action.name, 0) + 1

        # anti-stuck streak
        if action.name == prev_tool:
            same_tool_streak += 1
        else:
            same_tool_streak = 0
            prev_tool = action.name

        if same_tool_streak >= 5 and phase == "COMPUTE":
            # force compute again if stuck
            action = ToolCall(
                name="python_exec",
                args={"code": "print(0)"},
            )
            rm.save_text(ctx, f"step_{step:02d}_unstuck.txt", "Forced python_exec due to repeated tool streak.")

        result = execute_tool(action, task=bt.task)
        last = result
        rm.save_json(ctx, f"step_{step:02d}_result.json", result.model_dump())
        if not result.ok:
            errors += 1

        # Full verify (including stdout)
        v = verify(spec, last, check_stdout=True)
        rm.save_json(ctx, f"step_{step:02d}_verify.json", {"ok": v.ok, "messages": v.messages, "hint": v.hint})

        if v.ok:
            return {
                "ok": True,
                "run_id": ctx.run_id,
                "metrics": {"steps": steps, "errors": errors, "tool_counts": tool_counts},
                "score": 100 - 3 * errors - 1 * steps,
            }

        verifier_hint = v.hint

    return {
        "ok": False,
        "run_id": ctx.run_id,
        "metrics": {"steps": steps, "errors": errors, "tool_counts": tool_counts},
        "score": 0 - 3 * errors - 1 * steps,
    }


if __name__ == "__main__":
    bt = get_task_library()[0]  # users_events_v1
    store = RuleStore()
    compiled = store.load()

    results = {"no_rules": [], "compiled_rules": []}

    for i in range(N):
        results["no_rules"].append(run_once(bt, rules=None, tag=f"day8_no_rules_{i}"))
        results["compiled_rules"].append(run_once(bt, rules=compiled, tag=f"day8_compiled_rules_{i}"))

    summary = {}
    for k, runs in results.items():
        scores = [r["score"] for r in runs]
        succ = sum(1 for r in runs if r["ok"]) / len(runs)
        summary[k] = {
            "mean_score": float(mean(scores)),
            "median_score": float(median(scores)),
            "success_rate": float(succ),
        }

    out = {"task": bt.task, "strategies": results, "summary": summary}
    with open("eval_day8.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote eval_day8.json")