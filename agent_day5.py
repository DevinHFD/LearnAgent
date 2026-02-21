from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolCall, ToolResult
from agent_core.runtime.executor import execute_tool
from src.agent_core.llm.action_router import next_action
from src.agent_core.specs.task_spec import TaskSpec
from src.agent_core.verify.verifier import verify


MAX_STEPS = 20


def reset_env_for_day5():
    # Keep this tight & explicit for benchmark reproducibility
    for f in ["users.csv", "events.csv"]:
        if os.path.exists(f):
            os.remove(f)


def build_task_spec(task: str) -> TaskSpec:
    """
    Minimal heuristic spec builder.
    Later you can replace this with an LLM-based spec builder, but verifier stays the same.
    """
    t = task.lower()
    spec = TaskSpec(task=task)

    # Required files based on mentions
    for f in ["users.csv", "events.csv", "data.csv"]:
        if f in t:
            spec.required_files.append(f)

    # CSV schema & rows for our Day5 benchmark
    # users.csv: requires user_id, 3 rows
    if "users.csv" in t:
        spec.csv_required_columns["users.csv"] = ["user_id"]
        # if task mentions "3 users" or "1,2,3" we enforce 3
        if "3 users" in t or "1,2,3" in t or "1, 2, 3" in t:
            spec.csv_min_rows["users.csv"] = 3
        else:
            spec.csv_min_rows["users.csv"] = 1

    # events.csv: requires event_id,user_id, 2 rows (in benchmark)
    if "events.csv" in t:
        # Accept either "event_id,user_id" or "user_id,event_id" is NOT acceptable—force canonical
        spec.csv_required_columns["events.csv"] = ["event_id", "user_id"]
        if "two events" in t or "(1,1)" in t or "(2,3)" in t:
            spec.csv_min_rows["events.csv"] = 2
        else:
            spec.csv_min_rows["events.csv"] = 1

    # stdout expectations
    # For benchmark, we can optionally require exact answer "2" (comment/uncomment as you like)
    spec.stdout_is_number = True
    if "number of unique users" in t and "two events" in t:
        # This benchmark should be 2 unique users with events (user_id=1 and 3)
        spec.stdout_exact = "2"

    # Tools (hard whitelist)
    spec.allowed_tools = ["shell_exec", "python_exec", "file_write", "pip_install"]
    return spec


def planner(task: str) -> List[str]:
    """
    Simple planner (LLM optional).
    To keep Day5 deterministic and avoid new failure modes, we use a template plan.
    Later you can upgrade it to an LLM planner.
    """
    return [
        "Check whether required files exist; if missing, create them using file_write with correct headers and sample rows.",
        "If files exist, open and validate headers/rows quickly (cat/head).",
        "Use python_exec to read CSVs and compute the required value.",
        "Print ONLY the final numeric answer to stdout.",
    ]


def build_observation(task: str, plan: List[str], step_hint: str, verifier_hint: str, spec: TaskSpec) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Allowed tools: {spec.allowed_tools}\n"
        f"Plan:\n- " + "\n- ".join(plan) + "\n\n"
        f"Current step hint: {step_hint}\n"
        f"Verifier hint: {verifier_hint}\n\n"
        "RULES:\n"
        "- Output EXACTLY one JSON object: {\"name\": <tool>, \"args\": {...}}.\n"
        "- Tool name MUST be one of: shell_exec, python_exec, file_write, pip_install.\n"
        "- Do NOT output tool_call wrapper, do NOT nest tool calls inside args.\n"
        "- If task requires creating files ON DISK, you MUST use file_write (not python open()).\n"
        "- Keep each action minimal: do ONE action per step.\n"
    )


def run_day5(task: str):
    reset_env_for_day5()
    spec = build_task_spec(task)
    plan = planner(task)

    rm = RunManager()
    ctx = rm.start(tag="agent_day5")
    rm.save_text(ctx, "task.txt", task)
    rm.save_json(ctx, "task_spec.json", spec.__dict__)
    rm.save_json(ctx, "plan.json", {"plan": plan})

    history: List[Dict[str, Any]] = []
    last: Optional[ToolResult] = None
    verifier_hint = "Start by checking/creating required files."

    for step_i in range(1, MAX_STEPS + 1):
        step_hint = plan[min(step_i - 1, len(plan) - 1)]
        obs = build_observation(task, plan, step_hint, verifier_hint, spec)
        rm.save_text(ctx, f"step_{step_i:02d}_observation.txt", obs)

        action = next_action(task, obs, rules=None)  # keep rules None; Day5 uses verifier-first hints
        rm.save_json(ctx, f"step_{step_i:02d}_action.json", action.model_dump())

        # Hard enforce allowed tools (fail fast / safer)
        if action.name not in spec.allowed_tools:
            # Replace with a safe shell_exec that shows current dir (debug)
            action = ToolCall(name="shell_exec", args={"cmd": "pwd && ls -l"})
            rm.save_text(ctx, f"step_{step_i:02d}_tool_whitelist.txt", "Blocked invalid tool; replaced with safe shell_exec.")

        result = execute_tool(action, task=task)
        rm.save_json(ctx, f"step_{step_i:02d}_result.json", result.model_dump())

        history.append({"step": step_i, "action": action.model_dump(), "result": result.model_dump()})
        last = result

        # After every tool execution, run verifier
        v = verify(spec, last)
        rm.save_json(ctx, f"step_{step_i:02d}_verify.json", {"ok": v.ok, "messages": v.messages, "hint": v.hint})

        if v.ok:
            rm.save_text(ctx, "final.txt", "DONE")
            rm.save_json(ctx, "history.json", {"history": history})
            print(f"Day5 OK run_id: {ctx.run_id}")
            return True, ctx.run_id

        verifier_hint = v.hint

    rm.save_text(ctx, "final.txt", "FAILED: max steps reached")
    rm.save_json(ctx, "history.json", {"history": history})
    print(f"Day5 FAILED run_id: {ctx.run_id}")
    return False, ctx.run_id


if __name__ == "__main__":
    task = (
        "Create users.csv and events.csv on disk (must use file_write). "
        "users.csv must have column user_id with 3 users: 1,2,3. "
        "events.csv must have columns event_id,user_id with two events: (1,1) and (2,3). "
        "Then print the number of unique users who have at least one event."
    )
    ok, run_id = run_day5(task)