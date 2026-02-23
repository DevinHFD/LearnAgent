from __future__ import annotations

import os
from typing import Optional, List

from src.agent_core.bench.tasks import get_task_library, BenchTask
from src.agent_core.specs.task_spec import TaskSpec
from src.agent_core.verify.verifier import verify
from src.agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolCall, ToolResult
from src.agent_core.runtime.executor import execute_tool
from src.agent_core.llm.client import LLMClient
from src.agent_core.llm.action_router import next_action


MAX_STEPS = 30


def reset(files: list[str]):
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


def planner_llm(task: str) -> List[str]:
    client = LLMClient()
    sys = (
        "You are a planner. Produce a short step-by-step plan (3-6 steps).\n"
        "Return STRICT JSON: {\"plan\": [\"...\"]}."
    )
    raw = client.chat(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": task}],
        temperature=0,
    )
    try:
        import json
        obj = json.loads(raw)
        plan = obj.get("plan", [])
        return [str(x).strip() for x in plan if str(x).strip()][:6] or ["Create required files", "Compute answer"]
    except Exception:
        return ["Create required files", "Compute answer"]


def run(bt: BenchTask):
    spec = to_spec(bt)
    reset(spec.required_files or [])

    rm = RunManager()
    ctx = rm.start(tag=f"agent_day9_{bt.task_id}")
    rm.save_text(ctx, "task.txt", bt.task)
    rm.save_json(ctx, "task_spec.json", spec.__dict__)

    plan = planner_llm(bt.task)
    rm.save_json(ctx, "plan.json", {"plan": plan})

    last: Optional[ToolResult] = None
    verifier_hint = "Start."

    for step in range(1, MAX_STEPS + 1):
        obs = (
            f"Task:\n{bt.task}\n\n"
            f"Plan:\n- " + "\n- ".join(plan) + "\n\n"
            f"Allowed tools: {spec.allowed_tools}\n"
            f"Verifier hint: {verifier_hint}\n\n"
            "Executor: choose ONE tool call JSON only.\n"
            "No wrappers, no nesting, no other tool names.\n"
        )
        rm.save_text(ctx, f"step_{step:02d}_obs.txt", obs)

        action = next_action(bt.task, obs, rules=None)
        rm.save_json(ctx, f"step_{step:02d}_action.json", action.model_dump())

        if action.name not in spec.allowed_tools:
            action = ToolCall(name="shell_exec", args={"cmd": "pwd && ls -l"})

        result = execute_tool(action, task=bt.task)
        rm.save_json(ctx, f"step_{step:02d}_result.json", result.model_dump())
        last = result

        v = verify(spec, last)
        rm.save_json(ctx, f"step_{step:02d}_verify.json", {"ok": v.ok, "messages": v.messages, "hint": v.hint})
        if v.ok:
            rm.save_text(ctx, "final.txt", "DONE")
            print(f"[Day9] OK run_id={ctx.run_id}")
            return True
        verifier_hint = v.hint

    rm.save_text(ctx, "final.txt", "FAILED")
    print(f"[Day9] FAILED run_id={ctx.run_id}")
    return False


if __name__ == "__main__":
    bt = get_task_library()[0]
    run(bt)