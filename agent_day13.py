from __future__ import annotations

import os
from typing import Optional, List

from src.agent_core.bench.tasks import get_task_library, BenchTask
from src.agent_core.specs.task_spec import TaskSpec
from src.agent_core.verify.verifier import verify
from src.agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolCall, ToolResult
from src.agent_core.runtime.executor import execute_tool
from src.agent_core.llm.robust_action import robust_next_action
from src.agent_core.llm.critic import critique
from src.agent_core.llm.client import LLMClient


MAX_STEPS = 30


def reset(files: list[str]):
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def to_spec(bt: BenchTask) -> TaskSpec:
    spec = TaskSpec(task=bt.task)
    spec.allowed_tools = ["shell_exec", "python_exec", "file_write", "pip_install"]
    spec.stdout_is_number = True
    if getattr(bt, "expected_stdout", None) is not None:
        spec.stdout_exact = bt.expected_stdout
    if getattr(bt, "required_files", None):
        spec.required_files = list(bt.required_files)
    if getattr(bt, "csv_required_columns", None):
        spec.csv_required_columns = dict(bt.csv_required_columns)
    if getattr(bt, "csv_min_rows", None):
        spec.csv_min_rows = dict(bt.csv_min_rows)
    return spec


def planner(task: str) -> List[str]:
    client = LLMClient()
    sys = "Return STRICT JSON: {\"plan\":[\"...\"]} with 3-6 steps."
    raw = client.chat(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": task}],
        temperature=0,
    )
    import json
    obj = json.loads(raw)
    plan = obj.get("plan", [])
    return [str(x).strip() for x in plan if str(x).strip()][:6]


def allowed_for_phase(artifacts_ok: bool) -> List[str]:
    return ["python_exec"] if artifacts_ok else ["file_write", "shell_exec", "pip_install"]


def run(bt: BenchTask):
    spec = to_spec(bt)
    reset(spec.required_files or [])

    rm = RunManager()
    ctx = rm.start(tag=f"agent_day13_critic_{bt.task_id}")
    rm.save_text(ctx, "task.txt", bt.task)
    rm.save_json(ctx, "task_spec.json", spec.__dict__)

    plan = planner(bt.task)
    rm.save_json(ctx, "plan.json", {"plan": plan})

    last: Optional[ToolResult] = None
    last_action: Optional[ToolCall] = None
    hint = "Start."
    extra_instruction = ""

    for step in range(1, MAX_STEPS + 1):
        v_art = verify(spec, last=None, check_stdout=False)
        artifacts_ok = v_art.ok
        allowed = allowed_for_phase(artifacts_ok)

        obs = (
            f"Task:\n{bt.task}\n\n"
            f"Plan:\n- " + "\n- ".join(plan) + "\n\n"
            f"PHASE: {'COMPUTE' if artifacts_ok else 'ARTIFACTS'}\n"
            f"Allowed tools THIS STEP: {allowed}\n"
            f"Verifier hint: {hint}\n"
            f"GAPS: {v_art.gaps}\n"
            f"CRITIC_INSTRUCTION (if any): {extra_instruction}\n\n"
            "Return ONE JSON tool call only.\n"
        )
        rm.save_text(ctx, f"step_{step:02d}_obs.txt", obs)

        action = robust_next_action(rm, ctx, bt.task, obs, rules=None, allowed_tools=allowed, step=step)
        rm.save_json(ctx, f"step_{step:02d}_action.json", action.model_dump())
        last_action = action

        result = execute_tool(action, task=bt.task)
        rm.save_json(ctx, f"step_{step:02d}_result.json", result.model_dump())
        last = result

        v = verify(spec, last, check_stdout=True)
        rm.save_json(ctx, f"step_{step:02d}_verify.json", {"ok": v.ok, "hint": v.hint, "gaps": v.gaps, "messages": v.messages})

        if v.ok:
            rm.save_text(ctx, "final.txt", "DONE")
            print(f"[Day13] OK run_id={ctx.run_id}")
            return True

        # Critic generates a corrective instruction
        extra_instruction = critique(
            task=bt.task,
            action=last_action.model_dump(),
            result=result.model_dump(),
            gaps=v.gaps,
            hint=v.hint,
        )
        rm.save_text(ctx, f"step_{step:02d}_critic_instruction.txt", extra_instruction)
        hint = v.hint

    rm.save_text(ctx, "final.txt", "FAILED")
    print(f"[Day13] FAILED run_id={ctx.run_id}")
    return False


if __name__ == "__main__":
    bt = get_task_library()[1]
    run(bt)