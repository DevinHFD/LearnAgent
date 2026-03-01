from __future__ import annotations

import json
import os
from typing import Optional

from src.agent_core.bench.tasks import get_task_library
from src.agent_core.specs.task_spec import TaskSpec
from src.agent_core.verify.verifier import verify
from src.agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolResult, ToolCall
from src.agent_core.runtime.executor import execute_tool
from src.agent_core.llm.robust_action import robust_next_action
from src.agent_core.loop.state_machine import StateMachine, Phase


MAX_STEPS = 30


def to_spec(task: str) -> TaskSpec:
    s = TaskSpec(task=task)
    s.allowed_tools = ["shell_exec","python_exec","file_write","pip_install"]
    s.stdout_is_number = True
    return s


def reset(files: list[str]):
    for f in files:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    bt = get_task_library()[0]
    spec = to_spec(bt.task)
    if getattr(bt, "required_files", None):
        spec.required_files = list(bt.required_files)
    if getattr(bt, "csv_required_columns", None):
        spec.csv_required_columns = dict(bt.csv_required_columns)
    if getattr(bt, "csv_min_rows", None):
        spec.csv_min_rows = dict(bt.csv_min_rows)

    reset(spec.required_files or [])

    sm = StateMachine([
        Phase("ARTIFACTS", ["file_write","shell_exec","pip_install"], "Fix missing files/schema/rows on disk using file_write."),
        Phase("COMPUTE", ["python_exec"], "Read files from disk, compute final answer, print ONLY the number."),
    ])

    rm = RunManager()
    ctx = rm.start(tag="agent_day17_state_machine")
    rm.save_text(ctx, "task.txt", bt.task)
    rm.save_json(ctx, "task_spec.json", spec.__dict__)

    last: Optional[ToolResult] = None
    hint = "Start."

    for step in range(1, MAX_STEPS+1):
        v_art = verify(spec, last=None, check_stdout=False)
        phase_name = "COMPUTE" if v_art.ok else "ARTIFACTS"
        phase = sm.get(phase_name)

        obs = (
            f"Task:\n{bt.task}\n\n"
            f"PHASE: {phase.name}\n"
            f"Allowed tools: {phase.allowed_tools}\n"
            f"Phase instruction: {phase.instruction}\n"
            f"Verifier hint: {hint}\n"
            f"GAPS: {json.dumps(v_art.gaps, ensure_ascii=False)}\n\n"
            "Return ONE JSON tool call only.\n"
        )
        rm.save_text(ctx, f"step_{step:02d}_obs.txt", obs)

        action = robust_next_action(rm, ctx, bt.task, obs, rules=None, allowed_tools=phase.allowed_tools, step=step)
        rm.save_json(ctx, f"step_{step:02d}_action.json", action.model_dump())

        last = execute_tool(action, task=bt.task)
        rm.save_json(ctx, f"step_{step:02d}_result.json", last.model_dump())

        v = verify(spec, last, check_stdout=True)
        rm.save_json(ctx, f"step_{step:02d}_verify.json", {"ok": v.ok, "hint": v.hint, "gaps": v.gaps, "messages": v.messages})
        if v.ok:
            rm.save_text(ctx, "final.txt", "DONE")
            print(json.dumps({"ok": True, "run_id": ctx.run_id}, ensure_ascii=False))
            break
        hint = v.hint