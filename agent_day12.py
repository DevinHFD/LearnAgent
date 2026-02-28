from __future__ import annotations

import os
from typing import Optional, List

from src.agent_core.bench.tasks import get_task_library, BenchTask
from src.agent_core.specs.task_spec import TaskSpec
from src.agent_core.verify.verifier import verify
from src.agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolCall, ToolResult
from src.agent_core.runtime.executor import execute_tool
from src.agent_core.search.beam import propose_candidates, score_by_gaps


MAX_STEPS = 30
K = 4


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


def allowed_for_phase(artifacts_ok: bool) -> List[str]:
    return ["python_exec"] if artifacts_ok else ["file_write", "shell_exec", "pip_install"]


def run(bt: BenchTask):
    spec = to_spec(bt)
    reset(spec.required_files or [])

    rm = RunManager()
    ctx = rm.start(tag=f"agent_day12_beam_{bt.task_id}")
    rm.save_text(ctx, "task.txt", bt.task)
    rm.save_json(ctx, "task_spec.json", spec.__dict__)

    last: Optional[ToolResult] = None
    hint = "Start."

    for step in range(1, MAX_STEPS + 1):
        v_art = verify(spec, last=None, check_stdout=False)
        artifacts_ok = v_art.ok
        allowed = allowed_for_phase(artifacts_ok)

        obs = (
            f"Task:\n{bt.task}\n\n"
            f"PHASE: {'COMPUTE' if artifacts_ok else 'ARTIFACTS'}\n"
            f"Allowed tools THIS STEP: {allowed}\n"
            f"Hint: {hint}\n"
            f"GAPS: {v_art.gaps}\n\n"
            "Return JSON tool call only.\n"
        )
        rm.save_text(ctx, f"step_{step:02d}_obs.txt", obs)

        cands = propose_candidates(bt.task, obs, k=K)
        rm.save_json(ctx, f"step_{step:02d}_candidates.json", {"candidates": cands})

        best_action: Optional[ToolCall] = None
        best_score = -1e9
        best_reason = "none"

        # One-step lookahead: execute candidate, verify artifacts/stdout, pick best
        for i, c in enumerate(cands):
            if c["name"] not in allowed:
                continue
            action = ToolCall.model_validate(c)
            result = execute_tool(action, task=bt.task)

            v_full = verify(spec, result, check_stdout=True)
            s = score_by_gaps(v_full.gaps)

            rm.save_json(ctx, f"step_{step:02d}_cand_{i}_eval.json", {
                "action": action.model_dump(),
                "result": result.model_dump(),
                "verify_ok": v_full.ok,
                "gaps": v_full.gaps,
                "score": s,
            })

            if v_full.ok:
                rm.save_text(ctx, "final.txt", "DONE")
                print(f"[Day12] OK run_id={ctx.run_id}")
                return True

            if s > best_score:
                best_score = s
                best_action = action
                best_reason = f"best_score={s}"

        if best_action is None:
            best_action = ToolCall(name="shell_exec", args={"cmd": "pwd && ls -l"})
            best_reason = "fallback"

        rm.save_text(ctx, f"step_{step:02d}_chosen.txt", best_reason)
        last = execute_tool(best_action, task=bt.task)
        rm.save_json(ctx, f"step_{step:02d}_result.json", last.model_dump())

        v_after = verify(spec, last, check_stdout=True)
        rm.save_json(ctx, f"step_{step:02d}_verify.json", {"ok": v_after.ok, "hint": v_after.hint, "gaps": v_after.gaps})
        hint = v_after.hint

    rm.save_text(ctx, "final.txt", "FAILED")
    print(f"[Day12] FAILED run_id={ctx.run_id}")
    return False


if __name__ == "__main__":
    bt = get_task_library()[0]
    run(bt)