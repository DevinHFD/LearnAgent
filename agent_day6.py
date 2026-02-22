from __future__ import annotations

import os
from typing import Optional

from src.agent_core.bench.tasks import get_task_library, BenchTask
from src.agent_core.specs.task_spec import TaskSpec
from src.agent_core.verify.verifier import verify
from src.agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolCall, ToolResult
from src.agent_core.runtime.executor import execute_tool
from src.agent_core.llm.action_router import next_action


MAX_STEPS = 25


def reset_workspace(files: list[str]):
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def task_to_spec(bt: BenchTask) -> TaskSpec:
    spec = TaskSpec(task=bt.task)
    spec.allowed_tools = ["shell_exec", "python_exec", "file_write", "pip_install"]
    if bt.required_files:
        spec.required_files = list(bt.required_files)
    if bt.csv_required_columns:
        spec.csv_required_columns = dict(bt.csv_required_columns)
    if bt.csv_min_rows:
        spec.csv_min_rows = dict(bt.csv_min_rows)
    spec.stdout_is_number = True
    if bt.expected_stdout is not None:
        spec.stdout_exact = bt.expected_stdout
    return spec


def build_observation(bt: BenchTask, spec: TaskSpec, verifier_hint: str) -> str:
    return (
        f"TaskID: {bt.task_id}\n"
        f"Task:\n{bt.task}\n\n"
        f"Allowed tools: {spec.allowed_tools}\n"
        f"Verifier hint: {verifier_hint}\n\n"
        "Constraints:\n"
        "- Output EXACTLY one JSON object: {\"name\": <tool>, \"args\": {...}}.\n"
        "- Tool must be one of: shell_exec, python_exec, file_write, pip_install.\n"
        "- If task says CREATE ON DISK using file_write, you MUST use file_write.\n"
        "- Keep actions minimal: ONE tool call per step.\n"
        "- For python_exec, print ONLY the final answer (no extra text).\n"
    )


def run_task(bt: BenchTask):
    spec = task_to_spec(bt)

    reset_workspace(spec.required_files or [])

    rm = RunManager()
    ctx = rm.start(tag=f"agent_day6_{bt.task_id}")
    rm.save_text(ctx, "task.txt", bt.task)
    rm.save_json(ctx, "task_spec.json", spec.__dict__)

    history = []
    last: Optional[ToolResult] = None
    verifier_hint = "Start by checking/creating required files."

    for step in range(1, MAX_STEPS + 1):
        obs = build_observation(bt, spec, verifier_hint)
        rm.save_text(ctx, f"step_{step:02d}_obs.txt", obs)

        action = next_action(bt.task, obs, rules=None)
        rm.save_json(ctx, f"step_{step:02d}_action.json", action.model_dump())

        if action.name not in spec.allowed_tools:
            action = ToolCall(name="shell_exec", args={"cmd": "pwd && ls -l"})
            rm.save_text(ctx, f"step_{step:02d}_whitelist.txt", "Blocked invalid tool; replaced with safe shell_exec.")

        result = execute_tool(action, task=bt.task)
        rm.save_json(ctx, f"step_{step:02d}_result.json", result.model_dump())
        history.append({"step": step, "action": action.model_dump(), "result": result.model_dump()})
        last = result

        v = verify(spec, last)
        rm.save_json(ctx, f"step_{step:02d}_verify.json", {"ok": v.ok, "messages": v.messages, "hint": v.hint})

        if v.ok:
            rm.save_text(ctx, "final.txt", "DONE")
            rm.save_json(ctx, "history.json", {"history": history})
            print(f"[Day6] OK task={bt.task_id} run_id={ctx.run_id}")
            return True

        verifier_hint = v.hint

    rm.save_text(ctx, "final.txt", "FAILED max steps")
    rm.save_json(ctx, "history.json", {"history": history})
    print(f"[Day6] FAILED task={bt.task_id} run_id={ctx.run_id}")
    return False


if __name__ == "__main__":
    lib = get_task_library()
    # 跑前三个任务（可自行改顺序/数量）
    for bt in lib:
        run_task(bt)