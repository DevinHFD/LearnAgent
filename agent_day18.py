from __future__ import annotations

import json
from typing import Optional

from src.agent_core.bench.tasks import get_task_library
from src.agent_core.specs.task_spec import TaskSpec
from src.agent_core.verify.verifier import verify
from src.agent_core.runtime.run_manager import RunManager
from src.agent_core.schemas.tool import ToolResult
from src.agent_core.runtime.executor import execute_tool
from src.agent_core.llm.robust_action import robust_next_action
from src.agent_core.memory.sqlite_store import SQLiteMemoryStore


MAX_STEPS = 25


def to_spec(task: str) -> TaskSpec:
    s = TaskSpec(task=task)
    s.allowed_tools = ["shell_exec","python_exec","file_write","pip_install"]
    s.stdout_is_number = True
    return s


if __name__ == "__main__":
    bt = get_task_library()[0]
    spec = to_spec(bt.task)
    if getattr(bt, "required_files", None):
        spec.required_files = list(bt.required_files)
    if getattr(bt, "csv_required_columns", None):
        spec.csv_required_columns = dict(bt.csv_required_columns)
    if getattr(bt, "csv_min_rows", None):
        spec.csv_min_rows = dict(bt.csv_min_rows)

    rm = RunManager()
    ctx = rm.start(tag="agent_day18_sqlite_memory")

    store = SQLiteMemoryStore()
    last: Optional[ToolResult] = None
    hint = "Start."

    ok = False
    history = []

    for step in range(1, MAX_STEPS+1):
        v_art = verify(spec, last=None, check_stdout=False)
        allowed = ["python_exec"] if v_art.ok else ["file_write","shell_exec","pip_install"]

        obs = (
            f"Task:\n{bt.task}\n\nAllowed tools: {allowed}\n"
            f"Verifier hint: {hint}\nGAPS: {json.dumps(v_art.gaps, ensure_ascii=False)}\n\n"
            "Return ONE JSON tool call only.\n"
        )
        action = robust_next_action(rm, ctx, bt.task, obs, rules=None, allowed_tools=allowed, step=step)
        result = execute_tool(action, task=bt.task)
        last = result

        history.append({"step": step, "action": action.model_dump(), "result": result.model_dump()})

        v = verify(spec, last, check_stdout=True)
        hint = v.hint
        if v.ok:
            ok = True
            break

    payload = {"task": bt.task, "ok": ok, "history": history}
    store.add_episode(ctx.run_id, bt.task, ok, payload)

    print(json.dumps({"ok": ok, "run_id": ctx.run_id, "db": store.path}, ensure_ascii=False))