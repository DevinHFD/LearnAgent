from __future__ import annotations

import json
from typing import List, Optional, Tuple

from ..schemas.tool import ToolCall
from ..runtime.run_manager import RunManager
from ..llm.action_router import next_action as base_next_action
from .json_mode import repair_to_toolcall_json
from .normalize import normalize_toolcall_obj, ALLOWED


def robust_next_action(
    rm: RunManager,
    ctx,
    task: str,
    observation: str,
    rules: Optional[List[str]],
    allowed_tools: List[str],
    step: int,
) -> ToolCall:
    """
    1) call base next_action()
    2) log raw and parsed
    3) normalize wrapper/nesting
    4) if invalid => LLM repair => normalize => validate ToolCall
    """
    # base_next_action already returns ToolCall in your codebase,
    # but we still want to log a "raw-like" view to debug.
    try:
        action = base_next_action(task, observation, rules=rules)
        rm.save_json(ctx, f"step_{step:02d}_action_router.json", action.model_dump())
        if action.name in allowed_tools:
            return action
    except Exception as e:
        rm.save_text(ctx, f"step_{step:02d}_action_router_error.txt", repr(e))

    # If we are here, we need repair. We do a repair on a minimal "raw".
    raw_guess = ""
    try:
        raw_guess = json.dumps(getattr(action, "model_dump", lambda: {})(), ensure_ascii=False)
    except Exception:
        raw_guess = ""

    rm.save_text(ctx, f"step_{step:02d}_action_raw_guess.txt", raw_guess)

    fixed_obj = repair_to_toolcall_json(raw_guess)
    rm.save_json(ctx, f"step_{step:02d}_action_repair_obj.json", fixed_obj)

    fixed_obj = normalize_toolcall_obj(fixed_obj)
    rm.save_json(ctx, f"step_{step:02d}_action_normalized.json", fixed_obj)

    name = fixed_obj.get("name")
    if name not in ALLOWED or name not in allowed_tools:
        # safest fallback
        return ToolCall(name="shell_exec", args={"cmd": "pwd && ls -l"})

    return ToolCall.model_validate(fixed_obj)