from collections import Counter
from typing import Dict, Any, List

def compute_metrics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_counts = Counter()
    errors = 0
    repeats = 0
    guardrail_overrides = 0

    last_sig = None
    for h in history:
        action = h.get("action", {})
        result = h.get("result", {})

        name = action.get("name")
        tool_counts[name] += 1

        if not result.get("ok", False):
            errors += 1

        sig = str(action)
        if sig == last_sig:
            repeats += 1
        last_sig = sig

        # if you store override events in history, count them here.
        # otherwise, we’ll track via run files later (Day4+).

    return {
        "steps": len(history),
        "tool_counts": dict(tool_counts),
        "errors": errors,
        "repeats": repeats,
        "guardrail_overrides": guardrail_overrides,
    }
