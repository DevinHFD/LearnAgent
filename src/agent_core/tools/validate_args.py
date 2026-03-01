from __future__ import annotations

from typing import Any, Dict, Tuple


def _is_type(x: Any, t: str) -> bool:
    if t == "string":
        return isinstance(x, str)
    if t == "number":
        return isinstance(x, (int, float))
    if t == "integer":
        return isinstance(x, int) and not isinstance(x, bool)
    if t == "boolean":
        return isinstance(x, bool)
    if t == "object":
        return isinstance(x, dict)
    if t == "array":
        return isinstance(x, list)
    return True


def validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Minimal validator (enough to prevent common LLM mistakes).
    schema example:
      {"type":"object","required":["cmd"],"properties":{"cmd":{"type":"string"}}}
    """
    if schema.get("type") != "object":
        return True, "OK"

    if not isinstance(args, dict):
        return False, f"args must be object, got {type(args)}"

    required = schema.get("required", [])
    props = schema.get("properties", {})

    for k in required:
        if k not in args:
            return False, f"missing required arg: {k}"

    for k, v in args.items():
        if k in props:
            t = props[k].get("type")
            if t and not _is_type(v, t):
                return False, f"arg {k} type mismatch: expected {t}, got {type(v)}"
    return True, "OK"