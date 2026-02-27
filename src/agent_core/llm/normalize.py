from __future__ import annotations

from typing import Any, Dict


ALLOWED = {"shell_exec", "python_exec", "file_write", "pip_install"}


def normalize_toolcall_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize common wrapper/nesting mistakes:
    - {"name":"tool_call","args":{"name":"shell_exec","args":{...}}} => unwrap
    - {"name":"python_exec","args":{"name":"file_write","args":{...}}} => unwrap
    """
    if not isinstance(obj, dict):
        return {}

    name = obj.get("name")
    args = obj.get("args")

    # unwrap tool_call wrapper
    if name == "tool_call" and isinstance(args, dict) and "name" in args and "args" in args:
        obj = args
        name = obj.get("name")
        args = obj.get("args")

    # unwrap nested inside python_exec args
    if name == "python_exec" and isinstance(args, dict) and "name" in args and "args" in args:
        obj = args
        name = obj.get("name")
        args = obj.get("args")

    # if the model returned {"name": "...", "args":{"name":"file_write"...}} again
    if isinstance(args, dict) and "name" in args and "args" in args and name not in ALLOWED:
        obj = args

    # ensure minimal shape
    if not isinstance(obj.get("name"), str) or not isinstance(obj.get("args"), dict):
        return {}

    # keep only required keys
    return {"name": obj["name"], "args": obj["args"]}