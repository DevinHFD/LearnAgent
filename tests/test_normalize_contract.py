import json

# 如果你已经有 normalize_toolcall()，把 import 改成你的真实路径
# from src.agent_core.llm.normalize import normalize_toolcall

def normalize_toolcall(obj: dict) -> dict:
    """
    Minimal placeholder.
    Replace with your real normalize_toolcall if you already implemented it.
    """
    if obj.get("name") == "tool_call" and isinstance(obj.get("args"), dict):
        inner = obj["args"]
        if "name" in inner and "args" in inner:
            obj = inner
    if obj.get("name") == "python_exec" and isinstance(obj.get("args"), dict):
        a = obj["args"]
        if "name" in a and "args" in a:
            obj = a
    return obj


def test_unwrap_tool_call():
    raw = {"name": "tool_call", "args": {"name": "shell_exec", "args": {"cmd": "ls"}}}
    obj = normalize_toolcall(raw)
    assert obj["name"] == "shell_exec"
    assert obj["args"]["cmd"] == "ls"


def test_unwrap_nested_inside_python_exec():
    raw = {"name": "python_exec", "args": {"name": "file_write", "args": {"path": "a.txt", "content": "hi"}}}
    obj = normalize_toolcall(raw)
    assert obj["name"] == "file_write"
    assert obj["args"]["path"] == "a.txt"