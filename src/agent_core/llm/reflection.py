# src/agent_core/llm/reflection.py
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel, Field
from typing import List, Any

from .client import LLMClient

client = LLMClient()

class ReflectionOut(BaseModel):
    success_patterns: List[str] = Field(default_factory=list)
    failure_patterns: List[str] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)

SYSTEM = (
    "You are an agent learning from experience.\n"
    "Extract concise, actionable, generalizable rules.\n"
    "Return ONLY valid JSON.\n"
    f"Schema: {ReflectionOut.model_json_schema()}\n"
)

def _summarize_episodes(episodes: Any, max_steps_per_episode: int = 8):
    """
    Reduce token load: keep only the last N steps of each episode, and only key fields.
    """
    out = []
    for ep in episodes:
        hist = ep.get("history", [])
        hist = hist[-max_steps_per_episode:]
        slim = []
        for h in hist:
            slim.append({
                "step": h.get("step"),
                "action": h.get("action", {}),
                "result": {
                    "name": h.get("result", {}).get("name"),
                    "ok": h.get("result", {}).get("ok"),
                    "output": (h.get("result", {}).get("output") or "")[:300],
                    "error": (h.get("result", {}).get("error") or "")[:300],
                }
            })
        out.append({"task": ep.get("task", ""), "history": slim})
    return out

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def reflect(episodes) -> dict:
    slim = _summarize_episodes(episodes)

    raw = client.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": json.dumps(slim, ensure_ascii=False)}],
        temperature=0,
    )

    if not raw or not raw.strip():
        raise ValueError("Empty reflection output from LLM")

    try:
        out = ReflectionOut.model_validate(json.loads(raw))
        return out.model_dump()
    except Exception:
        fix = client.chat(
            [{"role": "system", "content": "Fix the JSON. Return ONLY corrected JSON."},
             {"role": "user", "content": raw}],
            temperature=0,
        )
        if not fix or not fix.strip():
            raise ValueError("Empty reflection fix output from LLM")

        out = ReflectionOut.model_validate(json.loads(fix))
        return out.model_dump()
