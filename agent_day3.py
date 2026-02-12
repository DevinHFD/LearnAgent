# agent_day3.py
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from agent_core.runtime.run_manager import RunManager
from agent_core.runtime.executor import execute_tool
from agent_core.llm.action_router import next_action
from agent_core.memory.episodic import EpisodicMemory
from agent_core.llm.reflection import reflect
from agent_core.runtime.guardrails import Guardrails
from agent_core.schemas.tool import ToolCall, ToolResult

MAX_STEPS = 12
REPEAT_ACTION_LIMIT = 2  # after 2 repeats, we inject anti-stuck notice (guardrails may also intervene)


# ---------------- DoneSpec: task-agnostic stop criteria ----------------
@dataclass
class DoneSpec:
    # If file_exists set: verify via shell ls -l <path>
    file_exists: Optional[str] = None
    # If stdout_is_number: success if stdout contains a number
    stdout_is_number: bool = False
    # If stdout_regex set: success if stdout matches regex
    stdout_regex: Optional[str] = None


def infer_done_spec(task: str) -> DoneSpec:
    """
    Heuristics for demos:
    - If task mentions save as <filename> or contains explicit filename like plot.png -> file_exists
    - If task mentions mean/average -> stdout_is_number
    - Else -> any non-empty stdout
    """
    t = task.lower()

    # "save ... as X"
    m = re.search(r"save (?:it )?as ([\w\-.\/]+)", t)
    if m:
        return DoneSpec(file_exists=m.group(1))

    # common explicit file mentions
    if "plot.png" in t:
        return DoneSpec(file_exists="plot.png")

    if ("mean" in t or "average" in t) and (".csv" in t or "data.csv" in t):
        return DoneSpec(stdout_is_number=True)

    # default: any stdout
    return DoneSpec(stdout_regex=r".+")


def check_done(done: DoneSpec, last_result: ToolResult) -> Tuple[bool, str]:
    out = (last_result.output or "").strip()

    if done.stdout_is_number:
        m = re.search(r"[-+]?\d+(\.\d+)?", out)
        if m:
            return True, f"OK: found numeric in stdout: {m.group(0)} (raw={out!r})"
        return False, f"Not numeric stdout: {out!r}"

    if done.stdout_regex:
        if re.search(done.stdout_regex, out):
            return True, f"OK: stdout matches /{done.stdout_regex}/ : {out!r}"
        return False, f"stdout does not match /{done.stdout_regex}/ : {out!r}"

    # file_exists is checked separately via verify_file_exists()
    return False, "DoneSpec requires file existence check."


def verify_file_exists(rm: RunManager, ctx, path: str, task: str) -> Tuple[bool, str]:
    call = ToolCall(name="shell_exec", args={"cmd": f"ls -l {path}"})
    res = execute_tool(call, task=task)
    rm.save_json(ctx, "done_verify_shell.json", {"call": call.model_dump(), "result": res.model_dump()})

    if res.ok and (res.output or "").strip() and "No such file" not in (res.output or ""):
        return True, f"OK: {path} exists: {res.output}"
    return False, f"FAIL: {path} not found: {res.error or res.output}"


# ---------------- Optional: context preview helpers ----------------
def maybe_add_file_preview(rm: RunManager, ctx, task: str, observation: str) -> str:
    """
    If task mentions a file (e.g., data.csv), preview a few lines to avoid schema guessing.
    This is a research-agent style “observe world state before acting”.
    """
    t = task.lower()
    # preview up to 5 lines for any mentioned .csv filename
    m = re.search(r"([\w\-.\/]+\.csv)", t)
    if not m:
        return observation

    fname = m.group(1)
    preview_call = ToolCall(name="shell_exec", args={"cmd": f"head -n 5 {fname} 2>/dev/null || true"})
    preview_res = execute_tool(preview_call, task=task)
    rm.save_json(ctx, "file_preview.json", {"call": preview_call.model_dump(), "result": preview_res.model_dump()})

    if preview_res.ok and (preview_res.output or "").strip():
        observation += f"\n\nFILE PREVIEW ({fname}):\n{preview_res.output.strip()}"
    return observation


# ---------------- Rule compilation: turn generic reflection into actionable policy ----------------
def compile_rules(task: str, learned_rules: List[str]) -> List[str]:
    base: List[str] = []

    t = task.lower()
    if ".csv" in t and ("mean" in t or "average" in t):
        base += [
            "If the required CSV file is missing, create a minimal valid sample using file_write.",
            "Preview the CSV (head -n 5) before choosing parsing logic; do not assume columns.",
            "Prefer Python standard library (csv) over pandas unless required.",
            "When using python_exec, ALWAYS print the final numeric mean to stdout.",
            "Avoid repeating the same shell command; if stuck, change strategy.",
        ]
    else:
        base += [
            "If a required file is missing, create a minimal valid sample using file_write.",
            "If a Python module is missing, use pip_install for that module.",
            "When using python_exec, print the final answer to stdout.",
            "Avoid repeating the same tool call; change strategy if stuck.",
        ]

    extras = [r.strip() for r in (learned_rules or []) if r and r.strip()]
    # keep it short to avoid prompt dilution
    out = (base + extras)[:10]
    return out

def require_artifacts(task: str) -> list[str]:
    """
    Infer required on-disk artifacts from task text (simple heuristic for Day4 benchmarks).
    Extend this list as you add more benchmark tasks.
    """
    t = task.lower()
    req = []
    for fname in ["users.csv", "events.csv", "data.csv", "config.json", "report.md"]:
        if fname in t:
            req.append(fname)
    return req



# ---------------- One episode: run a task with full trace + guardrails ----------------
def run_episode(task: str, rules: Optional[List[str]] = None, tag: str = "agent_day3", meta: dict | None = None):
    rm = RunManager()
    ctx = rm.start(tag=tag)

    rm.save_text(ctx, "task.txt", task)
    rm.save_json(ctx, "rules.json", {"rules": rules or []})
    if meta:
        rm.save_json(ctx, "meta.json", meta)

    done_spec = infer_done_spec(task)
    rm.save_json(ctx, "done_spec.json", done_spec.__dict__)

    guard = Guardrails()
    history: List[Dict[str, Any]] = []
    observation = "Task started."
    last_result: Optional[ToolResult] = None

    last_action_sig = None
    repeat_count = 0

    for step in range(1, MAX_STEPS + 1):
        # (Optional) add file preview to avoid schema guessing
        observation = maybe_add_file_preview(rm, ctx, task, observation)
        rm.save_text(ctx, f"step_{step:02d}_observation.txt", observation)

        # LLM proposes next action
        action = next_action(task, observation, rules)
        rm.save_json(ctx, f"step_{step:02d}_action.json", action.model_dump())

        # Soft repetition tracking (anti-stuck notice)
        action_sig = json.dumps(action.model_dump(), sort_keys=True)
        if action_sig == last_action_sig:
            repeat_count += 1
        else:
            repeat_count = 0
            last_action_sig = action_sig

        if repeat_count >= REPEAT_ACTION_LIMIT:
            rm.save_text(ctx, f"step_{step:02d}_anti_stuck.txt", "Repeated same action; anti-stuck notice injected.")
            observation += (
                "\n\nSYSTEM NOTICE: You are repeating the same action. Change strategy. "
                "If a file is missing, create it using file_write. "
                "If a module is missing, install it. "
                "Then proceed to compute/print the final answer."
            )

        # Hard guardrails (may override the action; MUST NOT hard-code domain content)
        guard.track(action)
        override = guard.intervene(action, last_result, observation)
        if override is not None:
            rm.save_json(ctx, f"step_{step:02d}_guardrail_override.json", override.model_dump())
            action = override

        # Execute tool (IMPORTANT: pass task so executor can expand __LLM_GENERATE_SAMPLE__)
        result = execute_tool(action, task=task)
        rm.save_json(ctx, f"step_{step:02d}_result.json", result.model_dump())

        history.append({"step": step, "action": action.model_dump(), "result": result.model_dump()})
        last_result = result

        # Update observation for next loop
        if not result.ok:
            observation = f"ERROR:\n{result.error}"
            continue
        observation = f"SUCCESS:\n{result.output}"

        # ----- DONE CHECKS -----

        # Case A: file_exists-based tasks (artifact-type tasks like report.md, plot.png)
        if done_spec.file_exists:
            ok, msg = verify_file_exists(rm, ctx, done_spec.file_exists, task=task)
            rm.save_text(ctx, f"step_{step:02d}_done_check.txt", msg)
            if ok:
                rm.save_text(ctx, "final.txt", f"DONE via file exists: {done_spec.file_exists}")
                return history, True, ctx.run_id
            continue

        # Case B: stdout-based tasks
        # ✅ Critical fix: only accept stdout-based completion from python_exec
        if action.name != "python_exec":
            rm.save_text(
                ctx,
                f"step_{step:02d}_done_check.txt",
                f"Not done: stdout-based completion requires python_exec, got {action.name}",
            )
            continue

        ok, msg = check_done(done_spec, result)
        rm.save_text(ctx, f"step_{step:02d}_done_check.txt", msg)

        if ok:
            # ✅ Artifact gate: require on-disk files mentioned in task (e.g., users.csv/events.csv)
            req = require_artifacts(task)
            if req:
                all_ok = True
                msgs = []
                for f in req:
                    ok_f, msg_f = verify_file_exists(rm, ctx, f, task=task)
                    msgs.append(msg_f)
                    all_ok = all_ok and ok_f

                rm.save_text(ctx, f"step_{step:02d}_artifact_check.txt", "\n".join(msgs))

                if not all_ok:
                    observation = (
                        "You must create the missing files ON DISK using file_write, "
                        "then read them from disk and print the final number."
                    )
                    continue

            rm.save_text(ctx, "final.txt", "DONE via stdout criteria (python_exec + artifacts)")
            return history, True, ctx.run_id

    rm.save_text(ctx, "final.txt", "FAILED: max steps reached")
    return history, False, ctx.run_id



if __name__ == "__main__":
    memory = EpisodicMemory()

    # ---- Choose a task ----
    task = "Read users.csv and events.csv, then print the number of unique users who have at least one event. If either file is missing, create minimal valid samples first"

    # Attempt 1
    hist1, ok1, run_id1 = run_episode(task, rules=None)
    memory.save_episode(task, hist1)
    print("attempt1 run_id:", run_id1, "ok:", ok1)

    # Reflection → rules
    learned = reflect(memory.load_all())
    Path("memory/reflection_latest.json").write_text(
        json.dumps(learned, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    compiled = compile_rules(task, learned.get("rules", []))
    print("learned rules:", learned.get("rules", []))
    print("compiled rules:", compiled)

    # Attempt 2 with memory/policy
    hist2, ok2, run_id2 = run_episode(task, rules=compiled)
    memory.save_episode(task, hist2)
    print("attempt2 run_id:", run_id2, "ok:", ok2)
