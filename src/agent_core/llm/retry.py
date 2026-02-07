import json
from tenacity import retry, stop_after_attempt, wait_fixed
from ..schemas.plan import ExperimentPlan
from .client import LLMClient
from ..runtime.run_manager import RunManager, RunContext, Timer

SYSTEM_TEMPLATE = "Return ONLY valid JSON following this schema:\n{}"

client = LLMClient()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def generate_plan(prompt: str, ctx: RunContext | None = None) -> ExperimentPlan:
    system = SYSTEM_TEMPLATE.format(ExperimentPlan.model_json_schema())

    rm = RunManager() if ctx else None
    if rm and ctx:
        rm.save_text(ctx, "prompt.txt", prompt)

    with Timer() as t:
        raw = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0,
        )

    if rm and ctx:
        rm.save_text(ctx, "raw.json", raw)
        rm.save_json(ctx, "meta.json", {"elapsed_s": t.elapsed_s, "model": client.model})

    try:
        data = json.loads(raw)
        plan = ExperimentPlan.model_validate(data)
        if rm and ctx:
            rm.save_json(ctx, "validated.json", plan.model_dump())
        return plan

    except Exception:
        fix = client.chat(
            [
                {"role": "system", "content": "Fix the JSON. Return only corrected JSON."},
                {"role": "user", "content": raw},
            ],
            temperature=0,
        )
        if rm and ctx:
            rm.save_text(ctx, "repair.json", fix)

        data = json.loads(fix)
        plan = ExperimentPlan.model_validate(data)
        if rm and ctx:
            rm.save_json(ctx, "validated.json", plan.model_dump())
        return plan

