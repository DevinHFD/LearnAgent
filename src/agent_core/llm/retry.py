import json
from tenacity import retry, stop_after_attempt, wait_fixed
from ..schemas.plan import ExperimentPlan
from .client import LLMClient

SYSTEM_TEMPLATE = "Return ONLY valid JSON following this schema:\n{}"

client = LLMClient()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def generate_plan(prompt: str) -> ExperimentPlan:
    system = SYSTEM_TEMPLATE.format(
        ExperimentPlan.model_json_schema()
    )

    raw = client.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ], temperature=0)

    try:
        data = json.loads(raw)
        return ExperimentPlan.model_validate(data)

    except Exception:
        fix = client.chat([
            {"role": "system", "content": "Fix the JSON. Return only corrected JSON."},
            {"role": "user", "content": raw}
        ], temperature=0)

        data = json.loads(fix)
        return ExperimentPlan.model_validate(data)
