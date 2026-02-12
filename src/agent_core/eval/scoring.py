from typing import Dict, Any

def score_run(metrics: Dict[str, Any], success: bool) -> float:
    steps = metrics["steps"]
    errors = metrics["errors"]
    repeats = metrics["repeats"]
    tool_counts = metrics["tool_counts"]
    pip_installs = tool_counts.get("pip_install", 0)
    guardrail_overrides = metrics.get("guardrail_overrides", 0)

    score = 100.0
    score -= 3.0 * steps
    score -= 10.0 * errors
    score -= 2.0 * repeats
    score -= 15.0 * pip_installs
    score -= 5.0 * guardrail_overrides
    score += 30.0 if success else -30.0
    return score
