from pydantic import BaseModel
from typing import List

class ExperimentPlan(BaseModel):
    goal: str
    steps: List[str]
