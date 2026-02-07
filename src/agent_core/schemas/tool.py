from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional

class ToolCall(BaseModel):
    name: Literal["python_exec"] = Field(..., description="Tool name")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

class ToolResult(BaseModel):
    name: str
    ok: bool
    output: str
    error: Optional[str] = None
