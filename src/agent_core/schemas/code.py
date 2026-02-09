from pydantic import BaseModel

class CodeBlock(BaseModel):
    code: str
