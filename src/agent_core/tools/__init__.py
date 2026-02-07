from typing import Dict, Callable, Any
from .python_exec import python_exec

TOOLS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "python_exec": python_exec,
}
