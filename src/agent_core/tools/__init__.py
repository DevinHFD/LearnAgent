from typing import Dict, Callable, Any
from .python_exec import python_exec
from .pip_install import pip_install

TOOLS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "python_exec": python_exec,
    "pip_install": pip_install,
}
