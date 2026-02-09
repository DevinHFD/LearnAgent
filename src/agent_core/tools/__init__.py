from typing import Dict, Callable, Any
from .python_exec import python_exec
from .pip_install import pip_install
from .file_write import file_write
from .shell_exec import shell_exec


TOOLS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "python_exec": python_exec,
    "pip_install": pip_install,
    "file_write": file_write,
    "shell_exec": shell_exec,
}
