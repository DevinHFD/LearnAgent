import subprocess
import sys
from typing import Dict, Any

def pip_install(args: Dict[str, Any]) -> str:
    """
    Install python packages into current venv.
    Args expects: {"packages": ["matplotlib", "numpy"]} or {"packages": "matplotlib"}
    """
    pkgs = args.get("packages")
    if isinstance(pkgs, str):
        packages = [pkgs]
    elif isinstance(pkgs, list) and all(isinstance(x, str) for x in pkgs):
        packages = pkgs
    else:
        raise ValueError("pip_install requires args['packages'] as str or list[str]")

    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    if proc.returncode != 0:
        raise RuntimeError(err or f"pip install failed: {packages}")
    return out or f"Installed: {', '.join(packages)}"
