import subprocess
from typing import Dict, Any

def shell_exec(args: Dict[str, Any]) -> str:
    """
    Execute shell command in current working directory.
    Args expects: {"cmd": "ls -l"} or {"cmd": ["ls", "-l"]}
    """
    cmd = args.get("cmd")

    if isinstance(cmd, str):
        shell = True
    elif isinstance(cmd, list):
        shell = False
    else:
        raise ValueError("shell_exec requires args['cmd'] as str or list")

    proc = subprocess.run(
        cmd,
        shell=shell,
        capture_output=True,
        text=True,
        timeout=30,
    )

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    if proc.returncode != 0:
        raise RuntimeError(err or f"Command failed: {cmd}")
    return out
