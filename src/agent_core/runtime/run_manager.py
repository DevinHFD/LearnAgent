import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

@dataclass
class RunContext:
    run_id: str
    run_dir: Path

class RunManager:
    def __init__(self, root: str = "runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def start(self, tag: str = "run") -> RunContext:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}_{tag}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return RunContext(run_id=run_id, run_dir=run_dir)

    def save_text(self, ctx: RunContext, name: str, text: str) -> None:
        (ctx.run_dir / name).write_text(text, encoding="utf-8")

    def save_json(self, ctx: RunContext, name: str, obj: Any) -> None:
        (ctx.run_dir / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_error(self, ctx: RunContext, err: str) -> None:
        p = ctx.run_dir / "errors.log"
        p.write_text(err + "\n", encoding="utf-8")

class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed_s = time.time() - self.t0
