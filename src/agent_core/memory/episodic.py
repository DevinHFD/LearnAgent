import json
from pathlib import Path
from typing import List, Dict, Any

class EpisodicMemory:
    def __init__(self, root: str = "memory"):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

    def save_episode(self, task: str, history: List[Dict[str, Any]]):
        idx = len(list(self.root.glob("episode_*.json"))) + 1
        p = self.root / f"episode_{idx:04d}.json"
        p.write_text(json.dumps({
            "task": task,
            "history": history
        }, ensure_ascii=False, indent=2))

    def load_all(self) -> List[Dict[str, Any]]:
        episodes = []
        for f in sorted(self.root.glob("episode_*.json")):
            episodes.append(json.loads(f.read_text()))
        return episodes
