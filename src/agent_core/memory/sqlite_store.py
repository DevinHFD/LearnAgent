from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeRow:
    run_id: str
    task: str
    ok: bool
    payload: Dict[str, Any]


class SQLiteMemoryStore:
    def __init__(self, path: str = "memory/learnagent.db"):
        self.path = path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT,
          task TEXT,
          ok INTEGER,
          payload TEXT
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ok ON episodes(ok);")
        conn.commit()
        conn.close()

    def add_episode(self, run_id: str, task: str, ok: bool, payload: Dict[str, Any]) -> None:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO episodes(run_id, task, ok, payload) VALUES(?,?,?,?)",
            (run_id, task, 1 if ok else 0, json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()

    def recent(self, limit: int = 20) -> List[EpisodeRow]:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute("SELECT run_id, task, ok, payload FROM episodes ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            out.append(EpisodeRow(run_id=r[0], task=r[1], ok=bool(r[2]), payload=json.loads(r[3] or "{}")))
        return out

    def by_task(self, task: str, limit: int = 50) -> List[EpisodeRow]:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            "SELECT run_id, task, ok, payload FROM episodes WHERE task=? ORDER BY id DESC LIMIT ?",
            (task, limit),
        )
        rows = cur.fetchall()
        conn.close()
        return [EpisodeRow(run_id=r[0], task=r[1], ok=bool(r[2]), payload=json.loads(r[3] or "{}")) for r in rows]