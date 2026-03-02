from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class OfflineAlertStore:
    def __init__(self, db_path: str = "buffer.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path), timeout=10)

    def init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_locally(self, payload: dict[str, Any]) -> int:
        payload_json = json.dumps(payload, ensure_ascii=False)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO pending_alerts (payload, created_at) VALUES (?, ?)",
                (payload_json, created_at),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_pending(self, limit: int = 100) -> list[tuple[int, dict[str, Any]]]:
        safe_limit = max(1, int(limit))
        rows: list[tuple[int, dict[str, Any]]] = []

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "SELECT id, payload FROM pending_alerts ORDER BY id ASC LIMIT ?",
                (safe_limit,),
            )
            for row_id, payload_text in cursor.fetchall():
                try:
                    payload_dict = json.loads(payload_text)
                    if isinstance(payload_dict, dict):
                        rows.append((int(row_id), payload_dict))
                except Exception:
                    continue

        return rows

    def remove_local_alert(self, row_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM pending_alerts WHERE id = ?", (int(row_id),))
            conn.commit()
