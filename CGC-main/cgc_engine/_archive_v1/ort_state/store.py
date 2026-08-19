import json
import os
import sqlite3
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ORTStateRecord:
    input_hash: str
    output_hash: str
    cache_hit: bool


class ORTStateStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ort_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    ep TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    output_hash TEXT NOT NULL,
                    outputs_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ort_runs_key ON ort_runs(model_sha256, ep, input_hash)"
            )
            conn.commit()
        finally:
            conn.close()

    def _lock_db(self) -> Tuple[int, Any]:
        lock_path = str(self.db_path) + ".lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd, fcntl

    def _unlock_db(self, fd: int, fcntl_mod: Any) -> None:
        try:
            fcntl_mod.flock(fd, fcntl_mod.LOCK_UN)
        finally:
            os.close(fd)

    def get_cached(
        self,
        *,
        model_sha256: str,
        ep: str,
        input_hash: str,
    ) -> Optional[Dict[str, Any]]:
        fd, fcntl_mod = self._lock_db()
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            try:
                conn.execute("PRAGMA busy_timeout=30000")
                cur = conn.execute(
                    "SELECT output_hash, outputs_json FROM ort_runs WHERE model_sha256=? AND ep=? AND input_hash=? ORDER BY id DESC LIMIT 1",
                    (model_sha256, ep, input_hash),
                )
                row = cur.fetchone()
                if not row:
                    return None
                out_hash, outputs_json = row
                return {"output_hash": str(out_hash), "outputs": json.loads(outputs_json)}
            finally:
                conn.close()
        finally:
            self._unlock_db(fd, fcntl_mod)

    def put(
        self,
        *,
        model_sha256: str,
        ep: str,
        input_hash: str,
        output_hash: str,
        outputs: Dict[str, Any],
    ) -> None:
        fd, fcntl_mod = self._lock_db()
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            try:
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute(
                    "INSERT INTO ort_runs(ts, model_sha256, ep, input_hash, output_hash, outputs_json) VALUES (?,?,?,?,?,?)",
                    (int(time.time()), str(model_sha256), str(ep), str(input_hash), str(output_hash), json.dumps(outputs, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()
        finally:
            self._unlock_db(fd, fcntl_mod)

    @staticmethod
    def sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
