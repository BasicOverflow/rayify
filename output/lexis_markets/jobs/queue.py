"""SQLite persistence for supervisor cron state and the pending work queue."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from lexis_markets.logging_setup import get_logger

logger = get_logger("jobs.queue")

SCHEMA = """
CREATE TABLE IF NOT EXISTS last_run (
    job_type TEXT PRIMARY KEY,
    last_run_at TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_pending_status ON pending (status, created_at);
"""


def wipe_supervisor_sqlite(path: str) -> None:
    """Delete and recreate an empty supervisor SQLite file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    sqlite3.connect(str(p)).close()


class SupervisorState:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_last_run(self, job_type: str, detail: dict | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(detail or {})
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO last_run (job_type, last_run_at, detail_json)
                VALUES (?, ?, ?)
                ON CONFLICT(job_type) DO UPDATE SET
                    last_run_at = excluded.last_run_at,
                    detail_json = excluded.detail_json
                """,
                (job_type, now, payload),
            )
        logger.info("last_run job_type=%s at=%s", job_type, now)

    def get_last_run(self, job_type: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT job_type, last_run_at, detail_json FROM last_run WHERE job_type = ?",
                (job_type,),
            ).fetchone()
        if not row:
            return None
        return {
            "job_type": row["job_type"],
            "last_run_at": row["last_run_at"],
            "detail": json.loads(row["detail_json"] or "{}"),
        }

    def enqueue_pending(self, job_type: str, payload: dict | None = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO pending (job_type, payload_json, status, created_at)
                VALUES (?, ?, 'pending', ?)
                """,
                (job_type, json.dumps(payload or {}), now),
            )
            item_id = int(cur.lastrowid)
        logger.info("pending enqueued id=%s job_type=%s", item_id, job_type)
        return item_id

    def fetch_pending(self, limit: int = 32) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, job_type, payload_json, status, created_at
                FROM pending
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "job_type": r["job_type"],
                "payload": json.loads(r["payload_json"] or "{}"),
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def mark_running(self, item_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE pending SET status = 'running', started_at = ? WHERE id = ?",
                (now, item_id),
            )

    def mark_complete(self, item_id: int, detail: dict | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        err = json.dumps(detail) if detail and detail.get("error") else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE pending
                SET status = 'complete', finished_at = ?, error = ?
                WHERE id = ?
                """,
                (now, err, item_id),
            )

    def mark_failed(self, item_id: int, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE pending
                SET status = 'failed', finished_at = ?, error = ?
                WHERE id = ?
                """,
                (now, error, item_id),
            )

    def has_active_job(self, job_type: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM pending
                WHERE job_type = ? AND status IN ('pending', 'running')
                LIMIT 1
                """,
                (job_type,),
            ).fetchone()
        return row is not None

    def requeue_interrupted(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE pending
                SET status = 'pending', started_at = NULL
                WHERE status = 'running'
                """
            )
            n = cur.rowcount
        if n:
            logger.info("requeued interrupted jobs count=%s", n)
        return n

    def recover_interrupted_jobs(self, *, seed_complete: bool) -> dict[str, int]:
        """Finish jobs already reflected in lake state; requeue the rest."""
        completed = 0
        requeued = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, job_type FROM pending WHERE status = 'running'"
            ).fetchall()
        for row in rows:
            item_id = int(row["id"])
            if row["job_type"] == "seed" and seed_complete:
                self.mark_complete(item_id)
                completed += 1
                logger.info("recover seed job id=%s already complete in lake", item_id)
                continue
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE pending
                    SET status = 'pending', started_at = NULL
                    WHERE id = ?
                    """,
                    (item_id,),
                )
            requeued += 1
        if requeued:
            logger.info("requeued interrupted jobs count=%s", requeued)
        return {"completed": completed, "requeued": requeued}

    def dedupe_pending_jobs(self, job_type: str) -> int:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id FROM pending
                WHERE job_type = ? AND status = 'pending'
                ORDER BY created_at ASC, id ASC
                """,
                (job_type,),
            ).fetchall()
        if len(rows) <= 1:
            return 0
        for row in rows[1:]:
            self.mark_failed(int(row["id"]), "duplicate pending job superseded")
        dropped = len(rows) - 1
        logger.info("deduped pending job_type=%s dropped=%s kept_id=%s", job_type, dropped, rows[0]["id"])
        return dropped

    def refresh_pending_catchup(
        self,
        job_type: str,
        *,
        target_key: str,
        target_value: str,
        run_key: str,
        mode: str = "catchup",
    ) -> bool:
        """Bump target on an active catch-up job when the calendar advanced while Ray was down.

        Returns True when a pending/running row's target was updated.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, payload_json FROM pending
                WHERE job_type = ? AND status IN ('pending', 'running')
                ORDER BY created_at DESC, id DESC
                """,
                (job_type,),
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"] or "{}")
            if payload.get("mode") != mode:
                continue
            if payload.get(target_key) == target_value and payload.get("run_key") == run_key:
                return False
            payload[target_key] = target_value
            payload["run_key"] = run_key
            payload["mode"] = mode
            with self._connect() as conn:
                conn.execute(
                    "UPDATE pending SET payload_json = ? WHERE id = ?",
                    (json.dumps(payload), int(row["id"])),
                )
            logger.info(
                "refreshed catchup id=%s job_type=%s %s=%s run_key=%s",
                row["id"],
                job_type,
                target_key,
                target_value,
                run_key,
            )
            return True
        return False

    def requeue_latest_failed(
        self,
        job_type: str,
        *,
        mode: str | None = None,
        payload_update: dict | None = None,
    ) -> int | None:
        """Requeue the newest failed job of ``job_type`` (optionally matching ``mode``)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, payload_json FROM pending
                WHERE job_type = ? AND status = 'failed'
                ORDER BY COALESCE(finished_at, created_at) DESC, id DESC
                LIMIT 16
                """,
                (job_type,),
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"] or "{}")
            if mode is not None and payload.get("mode") != mode:
                continue
            if payload_update:
                payload.update(payload_update)
            item_id = int(row["id"])
            now = datetime.now(timezone.utc).isoformat()
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE pending
                    SET status = 'pending', started_at = NULL, finished_at = NULL,
                        error = NULL, payload_json = ?
                    WHERE id = ?
                    """,
                    (json.dumps(payload), item_id),
                )
                # touch created_at so fetch_pending ordering stays fair
                conn.execute(
                    "UPDATE pending SET created_at = ? WHERE id = ?",
                    (now, item_id),
                )
            logger.info("requeued failed id=%s job_type=%s", item_id, job_type)
            return item_id
        return None
