"""Durable per-item job progress on the lake (MinIO) + optional PG stamps.

Long Ray jobs must resume after driver/cluster loss. Pattern:

1. Open ``JobProgress`` with a stable ``job`` name + ``scope`` fingerprint.
2. ``pending(items, key_fn)`` drops keys already done (lake file ∪ PG stamps).
3. After each successful batch, ``mark_done(keys)``.
4. ``complete()`` when the whole job finishes; ``reset()`` / ``fresh=True`` to redo.

Progress object: ``ops/progress/{job}/{scope}.json``.
PG backup stamp: ``series_meta.extras[job_done__{job}__{scope8}]`` so work already
written is visible even if the progress object is missing.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Iterable

from lexis_markets.lake import LakeStore, PgClient, get_json, put_json, utcnow
from lexis_markets.logging_setup import get_logger

logger = get_logger("jobs.progress")

PROGRESS_PREFIX = "ops/progress"


def scope_fingerprint(scope: dict) -> str:
    """Stable short hash of a JSON-serializable scope dict."""
    blob = json.dumps(scope, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def progress_object_key(job: str, scope_id: str) -> str:
    safe_job = job.replace("/", "_")
    return f"{PROGRESS_PREFIX}/{safe_job}/{scope_id}.json"


def stamp_extra_key(job: str, scope_id: str) -> str:
    return f"job_done__{job}__{scope_id[:12]}"


def _expand_done_keys(keys: Iterable[str]) -> set[str]:
    """Bare series_id and tip:/deep: variants are equivalent for resume."""
    out: set[str] = set()
    for k in keys:
        if not k:
            continue
        out.add(k)
        if k.startswith("tip:") or k.startswith("deep:"):
            sid = k.split(":", 1)[1]
            out.add(sid)
            out.add(f"tip:{sid}")
            out.add(f"deep:{sid}")
        else:
            out.add(f"tip:{k}")
            out.add(f"deep:{k}")
    return out


class JobProgress:
    """Resume cursor: set of completed item keys for one job+scope."""

    def __init__(
        self,
        lake: LakeStore,
        *,
        job: str,
        scope: dict,
        pg: PgClient | None = None,
    ):
        self.lake = lake
        self.job = job
        self.scope = dict(scope)
        self.scope_id = scope_fingerprint(self.scope)
        self.key = progress_object_key(job, self.scope_id)
        self.stamp_key = stamp_extra_key(job, self.scope_id)
        self.pg = pg
        self._done: set[str] = set()
        self._status = "running"
        self._total: int | None = None
        self._loaded = False

    def load(self) -> "JobProgress":
        if self.lake.exists(self.key):
            body = get_json(self.lake, self.key)
            self._done = _expand_done_keys(body.get("done") or [])
            self._status = body.get("status") or "running"
            self._total = body.get("total")
            logger.info(
                "job_progress resume job=%s scope=%s done=%s status=%s",
                self.job,
                self.scope_id,
                len(self._done),
                self._status,
            )
        else:
            logger.info(
                "job_progress new job=%s scope=%s key=%s",
                self.job,
                self.scope_id,
                self.key,
            )
        if self.pg is not None:
            stamped = self._load_pg_stamps()
            if stamped - self._done:
                logger.info(
                    "job_progress pg_stamps job=%s added=%s",
                    self.job,
                    len(stamped - self._done),
                )
                self._done |= _expand_done_keys(stamped)
                self._persist()
        self._loaded = True
        return self

    def _load_pg_stamps(self) -> set[str]:
        assert self.pg is not None
        rows = self.pg.fetchall(
            """
            SELECT series_id
            FROM series_meta
            WHERE extras ? %s
            """,
            (self.stamp_key,),
        )
        return {r["series_id"] for r in rows}

    def reset(self) -> None:
        self._done.clear()
        self._status = "running"
        self._total = None
        if self.lake.exists(self.key):
            self.lake.delete_keys([self.key])
        logger.info("job_progress reset job=%s scope=%s", self.job, self.scope_id)

    def done_ids(self) -> set[str]:
        if not self._loaded:
            self.load()
        return set(self._done)

    def is_complete(self) -> bool:
        if not self._loaded:
            self.load()
        return self._status == "complete"

    def pending(
        self,
        items: list[Any],
        key_fn: Callable[[Any], str] | None = None,
    ) -> list[Any]:
        if not self._loaded:
            self.load()
        key_fn = key_fn or (lambda x: x["series_id"] if isinstance(x, dict) else str(x))
        self._total = len(items)
        out = [it for it in items if key_fn(it) not in self._done]
        logger.info(
            "job_progress pending job=%s total=%s done=%s remaining=%s",
            self.job,
            len(items),
            len(self._done),
            len(out),
        )
        return out

    def mark_done(
        self,
        keys: Iterable[str],
        *,
        stamp_pg: bool = True,
        stamp_ids: Iterable[str] | None = None,
    ) -> None:
        if not self._loaded:
            self.load()
        batch = [k for k in keys if k and k not in self._done]
        if not batch:
            return
        self._done |= _expand_done_keys(batch)
        self._persist()
        if stamp_pg and self.pg is not None:
            if stamp_ids is not None:
                ids = [s for s in stamp_ids if s]
            else:
                ids = []
                for k in batch:
                    if k.startswith("tip:") or k.startswith("deep:"):
                        ids.append(k.split(":", 1)[1])
                    else:
                        ids.append(k)
            self._stamp_pg(sorted(set(ids)))

    def complete(self) -> None:
        if not self._loaded:
            self.load()
        self._status = "complete"
        self._persist()
        logger.info(
            "job_progress complete job=%s scope=%s done=%s",
            self.job,
            self.scope_id,
            len(self._done),
        )

    def _persist(self) -> None:
        put_json(
            self.lake,
            self.key,
            {
                "job": self.job,
                "scope": self.scope,
                "scope_id": self.scope_id,
                "stamp_key": self.stamp_key,
                "status": self._status,
                "total": self._total,
                "done_count": len(self._done),
                "done": sorted(self._done),
                "updated_at": utcnow().isoformat(),
            },
        )

    def _stamp_pg(self, series_ids: list[str]) -> None:
        assert self.pg is not None
        if not series_ids:
            return
        iso = utcnow().isoformat()
        # jsonb_build_object(key, value) — key is the stamp column name.
        self.pg.executemany(
            """
            UPDATE series_meta
            SET extras = COALESCE(extras, '{}'::jsonb)
                || jsonb_build_object(%s::text, %s::text)
            WHERE series_id = %s
            """,
            [(self.stamp_key, iso, sid) for sid in series_ids],
        )


def open_progress(
    cfg,
    job: str,
    scope: dict,
    *,
    fresh: bool = False,
    pg: PgClient | None = None,
) -> JobProgress:
    lake = LakeStore(cfg)
    client = pg if pg is not None else PgClient(cfg.postgres_url, pool_max=2)
    prog = JobProgress(lake, job=job, scope=scope, pg=client)
    if fresh:
        prog.reset()
    else:
        prog.load()
    return prog
