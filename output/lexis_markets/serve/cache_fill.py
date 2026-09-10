"""Bulk L3 fill via Serve HTTP. Resumes via ``jobs.progress`` (done-set of series_ids)."""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import TYPE_CHECKING

from lexis_markets.config import MarketsConfig
from lexis_markets.jobs.progress import open_progress
from lexis_markets.lake import PgClient
from lexis_markets.logging_setup import get_logger
from lexis_markets.registry.filters import SERIES_STATUSES, normalize_statuses
from lexis_markets.registry import uncached_ranges
from lexis_markets.serve.client import warm_series

if TYPE_CHECKING:
    from lexis_markets.jobs.progress import JobProgress

logger = get_logger("serve.cache_fill")

DEFAULT_TIP_DAYS = 60


def _universe_rows(
    pg: PgClient,
    *,
    statuses: list[str] | None = None,
    asset_classes: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    classes = asset_classes or ["equity", "etf", "macro"]
    status_list = normalize_statuses(statuses) or ["ACTIVE"]
    bad = set(status_list) - SERIES_STATUSES
    if bad:
        raise ValueError(f"invalid statuses: {sorted(bad)}")
    rows = pg.fetchall(
        """
        SELECT series_id, first_seen, last_seen, asset_class, status
        FROM series_meta
        WHERE status = ANY(%s)
          AND first_seen IS NOT NULL
          AND last_seen IS NOT NULL
          AND asset_class = ANY(%s)
        ORDER BY series_id
        """,
        (list(status_list), list(classes)),
    )
    if limit:
        rows = rows[: int(limit)]
    return rows


def _jobs_for_window(rows: list[dict], start_fn, end_fn) -> list[tuple[str, date, date]]:
    jobs: list[tuple[str, date, date]] = []
    for r in rows:
        start = start_fn(r)
        end = end_fn(r)
        if end < start:
            continue
        jobs.append((r["series_id"], start, end))
    return jobs


def _pending_jobs(
    pg: PgClient,
    jobs: list[tuple[str, date, date]],
    progress: "JobProgress",
    phase: str,
) -> list[tuple[str, date, date]]:
    keyed = progress.pending(jobs, key_fn=lambda j: f"{phase}:{j[0]}")
    out: list[tuple[str, date, date]] = []
    for sid, start, end in keyed:
        if uncached_ranges(pg, sid, start, end):
            out.append((sid, start, end))
        else:
            progress.mark_done([f"{phase}:{sid}"], stamp_pg=(phase == "deep"))
    return out


def fill_active_universe(
    cfg: MarketsConfig,
    *,
    tip_days: int | None = None,
    deepen: bool = True,
    asset_classes: list[str] | None = None,
    limit: int | None = None,
    statuses: list[str] | None = None,
    progress: "JobProgress | None" = None,
    fresh: bool = False,
) -> dict:
    """Warm L3 via Serve GET. Tip window first, then full ``first_seen..last_seen``."""
    tip_days = tip_days if tip_days is not None else int(
        os.environ.get("CACHE_FILL_TIP_DAYS", str(DEFAULT_TIP_DAYS))
    )
    pg = PgClient(cfg.postgres_url, pool_max=2)
    rows = _universe_rows(
        pg,
        statuses=statuses,
        asset_classes=asset_classes,
        limit=limit,
    )
    if not rows:
        return {"series": 0, "tip_jobs": 0, "deep_jobs": 0, "built_spans": 0}

    status_list = normalize_statuses(statuses) or ["ACTIVE"]
    own_progress = progress is None
    if progress is None:
        progress = open_progress(
            cfg,
            "cache_fill",
            {
                "statuses": sorted(status_list),
                "asset_classes": sorted(asset_classes or ["equity", "etf", "macro"]),
                "tip_days": tip_days,
                "deepen": bool(deepen),
                "limit": limit,
            },
            fresh=fresh,
            pg=pg,
        )

    def _run(jobs: list[tuple[str, date, date]], label: str) -> dict:
        if not jobs:
            return {"ok": 0, "skip": 0, "total": 0}
        work = _pending_jobs(pg, jobs, progress, label)
        logger.info("cache_fill %s jobs=%s remaining=%s", label, len(jobs), len(work))
        if not work:
            return {"ok": 0, "skip": 0, "total": 0}
        out = warm_series(cfg, work, include_rows=False, revision_mode="latest")
        progress.mark_done([f"{label}:{sid}" for sid, _, _ in work], stamp_pg=(label == "deep"))
        return out

    tip_jobs = _jobs_for_window(
        rows,
        lambda r: max(r["first_seen"], r["last_seen"] - timedelta(days=tip_days)),
        lambda r: r["last_seen"],
    )
    tip_results = _run(tip_jobs, "tip")

    deep_jobs: list[tuple[str, date, date]] = []
    deep_results: dict = {"ok": 0, "skip": 0, "total": 0}
    if deepen:
        deep_jobs = _jobs_for_window(
            rows,
            lambda r: r["first_seen"],
            lambda r: r["last_seen"],
        )
        deep_results = _run(deep_jobs, "deep")

    if own_progress:
        progress.complete()

    out = {
        "series": len(rows),
        "statuses": status_list,
        "tip_days": tip_days,
        "tip_jobs": len(tip_jobs),
        "deep_jobs": len(deep_jobs),
        "tip_ok": tip_results.get("ok"),
        "deep_ok": deep_results.get("ok"),
        "deepen": deepen,
        "progress_scope": progress.scope_id,
        "via": "serve_http",
    }
    logger.info("cache_fill done %s", out)
    return out
