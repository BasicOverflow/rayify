"""Align JW months for a YF wave. Serve tip warm is daily EOD only."""
from __future__ import annotations

from datetime import date, timedelta

from lexis_markets.config import MarketsConfig
from lexis_markets.eod.align import resolve_align_months, task_align_yf_month
from lexis_markets.jobs.clock import as_date
from lexis_markets.jobs.scheduler import plan_task_resources, run_batches
from lexis_markets.lake import PgClient, cfg_d_with_scratch
from lexis_markets.logging_setup import get_logger
from lexis_markets.serve.client import warm_series

logger = get_logger("eod.l3_warm")

DEFAULT_TIP_DAYS = 60


def align_symbols(cfg: MarketsConfig, symbols: list[str]) -> dict:
    names = sorted({str(s).upper() for s in symbols if s})
    if not names:
        return {"symbols": 0, "patched_rows": 0, "batches": 0, "targets": 0}
    cfg_d = cfg_d_with_scratch(cfg)
    pg = PgClient(cfg.postgres_url)
    targets, months = resolve_align_months(pg, names)
    if not months:
        logger.info("yf_align symbols=%s targets=%s months=0", len(names), len(targets))
        return {"symbols": len(names), "patched_rows": 0, "batches": 0, "targets": len(targets)}
    # One remote per month (batch_size=1). max_in_flight=0: not CPU-bound; Ray
    # still schedules on default num_cpus=1 so a node does not load every month at once.
    shape = plan_task_resources(batch_size=1, max_in_flight=0)
    results = run_batches(
        "yf_align",
        months,
        lambda batch: task_align_yf_month.remote(cfg_d, batch[0][0], batch[0][1], targets),
        shape,
    )
    patched = sum(int(r.get("patched_rows") or 0) for r in results)
    logger.info(
        "yf_align symbols=%s targets=%s months=%s patched_rows=%s",
        len(names),
        len(targets),
        len(results),
        patched,
    )
    return {
        "symbols": len(names),
        "patched_rows": patched,
        "batches": len(results),
        "targets": len(targets),
    }


def series_windows(
    pg: PgClient,
    series_ids: list[str],
    *,
    mode: str,
    tip_days: int = DEFAULT_TIP_DAYS,
) -> list[tuple[str, date, date]]:
    ids = [s for s in series_ids if s]
    if not ids:
        return []
    rows = pg.fetchall(
        """
        SELECT series_id, first_seen, last_seen
        FROM series_meta
        WHERE series_id = ANY(%s)
          AND first_seen IS NOT NULL
          AND last_seen IS NOT NULL
        """,
        (ids,),
    )
    jobs: list[tuple[str, date, date]] = []
    for r in rows:
        first = as_date(r["first_seen"])
        last = as_date(r["last_seen"])
        if first is None or last is None:
            continue
        if mode == "tip":
            start = max(first, last - timedelta(days=tip_days))
        else:
            start = first
        if start > last:
            continue
        jobs.append((r["series_id"], start, last))
    return jobs


def warm_series_ids(
    cfg: MarketsConfig,
    pg: PgClient,
    series_ids: list[str],
    *,
    mode: str,
    tip_days: int = DEFAULT_TIP_DAYS,
) -> dict:
    jobs = series_windows(pg, series_ids, mode=mode, tip_days=tip_days)
    if not jobs:
        return {"ok": 0, "skip": 0, "total": 0}
    price_jobs = [(s, a, b) for s, a, b in jobs if not str(s).startswith("macro:")]
    macro_jobs = [(s, a, b) for s, a, b in jobs if str(s).startswith("macro:")]
    out = {"ok": 0, "skip": 0, "total": 0}
    if price_jobs:
        part = warm_series(cfg, price_jobs, include_rows=False, revision_mode="latest")
        out["ok"] += int(part.get("ok") or 0)
        out["skip"] += int(part.get("skip") or 0)
        out["total"] += int(part.get("total") or 0)
    if macro_jobs:
        macro_mode = "as_of" if mode == "full" else "latest"
        part = warm_series(cfg, macro_jobs, include_rows=False, revision_mode=macro_mode)
        out["ok"] += int(part.get("ok") or 0)
        out["skip"] += int(part.get("skip") or 0)
        out["total"] += int(part.get("total") or 0)
    return out
