"""Ray tasks: incremental ALFRED vintage ingest for macro EOD / backfill."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import pandas as pd
import ray

from lexis_markets.lake import LakeStore, PgClient
from lexis_markets.registry import patch_macro_eod_registry, resolve_fred_vintage_jobs
from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import get_logger
from lexis_markets.fred.client import (
    fetch_fred_revisions_adaptive,
    gate_preferred_window_days,
    merge_fred_window_details,
    plan_fred_vintage_range,
    write_fred_vintage_frame,
)
from lexis_markets.ray.markets_actors import get_fred_gate_actor
from lexis_markets.jobs.scheduler import TaskShape, run_batches

logger = get_logger("fred.tasks")


def _detail_from_frame(
    series_id: str,
    symbol: str,
    df: pd.DataFrame,
    vintage_end: date,
    *,
    bisects: int = 0,
    windows_fetched: int = 0,
) -> dict:
    if df.empty:
        return {
            "series_id": series_id,
            "symbol": symbol,
            "rows": 0,
            "source": "fred",
            "series_type": "macro",
            "vintage_through": vintage_end.isoformat(),
            "bisects": bisects,
            "windows_fetched": windows_fetched,
        }
    ts = pd.to_datetime(df["ts"]).dt.date
    return {
        "series_id": series_id,
        "symbol": symbol,
        "rows": int(len(df)),
        "source": "fred",
        "series_type": "macro",
        "first": ts.min().isoformat(),
        "last": ts.max().isoformat(),
        "vintage_through": vintage_end.isoformat(),
        "bisects": bisects,
        "windows_fetched": windows_fetched,
    }


def _prepare_series_jobs(cfg: MarketsConfig, jobs: list[dict], rate_gate) -> list[dict]:
    """Clamp each series once on the driver and attach ALFRED vintage dates."""
    prepared: list[dict] = []
    for job in jobs:
        symbol = str(job["symbol"]).upper()
        v_start = date.fromisoformat(job["vintage_start"])
        v_end = date.fromisoformat(job["vintage_end"])
        plan = plan_fred_vintage_range(symbol, cfg.fred_api_key, v_start, v_end, rate_gate)
        if plan is None:
            prepared.append(
                {
                    **job,
                    "vintage_start": v_end.isoformat(),
                    "vintage_end": v_end.isoformat(),
                    "vintage_dates": [],
                    "empty_after_clamp": True,
                }
            )
            continue
        prepared.append(
            {
                **job,
                "vintage_start": plan.vintage_start.isoformat(),
                "vintage_end": plan.vintage_end.isoformat(),
                "vintage_dates": [d.isoformat() for d in plan.vintage_dates],
            }
        )
    return prepared


def ingest_fred_vintage_job(
    cfg: MarketsConfig,
    lake: LakeStore,
    job: dict,
    *,
    run_id: str,
    rate_gate,
) -> dict:
    symbol = str(job["symbol"]).upper()
    series_id = job["series_id"]
    v_start = date.fromisoformat(job["vintage_start"])
    v_end = date.fromisoformat(job["vintage_end"])
    if job.get("empty_after_clamp") or v_start > v_end:
        return _detail_from_frame(series_id, symbol, pd.DataFrame(), v_end)

    vintage_dates = [date.fromisoformat(d) for d in (job.get("vintage_dates") or [])]
    df, stats = fetch_fred_revisions_adaptive(
        symbol,
        cfg.fred_api_key,
        v_start,
        v_end,
        rate_gate,
        vintage_dates=vintage_dates,
        min_window_days=cfg.fred_vintage_min_days,
        max_window_days=cfg.fred_vintage_max_window_days,
    )
    windows_fetched = stats.windows
    detail = _detail_from_frame(
        series_id,
        symbol,
        df,
        v_end,
        bisects=stats.bisects,
        windows_fetched=windows_fetched,
    )
    if df.empty:
        return detail
    meta = write_fred_vintage_frame(lake, df, run_id=run_id)
    detail.update(
        {
            "rows": meta["rows"],
            "months_written": meta.get("months_written", 0),
        }
    )
    return detail


@ray.remote
def task_ingest_fred_vintage_batch(cfg_d: dict, jobs: list[dict], run_id: str) -> list[dict]:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    gate = get_fred_gate_actor(cfg)
    return [
        ingest_fred_vintage_job(cfg, lake, job, run_id=run_id, rate_gate=gate) for job in jobs
    ]


def ingest_fred_vintage_jobs(
    cfg: MarketsConfig,
    lake: LakeStore,
    pg: PgClient,
    jobs: list[dict],
    *,
    label: str = "fred_vintage",
) -> dict:
    if not jobs:
        return {"source": label, "symbols": 0, "rows": 0, "details": []}

    gate = get_fred_gate_actor(cfg)
    series_jobs = _prepare_series_jobs(cfg, jobs, gate)
    preferred = gate_preferred_window_days(gate, cfg.fred_vintage_max_window_days)
    cfg_d = cfg.to_dict()
    run_id = uuid4().hex[:12]
    shape = TaskShape(
        batch_size=max(1, cfg.fred_series_per_task),
        max_in_flight=max(1, cfg.fred_max_in_flight),
    )
    n_vintages = sum(len(j.get("vintage_dates") or []) for j in series_jobs)
    logger.info(
        "%s: series=%s vintage_dates=%s preferred_window_days=%s "
        "max_in_flight=%s batch_size=%s max_window_days=%s",
        label,
        len(series_jobs),
        n_vintages,
        preferred,
        shape.max_in_flight,
        shape.batch_size,
        cfg.fred_vintage_max_window_days,
    )

    results = run_batches(
        label,
        series_jobs,
        lambda batch: task_ingest_fred_vintage_batch.options(max_retries=2).remote(
            cfg_d, batch, run_id
        ),
        shape,
    )
    raw_details = [detail for batch in results for detail in batch]
    details = merge_fred_window_details(raw_details)
    fred_meta = [d for d in details if d.get("first")]
    if fred_meta:
        from lexis_markets.registry import seed_default_stitch, seed_from_details

        seed_from_details(pg, fred_meta)
        seed_default_stitch(pg)
    patch_macro_eod_registry(pg, details)
    rows = sum(int(d.get("rows") or 0) for d in details)
    ok = sum(1 for d in details if int(d.get("rows") or 0) > 0)
    bisects = sum(int(d.get("bisects") or 0) for d in details)
    if jobs and ok == 0 and rows == 0:
        raise RuntimeError(
            f"{label}: {len(jobs)} series scheduled but wrote 0 rows "
            "(check FRED output_type / API response shape); refusing silent empty backfill"
        )
    logger.info(
        "%s done ok=%s rows=%s bisects=%s preferred_window_days=%s",
        label,
        ok,
        rows,
        bisects,
        gate_preferred_window_days(gate, cfg.fred_vintage_max_window_days),
    )
    return {"source": label, "symbols": ok, "rows": rows, "details": details, "bisects": bisects}


def ingest_macro_eod(
    cfg: MarketsConfig,
    lake: LakeStore,
    pg: PgClient,
    *,
    fred_gate=None,
    target_date: date | None = None,
) -> dict:
    from lexis_markets.jobs.clock import eod_target_date

    target = target_date or eod_target_date()
    gate = fred_gate or get_fred_gate_actor(cfg)
    jobs = resolve_fred_vintage_jobs(pg, cfg, target, gate)
    if not jobs:
        logger.info("macro_eod: up to date target=%s", target)
        return {"source": "fred_eod", "symbols": 0, "rows": 0, "details": [], "target_date": target.isoformat()}

    out = ingest_fred_vintage_jobs(cfg, lake, pg, jobs, label="fred_eod")
    out["target_date"] = target.isoformat()
    logger.info("macro_eod done target=%s ok=%s rows=%s", target, out.get("symbols"), out.get("rows"))
    return out
