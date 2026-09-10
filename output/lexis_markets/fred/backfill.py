"""FRED / ALFRED vintage backfill: full history or gap fill through a target date."""
from __future__ import annotations

import time
from datetime import date

import ray

from lexis_markets.lake import PgClient, open_lake, seed_scratch_scope
from lexis_markets.lake.cluster import SEED_SCRATCH_KEY
from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.markets_actors import get_fred_gate_actor
from lexis_markets.jobs.scheduler import timed
from lexis_markets.kaggle.compact import compact_months, months_from_details
from lexis_markets.fred.tasks import ingest_fred_vintage_jobs
from lexis_markets.registry import resolve_fred_backfill_jobs
from lexis_markets.ray.runtime import DEFAULT_REMOTE_OPTS

logger = get_logger("pipelines.fred_backfill")


def run_fred_backfill(
    cfg: MarketsConfig,
    *,
    vintage_end: date | None = None,
    series_ids: list[str] | None = None,
    force: bool = False,
    job_limit: int | None = None,
    compact: bool = True,
) -> dict:
    from lexis_markets.jobs.clock import eod_target_date

    gate = get_fred_gate_actor(cfg)
    target = vintage_end or eod_target_date()
    lake = open_lake(cfg)
    pg = PgClient(cfg.postgres_url)
    jobs = resolve_fred_backfill_jobs(
        pg,
        cfg,
        vintage_end=target,
        fred_gate=gate,
        series_ids=series_ids,
        force=force,
    )
    if job_limit is not None:
        jobs = jobs[:job_limit]
    if not jobs:
        logger.info("fred_backfill: up to date through=%s", target)
        return {
            "source": "fred_backfill",
            "symbols": 0,
            "rows": 0,
            "details": [],
            "vintage_end": target.isoformat(),
            "jobs": 0,
        }

    t0 = time.perf_counter()
    with timed("fred_backfill_ingest"):
        out = ingest_fred_vintage_jobs(cfg, lake, pg, jobs, label="fred_backfill")

    details = out.get("details") or []
    months = months_from_details(details)
    if compact and months:
        with timed("fred_backfill_compact"):
            compact_months(cfg, months)

    elapsed = time.perf_counter() - t0
    logger.info(
        "fred_backfill done through=%s jobs=%s ok=%s rows=%s elapsed=%.1fs",
        target,
        len(jobs),
        out.get("symbols"),
        out.get("rows"),
        elapsed,
    )
    return {
        **out,
        "vintage_end": target.isoformat(),
        "jobs": len(jobs),
        "pipeline_elapsed_s": elapsed,
    }


@ray.remote(**DEFAULT_REMOTE_OPTS)
def remote_fred_backfill(cfg_d: dict, *, force: bool = False, compact: bool = True) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    with seed_scratch_scope(bool(cfg_d.get(SEED_SCRATCH_KEY))):
        return run_fred_backfill(cfg, force=force, compact=compact)
