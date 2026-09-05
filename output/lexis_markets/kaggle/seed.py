"""Full seed pipeline: bulk L1 ingest, registry, YF align, EOD backfill, compact.

``MARKETS_RESET=1`` truncates L2 tables before ingest. Test runs use
``MARKETS_TEST_PROFILE=1`` for small IO samples (see ``tests/README.md``).

Stage graph:
  1. Kaggle L1 (jakewright ∥ jacksoncrow) ∥ FRED backfill (macro; no equity registry needed)
  2. Equity/ETF registry from Kaggle details
  3. yfinance EOD (FRED already running)
  4. seed_align (patch jakewright closes from yfinance already in L1)
  5. Join FRED → compact + final registry
"""
from __future__ import annotations

import time
from datetime import date

import ray

from lexis_markets.lake import LakeStore, PgClient, ensure_schema, put_json, utcnow
from lexis_markets.registry import (
    ensure_eod_aliases,
    seed_default_stitch,
    seed_from_details,
)
from lexis_markets.registry.universe import sync_live_universe
from lexis_markets.cleanup import prune_ray_worker_disk, purge_seed_build_artifacts
from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.gate_client import GateHandles, refresh_gates, resolve_gates
from lexis_markets.jobs.scheduler import plan_task_resources, run_batches, timed
from lexis_markets.kaggle.compact import compact_all_l1, compact_months, months_from_details
from lexis_markets.kaggle.bulk import (
    _remote_ingest_jacksoncrow,
    _remote_ingest_jakewright,
)
from lexis_markets.fred.backfill import remote_fred_backfill
from lexis_markets.eod.align import has_yfinance_coverage, task_align_yf_batch
from lexis_markets.eod.ingest import ingest_eod
from lexis_markets.serve.app import deploy_serve

logger = get_logger("pipelines.seed")

SEED_COMPLETE_MARKER = "ops/markers/seed_complete.json"
SEED_PROGRESS_MARKER = "ops/markers/seed_progress.json"

# Align is S3/CPU bound (no yfinance HTTP). Fan out after EOD has written YF rows.
ALIGN_BATCH_SIZE = 200
ALIGN_MAX_IN_FLIGHT = 16



def is_seed_complete(cfg: MarketsConfig) -> bool:
    return LakeStore(cfg).exists(SEED_COMPLETE_MARKER)


def record_seed_progress(lake: LakeStore, stage: str, **detail) -> None:
    put_json(
        lake,
        SEED_PROGRESS_MARKER,
        {"stage": stage, "updated_at": utcnow().isoformat(), **detail},
    )


def mark_seed_complete(cfg: MarketsConfig, detail: dict) -> None:
    lake = LakeStore(cfg)
    if lake.exists(SEED_PROGRESS_MARKER):
        lake.delete_keys([SEED_PROGRESS_MARKER])
    put_json(
        lake,
        SEED_COMPLETE_MARKER,
        {"stage": "complete", "completed_at": utcnow().isoformat(), **detail},
    )


def _clear_l2(pg: PgClient) -> int:
    tables = (
        "series_cache_span",
        "series_cache_meta",
        "dataset_jobs",
        "stitch_segments",
        "symbol_aliases",
        "series_links",
        "series_meta",
        "symbol_month_coverage",
        "l1_month_manifest",
    )
    n = 0
    for t in tables:
        pg.execute(f"TRUNCATE {t} CASCADE")
        n += 1
    return n


def run_seed(
    cfg: MarketsConfig,
    *,
    reset: bool = False,
    deploy_serve_app: bool = True,
    gates: GateHandles | None = None,
    eod_target_date: date | None = None,
) -> dict:
    t0 = time.perf_counter()
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    cfg_d = cfg.to_dict()
    all_details: list[dict] = []
    gates = resolve_gates(cfg, gates)

    with timed("ensure_schema"):
        ensure_schema(pg)
    record_seed_progress(lake, "ensure_schema")

    if reset:
        if lake.exists(SEED_COMPLETE_MARKER):
            lake.delete_keys([SEED_COMPLETE_MARKER])
        if lake.exists(SEED_PROGRESS_MARKER):
            lake.delete_keys([SEED_PROGRESS_MARKER])
        with timed("clear_l2"):
            n = _clear_l2(pg)
            logger.info("reset truncated %s tables", n)

    if deploy_serve_app:
        with timed("deploy_serve"):
            deploy_serve(cfg)

    # FRED is independent of Kaggle equity L1 / registry — start it with JW∥JC.
    # compact=False: do not compact/delete shared month prefixes while EOD may write later.
    fred_ref = remote_fred_backfill.remote(cfg_d, compact=False)
    logger.info("seed: fred backfill launched in parallel with kaggle L1")
    record_seed_progress(lake, "fred_launched")

    with timed("kaggle_ingest"):
        jw_ref = _remote_ingest_jakewright.remote(cfg_d)
        jc_ref = _remote_ingest_jacksoncrow.remote(cfg_d)
        jw_out, jc_out = ray.get([jw_ref, jc_ref])
        all_details.extend(jw_out.get("details") or [])
        all_details.extend(jc_out.get("details") or [])
    record_seed_progress(lake, "kaggle_ingest", details=len(all_details))

    gates = refresh_gates(cfg)

    with timed("registry_seed"):
        meta = seed_from_details(pg, all_details)
        stitch_n = seed_default_stitch(pg)
        uni = sync_live_universe(pg)
        logger.info(
            "registry series=%s stitch=%s jakewright_only=%s",
            meta["series"],
            stitch_n,
            uni["jakewright_only"],
        )
    record_seed_progress(lake, "registry_seed", series=meta["series"])

    ensure_eod_aliases(pg)
    with timed("eod_backfill"):
        limit = cfg.yf_backfill_limit()
        eod_out = ingest_eod(
            cfg,
            lake,
            pg,
            yf_gate=gates.yf_gate,
            limit=limit,
            target_date=eod_target_date,
        )
        all_details.extend(eod_out.get("details") or [])
    record_seed_progress(lake, "eod_backfill", rows=eod_out.get("rows"))

    # Align after EOD so L1 already holds yfinance rows to patch jakewright against.
    with timed("seed_align"):
        symbols = sorted(
            {
                str(d["symbol"]).upper()
                for d in all_details
                if d.get("source") == "jakewright" and d.get("symbol")
            }
        )
        symbols = cfg.seed_align_symbols(symbols)
        if not symbols:
            logger.info("seed_align: skip (no symbols)")
        elif not has_yfinance_coverage(pg):
            logger.info("seed_align: skip (no yfinance coverage in L1)")
        else:
            logger.info(
                "seed_align: %s symbols batch=%s in_flight=%s",
                len(symbols),
                ALIGN_BATCH_SIZE,
                ALIGN_MAX_IN_FLIGHT,
            )
            shape = plan_task_resources(
                batch_size=ALIGN_BATCH_SIZE,
                max_in_flight=ALIGN_MAX_IN_FLIGHT,
            )
            results = run_batches(
                "seed_align",
                symbols,
                lambda batch: task_align_yf_batch.remote(cfg_d, batch),
                shape,
            )
            patched = sum(int(r.get("patched_rows") or 0) for r in results)
            logger.info("seed_align: done patched_rows=%s batches=%s", patched, len(results))
    record_seed_progress(lake, "seed_align")

    with timed("fred_join"):
        fred_out = ray.get(fred_ref)
        all_details.extend(fred_out.get("details") or [])
        logger.info(
            "seed: fred joined ok=%s rows=%s",
            fred_out.get("symbols"),
            fred_out.get("rows"),
        )
    record_seed_progress(lake, "fred_join", rows=fred_out.get("rows"))

    with timed("compact_l1"):
        months = months_from_details(all_details)
        if months:
            compact_months(cfg, months)
        else:
            compact_all_l1(cfg, lake)
    record_seed_progress(lake, "compact_l1")

    with timed("registry_final"):
        meta = seed_from_details(pg, all_details)
        seed_default_stitch(pg)
        logger.info("registry final series=%s", meta["series"])

    with timed("purge_build_artifacts"):
        cleared = purge_seed_build_artifacts(lake, pg)
        logger.info("cleanup lake artifacts %s", cleared)

    with timed("prune_worker_disk"):
        try:
            worker_out = prune_ray_worker_disk()
            freed = sum(float(r.get("freed_mb") or 0) for r in worker_out)
            logger.info(
                "cleanup worker_disk freed_mb=%.1f nodes=%s",
                freed,
                len(worker_out),
            )
        except Exception:
            logger.exception("cleanup worker_disk failed")
            worker_out = None
            freed = 0.0

    elapsed = time.perf_counter() - t0
    rows = sum(int(d.get("rows") or 0) for d in all_details)
    out = {
        "elapsed_s": elapsed,
        "details_count": len(all_details),
        "rows": rows,
        "series": meta["series"],
        "cleanup": cleared,
        "worker_disk_freed_mb": freed,
        "worker_disk_nodes": len(worker_out or []),
    }
    mark_seed_complete(cfg, out)
    logger.info("seed complete elapsed=%.1fs rows=%s", elapsed, rows)
    return out
