"""Full seed pipeline: worker-local L1, then compact + quality L3, then MinIO flush.

Test runs use ``MARKETS_TEST_PROFILE=1`` for small IO samples (see ``tests/README.md``).

Stage graph:
  1. Scratch actors on Ray workers (no MinIO working store)
  2. Kaggle L1 (jakewright ∥ jacksoncrow) ∥ FRED backfill
  3. Compact Kaggle months, equity/ETF registry
  4. yfinance EOD (align JW as each wave lands; no Serve L3 warm)
  5. Join FRED → compact remaining months
  6. Cluster materialize: stitch+quality on workers, L3 on scratch
  7. Drop staging/parts, flush compacted L1 + L3 to MinIO, mark complete
"""
from __future__ import annotations

import time
from datetime import date

import ray

from lexis_markets.lake import (
    LakeStore,
    PgClient,
    cfg_d_with_scratch,
    ensure_schema,
    flush_seed_lake,
    lake_from_cfg_d,
    open_lake,
    put_json,
    seed_scratch_scope,
    start_seed_lake,
    stop_seed_lake,
    utcnow,
)
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
from lexis_markets.jobs.scheduler import timed
from lexis_markets.kaggle.compact import compact_all_l1, compact_months, months_from_details
from lexis_markets.kaggle.ingest import ingest_jacksoncrow, ingest_jakewright
from lexis_markets.fred.backfill import remote_fred_backfill
from lexis_markets.ray.runtime import DEFAULT_REMOTE_OPTS
from lexis_markets.eod.ingest import ingest_eod
from lexis_markets.eod.materialize import run_materialize
from lexis_markets.serve.app import deploy_serve

logger = get_logger("pipelines.seed")

SEED_COMPLETE_MARKER = "ops/markers/seed_complete.json"
SEED_PROGRESS_MARKER = "ops/markers/seed_progress.json"


@ray.remote(**DEFAULT_REMOTE_OPTS)
def _remote_ingest_jakewright(cfg_d: dict) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    return ingest_jakewright(cfg, lake_from_cfg_d(cfg_d))


@ray.remote(**DEFAULT_REMOTE_OPTS)
def _remote_ingest_jacksoncrow(cfg_d: dict) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    return ingest_jacksoncrow(cfg, lake_from_cfg_d(cfg_d))


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


def run_seed(
    cfg: MarketsConfig,
    *,
    deploy_serve_app: bool = True,
    gates: GateHandles | None = None,
    eod_target_date: date | None = None,
) -> dict:
    t0 = time.perf_counter()
    minio = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    all_details: list[dict] = []
    gates = resolve_gates(cfg, gates)

    with timed("ensure_schema"):
        ensure_schema(pg)
    record_seed_progress(minio, "ensure_schema")

    if deploy_serve_app:
        with timed("deploy_serve"):
            deploy_serve(cfg)

    flush_out = {"workers": 0, "files": 0, "bytes": 0}
    mat = {"shards": 0, "keys": 0, "series": 0, "ok": 0, "rows": 0}
    cleared: dict = {}
    meta = {"series": 0}
    n_workers = 0
    fred_ref = None
    jw_ref = None
    jc_ref = None

    try:
        with seed_scratch_scope(True):
            with timed("seed_lake"):
                stop_seed_lake()
                n_workers = start_seed_lake(cfg)
                logger.info("seed lake workers=%s", n_workers)
            lake = open_lake(cfg, scratch=True)
            cfg_d = cfg_d_with_scratch(cfg)
            record_seed_progress(minio, "seed_lake", workers=n_workers)
            # FRED is independent of Kaggle equity L1 / registry — start it with JW∥JC.
            # compact=False: do not compact/delete shared month prefixes while EOD may write later.
            fred_ref = remote_fred_backfill.remote(cfg_d, compact=False, force=True)
            logger.info("seed: fred backfill launched in parallel with kaggle L1")
            record_seed_progress(minio, "fred_launched")

            with timed("kaggle_ingest"):
                jw_ref = _remote_ingest_jakewright.remote(cfg_d)
                jc_ref = _remote_ingest_jacksoncrow.remote(cfg_d)
                jw_out, jc_out = ray.get([jw_ref, jc_ref])
                all_details.extend(jw_out.get("details") or [])
                all_details.extend(jc_out.get("details") or [])
            record_seed_progress(minio, "kaggle_ingest", details=len(all_details))

            gates = refresh_gates(cfg)

            with timed("kaggle_compact"):
                kaggle_months = months_from_details(all_details)
                if kaggle_months:
                    compact_months(cfg, kaggle_months)
            record_seed_progress(minio, "kaggle_compact", months=len(kaggle_months))

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
            record_seed_progress(minio, "registry_seed", series=meta["series"])

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
                    l3_warm="off",
                )
                all_details.extend(eod_out.get("details") or [])
            record_seed_progress(minio, "eod_backfill", rows=eod_out.get("rows"))

            with timed("fred_join"):
                fred_out = ray.get(fred_ref)
                all_details.extend(fred_out.get("details") or [])
                logger.info(
                    "seed: fred joined ok=%s rows=%s",
                    fred_out.get("symbols"),
                    fred_out.get("rows"),
                )
            record_seed_progress(minio, "fred_join", rows=fred_out.get("rows"))

            with timed("compact_l1"):
                months = months_from_details(all_details)
                if months:
                    compact_months(cfg, months)
                else:
                    compact_all_l1(cfg, lake)
            record_seed_progress(minio, "compact_l1")

            with timed("registry_final"):
                meta = seed_from_details(pg, all_details)
                seed_default_stitch(pg)
                logger.info("registry final series=%s", meta["series"])

            with timed("cluster_materialize"):
                mat = run_materialize(cfg)
                logger.info("cluster_materialize %s", mat)
            record_seed_progress(
                minio, "cluster_materialize", **{k: mat.get(k) for k in ("ok", "rows", "shards")}
            )

            with timed("purge_build_artifacts"):
                cleared = purge_seed_build_artifacts(lake, pg)
                logger.info("cleanup lake artifacts %s", cleared)

            with timed("flush_minio"):
                flush_out = flush_seed_lake(cfg)
                logger.info("flush_minio %s", flush_out)
            record_seed_progress(minio, "flush_minio", **flush_out)
    finally:
        for ref in (jw_ref, jc_ref, fred_ref):
            if ref is None:
                continue
            try:
                ray.cancel(ref, force=True)
            except Exception as exc:
                logger.warning("cancel seed child: %s", exc)
        stop_seed_lake()

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
        "materialize": mat,
        "flush": flush_out,
    }
    mark_seed_complete(cfg, out)
    logger.info("seed complete elapsed=%.1fs rows=%s", elapsed, rows)
    return out
