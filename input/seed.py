from __future__ import annotations

import os
import time
from contextlib import contextmanager
from uuid import uuid4

from lexis_markets.api import deploy_markets_api
from lexis_markets.cluster import init_ray, plan_task_resources, run_batches
from lexis_markets.compact import compact_all_l1
from lexis_markets.config import MarketsConfig
from lexis_markets.eod import YfRateGate, ingest_eod
from lexis_markets.ingest import (
    FredRateGate,
    bootstrap_fred_done_from_lake,
    fred_done_details,
    fred_done_symbols,
    ingest_jacksoncrow,
    ingest_jakewright,
    resolve_fred_series_ids,
    task_ingest_fred_batch,
)
from lexis_markets.registry import ensure_eod_aliases, seed_default_stitch, seed_from_details
from lexis_markets.universe import sync_live_universe
from lexis_markets.cleanup import prune_ray_worker_disk, purge_seed_build_artifacts
from lexis_markets.reset import clear_ingest_markers, clear_l2
from lexis_markets.stitch_cache import build_stitched_months, months_from_l1_manifest
from lexis_markets.storage import LakeStore, PgClient, ensure_schema

TIMINGS: dict[str, float] = {}


@contextmanager
def step(label: str):
    t0 = time.perf_counter()
    yield
    s = time.perf_counter() - t0
    TIMINGS[label] = s
    print(f"{label}: {s:.1f}s", flush=True)


def main():
    cfg = MarketsConfig.from_env()
    init_ray(cfg)
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    cfg_d = cfg.to_dict()
    all_details: list[dict] = []

    if os.environ.get("MARKETS_RESET") == "1":
        n = clear_ingest_markers(lake)
        print(f"reset: cleared {n} lake marker keys")

    with step("ensure_schema"):
        ensure_schema(pg)

    with step("clear_l2"):
        n = clear_l2(pg)
        print(f"reset: truncated {n} L2 tables (calendars kept)")

    with step("deploy_serve"):
        from lexis_markets.api import ensure_markets_api

        try:
            deploy_markets_api(cfg)
        except RuntimeError as e:
            print(f"serve: deploy failed ({e}); ensure fallback", flush=True)
            ensure_markets_api(cfg)
        print(f"serve: {cfg.ray_serve_url} bucket={cfg.s3_bucket}")

    with step("ingest_jakewright"):
        all_details.extend(ingest_jakewright(cfg, lake).get("details") or [])

    with step("ingest_jacksoncrow"):
        all_details.extend(ingest_jacksoncrow(cfg, lake).get("details") or [])

    with step("registry_seed_mid"):
        meta = seed_from_details(pg, all_details)
        stitch_n = seed_default_stitch(pg)
        print(
            f"registry: series={meta['series']} aliases={meta['aliases']} "
            f"stitch={stitch_n} gap_nonzero={meta['gap_nonzero']} gap_total={meta['gap_total']}"
        )
        uni = sync_live_universe(pg)
        print(f"universe: UNSUPPORTED jakewright_only={uni['jakewright_only']} non_us={uni['non_us_listing']}")

    with step("ensure_eod_aliases"):
        ensure_eod_aliases(pg)
        seed_default_stitch(pg)

    yf_gate = YfRateGate.remote(cfg.eod_pace_seconds)
    with step("eod_backfill"):
        eod_out = ingest_eod(
            cfg,
            lake,
            pg,
            yf_gate=yf_gate,
        )
    all_details.extend(eod_out.get("details") or [])

    with step("fred_ingest"):
        fred_shape = plan_task_resources(batch_size=4)
        fred_run_id = uuid4().hex[:12]
        rate_gate = FredRateGate.remote()
        bootstrap_fred_done_from_lake(cfg_d)
        done = fred_done_symbols(lake)
        fred_ids = resolve_fred_series_ids(cfg, rate_gate=rate_gate)
        pending = [s for s in fred_ids if s not in done]
        all_details.extend(fred_done_details(lake))
        for r in run_batches(
            "fred",
            pending,
            lambda batch: task_ingest_fred_batch.options(max_retries=2).remote(
                cfg_d, batch, fred_run_id, rate_gate
            ),
            fred_shape,
        ):
            all_details.extend(r["details"])

    with step("compact_l1"):
        compact_all_l1(cfg, lake)

    with step("registry_seed_final"):
        meta = seed_from_details(pg, all_details)
        stitch_n = seed_default_stitch(pg)
        print(
            f"registry: series={meta['series']} aliases={meta['aliases']} "
            f"stitch={stitch_n} gap_nonzero={meta['gap_nonzero']} gap_total={meta['gap_total']}"
        )

    with step("build_stitched_cache"):
        l3_months = months_from_l1_manifest(pg)
        cache_out = build_stitched_months(cfg, l3_months, pg=pg)
        print(f"stitch_cache: months={len(cache_out)} rows={sum(r.get('rows', 0) for r in cache_out)}", flush=True)

    with step("purge_build_artifacts"):
        cleared = purge_seed_build_artifacts(lake, pg)
        print(f"cleanup: {cleared}", flush=True)

    with step("prune_worker_disk"):
        try:
            worker_out = prune_ray_worker_disk()
        except Exception as e:
            print(f"cleanup: worker_disk skipped ({e})", flush=True)
            worker_out = None
        if worker_out:
            freed = sum(r.get("freed_mb", 0) for r in worker_out)
            print(f"cleanup: worker_disk freed_mb={freed:.1f} nodes={len(worker_out)}", flush=True)

    total_s = sum(TIMINGS.values())
    print("timing_summary:", flush=True)
    for label, s in sorted(TIMINGS.items(), key=lambda x: -x[1]):
        pct = 100 * s / total_s if total_s else 0
        print(f"  {label}: {s:.1f}s ({pct:.0f}%)", flush=True)
    print(f"  total: {total_s:.1f}s", flush=True)
    print("seeded=ok")


if __name__ == "__main__":
    main()
