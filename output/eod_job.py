"""Scheduled / manual EOD: entity detect, gap scan, ingest, compact, registry patch."""

from __future__ import annotations

import argparse
import time

from lexis_markets.cluster import init_ray, timed
from lexis_markets.compact import compact_months, months_from_details
from lexis_markets.config import MarketsConfig
from lexis_markets.eod import YfRateGate, ingest_eod, reset_eod_fill_pointers, resolve_eod_targets
from lexis_markets.registry import patch_eod_registry
from lexis_markets.stitch_cache import build_stitched_months
from lexis_markets.storage import LakeStore, PgClient


def _compact_patch_l3(
    cfg: MarketsConfig,
    pg: PgClient,
    details: list[dict],
    *,
    pass_label: str,
) -> None:
    if not details:
        return
    months = months_from_details(details)
    if months:
        with timed(f"compact_{pass_label}"):
            compact_months(cfg, months)
        print(f"eod_job: compact {pass_label} months={len(months)}", flush=True)
    with timed(f"patch_registry_{pass_label}"):
        patch = patch_eod_registry(pg, details)
    print(f"eod_job: registry {pass_label} patched={patch['updated']}", flush=True)
    if months:
        with timed(f"l3_{pass_label}"):
            l3_out = build_stitched_months(cfg, months, pg=pg)
        print(f"eod_job: l3 {pass_label} rebuilt={len(l3_out)}", flush=True)


def run_eod_job(
    cfg: MarketsConfig,
    *,
    max_passes: int = 3,
    stale_retries: int | None = None,
    reset_pointers: bool = False,
) -> dict:
    init_ray(cfg)
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    stale_retries = cfg.eod_stale_retries if stale_retries is None else stale_retries

    if reset_pointers:
        n = reset_eod_fill_pointers(pg)
        print(f"eod_job: reset {n} fill pointers", flush=True)

    stale_before = len(resolve_eod_targets(pg))
    print(
        f"eod_job: stale_before={stale_before} max_passes={max_passes} stale_retries={stale_retries}",
        flush=True,
    )

    yf_gate = YfRateGate.remote(cfg.eod_pace_seconds)
    last_out: dict = {"source": "eod", "symbols": 0, "rows": 0, "details": []}

    for pass_n in range(1, max_passes + 1):
        stale = len(resolve_eod_targets(pg))
        if pass_n > 1 and stale == 0:
            break
        print(f"eod_job: pass {pass_n}/{max_passes} stale={stale}", flush=True)
        last_out = ingest_eod(cfg, lake, pg, yf_gate=yf_gate)

        details = last_out.get("details") or []
        _compact_patch_l3(cfg, pg, details, pass_label=f"pass{pass_n}")

        stale_after = len(resolve_eod_targets(pg))
        failed = int(last_out.get("failed") or 0)
        print(
            f"eod_job: pass={pass_n} ok={last_out.get('symbols')} rows={last_out.get('rows')} "
            f"failed={failed} stale_after={stale_after}",
            flush=True,
        )
        if stale_after == 0:
            break

    for retry in range(1, stale_retries + 1):
        stale = len(resolve_eod_targets(pg))
        if stale == 0:
            break
        print(f"eod_job: stale_retry {retry}/{stale_retries} stale={stale}", flush=True)
        last_out = ingest_eod(cfg, lake, pg, yf_gate=yf_gate)
        details = last_out.get("details") or []
        _compact_patch_l3(cfg, pg, details, pass_label=f"stale{retry}")

    last_out["stale_before"] = stale_before
    last_out["stale_after"] = len(resolve_eod_targets(pg))
    return last_out


def main():
    p = argparse.ArgumentParser(description="Lexis Markets EOD job (detect + gap scan + backfill)")
    p.add_argument("--reset-pointers", action="store_true", help="reset eod_filled_through drift")
    p.add_argument("--max-passes", type=int, default=3, help="retry passes when failures remain")
    p.add_argument("--stale-retries", type=int, default=None, help="extra passes for remaining stale symbols")
    args = p.parse_args()

    cfg = MarketsConfig.from_env()
    t0 = time.perf_counter()
    out = run_eod_job(
        cfg,
        max_passes=args.max_passes,
        stale_retries=args.stale_retries,
        reset_pointers=args.reset_pointers,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"eod_job=ok elapsed={elapsed:.1f}s stale_after={out.get('stale_after')} "
        f"rows={out.get('rows')} entity_new={(out.get('entity_detect') or {}).get('registered', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
