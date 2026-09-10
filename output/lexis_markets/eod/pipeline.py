"""Daily EOD pipeline: entity detect, marketparquet + yfinance fill, macro FRED, compact."""
from __future__ import annotations

import time
from datetime import date

from lexis_markets.lake import LakeStore, PgClient
from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.gate_client import GateHandles, resolve_gates
from lexis_markets.jobs.scheduler import timed
from lexis_markets.kaggle.compact import compact_months, months_from_details
from lexis_markets.eod.ingest import ingest_eod
from lexis_markets.fred.tasks import ingest_macro_eod

logger = get_logger("eod.pipeline")


def run_eod(
    cfg: MarketsConfig,
    *,
    limit: int | None = None,
    gates: GateHandles | None = None,
    target_date: date | None = None,
) -> dict:
    t0 = time.perf_counter()
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    gates = resolve_gates(cfg, gates)

    with timed("eod_ingest"):
        out = ingest_eod(
            cfg, lake, pg, yf_gate=gates.yf_gate, limit=limit, target_date=target_date,
            l3_warm="off" if cfg.test.enabled else "tip",
        )

    with timed("macro_eod"):
        macro_out = ingest_macro_eod(
            cfg, lake, pg, fred_gate=gates.fred_gate, target_date=target_date
        )

    details = (out.get("details") or []) + (macro_out.get("details") or [])
    months = months_from_details(details)
    if months:
        with timed("compact"):
            compact_months(cfg, months)

    elapsed = time.perf_counter() - t0
    logger.info(
        "eod pipeline done elapsed=%.1fs equity_ok=%s macro_ok=%s",
        elapsed,
        out.get("symbols"),
        macro_out.get("symbols"),
    )
    return {
        **out,
        "macro_eod": macro_out,
        "pipeline_elapsed_s": elapsed,
    }
