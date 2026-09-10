"""Drain the SQLite pending queue into Ray remote jobs with bounded concurrency."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import ray

from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.gate_client import GateHandles
from lexis_markets.ray.runtime import DEFAULT_REMOTE_OPTS, max_in_flight, ray_cluster_ready
from lexis_markets.jobs.queue import SupervisorState

logger = get_logger("jobs.dispatch")


@dataclass
class WorkItem:
    item_id: int
    job_type: str
    payload: dict


def run_eod_job(cfg: MarketsConfig, payload: dict, *, gates: GateHandles) -> dict:
    from lexis_markets.eod.pipeline import run_eod

    limit = payload.get("limit")
    target_raw = payload.get("target_date")
    target_date = date.fromisoformat(target_raw) if target_raw else None
    return run_eod(
        cfg,
        limit=limit,
        gates=gates,
        target_date=target_date,
    )


def run_seed_job(cfg: MarketsConfig, payload: dict, *, gates: GateHandles) -> dict:
    from lexis_markets.kaggle.seed import run_seed

    return run_seed(cfg, deploy_serve_app=False, gates=gates)


def run_fred_backfill_job(cfg: MarketsConfig, payload: dict) -> dict:
    from lexis_markets.fred.backfill import run_fred_backfill

    vintage_raw = payload.get("vintage_end")
    vintage_end = date.fromisoformat(vintage_raw) if vintage_raw else None
    try:
        return run_fred_backfill(
            cfg,
            vintage_end=vintage_end,
            force=bool(payload.get("force")),
            compact=payload.get("compact", True),
        )
    except RuntimeError as exc:
        # Catch-up may target index series (SP500/DJIA) whose ALFRED windows 400-out.
        if payload.get("mode") == "catchup" and "wrote 0 rows" in str(exc):
            logger.warning("fred catchup empty through=%s: %s", vintage_end, exc)
            return {
                "source": "fred_backfill",
                "symbols": 0,
                "rows": 0,
                "details": [],
                "catchup_empty": True,
                "vintage_end": vintage_end.isoformat() if vintage_end else None,
            }
        raise


def dispatch_work_item(cfg: MarketsConfig, item: WorkItem, *, gates: GateHandles) -> dict:
    if item.job_type == "eod":
        return run_eod_job(cfg, item.payload, gates=gates)
    if item.job_type == "seed":
        return run_seed_job(cfg, item.payload, gates=gates)
    if item.job_type == "fred_backfill":
        return run_fred_backfill_job(cfg, item.payload)
    raise ValueError(f"unknown job_type: {item.job_type}")


def submit_loop(
    cfg: MarketsConfig,
    state: SupervisorState,
    *,
    gates: GateHandles,
    max_in_flight_jobs: int | None = None,
    poll_seconds: float = 2.0,
) -> int:
    if not ray_cluster_ready():
        logger.warning("submit_loop skipped: Ray cluster unavailable; leaving pending queue intact")
        return 0

    cap = max_in_flight_jobs if max_in_flight_jobs is not None else max_in_flight()
    pending_items = state.fetch_pending(limit=None if cap <= 0 else cap)
    if not pending_items:
        return 0

    in_flight: dict[int, tuple[WorkItem, ray.ObjectRef]] = {}
    cfg_d = cfg.to_dict()
    idx = 0

    while idx < len(pending_items) or in_flight:
        while idx < len(pending_items) and (cap <= 0 or len(in_flight) < cap):
            item_dict = pending_items[idx]
            item = WorkItem(
                item_id=item_dict["id"],
                job_type=item_dict["job_type"],
                payload=item_dict["payload"],
            )
            state.mark_running(item.item_id)
            ref = _remote_dispatch.remote(
                cfg_d, item.item_id, item.job_type, item.payload, gates.yf_gate, gates.fred_gate
            )
            in_flight[item.item_id] = (item, ref)
            idx += 1

        if not in_flight:
            break

        ready, _ = ray.wait(list(ref for _, ref in in_flight.values()), num_returns=1, timeout=poll_seconds)
        if not ready:
            continue

        ref = ready[0]
        item_id = next(i for i, (_, r) in in_flight.items() if r == ref)
        item, _ = in_flight.pop(item_id)
        try:
            result = ray.get(ref)
            state.mark_complete(item.item_id)
            detail = {"item_id": item.item_id, **result}
            for key in ("run_key", "mode", "target_date", "vintage_end"):
                if key in item.payload:
                    detail.setdefault(key, item.payload[key])
            state.record_last_run(item.job_type, detail)
            if item.job_type == "eod" and item.payload.get("mode") == "catchup":
                state.record_last_run("eod_catchup", detail)
            if item.job_type == "fred_backfill" and item.payload.get("mode") == "catchup":
                state.record_last_run("fred_catchup", detail)
            logger.info("work complete id=%s type=%s", item.item_id, item.job_type)
        except Exception as exc:
            state.mark_failed(item.item_id, str(exc))
            logger.exception("work failed id=%s type=%s", item.item_id, item.job_type)

    return len(pending_items)


@ray.remote(**DEFAULT_REMOTE_OPTS)
def _remote_dispatch(
    cfg_d: dict, item_id: int, job_type: str, payload: dict, yf_gate, fred_gate
) -> dict:
    from lexis_markets.lake import reset_seed_scratch

    reset_seed_scratch()
    cfg = MarketsConfig.from_dict(cfg_d)
    item = WorkItem(item_id=item_id, job_type=job_type, payload=payload)
    gates = GateHandles(yf_gate=yf_gate, fred_gate=fred_gate)
    return dispatch_work_item(cfg, item, gates=gates)
