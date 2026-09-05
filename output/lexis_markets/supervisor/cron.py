"""ET market-hours cron: seed, ASAP historical catch-up, daily EOD, quality, disk prune.

On first start (or after ``MARKETS_RESET=1``), enqueues a one-time seed job.

After seed: missing history through the day *before* the latest session target is
enqueued immediately (EOD + FRED). Only the latest session day's EOD waits for the
post-close window (16:00 ET + ``MARKETS_EOD_DELAY_HOURS``). Quality stays on that
same window. Worker disk prune runs on a fixed interval.
"""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from lexis_markets.cleanup import prune_ray_worker_disk
from lexis_markets.config import MarketsConfig
from lexis_markets.eod.ingest import EOD_MARKER_PREFIX, eod_marker_key
from lexis_markets.logging_setup import get_logger
from lexis_markets.kaggle.seed import is_seed_complete
from lexis_markets.kaggle.status import assess_markets_state
from lexis_markets.jobs.clock import eod_target_date
from lexis_markets.jobs.queue import SupervisorState
from lexis_markets.lake import LakeStore

logger = get_logger("supervisor.cron")

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)  # NYSE regular session close (ET)
DISK_PRUNE_EVERY = timedelta(hours=6)


def _now_et(now: datetime | None = None) -> datetime:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(ET)


def eod_window_open(cfg: MarketsConfig, now: datetime | None = None) -> bool:
    """True after 16:00 ET plus MARKETS_EOD_DELAY_HOURS."""
    now_et = _now_et(now)
    close_dt = datetime.combine(now_et.date(), MARKET_CLOSE, tzinfo=ET)
    open_at = close_dt + timedelta(hours=cfg.eod_delay_hours)
    return now_et >= open_at


def latest_session_target(now: datetime | None = None) -> date:
    """Most recent session day the daily cron is responsible for (yesterday ET)."""
    return eod_target_date(_now_et(now).date())


def catchup_through_date(now: datetime | None = None) -> date:
    """Last day historical catch-up may fill without waiting for the EOD window."""
    return latest_session_target(now) - timedelta(days=1)


def latest_eod_marker_date(lake: LakeStore) -> date | None:
    keys = lake.list_keys(f"{EOD_MARKER_PREFIX}/")
    found: list[date] = []
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        if not name.endswith(".json"):
            continue
        try:
            found.append(date.fromisoformat(name[:-5]))
        except ValueError:
            continue
    return max(found) if found else None


def needs_eod_through(lake: LakeStore, through: date) -> bool:
    """True when the lake has no EOD marker on/after ``through``."""
    last = latest_eod_marker_date(lake)
    if last is not None and last >= through:
        return False
    return not lake.exists(eod_marker_key(through))


def pipeline_ready(cfg: MarketsConfig, state: SupervisorState) -> bool:
    return is_seed_complete(cfg) and not state.has_active_job("seed")


def _reset_seed_already_done(state: SupervisorState, cfg: MarketsConfig) -> bool:
    """True when a MARKETS_RESET=1 seed already finished and lake is populated."""
    last = state.get_last_run("seed")
    if not last or not last.get("detail", {}).get("reset"):
        return False
    return is_seed_complete(cfg)


def maybe_schedule_seed(cfg: MarketsConfig, state: SupervisorState) -> bool:
    if state.has_active_job("seed"):
        return False

    reset = os.environ.get("MARKETS_RESET") == "1"
    if reset:
        if _reset_seed_already_done(state, cfg):
            return False
    elif is_seed_complete(cfg):
        return False

    state.enqueue_pending("seed", {"reset": reset})
    logger.info("cron scheduled seed reset=%s", reset)
    return True


def maybe_schedule_eod_catchup(
    cfg: MarketsConfig, state: SupervisorState, now: datetime | None = None
) -> bool:
    """Enqueue EOD through day-before-latest as soon as seed is ready (no window gate).

    While Ray is down, a pending catch-up job stays in SQLite; its target is refreshed
    as the calendar advances. Failed catch-ups are requeued when markers still lag.
    """
    through = catchup_through_date(now)
    lake = LakeStore(cfg)
    if not needs_eod_through(lake, through):
        return False

    run_key = f"eod_catchup:{through.isoformat()}"
    if state.has_active_job("eod"):
        return state.refresh_pending_catchup(
            "eod",
            target_key="target_date",
            target_value=through.isoformat(),
            run_key=run_key,
        )

    requeued = state.requeue_latest_failed(
        "eod",
        mode="catchup",
        payload_update={
            "target_date": through.isoformat(),
            "run_key": run_key,
            "mode": "catchup",
        },
    )
    if requeued is not None:
        state.record_last_run(
            "eod_catchup",
            {
                "scheduled": True,
                "run_key": run_key,
                "target_date": through.isoformat(),
                "requeued": True,
            },
        )
        logger.info("requeued eod catchup id=%s target=%s", requeued, through)
        return True

    state.enqueue_pending(
        "eod",
        {
            "target_date": through.isoformat(),
            "run_key": run_key,
            "mode": "catchup",
        },
    )
    state.record_last_run(
        "eod_catchup",
        {"scheduled": True, "run_key": run_key, "target_date": through.isoformat()},
    )
    logger.info("scheduled eod catchup target=%s run_key=%s", through, run_key)
    return True


def maybe_schedule_fred_catchup(
    cfg: MarketsConfig, state: SupervisorState, now: datetime | None = None
) -> bool:
    """Enqueue FRED vintage gap-fill through catch-up ceiling (no window gate)."""
    through = catchup_through_date(now)
    run_key = f"fred_catchup:{through.isoformat()}"

    if state.has_active_job("fred_backfill"):
        return state.refresh_pending_catchup(
            "fred_backfill",
            target_key="vintage_end",
            target_value=through.isoformat(),
            run_key=run_key,
        )

    requeued = state.requeue_latest_failed(
        "fred_backfill",
        mode="catchup",
        payload_update={
            "vintage_end": through.isoformat(),
            "run_key": run_key,
            "mode": "catchup",
        },
    )
    if requeued is not None:
        state.record_last_run(
            "fred_catchup",
            {
                "scheduled": True,
                "run_key": run_key,
                "vintage_end": through.isoformat(),
                "requeued": True,
            },
        )
        logger.info("requeued fred catchup id=%s through=%s", requeued, through)
        return True

    last = state.get_last_run("fred_catchup")
    if last and last.get("detail", {}).get("run_key") == run_key:
        return False

    state.enqueue_pending(
        "fred_backfill",
        {
            "vintage_end": through.isoformat(),
            "run_key": run_key,
            "mode": "catchup",
        },
    )
    state.record_last_run(
        "fred_catchup",
        {"scheduled": True, "run_key": run_key, "vintage_end": through.isoformat()},
    )
    logger.info("scheduled fred catchup through=%s run_key=%s", through, run_key)
    return True


def maybe_schedule_eod(cfg: MarketsConfig, state: SupervisorState, now: datetime | None = None) -> bool:
    """Cron: latest session day only, after the post-close delay window."""
    if not eod_window_open(cfg, now):
        return False
    if state.has_active_job("eod"):
        return False

    now_et = _now_et(now)
    run_key = f"eod:{now_et.date().isoformat()}"
    last = state.get_last_run("eod")
    if last and last.get("detail", {}).get("run_key") == run_key:
        return False

    target = latest_session_target(now)
    state.enqueue_pending(
        "eod",
        {"target_date": target.isoformat(), "run_key": run_key, "mode": "daily"},
    )
    state.record_last_run(
        "eod",
        {"scheduled": True, "run_key": run_key, "target_date": target.isoformat()},
    )
    logger.info("cron scheduled eod target=%s run_key=%s", target, run_key)
    return True


def maybe_schedule_quality(cfg: MarketsConfig, state: SupervisorState, now: datetime | None = None) -> bool:
    now_et = _now_et(now)
    run_key = f"quality:{now_et.date().isoformat()}"
    last = state.get_last_run("quality")
    if last and last.get("detail", {}).get("run_key") == run_key:
        return False

    if not eod_window_open(cfg, now):
        return False

    payload: dict = {"run_key": run_key}
    if cfg.test.enabled:
        payload["limit"] = cfg.test.quality_limit
    state.enqueue_pending("quality", payload)
    state.record_last_run("quality", {"scheduled": True, "run_key": run_key})
    logger.info("cron scheduled quality run_key=%s", run_key)
    return True


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def maybe_prune_worker_disk(state: SupervisorState, now: datetime | None = None) -> dict | None:
    """Drop stale Ray sessions / trim huge logs on every alive node every DISK_PRUNE_EVERY."""
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    last = state.get_last_run("disk_prune")
    last_at = _parse_iso((last or {}).get("last_run_at"))
    if last_at and (now_utc - last_at) < DISK_PRUNE_EVERY:
        return None

    try:
        worker_out = prune_ray_worker_disk()
    except Exception:
        logger.exception("cron disk_prune failed")
        state.record_last_run("disk_prune", {"ok": False})
        return None

    freed = sum(float(r.get("freed_mb") or 0) for r in worker_out)
    detail = {
        "ok": True,
        "freed_mb": freed,
        "nodes": len(worker_out),
        "per_node": worker_out,
    }
    state.record_last_run("disk_prune", detail)
    logger.info("cron disk_prune freed_mb=%.1f nodes=%s", freed, len(worker_out))
    return detail


def cron_tick(cfg: MarketsConfig, state: SupervisorState, now: datetime | None = None) -> dict:
    markets = assess_markets_state(cfg)
    seed_scheduled = maybe_schedule_seed(cfg, state)
    disk_prune = maybe_prune_worker_disk(state, now)
    if not pipeline_ready(cfg, state):
        return {
            "markets": markets.to_log_dict(),
            "seed_scheduled": seed_scheduled,
            "seed_complete": markets.seed_complete,
            "eod_catchup_scheduled": False,
            "fred_catchup_scheduled": False,
            "eod_scheduled": False,
            "quality_scheduled": False,
            "disk_prune": disk_prune,
        }

    eod_catchup = maybe_schedule_eod_catchup(cfg, state, now)
    fred_catchup = maybe_schedule_fred_catchup(cfg, state, now)
    eod = maybe_schedule_eod(cfg, state, now)
    quality = maybe_schedule_quality(cfg, state, now)
    return {
        "markets": markets.to_log_dict(),
        "seed_scheduled": seed_scheduled,
        "seed_complete": True,
        "eod_catchup_scheduled": eod_catchup,
        "fred_catchup_scheduled": fred_catchup,
        "eod_scheduled": eod,
        "quality_scheduled": quality,
        "disk_prune": disk_prune,
    }
