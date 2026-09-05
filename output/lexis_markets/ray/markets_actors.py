"""Gate actor lifecycle: reset on supervisor boot, reuse within a session."""

from __future__ import annotations

import ray
from ray.exceptions import ActorAlreadyExistsError, RayActorError, RayTaskError

from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.gate_actors import FredGateActor, YfinanceGateActor

logger = get_logger("ray.markets_actors")

YF_GATE_NAME = "lexis-markets-yf-gate"
FRED_GATE_NAME = "lexis-markets-fred-gate"
_GATE_CLASS_MARKERS = ("YfinanceGateActor", "FredGateActor")

_yf_gate = None
_fred_gate = None


def _clear_gate_cache() -> None:
    global _yf_gate, _fred_gate
    _yf_gate = None
    _fred_gate = None


def _gate_options(name: str, cfg: MarketsConfig) -> dict:
    return {"name": name, "namespace": cfg.ray_namespace}


_GATE_PROBE_ERRORS = (RayActorError, ActorAlreadyExistsError, AttributeError, RayTaskError)


def reset_gate_actors(cfg: MarketsConfig) -> int:
    """Kill every live YF/Fred gate actor in this namespace.

    Only for supervisor bootstrap/shutdown. Do not call from ingest pipelines —
    ``get_*_gate_actor`` already recreates dead or misconfigured gates.
    """
    from ray.util.state import list_actors

    _clear_gate_cache()
    killed = 0
    for name in (YF_GATE_NAME, FRED_GATE_NAME):
        try:
            handle = ray.get_actor(name, namespace=cfg.ray_namespace)
            ray.kill(handle, no_restart=True)
            killed += 1
        except ValueError:
            pass
        except Exception as exc:
            logger.debug("reset_gate_actors skip name=%s err=%s", name, exc)

    try:
        actors = list_actors(
            filters=[("ray_namespace", "=", cfg.ray_namespace), ("state", "=", "ALIVE")],
            limit=1000,
        )
    except Exception as exc:
        logger.warning("reset_gate_actors scan skipped: %s", exc)
        if killed:
            logger.info("reset_gate_actors killed=%s namespace=%s", killed, cfg.ray_namespace)
        return killed

    for actor in actors or []:
        cls = actor.class_name or ""
        if not any(marker in cls for marker in _GATE_CLASS_MARKERS):
            continue
        try:
            if actor.name:
                handle = ray.get_actor(actor.name, namespace=cfg.ray_namespace)
                ray.kill(handle, no_restart=True)
            else:
                from ray._private.worker import global_worker
                from ray.actor import ActorID

                global_worker.core_worker.kill_actor(ActorID.from_hex(actor.actor_id), True)
            killed += 1
        except ValueError:
            pass
        except Exception as exc:
            logger.debug("reset_gate_actors skip id=%s err=%s", actor.actor_id, exc)

    if killed:
        logger.info("reset_gate_actors killed=%s namespace=%s", killed, cfg.ray_namespace)
    return killed


def bootstrap_gate_actors(cfg: MarketsConfig) -> tuple[object, object]:
    """Supervisor startup: clear stale gates and create fresh non-detached singletons."""
    global _yf_gate, _fred_gate
    reset_gate_actors(cfg)
    _yf_gate = YfinanceGateActor.options(**_gate_options(YF_GATE_NAME, cfg)).remote(
        cfg.eod_pace_seconds,
        cfg.yf_pace_max_seconds,
        cfg.yf_pace_backoff_factor,
        cfg.yf_pace_recovery_step,
        cfg.eod_chunk_size,
        cfg.eod_chunk_size_min,
        cfg.eod_start_slop_days,
        cfg.eod_start_slop_days_min,
    )
    _fred_gate = FredGateActor.options(**_gate_options(FRED_GATE_NAME, cfg)).remote(
        cfg.fred_pace_per_minute,
        cfg.fred_vintage_max_window_days,
        cfg.fred_vintage_min_days,
    )
    logger.info("bootstrap_gate_actors yf=%s fred=%s", YF_GATE_NAME, FRED_GATE_NAME)
    return _yf_gate, _fred_gate


def get_yf_gate_actor(cfg: MarketsConfig):
    """Return the session gate: local cache, supervisor-named actor, or a new CLI-scoped actor."""
    global _yf_gate
    if _yf_gate is not None:
        try:
            ray.get(_yf_gate.interval.remote())
            ray.get(_yf_gate.batch_knobs.remote())
            return _yf_gate
        except _GATE_PROBE_ERRORS:
            _yf_gate = None
    try:
        actor = ray.get_actor(YF_GATE_NAME, namespace=cfg.ray_namespace)
        try:
            ray.get(actor.interval.remote())
            ray.get(actor.batch_knobs.remote())
        except (AttributeError, RayActorError, RayTaskError):
            logger.warning("yf gate actor stale or dead; recreating")
            try:
                ray.kill(actor, no_restart=True)
            except (ValueError, RayActorError):
                pass
            raise ValueError("yf gate stale actor") from None
        _yf_gate = actor
        return _yf_gate
    except (ValueError, RayActorError):
        try:
            _yf_gate = YfinanceGateActor.options(**_gate_options(YF_GATE_NAME, cfg)).remote(
                cfg.eod_pace_seconds,
                cfg.yf_pace_max_seconds,
                cfg.yf_pace_backoff_factor,
                cfg.yf_pace_recovery_step,
                cfg.eod_chunk_size,
                cfg.eod_chunk_size_min,
                cfg.eod_start_slop_days,
                cfg.eod_start_slop_days_min,
            )
        except ActorAlreadyExistsError:
            _yf_gate = ray.get_actor(YF_GATE_NAME, namespace=cfg.ray_namespace)
        return _yf_gate


def get_fred_gate_actor(cfg: MarketsConfig):
    global _fred_gate
    if _fred_gate is not None:
        try:
            ray.get(_fred_gate.pace_per_minute.remote())
            ray.get(_fred_gate.preferred_window_days.remote())
            return _fred_gate
        except _GATE_PROBE_ERRORS:
            _fred_gate = None
    try:
        actor = ray.get_actor(FRED_GATE_NAME, namespace=cfg.ray_namespace)
        try:
            pace = ray.get(actor.pace_per_minute.remote())
            if abs(pace - cfg.fred_pace_per_minute) > 0.01:
                logger.warning(
                    "fred gate pace mismatch actor=%.1f cfg=%.1f; recreating",
                    pace,
                    cfg.fred_pace_per_minute,
                )
                ray.kill(actor, no_restart=True)
                raise ValueError("fred gate pace mismatch")
            # Old actors lack window-sizer methods; recreate so adaptive packing works.
            ray.get(actor.preferred_window_days.remote())
        except (AttributeError, RayActorError, RayTaskError):
            logger.warning("fred gate actor stale or dead; recreating")
            try:
                ray.kill(actor, no_restart=True)
            except (ValueError, RayActorError):
                pass
            raise ValueError("fred gate stale actor") from None
        _fred_gate = actor
        return _fred_gate
    except (ValueError, RayActorError):
        try:
            _fred_gate = FredGateActor.options(**_gate_options(FRED_GATE_NAME, cfg)).remote(
                cfg.fred_pace_per_minute,
                cfg.fred_vintage_max_window_days,
                cfg.fred_vintage_min_days,
            )
        except ActorAlreadyExistsError:
            _fred_gate = ray.get_actor(FRED_GATE_NAME, namespace=cfg.ray_namespace)
        return _fred_gate
