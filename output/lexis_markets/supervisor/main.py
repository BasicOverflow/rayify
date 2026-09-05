"""Supervisor entrypoint: seed bootstrap, cron scheduling, pending job submission.

Runs inside the Docker supervisor container. Keeps cron + NAS/SQLite alive even when
the Ray cluster is offline; reconnects and drains the pending queue when Ray returns.
"""
from __future__ import annotations

import signal
import time

from lexis_markets.lake import PgClient, ensure_schema
from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import configure_logging, get_logger
from lexis_markets.ray.gate_client import GateHandles
from lexis_markets.ray.markets_actors import bootstrap_gate_actors
from lexis_markets.ray.runtime import init_ray, ray_cluster_ready
from lexis_markets.serve.app import deploy_serve
from lexis_markets.kaggle.status import assess_markets_state
from lexis_markets.supervisor.cron import cron_tick
from lexis_markets.jobs.queue import SupervisorState
from lexis_markets.jobs.dispatch import submit_loop

logger = get_logger("supervisor.main")

# How long the main loop sleeps between cron + submit passes.
POLL_SECONDS = 30.0
_running = True


def _handle_stop(signum, frame):
    global _running
    logger.info("supervisor stop signal=%s", signum)
    _running = False
    try:
        from lexis_markets.ray.markets_actors import reset_gate_actors

        reset_gate_actors(MarketsConfig.from_env())
    except Exception:
        logger.exception("supervisor gate reset on stop failed")


def _try_connect_ray(cfg: MarketsConfig) -> GateHandles | None:
    """Connect, bootstrap gates, deploy Serve. Returns None while the cluster is down."""
    try:
        init_ray(cfg)
        yf_gate, fred_gate = bootstrap_gate_actors(cfg)
        gates = GateHandles(yf_gate=yf_gate, fred_gate=fred_gate)
        deploy_serve(cfg)
        logger.info("ray connected namespace=%s", cfg.ray_namespace)
        return gates
    except Exception:
        logger.exception("ray unavailable; supervisor continues cron against NAS")
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        return None


def main() -> None:
    configure_logging()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    cfg = MarketsConfig.from_env()
    state = SupervisorState(cfg.supervisor_state_path)

    logger.info("supervisor starting state=%s", cfg.supervisor_state_path)

    pg = PgClient(cfg.postgres_url)
    ensure_schema(pg)
    logger.info("schema ready postgres connected")

    markets = assess_markets_state(cfg)
    logger.info(
        "markets boot state seed_complete=%s seed_stage=%s jw=%s jc=%s "
        "registry_series=%s macro_series=%s last_eod=%s resume=%s",
        markets.seed_complete,
        markets.seed_stage,
        "done" if markets.jakewright.done else "pending",
        "done" if markets.jacksoncrow.done else "pending",
        markets.registry_series,
        markets.macro_series,
        markets.last_eod_date,
        markets.resume,
    )

    state.dedupe_pending_jobs("seed")
    recovered = state.recover_interrupted_jobs(seed_complete=markets.seed_complete)
    if recovered["completed"] or recovered["requeued"]:
        logger.info("supervisor recovered interrupted=%s", recovered)

    gates = _try_connect_ray(cfg)

    while _running:
        try:
            if gates is None or not ray_cluster_ready():
                if gates is not None:
                    logger.warning("ray connection lost; pausing submit until reconnect")
                    gates = None
                gates = _try_connect_ray(cfg)

            cron_tick(cfg, state)
            if gates is not None and ray_cluster_ready():
                n = submit_loop(cfg, state, gates=gates)
                if n:
                    logger.info("supervisor submitted=%s pending jobs", n)
            else:
                pending = state.fetch_pending(limit=8)
                if pending:
                    logger.info(
                        "ray offline; %s job(s) waiting in SQLite (next=%s)",
                        len(pending),
                        pending[0]["job_type"],
                    )
        except Exception:
            logger.exception("supervisor loop error")
            if not ray_cluster_ready():
                gates = None
        time.sleep(POLL_SECONDS)

    logger.info("supervisor stopped")


if __name__ == "__main__":
    main()
