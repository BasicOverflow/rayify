"""Ray cluster connection and worker runtime_env assembly."""
from __future__ import annotations

import os
from pathlib import Path

import ray

from lexis_markets.config import MarketsConfig, OUTPUT_DIR
from lexis_markets.logging_setup import get_logger

logger = get_logger("ray.runtime")

# Lexis default: SPREAD is the high-level Ray strategy for cross-node fan-out.
# Do not round-robin NodeAffinity — Ray docs call that a last-resort low-level
# escape hatch that blocks scheduler optimizations.
#
# Placement constraints (Ray-supported):
# - label_selector !headgroup: workload workers carry ray.io/node-group; head is
#   headgroup; vram-monitor pods have no node-group and do not match.
# - IO tasks request a tiny CPU > 0: vram monitors advertise memory only (no CPU),
#   so they are hard-infeasible. num_cpus=0 would be schedulable on them.
# Ops that must pin to head / every node (Serve restart, disk prune) use
# OPS_REMOTE_OPTS — no label_selector — plus explicit NodeAffinity.
WORKER_LABEL_SELECTOR = {"ray.io/node-group": "!headgroup"}
DEFAULT_REMOTE_OPTS = {
    "scheduling_strategy": "SPREAD",
    "label_selector": WORKER_LABEL_SELECTOR,
}
OPS_REMOTE_OPTS = {"scheduling_strategy": "SPREAD"}
IO_TASK_CPUS = 0.001
IO_REMOTE_OPTS = {**DEFAULT_REMOTE_OPTS, "num_cpus": IO_TASK_CPUS}
CPU_REMOTE_OPTS = {**DEFAULT_REMOTE_OPTS, "num_cpus": 1}

WORKER_PIP = [
    "pandas", "pyarrow", "requests", "boto3", "psycopg[binary]", "python-dotenv",
    "fastapi==0.116.0", "starlette==0.46.2", "pydantic", "yfinance",
    # Bump these when the cluster pip URI cache goes stale (missing virtualenv for hashed URI).
    "certifi>=2025.8.3",
    "packaging>=25.0",
    "urllib3>=2.5.0",
]
RUNTIME_EXCLUDES = [
    "cli/**", "**/__pycache__/**", "**/*.pyc", "**/*.png", "**/*.log", "**/*.csv",
    ".supervisor/**", ".pytest_cache/**", "tests/**", "l1_inspect/**", "plot_samples/**",
]


def _runtime_pip_mode() -> str:
    return os.environ.get("MARKETS_RUNTIME_PIP", "auto").strip().lower()


def _pip_cache_miss(exc: BaseException) -> bool:
    msg = str(exc)
    cause = str(exc.__cause__) if getattr(exc, "__cause__", None) else ""
    ctx = str(exc.__context__) if getattr(exc, "__context__", None) else ""
    blob = "\n".join((msg, cause, ctx))
    needles = (
        "pip://",
        "runtime_env `pip`",
        "Failed to create runtime_env",
        "does not exist on the cluster",
        "Local directory",
    )
    return any(n in blob for n in needles)


def _worker_env_vars(cfg: MarketsConfig) -> dict[str, str]:
    env: dict[str, str] = {}
    if cfg.ray_namespace:
        env["RAY_NAMESPACE"] = cfg.ray_namespace
    if cfg.lake_prefix:
        env["MARKETS_LAKE_PREFIX"] = cfg.lake_prefix
    for key in (
        "QUALITY_INLINE_REPAIR",
        "STITCH_MONTH_WORKERS",
        "PG_POOL_MAX",
    ):
        val = os.environ.get(key)
        if val is not None and val != "":
            env[key] = val
    if cfg.test.enabled:
        env["MARKETS_TEST_PROFILE"] = "1"
        env["MARKETS_TEST_YF_LIMIT"] = str(cfg.test.yf_limit)
        env["MARKETS_TEST_YF_SEED"] = str(cfg.test.yf_seed)
        env["MARKETS_TEST_QUALITY_LIMIT"] = str(cfg.test.quality_limit)
        if cfg.test.test_symbols:
            env["MARKETS_TEST_SYMBOLS"] = ",".join(cfg.test.test_symbols)
        if cfg.test.eod_yf_limit != cfg.test.yf_limit:
            env["MARKETS_TEST_EOD_YF_LIMIT"] = str(cfg.test.eod_yf_limit)
        if cfg.test.eod_full_path:
            env["MARKETS_TEST_EOD_FULL_PATH"] = "1"
        env["MARKETS_TEST_EOD_START"] = cfg.test.eod_backfill_start.isoformat()
        env["MARKETS_TEST_FRED_SERIES"] = ",".join(cfg.test.fred_series)
        env["MARKETS_TEST_ALIGN_SYMBOLS"] = ",".join(cfg.test.align_symbols)
    return env


def _build_runtime_env(
    cfg: MarketsConfig,
    *,
    include_pip: bool,
    pip_extra: list[str] | None = None,
) -> dict:
    runtime_env = dict(cfg.runtime_env) if cfg.runtime_env else {}
    runtime_env.setdefault("working_dir", str(OUTPUT_DIR))
    runtime_env.setdefault("excludes", RUNTIME_EXCLUDES)
    env_vars = dict(runtime_env.get("env_vars") or {})
    env_vars.update(_worker_env_vars(cfg))
    if env_vars:
        runtime_env["env_vars"] = env_vars
    if include_pip:
        pip = list(runtime_env.get("pip") or [])
        for pkg in WORKER_PIP + list(pip_extra or []):
            if pkg not in pip:
                pip.append(pkg)
        runtime_env["pip"] = pip
    else:
        runtime_env.pop("pip", None)
    return runtime_env


def init_ray(cfg: MarketsConfig):
    mode = _runtime_pip_mode()
    # (include_pip, label, pip_extra)
    if mode in ("0", "false", "never", "no"):
        attempts = [(False, "pip=never", None)]
    elif mode in ("1", "true", "always", "yes"):
        attempts = [(True, "pip=always", None)]
    else:
        attempts = [
            # Prefer a salted pip list first so a stale hashed URI is not reused.
            (True, "pip=auto-rebuild", ["setuptools>=75", "wheel>=0.45"]),
            (True, "pip=auto", None),
            # Never fall back to bare working_dir: seed imports yfinance at module load.
        ]

    last_exc: BaseException | None = None
    for include_pip, label, pip_extra in attempts:
        runtime_env = _build_runtime_env(cfg, include_pip=include_pip, pip_extra=pip_extra)
        try:
            ctx = ray.init(
                address=cfg.ray_address,
                namespace=cfg.ray_namespace,
                ignore_reinit_error=True,
                runtime_env=runtime_env,
            )
            if label != "pip=auto":
                logger.info("init_ray mode=%s", label)
            logger.info("connected namespace=%s", cfg.ray_namespace)
            return ctx
        except (ConnectionAbortedError, ConnectionError, RuntimeError, ValueError, OSError) as exc:
            last_exc = exc
            if include_pip and mode == "auto" and _pip_cache_miss(exc):
                logger.warning("init_ray stale pip cache (%s), trying next attempt", label)
                try:
                    ray.shutdown()
                except Exception:
                    pass
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("init_ray: no connection attempts configured")


def ray_cluster_ready() -> bool:
    """True when this process is connected to a live Ray cluster (workers may still be empty)."""
    try:
        if not ray.is_initialized():
            return False
        # cluster_resources raises / returns {} when the GCS is unreachable.
        ray.cluster_resources()
        return True
    except Exception:
        return False


def max_in_flight() -> int:
    """Pending-task cap. 0 = unlimited; Ray schedules by CPU, no headroom fraction."""
    return 0
