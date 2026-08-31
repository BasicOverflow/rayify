from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ray
from ray._private.worker import global_worker
from ray._raylet import ActorID
from ray.util.state import list_actors

from lexis_markets.config import MarketsConfig

OUTPUT_DIR = Path(__file__).resolve().parent.parent
SERVE_APP_NAME = "lexis-markets"
WORKER_PIP = [
    "pandas",
    "pyarrow",
    "requests",
    "boto3",
    "psycopg[binary]",
    "python-dotenv",
    "fastapi==0.116.0",
    "starlette==0.46.2",
    "pydantic",
    "yfinance",
    "matplotlib",
    "httpx",
]
RUNTIME_EXCLUDES = [
    "l1_inspect/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.png",
    "**/*.log",
    "**/*.csv",
    "_ssh_*.py",
]


def _runtime_pip_mode() -> str:
    return os.environ.get("MARKETS_RUNTIME_PIP", "auto").strip().lower()


def _pip_cache_miss(exc: BaseException) -> bool:
    msg = str(exc)
    needles = (
        "pip://",
        "runtime_env `pip`",
        "Failed to create runtime_env",
        "does not exist on the cluster",
    )
    return any(n in msg for n in needles)


def _build_runtime_env(cfg: MarketsConfig, *, include_pip: bool) -> dict:
    runtime_env = dict(cfg.runtime_env) if cfg.runtime_env else {}
    runtime_env.setdefault("working_dir", str(OUTPUT_DIR))
    runtime_env.setdefault("excludes", RUNTIME_EXCLUDES)
    if include_pip:
        pip = list(runtime_env.get("pip") or [])
        for pkg in WORKER_PIP:
            if pkg not in pip:
                pip.append(pkg)
        runtime_env["pip"] = pip
    else:
        runtime_env.pop("pip", None)
    return runtime_env


def init_ray(cfg: MarketsConfig):
    mode = _runtime_pip_mode()
    if mode in ("0", "false", "never", "no"):
        attempts = [(False, "pip=never")]
    elif mode in ("1", "true", "always", "yes"):
        attempts = [(True, "pip=always")]
    else:
        attempts = [(True, "pip=auto"), (False, "pip=auto-fallback")]

    last_exc: BaseException | None = None
    for include_pip, label in attempts:
        runtime_env = _build_runtime_env(cfg, include_pip=include_pip)
        try:
            ctx = ray.init(
                address=cfg.ray_address,
                namespace=cfg.ray_namespace,
                ignore_reinit_error=True,
                runtime_env=runtime_env,
            )
            if label != "pip=auto":
                print(f"init_ray: {label}", flush=True)
            purge_namespace(cfg.ray_namespace)
            return ctx
        except (ConnectionAbortedError, RuntimeError, ValueError) as exc:
            last_exc = exc
            if include_pip and mode == "auto" and _pip_cache_miss(exc):
                print(
                    "init_ray: stale cluster pip cache, retrying without runtime_env pip "
                    "(set MARKETS_RUNTIME_PIP=always to require reinstall, or restart Ray head "
                    "to purge /tmp/ray/session_latest/runtime_resources)",
                    flush=True,
                )
                try:
                    ray.shutdown()
                except Exception:
                    pass
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("init_ray: no connection attempts configured")


def purge_namespace(namespace: str) -> None:
    """Kill stale actors left in this namespace from prior runs."""
    killed = 0
    try:
        actors = list_actors(
            filters=[("ray_namespace", "=", namespace), ("state", "=", "ALIVE")],
            limit=10_000,
        )
    except Exception as e:
        print(f"purge: actor scan skipped ({e})")
        actors = []
    for a in actors:
        try:
            if a.name:
                ray.kill(ray.get_actor(a.name, namespace=namespace), no_restart=True)
            else:
                global_worker.core_worker.kill_actor(ActorID.from_hex(a.actor_id), True)
            killed += 1
        except ValueError:
            pass

    if killed:
        print(f"purge: killed {killed} stale actor(s) in namespace={namespace}")


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t0:.1f}s")


@dataclass
class TaskShape:
    batch_size: int
    max_in_flight: int = 0


def plan_task_resources(*, batch_size: int = 50, **_) -> TaskShape:
    print(f"resources: ray=unlimited batch_size={batch_size}", flush=True)
    return TaskShape(batch_size=batch_size)


def chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_batches(label: str, items: list, submit_fn: Callable, shape: TaskShape) -> list:
    batches = chunks(items, shape.batch_size)
    total_batches = len(batches)
    total_items = len(items)
    results: list = []
    pending: list = []
    next_i = 0
    done_batches = 0
    done_items = 0
    t0 = time.perf_counter()
    cap = shape.max_in_flight if shape.max_in_flight > 0 else len(batches)
    while next_i < len(batches) or pending:
        while next_i < len(batches) and len(pending) < cap:
            pending.append(submit_fn(batches[next_i]))
            next_i += 1
        ready, pending = ray.wait(pending, num_returns=1)
        batch = ray.get(ready[0])
        results.append(batch)
        done_batches += 1
        done_items += len(batch) if isinstance(batch, list) else 1
        elapsed = time.perf_counter() - t0
        rate = done_items / elapsed if elapsed else 0
        print(
            f"{label}: {done_batches}/{total_batches} batches "
            f"{done_items}/{total_items} ({rate:.1f}/s, {elapsed:.0f}s)",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    avg_batch = total_items / total_batches if total_batches else 0
    print(f"{label}: done {elapsed:.1f}s batches={total_batches} avg_batch={avg_batch:.1f}", flush=True)
    return results
