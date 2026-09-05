"""Lake build-artifact purge and per-node Ray worker disk pruning.

Worker prune runs once per alive Ray node (NodeAffinity). It drops stale
``session_*`` dirs, truncates oversized session logs, and removes aged
``runtime_resources`` so emptyDir / overlay boot disks do not fill up.
"""
from __future__ import annotations

import glob
import os
import shutil
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from lexis_markets.lake import LakeStore, PgClient, delete_prefix, month_prefix
from lexis_markets.kaggle.ingest import (
    JC_CACHE_KEY,
    JC_STAGING,
    JC_STAGING_MARKER,
    JW_CACHE_KEY,
    JW_STAGING,
    JW_STAGING_MARKER,
)

FRED_DONE_PREFIX = "ops/fred/done/"

STAGING_SOURCES = (
    (JW_STAGING, JW_STAGING_MARKER, JW_CACHE_KEY),
    (JC_STAGING, JC_STAGING_MARKER, JC_CACHE_KEY),
)

DEFAULT_LOG_KEEP_BYTES = 50 * 1024 * 1024
DEFAULT_LOG_TRIGGER_BYTES = 200 * 1024 * 1024


def clear_kaggle_staging(lake: LakeStore, staging_prefix: str, staging_marker: str) -> int:
    n = delete_prefix(lake, staging_prefix)
    if lake.exists(staging_marker):
        lake.delete_keys([staging_marker])
        n += 1
    return n


def clear_kaggle_zip(lake: LakeStore, cache_key: str) -> int:
    if lake.exists(cache_key):
        lake.delete_keys([cache_key])
        return 1
    return 0


def clear_all_kaggle_build(lake: LakeStore, *, include_zips: bool = True) -> dict:
    out: dict = {}
    for staging_prefix, staging_marker, cache_key in STAGING_SOURCES:
        label = staging_prefix.split("/")[2]
        cleared = clear_kaggle_staging(lake, staging_prefix, staging_marker)
        out[f"{label}_staging"] = cleared
        if include_zips:
            out[f"{label}_zip"] = clear_kaggle_zip(lake, cache_key)
    return out


def clear_fred_done_markers(lake: LakeStore) -> int:
    return delete_prefix(lake, FRED_DONE_PREFIX)


def sweep_layer1_parts(lake: LakeStore, pg: PgClient) -> int:
    rows = pg.fetchall("SELECT year, month, compacted_key FROM l1_month_manifest")
    n = 0
    for r in rows:
        prefix = month_prefix(r["year"], r["month"])
        keys = [
            k
            for k in lake.list_keys(prefix)
            if "/part-" in k and k.endswith(".parquet") and k != r["compacted_key"]
        ]
        if keys:
            lake.delete_keys(keys)
            n += len(keys)
    return n


def purge_seed_build_artifacts(
    lake: LakeStore,
    pg: PgClient,
    *,
    include_kaggle_zips: bool = True,
    include_fred_done: bool = True,
) -> dict:
    out = clear_all_kaggle_build(lake, include_zips=include_kaggle_zips)
    if include_fred_done:
        out["fred_done"] = clear_fred_done_markers(lake)
    out["l1_parts"] = sweep_layer1_parts(lake, pg)
    return out


def _dir_bytes(path: str) -> int:
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def _resolve_current_session(ray_tmp: str = "/tmp/ray") -> str | None:
    latest = os.path.join(ray_tmp, "session_latest")
    if os.path.islink(latest) or os.path.exists(latest):
        try:
            return os.path.realpath(latest)
        except OSError:
            pass
    sessions = sorted(glob.glob(os.path.join(ray_tmp, "session_*")))
    return sessions[-1] if sessions else None


def _prune_old_sessions(ray_tmp: str, current: str | None) -> tuple[int, int]:
    removed = 0
    freed = 0
    for path in glob.glob(os.path.join(ray_tmp, "session_*")):
        if not os.path.isdir(path):
            continue
        if current and os.path.realpath(path) == current:
            continue
        try:
            freed += _dir_bytes(path)
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        except OSError:
            pass
    return removed, freed


def _truncate_large_logs(
    session_dir: str,
    *,
    trigger_bytes: int,
    keep_bytes: int,
) -> tuple[int, int]:
    trimmed = 0
    freed = 0
    logs = os.path.join(session_dir, "logs")
    if not os.path.isdir(logs):
        return 0, 0
    for dp, _, files in os.walk(logs):
        for name in files:
            path = os.path.join(dp, name)
            try:
                sz = os.path.getsize(path)
            except OSError:
                continue
            if sz < trigger_bytes:
                continue
            try:
                with open(path, "rb") as fh:
                    if sz > keep_bytes:
                        fh.seek(-keep_bytes, os.SEEK_END)
                    tail = fh.read()
                with open(path, "wb") as fh:
                    fh.write(b"...truncated...\n")
                    fh.write(tail)
                freed += sz - os.path.getsize(path)
                trimmed += 1
            except OSError:
                pass
    return trimmed, freed


def _prune_runtime_resources(session_dir: str, cutoff: float) -> tuple[int, int]:
    removed = 0
    freed = 0
    rt = os.path.join(session_dir, "runtime_resources")
    if not os.path.isdir(rt):
        return 0, 0
    for name in os.listdir(rt):
        path = os.path.join(rt, name)
        try:
            if os.path.getmtime(path) >= cutoff:
                continue
            if os.path.isdir(path):
                freed += _dir_bytes(path)
                shutil.rmtree(path, ignore_errors=True)
            else:
                freed += os.path.getsize(path)
                os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed, freed


@ray.remote(num_cpus=0.01, memory=64 * 1024 * 1024)
def _prune_worker_tmp(
    max_age_s: int = 3600,
    max_log_mb: int = 200,
    log_keep_mb: int = 50,
    drop_old_sessions: bool = True,
) -> dict:
    ray_tmp = "/tmp/ray"
    if not os.path.isdir(ray_tmp):
        return {
            "node_ip": ray.util.get_node_ip_address(),
            "skipped": "no_/tmp/ray",
            "freed_mb": 0.0,
        }

    current = _resolve_current_session(ray_tmp)
    sessions_removed = 0
    logs_trimmed = 0
    rr_removed = 0
    freed = 0

    if drop_old_sessions:
        n, b = _prune_old_sessions(ray_tmp, current)
        sessions_removed = n
        freed += b

    if current and os.path.isdir(current):
        n, b = _truncate_large_logs(
            current,
            trigger_bytes=max(1, max_log_mb) * 1024 * 1024,
            keep_bytes=max(1, log_keep_mb) * 1024 * 1024,
        )
        logs_trimmed = n
        freed += b
        n, b = _prune_runtime_resources(current, time.time() - max_age_s)
        rr_removed = n
        freed += b

    return {
        "node_ip": ray.util.get_node_ip_address(),
        "current_session": os.path.basename(current) if current else None,
        "sessions_removed": sessions_removed,
        "logs_trimmed": logs_trimmed,
        "runtime_resources_removed": rr_removed,
        "freed_mb": round(freed / (1024 * 1024), 1),
    }


def prune_ray_worker_disk(
    *,
    max_age_s: int = 3600,
    max_log_mb: int = 200,
    log_keep_mb: int = 50,
    drop_old_sessions: bool = True,
) -> list[dict]:
    """Pin a prune task on every alive node and return per-node stats.

    Dead/unscheduable nodes are skipped (soft failure). A single vanished node
    must not fail seed/cron — that was ``TaskUnschedulableError`` with ``soft=False``.
    """
    from lexis_markets.logging_setup import get_logger

    logger = get_logger("cleanup")
    refs = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_id = node.get("NodeID")
        if not node_id:
            continue
        refs.append(
            _prune_worker_tmp.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            ).remote(max_age_s, max_log_mb, log_keep_mb, drop_old_sessions)
        )
    if not refs:
        return []
    out: list[dict] = []
    for ref in refs:
        try:
            out.append(ray.get(ref))
        except Exception as exc:
            logger.warning("prune_ray_worker_disk: skip node (%s)", exc)
    return out
