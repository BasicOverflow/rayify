from __future__ import annotations

import glob
import os
import shutil
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from lexis_markets.storage import LakeStore, PgClient, month_prefix

FRED_DONE_PREFIX = "ops/fred/done/"
JW_STAGING = "ops/staging/jakewright/parts/"
JC_STAGING = "ops/staging/jacksoncrow/parts/"
JW_STAGING_MARKER = "ops/staging/jakewright.json"
JC_STAGING_MARKER = "ops/staging/jacksoncrow.json"
JW_CACHE_KEY = "ops/cache/jakewright.zip"
JC_CACHE_KEY = "ops/cache/jacksoncrow.zip"

STAGING_SOURCES = (
    (JW_STAGING, JW_STAGING_MARKER, JW_CACHE_KEY),
    (JC_STAGING, JC_STAGING_MARKER, JC_CACHE_KEY),
)


def delete_prefix(lake: LakeStore, prefix: str) -> int:
    keys = lake.list_keys(prefix)
    if keys:
        lake.delete_keys(keys)
    return len(keys)


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


def clear_kaggle_source_build(lake: LakeStore, staging_prefix: str, staging_marker: str, cache_key: str) -> dict:
    return {
        "staging": clear_kaggle_staging(lake, staging_prefix, staging_marker),
        "zip": clear_kaggle_zip(lake, cache_key),
    }


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


@ray.remote(num_cpus=0.01, memory=64 * 1024 * 1024)
def _prune_worker_tmp(max_age_s: int = 3600) -> dict:
    removed = 0
    freed = 0
    cutoff = time.time() - max_age_s
    for base in glob.glob("/tmp/ray/session_*"):
        rt = os.path.join(base, "runtime_resources")
        if not os.path.isdir(rt):
            continue
        for name in os.listdir(rt):
            path = os.path.join(rt, name)
            try:
                if os.path.getmtime(path) >= cutoff:
                    continue
                if os.path.isdir(path):
                    freed += sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, files in os.walk(path)
                        for f in files
                    )
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    freed += os.path.getsize(path)
                    os.remove(path)
                removed += 1
            except OSError:
                pass
    return {"removed": removed, "freed_mb": round(freed / (1024 * 1024), 1)}


def prune_ray_worker_disk(*, max_age_s: int = 3600) -> list[dict]:
    refs = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_id = node.get("NodeID")
        if not node_id:
            continue
        refs.append(
            _prune_worker_tmp.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=True),
            ).remote(max_age_s)
        )
    if not refs:
        return []
    return ray.get(refs)
