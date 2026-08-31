from __future__ import annotations

import json

import ray

from lexis_markets.config import MarketsConfig
from lexis_markets.stitch_cache import ensure_stitched_cache
from lexis_markets.storage import LakeStore, PgClient, utcnow


def build_dataset_snapshot(cfg: MarketsConfig, job_id: str, spec: dict) -> dict:
    pg = PgClient(cfg.postgres_url)
    lake = LakeStore(cfg)
    pg.execute("UPDATE dataset_jobs SET status = 'running' WHERE job_id = %s", (job_id,))

    from lexis_markets.api import MarketsService

    if spec.get("revision_mode", "as_of") == "as_of":
        start_s, end_s = spec["start"], spec["end"]
        from datetime import date

        ensure_stitched_cache(cfg, pg, lake, date.fromisoformat(start_s), date.fromisoformat(end_s))

    svc = MarketsService(cfg)
    bars = svc.query_stitched_bars(spec)
    prefix = f"layer_3/datasets/{job_id}/"
    svc.lake.put_df_parquet(f"{prefix}part-000.parquet", bars)
    manifest = {
        "job_id": job_id,
        "status": "complete",
        "s3_prefix": lake.uri(prefix),
        "files": ["part-000.parquet"],
        "series_count": int(bars["series_id"].nunique()) if not bars.empty else 0,
        "row_count": len(bars),
    }
    lake.put_bytes(f"{prefix}manifest.json", json.dumps(manifest, indent=2).encode(), "application/json")
    pg.execute(
        """
        UPDATE dataset_jobs
        SET status = 'complete', s3_prefix = %s, series_count = %s, row_count = %s, finished_at = %s
        WHERE job_id = %s
        """,
        (manifest["s3_prefix"], manifest["series_count"], manifest["row_count"], utcnow(), job_id),
    )
    return manifest


@ray.remote
def task_build_dataset_snapshot(cfg_d: dict, job_id: str, spec: dict) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    try:
        return build_dataset_snapshot(cfg, job_id, spec)
    except Exception as e:
        pg = PgClient(cfg.postgres_url)
        pg.execute(
            "UPDATE dataset_jobs SET status = 'failed', error = %s, finished_at = %s WHERE job_id = %s",
            (str(e), utcnow(), job_id),
        )
        raise
