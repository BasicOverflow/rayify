"""Ray remote task: export a multi-series dataset to ``layer_3/outputs/{job_id}/``."""
from __future__ import annotations

import json

import ray

from lexis_markets.lake import LakeStore, PgClient, put_json, utcnow
from lexis_markets.config import MarketsConfig, OUTPUTS_PREFIX
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.runtime import DEFAULT_REMOTE_OPTS

logger = get_logger("serve.export")


def build_dataset_parquet(cfg: MarketsConfig, job_id: str, spec: dict) -> dict:
    from lexis_markets.domain.recipes import bars_to_wide_matrix, parse_nan_policy
    from lexis_markets.serve.handlers import MarketsService

    pg = PgClient(cfg.postgres_url)
    lake = LakeStore(cfg)
    pg.execute("UPDATE dataset_jobs SET status = 'running' WHERE job_id = %s", (job_id,))

    svc = MarketsService(cfg)
    bars = svc.query_stitched_bars(spec)
    if spec.get("mode") == "wide_matrix":
        bars = bars_to_wide_matrix(
            bars,
            value_col=spec.get("value_col") or "close",
            column_key=spec.get("column_key") or "series_id",
            nan_policy=parse_nan_policy(spec.get("nan_policy")),
        )
    prefix = f"{OUTPUTS_PREFIX}{job_id}/"
    dataset_key = f"{prefix}dataset.parquet"
    lake.put_df_parquet(dataset_key, bars)

    series_count = 0
    if not bars.empty:
        if "series_id" in bars.columns:
            series_count = int(bars["series_id"].nunique())
        else:
            # wide_matrix: columns are tickers (+ ts)
            series_count = max(0, len(bars.columns) - (1 if "ts" in bars.columns else 0))

    manifest = {
        "job_id": job_id,
        "status": "complete",
        "s3_prefix": lake.uri(prefix),
        "files": ["dataset.parquet"],
        "series_count": series_count,
        "row_count": len(bars),
        "mode": spec.get("mode"),
        "nan_policy": spec.get("nan_policy"),
    }
    put_json(lake, f"{prefix}manifest.json", manifest)
    pg.execute(
        """
        UPDATE dataset_jobs
        SET status = 'complete', s3_prefix = %s, series_count = %s, row_count = %s, finished_at = %s
        WHERE job_id = %s
        """,
        (manifest["s3_prefix"], manifest["series_count"], manifest["row_count"], utcnow(), job_id),
    )
    logger.info(
        "dataset_export job=%s mode=%s series=%s rows=%s",
        job_id,
        spec.get("mode"),
        manifest["series_count"],
        manifest["row_count"],
    )
    return manifest


@ray.remote(**DEFAULT_REMOTE_OPTS)
def task_build_dataset(cfg_d: dict, job_id: str, spec: dict) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    try:
        return build_dataset_parquet(cfg, job_id, spec)
    except Exception as exc:
        pg = PgClient(cfg.postgres_url)
        pg.execute(
            "UPDATE dataset_jobs SET status = 'failed', error = %s, finished_at = %s WHERE job_id = %s",
            (str(exc), utcnow(), job_id),
        )
        logger.exception("dataset_export failed job=%s", job_id)
        raise
