from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from ray import serve

from lexis_markets.cluster import SERVE_APP_NAME
from lexis_markets.config import MarketsConfig
from lexis_markets.dataset_jobs import build_dataset_snapshot, task_build_dataset_snapshot
from lexis_markets.quality import apply_min_volume, series_quality_window
from lexis_markets.stitch_cache import (
    cache_status,
    ensure_stitched_cache,
    l1_fingerprint,
    read_stitched_cache,
)
from lexis_markets.stitch_engine import clip_range, resample_bars, stitch_live
from lexis_markets.storage import LakeStore, PgClient, get_json

SERVE_ROUTE_PREFIX = "/markets"
app = FastAPI(title="Lexis Markets")


class DatasetSpec(BaseModel):
    series_ids: list[str] | None = None
    asset_classes: list[str] | None = None
    symbols: list[str] | None = None
    start: str
    end: str
    granularity: str = "daily"
    revision_mode: str = "as_of"
    include_delisted: bool = True
    min_volume: float | None = None


class CacheWarmSpec(BaseModel):
    start: str
    end: str
    force: bool = False


def _spec_hash(spec: dict, pg: PgClient) -> str:
    start = date.fromisoformat(spec["start"])
    end = date.fromisoformat(spec["end"])
    fp = l1_fingerprint(pg, start, end)
    payload = json.dumps({"spec": spec, "fp": fp}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_series_ids(pg: PgClient, spec: dict) -> list[str]:
    if spec.get("series_ids"):
        return list(spec["series_ids"])
    if spec.get("symbols"):
        rows = pg.fetchall(
            """
            SELECT DISTINCT series_id FROM symbol_aliases
            WHERE UPPER(source_symbol) = ANY(%s)
            """,
            ([s.upper() for s in spec["symbols"]],),
        )
        return [r["series_id"] for r in rows]
    sql = "SELECT series_id FROM series_meta WHERE 1=1"
    params: list = []
    if spec.get("asset_classes"):
        sql += " AND asset_class = ANY(%s)"
        params.append(spec["asset_classes"])
    if not spec.get("include_delisted", True):
        sql += " AND status = 'ACTIVE'"
    rows = pg.fetchall(sql, params or None)
    return [r["series_id"] for r in rows]


def _prefix_to_key(lake: LakeStore, s3_prefix: str) -> str:
    needle = f"s3://{lake.bucket}/"
    if s3_prefix.startswith(needle):
        return s3_prefix[len(needle) :]
    return s3_prefix.rstrip("/") + "/"


def manifest_for_prefix(lake: LakeStore, s3_prefix: str) -> dict | None:
    key = _prefix_to_key(lake, s3_prefix)
    manifest_key = f"{key}manifest.json" if key.endswith("/") else f"{key}/manifest.json"
    if not lake.exists(manifest_key):
        return None
    return get_json(lake, manifest_key)


class MarketsService:
    def __init__(self, cfg: MarketsConfig):
        self.cfg = cfg
        self.pg = PgClient(cfg.postgres_url)
        self.lake = LakeStore(cfg)


    def query_stitched_bars(self, spec: dict) -> pd.DataFrame:
        series_ids = resolve_series_ids(self.pg, spec)
        start = date.fromisoformat(spec["start"])
        end = date.fromisoformat(spec["end"])
        start, end = clip_range(self.pg, series_ids, start, end)
        revision_mode = spec.get("revision_mode", "as_of")
        granularity = spec.get("granularity", "daily")

        if revision_mode == "as_of":
            ensure_stitched_cache(self.cfg, self.pg, self.lake, start, end)
            cached = read_stitched_cache(
                self.lake,
                self.pg,
                series_ids,
                start,
                end,
                max_workers=self.cfg.minio_max_workers,
            )
            if cached is not None:
                bars = resample_bars(cached, granularity)
                return apply_min_volume(bars, spec.get("min_volume"))

        bars = stitch_live(
            self.lake,
            self.pg,
            series_ids,
            start,
            end,
            revision_mode=revision_mode,
        )
        bars = resample_bars(bars, granularity)
        return apply_min_volume(bars, spec.get("min_volume"))


    def series_meta_row(self, series_id: str) -> dict | None:
        return self.pg.fetchone(
            """
            SELECT series_id, calendar_id, gap_count, disagreement_count, quality_score,
                   first_seen, last_seen, status
            FROM series_meta WHERE series_id = %s
            """,
            (series_id,),
        )


    def query_series(self, series_id: str, start: str, end: str, spec: dict | None = None) -> dict:
        body = {
            "series_ids": [series_id],
            "start": start,
            "end": end,
            "granularity": (spec or {}).get("granularity", "daily"),
            "revision_mode": (spec or {}).get("revision_mode", "as_of"),
        }
        bars = self.query_stitched_bars(body)
        meta = self.series_meta_row(series_id)
        calendar_id = (meta or {}).get("calendar_id") or "nyse"
        return {
            "series_id": series_id,
            "start": start,
            "end": end,
            "quality": series_quality_window(bars, calendar_id, meta),
            "rows": _rows_json(bars),
        }


    def find_reusable_dataset(self, spec: dict) -> dict | None:
        sh = _spec_hash(spec, self.pg)
        row = self.pg.fetchone(
            """
            SELECT job_id, status, s3_prefix, series_count, row_count, finished_at
            FROM dataset_jobs
            WHERE spec_hash = %s AND status = 'complete'
            ORDER BY finished_at DESC NULLS LAST
            LIMIT 1
            """,
            (sh,),
        )
        if not row:
            return None
        manifest = manifest_for_prefix(self.lake, row["s3_prefix"]) or {}
        return {
            "job_id": str(row["job_id"]),
            "status": row["status"],
            "s3_prefix": row["s3_prefix"],
            "files": manifest.get("files", ["part-000.parquet"]),
            "series_count": row["series_count"],
            "row_count": row["row_count"],
            "reused": True,
        }


    def build_snapshot(self, job_id: str, spec: dict) -> dict:
        return build_dataset_snapshot(self.cfg, job_id, spec)


    def enqueue_dataset_job(self, spec: dict) -> dict:
        job_id = str(uuid4())
        sh = _spec_hash(spec, self.pg)
        prefix = f"s3://{self.lake.bucket}/layer_3/datasets/{job_id}/"
        self.pg.execute(
            """
            INSERT INTO dataset_jobs (job_id, status, spec, spec_hash, s3_prefix)
            VALUES (%s, 'queued', %s::jsonb, %s, %s)
            """,
            (job_id, json.dumps(spec), sh, prefix),
        )
        return {"job_id": job_id, "status": "queued", "s3_prefix": prefix}


    def create_dataset(self, spec: dict, *, sync: bool = False) -> dict:
        reused = self.find_reusable_dataset(spec)
        if reused:
            return reused
        body = self.enqueue_dataset_job(spec)
        job_id = body["job_id"]
        if sync:
            manifest = self.build_snapshot(job_id, spec)
            return {"job_id": job_id, "reused": False, **manifest}
        task_build_dataset_snapshot.remote(self.cfg.to_dict(), job_id, spec)
        return {
            "job_id": job_id,
            "status": "queued",
            "s3_prefix": body["s3_prefix"],
            "files": ["part-000.parquet"],
            "reused": False,
        }


    def get_dataset_job(self, job_id: str) -> dict:
        row = self.pg.fetchone("SELECT * FROM dataset_jobs WHERE job_id = %s", (job_id,))
        if not row:
            raise HTTPException(404, "job not found")
        manifest = manifest_for_prefix(self.lake, row["s3_prefix"]) if row.get("s3_prefix") else None
        out = {
            "job_id": str(row["job_id"]),
            "status": row["status"],
            "s3_prefix": row["s3_prefix"],
            "series_count": row["series_count"],
            "row_count": row["row_count"],
            "error": row.get("error"),
        }
        if manifest:
            out["files"] = manifest.get("files", [])
        elif row["status"] == "complete":
            out["files"] = ["part-000.parquet"]
        return out


    def warm_cache(self, start: str, end: str, *, force: bool = False) -> dict:
        warmed = ensure_stitched_cache(
            self.cfg,
            self.pg,
            self.lake,
            date.fromisoformat(start),
            date.fromisoformat(end),
            force=force,
        )
        return {**cache_status(self.pg), **warmed}


def query_stitched_bars(cfg: MarketsConfig, lake: LakeStore, pg: PgClient, spec: dict) -> pd.DataFrame:
    svc = MarketsService(cfg)
    svc.pg = pg
    svc.lake = lake
    return svc.query_stitched_bars(spec)


def build_snapshot(cfg: MarketsConfig, lake: LakeStore, pg: PgClient, job_id: str, spec: dict) -> dict:
    return build_dataset_snapshot(cfg, job_id, spec)


def enqueue_dataset_job(pg: PgClient, lake: LakeStore, spec: dict) -> dict:
    cfg = MarketsConfig.from_env()
    svc = MarketsService(cfg)
    svc.pg = pg
    svc.lake = lake
    return svc.enqueue_dataset_job(spec)


def _rows_json(bars: pd.DataFrame) -> list[dict]:
    records = bars.to_dict(orient="records")
    for r in records:
        ts = r.get("ts")
        if hasattr(ts, "isoformat"):
            r["ts"] = ts.isoformat()
        for k, v in r.items():
            if isinstance(v, float) and pd.isna(v):
                r[k] = None
    return records


@serve.deployment(name=SERVE_APP_NAME, num_replicas=1, max_ongoing_requests=32)
@serve.ingress(app)
class MarketsApi:
    def __init__(self, cfg_d: dict):
        self.svc = MarketsService(MarketsConfig.from_dict(cfg_d))


    @app.get("/health")
    async def health(self):
        return {"ok": True, "service": "lexis-markets"}


    @app.get("/v1/cache/status")
    async def cache_status_route(self):
        return await asyncio.to_thread(cache_status, self.svc.pg)


    @app.post("/v1/cache/warm")
    async def cache_warm(self, spec: CacheWarmSpec):
        return await asyncio.to_thread(self.svc.warm_cache, spec.start, spec.end, force=spec.force)


    @app.post("/v1/datasets")
    async def create_dataset(self, spec: DatasetSpec, sync: bool = Query(False)):
        body = spec.model_dump()
        return await asyncio.to_thread(self.svc.create_dataset, body, sync=sync)


    @app.get("/v1/datasets/{job_id}")
    async def get_dataset(self, job_id: str):
        return await asyncio.to_thread(self.svc.get_dataset_job, job_id)


    @app.get("/v1/series/{series_id:path}")
    async def get_series(self, series_id: str, start: str, end: str):
        return await asyncio.to_thread(self.svc.query_series, series_id, start, end)


def deploy_markets_api(cfg: MarketsConfig):
    import time

    try:
        serve.delete(SERVE_APP_NAME, _blocking=True)
        time.sleep(3)
    except Exception:
        pass

    return serve.run(
        MarketsApi.bind(cfg.to_dict()),
        name=SERVE_APP_NAME,
        route_prefix=SERVE_ROUTE_PREFIX,
    )


def ensure_markets_api(cfg: MarketsConfig):
    import httpx

    url = f"{cfg.ray_serve_url.rstrip('/')}/health"
    try:
        if httpx.get(url, timeout=10.0).status_code == 200:
            return
    except Exception:
        pass
    deploy_markets_api(cfg)
