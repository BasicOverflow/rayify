from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import pandas as pd
import psycopg
import pyarrow.fs as pafs
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from psycopg.rows import dict_row

from lexis_markets.config import MarketsConfig

OBS_COLUMNS = [
    "source",
    "source_symbol",
    "series_type",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "dividend",
    "split",
    "currency",
    "fetched_at",
    "realtime_start",
    "realtime_end",
    "extras",
]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calendars (
    calendar_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    timezone TEXT NOT NULL,
    rules JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS series_meta (
    series_id TEXT PRIMARY KEY,
    canonical_symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    series_type TEXT,
    exchange TEXT,
    country TEXT,
    currency TEXT DEFAULT 'USD',
    calendar_id TEXT REFERENCES calendars(calendar_id),
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    first_seen DATE,
    last_seen DATE,
    gap_count INT NOT NULL DEFAULT 0,
    disagreement_count INT NOT NULL DEFAULT 0,
    suspicious_count INT NOT NULL DEFAULT 0,
    quality_score DOUBLE PRECISION,
    extras JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS symbol_aliases (
    id BIGSERIAL PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    source TEXT NOT NULL,
    source_symbol TEXT NOT NULL,
    valid_from DATE,
    valid_to DATE,
    UNIQUE (series_id, source, source_symbol)
);

CREATE TABLE IF NOT EXISTS series_links (
    id BIGSERIAL PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    link_type TEXT NOT NULL,
    related_series_id TEXT,
    effective_date DATE,
    note TEXT
);

CREATE TABLE IF NOT EXISTS stitch_segments (
    id BIGSERIAL PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    source TEXT NOT NULL,
    source_symbol TEXT NOT NULL,
    valid_from DATE,
    valid_to DATE,
    role TEXT NOT NULL DEFAULT 'primary',
    method TEXT
);

CREATE TABLE IF NOT EXISTS stitch_decisions (
    id BIGSERIAL PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series_meta(series_id),
    ts DATE NOT NULL,
    winner_source TEXT NOT NULL,
    loser_source TEXT NOT NULL,
    field TEXT NOT NULL DEFAULT 'close',
    winner_value DOUBLE PRECISION,
    loser_value DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS dataset_jobs (
    job_id UUID PRIMARY KEY,
    status TEXT NOT NULL,
    spec JSONB NOT NULL,
    spec_hash TEXT,
    s3_prefix TEXT,
    series_count INT,
    row_count INT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS l1_month_manifest (
    year INT NOT NULL,
    month INT NOT NULL,
    compacted_key TEXT NOT NULL,
    row_count BIGINT,
    symbol_count INT,
    compacted_at TIMESTAMPTZ,
    PRIMARY KEY (year, month)
);

CREATE TABLE IF NOT EXISTS symbol_month_coverage (
    source TEXT NOT NULL,
    source_symbol TEXT NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    PRIMARY KEY (source, source_symbol, year, month)
);

CREATE TABLE IF NOT EXISTS l3_month_manifest (
    year INT NOT NULL,
    month INT NOT NULL,
    stitched_key TEXT NOT NULL,
    row_count BIGINT,
    series_count INT,
    l1_fp TEXT,
    stitch_rule TEXT,
    PRIMARY KEY (year, month)
);

INSERT INTO calendars (calendar_id, name, timezone, rules)
VALUES
    ('nyse', 'NYSE equity sessions', 'America/New_York', '{"kind":"equity_session"}'::jsonb),
    ('fred_native', 'FRED native frequency', 'America/New_York', '{"kind":"fred_native"}'::jsonb)
ON CONFLICT (calendar_id) DO NOTHING;
"""

INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_stitch_segments_lookup
    ON stitch_segments (series_id);
CREATE INDEX IF NOT EXISTS idx_series_meta_asset_status
    ON series_meta (asset_class, status);
CREATE INDEX IF NOT EXISTS idx_symbol_aliases_upper_symbol
    ON symbol_aliases (UPPER(source_symbol));
CREATE INDEX IF NOT EXISTS idx_dataset_jobs_spec_hash_complete
    ON dataset_jobs (spec_hash) WHERE status = 'complete';
"""

MIGRATION_SQL = """
ALTER TABLE stitch_segments DROP COLUMN IF EXISTS universe_version;
ALTER TABLE stitch_segments DROP COLUMN IF EXISTS rule_version;
ALTER TABLE stitch_decisions DROP COLUMN IF EXISTS universe_version;
ALTER TABLE stitch_decisions DROP COLUMN IF EXISTS rule_version;
ALTER TABLE dataset_jobs DROP COLUMN IF EXISTS universe_version;
ALTER TABLE dataset_jobs DROP COLUMN IF EXISTS rule_version;
ALTER TABLE l3_month_manifest ADD COLUMN IF NOT EXISTS l1_fp TEXT;
ALTER TABLE l3_month_manifest ADD COLUMN IF NOT EXISTS stitch_rule TEXT;
DROP INDEX IF EXISTS idx_stitch_segments_lookup;
"""

STITCHED_BAR_COLUMNS = [
    "series_id",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "source",
    "source_count",
    "data_quality",
]


class PgClient:
    def __init__(self, url: str):
        self.url = url


    @contextmanager
    def connect(self):
        conn = psycopg.connect(self.url, row_factory=dict_row)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


    def execute(self, sql: str, params=None):
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)


    def executemany(self, sql: str, params_seq):
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params_seq)


    def fetchall(self, sql: str, params=None) -> list[dict]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())


    def fetchone(self, sql: str, params=None) -> dict | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()


def ensure_schema(pg: PgClient) -> None:
    with pg.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute(MIGRATION_SQL)
            cur.execute(INDEXES_SQL)


def write_parquet_lake(
    lake: LakeStore,
    key: str,
    df: pd.DataFrame,
    *,
    sort_by: list[str] | None = None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = df.copy()
    if sort_by:
        out = out.sort_values(sort_by)
    table = pa.Table.from_pandas(out, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(
        table,
        buf,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=256_000,
    )
    lake.put_bytes(key, buf.getvalue(), "application/octet-stream")


class LakeStore:
    def __init__(self, cfg: MarketsConfig):
        self.bucket = cfg.s3_bucket
        self._s3_endpoint = cfg.s3_endpoint
        self._s3_access_key = cfg.s3_access_key
        self._s3_secret_key = cfg.s3_secret_key
        self._s3_region = cfg.s3_region
        self.client = boto3.client(
            "s3",
            endpoint_url=cfg.s3_endpoint,
            aws_access_key_id=cfg.s3_access_key,
            aws_secret_access_key=cfg.s3_secret_key,
            region_name=cfg.s3_region,
            config=Config(
                s3={"addressing_style": "path"},
                retries={"max_attempts": 10, "mode": "adaptive"},
                connect_timeout=60,
                read_timeout=300,
            ),
        )


    def pyarrow_fs(self) -> pafs.S3FileSystem:
        u = urlparse(self._s3_endpoint)
        return pafs.S3FileSystem(
            access_key=self._s3_access_key,
            secret_key=self._s3_secret_key,
            region=self._s3_region,
            endpoint_override=u.netloc,
            scheme=u.scheme or "http",
            force_virtual_addressing=False,
        )


    def s3_path(self, key: str) -> str:
        return f"{self.bucket}/{key.lstrip('/')}"


    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)


    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream"):
        self.client.upload_file(
            Filename=str(path),
            Bucket=self.bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )


    def put_df_parquet(self, key: str, df: pd.DataFrame):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.put_bytes(key, buf.getvalue(), "application/octet-stream")


    def get_bytes(self, key: str, *, attempts: int = 6) -> bytes:
        last = None
        for i in range(attempts):
            try:
                obj = self.client.get_object(Bucket=self.bucket, Key=key)
                return obj["Body"].read()
            except (BotoCoreError, ClientError, OSError, ConnectionError) as e:
                last = e
                time.sleep(min(2 ** i, 20))
        raise last


    def download_file(self, key: str, path: Path, *, attempts: int = 8):
        path.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for i in range(attempts):
            try:
                self.client.download_file(self.bucket, key, str(path))
                return
            except (BotoCoreError, ClientError, OSError, ConnectionError) as e:
                last = e
                time.sleep(min(2 ** i, 30))
        raise last


    def get_df_parquet(
        self,
        key: str,
        *,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        return pd.read_parquet(
            io.BytesIO(self.get_bytes(key)),
            columns=columns,
            filters=filters,
            engine="pyarrow",
        )


    def get_dfs_parquet_parallel(
        self, keys: list[str], *, columns: list[str] | None = None, max_workers: int = 16
    ) -> list[pd.DataFrame]:
        if not keys:
            return []
        workers = min(max_workers, len(keys))

        def load(key: str) -> pd.DataFrame:
            return self.get_df_parquet(key, columns=columns)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(load, keys))


    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


    def has_prefix(self, prefix: str) -> bool:
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
        return bool(resp.get("Contents"))


    def list_keys(self, prefix: str, *, attempts: int = 8) -> list[str]:
        last = None
        for i in range(attempts):
            try:
                keys: list[str] = []
                token = None
                while True:
                    kwargs = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": 1000}
                    if token:
                        kwargs["ContinuationToken"] = token
                    resp = self.client.list_objects_v2(**kwargs)
                    for item in resp.get("Contents", []):
                        keys.append(item["Key"])
                    if not resp.get("IsTruncated"):
                        break
                    token = resp["NextContinuationToken"]
                return keys
            except (BotoCoreError, ClientError, OSError, ConnectionError) as e:
                last = e
                time.sleep(min(2 ** i, 30))
        raise last


    def list_common_prefixes(self, prefix: str, *, attempts: int = 8) -> list[str]:
        last = None
        for i in range(attempts):
            try:
                out: list[str] = []
                token = None
                while True:
                    kwargs = {"Bucket": self.bucket, "Prefix": prefix, "Delimiter": "/", "MaxKeys": 1000}
                    if token:
                        kwargs["ContinuationToken"] = token
                    resp = self.client.list_objects_v2(**kwargs)
                    for p in resp.get("CommonPrefixes", []):
                        out.append(p["Prefix"])
                    if not resp.get("IsTruncated"):
                        break
                    token = resp["NextContinuationToken"]
                return out
            except (BotoCoreError, ClientError, OSError, ConnectionError) as e:
                last = e
                time.sleep(min(2 ** i, 30))
        raise last


    def delete_keys(self, keys: Iterable[str]):
        batch = list(keys)
        for i in range(0, len(batch), 1000):
            chunk = batch[i : i + 1000]
            if not chunk:
                continue
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in chunk]},
            )


    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"


def put_json(lake: LakeStore, key: str, obj) -> None:
    lake.put_bytes(key, json.dumps(obj).encode(), "application/json")


def get_json(lake: LakeStore, key: str):
    return json.loads(lake.get_bytes(key))


def normalize_obs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OBS_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[OBS_COLUMNS]
    out["ts"] = pd.to_datetime(out["ts"]).dt.date
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], utc=True)
    return out


def month_in_range(y: int, m: int, start: date, end: date) -> bool:
    return (y, m) >= (start.year, start.month) and (y, m) <= (end.year, end.month)


def compacted_data_key(lake: LakeStore, pg: PgClient, year: int, month: int) -> str | None:
    row = pg.fetchone(
        "SELECT compacted_key FROM l1_month_manifest WHERE year = %s AND month = %s",
        (year, month),
    )
    if row:
        return row["compacted_key"]
    prefix = month_prefix(year, month)
    keys = sorted(k for k in lake.list_keys(prefix) if "/compacted-" in k and k.endswith(".parquet"))
    return keys[-1] if keys else None


def month_prefix(year: int, month: int) -> str:
    return f"layer_1/year={year:04d}/month={month:02d}/"


def part_key(year: int, month: int, run_id: str, shard: str) -> str:
    return f"{month_prefix(year, month)}part-{run_id}-{shard}.parquet"


def months_in_range(start: date, end: date) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def symbol_stats(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    rows = []
    for (sym, stype, source), g in df.groupby(["source_symbol", "series_type", "source"], sort=False):
        ts = pd.to_datetime(g["ts"])
        rows.append(
            {
                "symbol": str(sym).upper(),
                "source": str(source),
                "series_type": str(stype),
                "rows": int(len(g)),
                "unique_days": int(ts.dt.normalize().nunique()),
                "months_written": int(ts.dt.to_period("M").nunique()),
                "first": str(ts.min().date()),
                "last": str(ts.max().date()),
            }
        )
    return rows


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ObsWriter:
    def __init__(self, lake: LakeStore, *, run_id: str | None = None):
        self.lake = lake
        self.run_id = run_id or uuid4().hex[:12]


    def write_parts(self, df: pd.DataFrame, *, shard: str) -> dict:
        df = normalize_obs(df)
        if df.empty:
            return {"keys": [], "months_written": 0, "rows": 0, "details": []}
        keys: list[str] = []
        rows = 0
        work = df.copy()
        work["_y"] = pd.to_datetime(work["ts"]).dt.year
        work["_m"] = pd.to_datetime(work["ts"]).dt.month
        for (y, m), part in work.groupby(["_y", "_m"], sort=False):
            key = part_key(int(y), int(m), self.run_id, shard)
            self.lake.put_df_parquet(key, part.drop(columns=["_y", "_m"]))
            keys.append(key)
            rows += len(part)
        return {
            "keys": keys,
            "months_written": len(keys),
            "rows": rows,
            "details": symbol_stats(df),
        }
