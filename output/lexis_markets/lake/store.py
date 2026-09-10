"""MinIO (L1/L3) and Postgres adapters.

``LakeStore`` reads/writes parquet under the configured bucket.
``L1Writer`` shards ingest output into ``layer_1/year=YYYY/month=MM/part-*.parquet``.
``PgClient`` is a thin psycopg wrapper used by registry and pipelines.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from uuid import uuid4
from queue import Empty, Full, LifoQueue

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from lexis_markets.config import MarketsConfig
from lexis_markets.domain.raw_bar import RAW_BAR_COLUMNS

__all__ = [
    "LakeStore",
    "L1Writer",
    "PgClient",
    "ensure_schema",
    "write_parquet_lake",
    "normalize_raw_bars",
    "put_json",
    "get_json",
    "utcnow",
    "month_prefix",
    "part_key",
    "months_in_range",
    "month_in_range",
    "compacted_data_keys",
    "symbol_stats",
    "delete_prefix",
    "RAW_BAR_COLUMNS",
]

_S3_TRANSIENT = (BotoCoreError, ClientError, OSError, ConnectionError)
_PG_ATTEMPTS = 5
_S3_ATTEMPTS = 6


def _s3_not_found(exc: ClientError, *, allow_403_as_missing: bool = False) -> bool:
    code = exc.response.get("Error", {}).get("Code", "")
    if code in ("404", "NoSuchKey", "NotFound"):
        return True
    # Prefix-scoped MinIO creds may return 403 (not 404) for keys that do not exist yet.
    if allow_403_as_missing and exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 403:
        return True
    return False


def _retry_s3(fn, *, attempts: int = _S3_ATTEMPTS, max_sleep: int = 20):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except _S3_TRANSIENT as e:
            last = e
            time.sleep(min(2**i, max_sleep))
    raise last


class PgClient:
    """Pooled psycopg client.

    Reuses a small Lifo queue of connections per client instead of
    connect/close per query. Cap via ``PG_POOL_MAX`` (default 4) so Ray
    workers cannot open one socket per month ThreadPool worker.
    """

    def __init__(self, url: str, *, pool_max: int | None = None):
        self.url = url
        self._pool_max = max(1, int(pool_max if pool_max is not None else os.environ.get("PG_POOL_MAX", "4")))
        self._pool: LifoQueue = LifoQueue(maxsize=self._pool_max)
        self._created = 0
        self._lock = threading.Lock()

    def _retry(self, fn, *, attempts: int = _PG_ATTEMPTS):
        import psycopg

        last = None
        for i in range(attempts):
            try:
                return fn()
            except (psycopg.OperationalError, psycopg.InterfaceError) as e:
                last = e
                time.sleep(min(2**i, 20))
        raise last

    def _open(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.url, row_factory=dict_row)

    def _acquire(self):
        import psycopg

        while True:
            try:
                conn = self._pool.get_nowait()
            except Empty:
                with self._lock:
                    if self._created < self._pool_max:
                        self._created += 1
                        return self._open()
                conn = self._pool.get()
            if conn.closed:
                with self._lock:
                    self._created = max(0, self._created - 1)
                    if self._created < self._pool_max:
                        self._created += 1
                        return self._open()
                continue
            try:
                if conn.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS:
                    conn.rollback()
                return conn
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                with self._lock:
                    self._created = max(0, self._created - 1)
                    if self._created < self._pool_max:
                        self._created += 1
                        return self._open()

    def _release(self, conn, *, discard: bool = False):
        if discard or conn.closed:
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._created = max(0, self._created - 1)
            return
        try:
            self._pool.put_nowait(conn)
        except Full:
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._created = max(0, self._created - 1)

    def close(self) -> None:
        while True:
            try:
                conn = self._pool.get_nowait()
            except Empty:
                break
            try:
                conn.close()
            except Exception:
                pass
        with self._lock:
            self._created = 0

    @contextmanager
    def connect(self):
        import psycopg

        conn = self._acquire()
        discard = False
        try:
            yield conn
            conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                discard = True
            if isinstance(e, (psycopg.OperationalError, psycopg.InterfaceError)):
                discard = True
            raise
        finally:
            self._release(conn, discard=discard)

    def execute(self, sql: str, params=None):
        def _go():
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)

        return self._retry(_go)

    def execute_autocommit(self, sql: str):
        """Run a statement outside a transaction (ALTER SYSTEM, VACUUM, etc.)."""

        def _go():
            conn = self._acquire()
            discard = False
            try:
                prev = conn.autocommit
                conn.autocommit = True
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql)
                finally:
                    conn.autocommit = prev
            except Exception:
                discard = True
                raise
            finally:
                self._release(conn, discard=discard)

        return self._retry(_go)

    def executemany(self, sql: str, params_seq):
        def _go():
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.executemany(sql, params_seq)

        return self._retry(_go)

    def fetchall(self, sql: str, params=None) -> list[dict]:
        def _go():
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return list(cur.fetchall())

        return self._retry(_go)

    def fetchone(self, sql: str, params=None) -> dict | None:
        def _go():
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchone()

        return self._retry(_go)

    def run(self, fn):
        """Run ``fn()`` with retry on transient disconnect (for multi-statement transactions)."""
        return self._retry(fn)


def ensure_schema(pg: PgClient) -> None:
    from lexis_markets.registry import apply_schema

    apply_schema(pg)


class LakeStore:
    def __init__(self, cfg: MarketsConfig):
        self.bucket = cfg.s3_bucket
        self.prefix = (cfg.lake_prefix or "").strip()
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix = f"{self.prefix}/"
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

    def key(self, rel: str) -> str:
        rel = rel.lstrip("/")
        if not self.prefix:
            return rel
        if rel.startswith(self.prefix):
            return rel
        return f"{self.prefix}{rel}"

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        key = self.key(key)
        _retry_s3(
            lambda: self.client.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
            )
        )

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream"):
        key = self.key(key)
        _retry_s3(
            lambda: self.client.upload_file(
                Filename=str(path),
                Bucket=self.bucket,
                Key=key,
                ExtraArgs={"ContentType": content_type},
            ),
            max_sleep=30,
        )

    def put_df_parquet(self, key: str, df: pd.DataFrame):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.put_bytes(key, buf.getvalue(), "application/octet-stream")

    def get_bytes(self, key: str, *, attempts: int = _S3_ATTEMPTS) -> bytes:
        key = self.key(key)
        return _retry_s3(
            lambda: self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read(),
            attempts=attempts,
        )

    def download_file(self, key: str, path: Path, *, attempts: int = 8):
        key = self.key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for i in range(attempts):
            try:
                self.client.download_file(self.bucket, key, str(path))
                return
            except (BotoCoreError, ClientError, OSError, ConnectionError) as e:
                last = e
                time.sleep(min(2**i, 30))
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

    def exists(self, key: str, *, attempts: int = 4) -> bool:
        key = self.key(key)
        allow_403 = bool(self.prefix)
        last = None
        for i in range(attempts):
            try:
                self.client.head_object(Bucket=self.bucket, Key=key)
                return True
            except ClientError as e:
                if _s3_not_found(e, allow_403_as_missing=allow_403):
                    return False
                last = e
                time.sleep(min(2**i, 20))
            except (BotoCoreError, OSError, ConnectionError) as e:
                last = e
                time.sleep(min(2**i, 20))
        raise last

    def list_keys(self, prefix: str, *, attempts: int = 8) -> list[str]:
        prefix = self.key(prefix)
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
                time.sleep(min(2**i, 30))
        raise last

    def list_common_prefixes(self, prefix: str, *, attempts: int = 8) -> list[str]:
        prefix = self.key(prefix)
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
                time.sleep(min(2**i, 30))
        raise last

    def delete_keys(self, keys: Iterable[str]):
        batch = [self.key(k) for k in keys]
        for i in range(0, len(batch), 1000):
            chunk = batch[i : i + 1000]
            if chunk:
                _retry_s3(
                    lambda c=chunk: self.client.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": [{"Key": k} for k in c]},
                    ),
                    max_sleep=30,
                )

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self.key(key)}"


def put_json(lake: LakeStore, key: str, obj) -> None:
    lake.put_bytes(key, json.dumps(obj).encode(), "application/json")


def get_json(lake: LakeStore, key: str):
    return json.loads(lake.get_bytes(key))


def normalize_raw_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RAW_BAR_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[RAW_BAR_COLUMNS]
    out["ts"] = pd.to_datetime(out["ts"]).dt.date
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], utc=True)
    return out


def month_in_range(y: int, m: int, start: date, end: date) -> bool:
    return (y, m) >= (start.year, start.month) and (y, m) <= (end.year, end.month)


def compacted_data_keys(
    lake: LakeStore,
    pg: PgClient,
    months: list[tuple[int, int]],
) -> dict[tuple[int, int], str]:
    """Resolve compacted L1 keys for many months in one PG round-trip (+ S3 fallback)."""
    out: dict[tuple[int, int], str] = {}
    if not months:
        return out
    years = [y for y, _ in months]
    mos = [m for _, m in months]
    rows = pg.fetchall(
        """
        SELECT m.year, m.month, l.compacted_key
        FROM unnest(%s::int[], %s::int[]) AS m(year, month)
        LEFT JOIN l1_month_manifest l ON l.year = m.year AND l.month = m.month
        """,
        (years, mos),
    )
    missing: list[tuple[int, int]] = []
    for r in rows:
        ym = (int(r["year"]), int(r["month"]))
        key = r.get("compacted_key")
        if key:
            out[ym] = key
        else:
            missing.append(ym)
    for y, m in missing:
        prefix = month_prefix(y, m)
        keys = sorted(k for k in lake.list_keys(prefix) if "/compacted-" in k and k.endswith(".parquet"))
        if keys:
            out[(y, m)] = keys[-1]
    return out


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


def delete_prefix(lake: LakeStore, prefix: str) -> int:
    keys = lake.list_keys(prefix)
    if keys:
        lake.delete_keys(keys)
    return len(keys)


def write_parquet_lake(
    lake: LakeStore,
    key: str,
    df: pd.DataFrame,
    *,
    sort_by: list[str] | None = None,
) -> None:
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


class L1Writer:
    def __init__(self, lake: LakeStore, *, run_id: str | None = None):
        self.lake = lake
        self.run_id = run_id or uuid4().hex[:12]

    def write_parts(self, df: pd.DataFrame, *, shard: str) -> dict:
        df = normalize_raw_bars(df)
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
