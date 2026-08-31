from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RULE_VERSION = "exchange_priority_v1"
SOURCE_PRIORITY = ("jakewright", "jacksoncrow", "marketparquet", "yfinance", "fred")
FILL_SOURCES = frozenset({"marketparquet", "yfinance"})
PRICE_COLS = ("open", "high", "low", "close")
CALIBRATION_MIN_RATIO = 0.5
CALIBRATION_MAX_RATIO = 2.0
CALIBRATION_MIN_OVERLAP_DAYS = 2

# Markets-relevant FRED seed (Athena-style core + common rates). Expand via releases.
DEFAULT_FRED_SERIES = (
    "DGS10",
    "DGS2",
    "DGS30",
    "DFF",
    "T10Y2Y",
    "T10YIE",
    "CPIAUCSL",
    "PCEPI",
    "UNRATE",
    "PAYEMS",
    "INDPRO",
    "HOUST",
    "RSAFS",
    "UMCSENT",
    "VIXCLS",
    "DTWEXBGS",
    "DEXUSEU",
    "DEXCHUS",
    "DCOILWTICO",
    "M2SL",
    "WALCL",
    "BAMLH0A0HYM2",
    "TEDRATE",
    "SOFR",
    "FEDFUNDS",
    "GS10",
    "AAA",
    "BAA",
    "SP500",
)

FRED_RELEASE_IDS = (18,)  # H.15 Selected Interest Rates


@dataclass
class MarketsConfig:
    ray_address: str
    ray_namespace: str
    ray_serve_url: str
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    s3_region: str
    postgres_url: str
    fred_api_key: str
    kaggle_api_token: str
    runtime_env: dict = field(default_factory=dict)
    fred_series: tuple[str, ...] = DEFAULT_FRED_SERIES
    fred_release_ids: tuple[int, ...] = FRED_RELEASE_IDS
    eod_pace_seconds: float = 5.0
    eod_chunk_size: int = 25
    eod_start_slop_days: int = 90
    discover_min_history_days: int = 252
    minio_max_workers: int = 8
    eod_stale_retries: int = 5


    @classmethod
    def from_env(cls) -> MarketsConfig:
        runtime_raw = os.environ.get("RAY_RUNTIME_ENV", "").strip()
        runtime_env = json.loads(runtime_raw) if runtime_raw else {}
        pg = os.environ.get("POSTGRES_FAST_URL") or os.environ["DATABASE_URL"]
        s3_endpoint = (
            os.environ.get("MINIO_FAST_ENDPOINT")
            or os.environ.get("S3_ENDPOINT_URL")
            or os.environ["MINIO_ENDPOINT"]
        )
        s3_bucket = os.environ.get("MINIO_FAST_BUCKET") or os.environ["MINIO_BUCKET"]
        return cls(
            ray_address=os.environ["RAY_ADDRESS"],
            ray_namespace=os.environ["RAY_NAMESPACE"],
            # App lives at Serve route_prefix /markets (Hive keeps /v1 and /{model_id}).
            ray_serve_url=os.environ.get("RAY_SERVE_URL", "http://10.0.1.52:8000/markets"),
            s3_endpoint=s3_endpoint,
            s3_access_key=os.environ.get("AWS_ACCESS_KEY_ID") or os.environ["MINIO_ACCESS_KEY"],
            s3_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ["MINIO_SECRET_KEY"],
            s3_bucket=s3_bucket,
            s3_region=os.environ.get("MINIO_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            postgres_url=pg,
            fred_api_key=os.environ["FRED_API_KEY"],
            kaggle_api_token=os.environ["KAGGLE_API_TOKEN"],
            runtime_env=runtime_env,
            eod_pace_seconds=float(os.environ.get("EOD_PACE_SECONDS") or 5.0),
            eod_chunk_size=int(os.environ.get("EOD_CHUNK_SIZE") or 25),
            eod_start_slop_days=int(os.environ.get("EOD_START_SLOP_DAYS") or 90),
            discover_min_history_days=int(os.environ.get("DISCOVER_MIN_HISTORY_DAYS") or 252),
            minio_max_workers=int(os.environ.get("MINIO_MAX_WORKERS") or 8),
            eod_stale_retries=int(os.environ.get("EOD_STALE_RETRIES") or 5),
        )


    def to_dict(self) -> dict:
        return {
            "ray_address": self.ray_address,
            "ray_namespace": self.ray_namespace,
            "ray_serve_url": self.ray_serve_url,
            "s3_endpoint": self.s3_endpoint,
            "s3_access_key": self.s3_access_key,
            "s3_secret_key": self.s3_secret_key,
            "s3_bucket": self.s3_bucket,
            "s3_region": self.s3_region,
            "postgres_url": self.postgres_url,
            "fred_api_key": self.fred_api_key,
            "kaggle_api_token": self.kaggle_api_token,
            "runtime_env": self.runtime_env,
            "fred_series": list(self.fred_series),
            "fred_release_ids": list(self.fred_release_ids),
            "eod_pace_seconds": self.eod_pace_seconds,
            "eod_chunk_size": self.eod_chunk_size,
            "eod_start_slop_days": self.eod_start_slop_days,
            "discover_min_history_days": self.discover_min_history_days,
            "minio_max_workers": self.minio_max_workers,
            "eod_stale_retries": self.eod_stale_retries,
        }


    @classmethod
    def from_dict(cls, d: dict) -> MarketsConfig:
        return cls(
            ray_address=d["ray_address"],
            ray_namespace=d["ray_namespace"],
            ray_serve_url=d["ray_serve_url"],
            s3_endpoint=d["s3_endpoint"],
            s3_access_key=d["s3_access_key"],
            s3_secret_key=d["s3_secret_key"],
            s3_bucket=d["s3_bucket"],
            s3_region=d["s3_region"],
            postgres_url=d["postgres_url"],
            fred_api_key=d["fred_api_key"],
            kaggle_api_token=d["kaggle_api_token"],
            runtime_env=d.get("runtime_env") or {},
            fred_series=tuple(d.get("fred_series") or ()),
            fred_release_ids=tuple(d.get("fred_release_ids") or ()),
            eod_pace_seconds=float(d.get("eod_pace_seconds", 5.0)),
            eod_chunk_size=int(d.get("eod_chunk_size", 25)),
            eod_start_slop_days=int(d.get("eod_start_slop_days", 90)),
            discover_min_history_days=int(d.get("discover_min_history_days", 252)),
            minio_max_workers=int(d.get("minio_max_workers", 8)),
            eod_stale_retries=int(d.get("eod_stale_retries", 5)),
        )
