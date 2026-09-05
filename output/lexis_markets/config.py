"""Environment-backed configuration for Lexis Markets v4.

Loads cluster credentials and tuning knobs from the repo root ``.env``.
Layer paths (L1 lake, L3 cache/outputs) and Serve route names live here.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SOURCE_PRIORITY = ("jakewright", "jacksoncrow", "marketparquet", "yfinance", "fred")
# Sources whose gaps may be scaled onto the prior segment during stitch calibration.
FILL_SOURCES = frozenset({"marketparquet", "yfinance"})
PRICE_COLS = ("open", "high", "low", "close")
CALIBRATION_MIN_RATIO = 0.5
CALIBRATION_MAX_RATIO = 2.0
CALIBRATION_MIN_OVERLAP_DAYS = 2

DEFAULT_FRED_SERIES = (
    # rates / policy / curves
    "DFF", "EFFR", "SOFR", "DPRIME",
    "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
    "T10Y2Y", "T10Y3M", "T10YFF", "T10YIE", "T5YIE", "T5YIFR", "DFII5", "DFII10",
    "MORTGAGE30US",
    # credit / financial conditions
    "BAMLH0A0HYM2", "BAMLC0A0CM", "STLFSI4", "NFCI", "ANFCI",
    # money / credit aggregates
    "BUSLOANS", "TOTCI", "DRALACBS", "M2SL", "BOGMBASE", "TOTRESNS", "WALCL",
    "RRPONTSYD", "WTREGEN",
    # inflation / prices
    "CPIAUCSL", "CPILFESL", "CPITRNSL", "CPIENGSL", "CPIHOSSL", "CPIFABSL",
    "PCEPI", "PCEPILFE", "PPIFIS", "PPIACO",
    # labor
    "UNRATE", "PAYEMS", "ICSA", "CCSA", "CIVPART", "EMRATIO", "AHETPI",
    # activity / sentiment / trade
    "INDPRO", "TCU", "UMCSENT", "RSAFS", "DGORDER", "GDPC1", "GDPPOT", "PCEC96",
    "DSPIC96", "HOUST", "PERMIT", "BOPGSTB",
    # FX
    "DTWEXBGS", "DTWEXAFEGS", "DTWEXEMEGS",
    "DEXUSEU", "DEXJPUS", "DEXCHUS", "DEXCAUS", "DEXUSUK", "DEXSZUS", "DEXMXUS",
    "DEXKOUS", "DEXINUS", "DEXBZUS", "DEXUSAL", "DEXUSNZ", "DEXTHUS", "DEXSDUS",
    # equities / vol / policy uncertainty
    "SP500", "DJIA", "NASDAQCOM", "VIXCLS", "USEPUINDXD", "GVZCLS", "OVXCLS",
    # energy
    "DCOILWTICO", "DCOILBRENTEU", "DHHNGSP", "DHOILNYH", "DJFUELUSGULF",
    "DPROPANEMBTX", "GASREGW", "GASDESW", "POILWTIUSDM", "POILBREUSDM",
    # metals / commodities (spot + IMF / BLS / USDA proxies)
    "IR14270", "PCOPPUSDM", "PALUMUSDM", "PNICKUSDM", "PZINCUSDM", "PLEADUSDM",
    "PIORECRUSDM", "PURANUSDM", "WPU081",
    "PWHEAMTUSDM", "PMAIZMTUSDM", "PSOYBUSDM", "PCOTTINDUSDM", "PCOFFROBUSDM",
    "PSUGAISAUSDM", "PCOCOUSDM", "PBEEFUSDM", "PPORKUSDM", "PRUBBUSDM",
)
FRED_RELEASE_IDS: tuple[int, ...] = ()
# Adaptive ALFRED bisect stops splitting below this window (days).
FRED_VINTAGE_MIN_DAYS_DEFAULT = 31
FRED_VINTAGE_MAX_WINDOW_DAYS_DEFAULT = 1825  # ~5y Ray fan-out windows
FRED_VINTAGE_START_DEFAULT = "1970-01-01"
FRED_PACE_PER_MINUTE_DEFAULT = 120.0
FRED_MAX_IN_FLIGHT_DEFAULT = 28
EOD_MAX_IN_FLIGHT_DEFAULT = 0
FRED_SERIES_PER_TASK_DEFAULT = 1
EOD_PACE_SECONDS_DEFAULT = 2.5
EOD_CHUNK_SIZE_DEFAULT = 150
EOD_CHUNK_SIZE_MIN_DEFAULT = 30
EOD_START_SLOP_DAYS_DEFAULT = 150
EOD_START_SLOP_DAYS_MIN_DEFAULT = 30
YF_PACE_MAX_SECONDS_DEFAULT = 30.0
YF_WAVE_JOBS_DEFAULT = 8

SERVE_APP_NAME = "lexis-markets"
SERVE_ROUTE_PREFIX = "/markets"
CACHE_PREFIX = "layer_3/cache/"
OUTPUTS_PREFIX = "layer_3/outputs/"


def series_id_to_cache_key(series_id: str) -> str:
    return series_id.replace(":", "__")


def cache_object_key(series_id: str) -> str:
    return f"{CACHE_PREFIX}{series_id_to_cache_key(series_id)}.parquet"


TEST_FRED_SERIES_DEFAULT = ("DFF", "DGS10", "UNRATE", "CPIAUCSL", "SP500")
TEST_IO_MAX = 10
TEST_EOD_IO_MAX = 100


@dataclass
class TestProfile:
    """Deterministic small samples for pytest (MARKETS_TEST_PROFILE=1)."""

    enabled: bool = False
    yf_limit: int = 5
    yf_seed: int = 42
    fred_series: tuple[str, ...] = TEST_FRED_SERIES_DEFAULT
    align_symbols: tuple[str, ...] = ("AAPL",)
    quality_limit: int = 5
    test_symbols: tuple[str, ...] | None = None
    eod_yf_limit: int = 5
    eod_full_path: bool = False
    eod_backfill_start: date = field(
        default_factory=lambda: date.today() - timedelta(days=90)
    )

    @classmethod
    def from_env(cls) -> TestProfile:
        enabled = os.environ.get("MARKETS_TEST_PROFILE", "").strip() in ("1", "true", "yes")
        yf_limit = min(int(os.environ.get("MARKETS_TEST_YF_LIMIT", "5") or 5), TEST_IO_MAX)
        yf_seed = int(os.environ.get("MARKETS_TEST_YF_SEED", "42") or 42)
        sym_raw = os.environ.get("MARKETS_TEST_SYMBOLS", "").strip()
        test_symbols = tuple(s.strip().upper() for s in sym_raw.split(",") if s.strip()) or None
        if test_symbols:
            test_symbols = test_symbols[:TEST_IO_MAX]
        fred_raw = os.environ.get("MARKETS_TEST_FRED_SERIES", "").strip()
        fred = tuple(s.strip().upper() for s in fred_raw.split(",") if s.strip()) if fred_raw else TEST_FRED_SERIES_DEFAULT
        fred = fred[:TEST_IO_MAX]
        align_raw = os.environ.get("MARKETS_TEST_ALIGN_SYMBOLS", "AAPL").strip()
        align = tuple(s.strip().upper() for s in align_raw.split(",") if s.strip())
        quality = min(int(os.environ.get("MARKETS_TEST_QUALITY_LIMIT", "5") or 5), TEST_IO_MAX)
        eod_start_raw = os.environ.get("MARKETS_TEST_EOD_START", "").strip()
        eod_backfill_start = (
            date.fromisoformat(eod_start_raw)
            if eod_start_raw
            else date.today() - timedelta(days=90)
        )
        eod_raw = os.environ.get("MARKETS_TEST_EOD_YF_LIMIT", "").strip()
        eod_yf_limit = (
            min(int(eod_raw), TEST_EOD_IO_MAX)
            if eod_raw
            else yf_limit
        )
        eod_full_path = os.environ.get("MARKETS_TEST_EOD_FULL_PATH", "").strip() in (
            "1",
            "true",
            "yes",
        )
        return cls(
            enabled=enabled,
            yf_limit=yf_limit,
            yf_seed=yf_seed,
            fred_series=fred,
            align_symbols=align,
            test_symbols=test_symbols,
            quality_limit=quality,
            eod_yf_limit=eod_yf_limit,
            eod_full_path=eod_full_path,
            eod_backfill_start=eod_backfill_start,
        )

    def _hash_rank(self, key: str, *, seed: int) -> int:
        return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16)

    def sample_targets(self, targets: list[dict]) -> list[dict]:
        if not self.enabled:
            return targets
        if self.test_symbols:
            allowed = set(self.test_symbols)
            picked = [t for t in targets if str(t.get("symbol", t.get("series_id", ""))).upper() in allowed]
            if picked:
                return picked[: self.yf_limit]
        if len(targets) <= self.yf_limit:
            return targets
        keyed = sorted(
            targets,
            key=lambda t: self._hash_rank(
                str(t.get("series_id", t.get("symbol", ""))),
                seed=self.yf_seed,
            ),
        )
        return keyed[: self.yf_limit]

    def sample_eod_targets(self, targets: list[dict], *, symbol_filter: tuple[str, ...] | None) -> list[dict]:
        """Hash-sample EOD stale targets; separate cap from general yf_limit."""
        if not self.enabled:
            return targets
        limit = self.eod_yf_limit
        if symbol_filter and len(symbol_filter) >= limit:
            allowed = set(symbol_filter)
            picked = [
                t
                for t in targets
                if str(t.get("symbol", t.get("series_id", ""))).upper() in allowed
            ]
            if picked:
                return picked[:limit]
        if len(targets) <= limit:
            return targets
        keyed = sorted(
            targets,
            key=lambda t: self._hash_rank(
                str(t.get("series_id", t.get("symbol", ""))),
                seed=self.yf_seed,
            ),
        )
        return keyed[:limit]

    def fred_series_ids(self, fallback: tuple[str, ...]) -> tuple[str, ...]:
        if not self.enabled:
            return fallback
        return self.fred_series

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "yf_limit": self.yf_limit,
            "yf_seed": self.yf_seed,
            "fred_series": list(self.fred_series),
            "align_symbols": list(self.align_symbols),
            "quality_limit": self.quality_limit,
            "test_symbols": list(self.test_symbols) if self.test_symbols else None,
            "eod_yf_limit": self.eod_yf_limit,
            "eod_full_path": self.eod_full_path,
            "eod_backfill_start": self.eod_backfill_start.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> TestProfile:
        if not d:
            return cls.from_env()
        syms = d.get("test_symbols")
        return cls(
            enabled=bool(d.get("enabled")),
            yf_limit=min(int(d.get("yf_limit", 5)), TEST_IO_MAX),
            yf_seed=int(d.get("yf_seed", 42)),
            fred_series=tuple(d.get("fred_series") or TEST_FRED_SERIES_DEFAULT),
            align_symbols=tuple(d.get("align_symbols") or ("AAPL",)),
            quality_limit=min(int(d.get("quality_limit", 5)), TEST_IO_MAX),
            test_symbols=tuple(syms) if syms else None,
            eod_yf_limit=min(int(d.get("eod_yf_limit", d.get("yf_limit", 5))), TEST_EOD_IO_MAX),
            eod_full_path=bool(d.get("eod_full_path")),
            eod_backfill_start=date.fromisoformat(
                d.get("eod_backfill_start", (date.today() - timedelta(days=90)).isoformat())
            ),
        )


@dataclass
class DevProfile:
    """Hash-sampled symbol subsets for local/dev runs."""

    yf_backfill_limit: int | None = None
    yf_backfill_seed: int = 0
    fred_backfill_limit: int | None = None
    fred_backfill_seed: int = 0

    @classmethod
    def from_env(cls) -> DevProfile:
        yf_raw = os.environ.get("MARKETS_DEV_YF_LIMIT", "").strip()
        yf_limit = int(yf_raw) if yf_raw else None
        yf_seed = int(os.environ.get("MARKETS_DEV_YF_SEED", "0") or 0)
        fred_raw = os.environ.get("MARKETS_DEV_FRED_LIMIT", "").strip()
        fred_limit = int(fred_raw) if fred_raw else None
        fred_seed = int(os.environ.get("MARKETS_DEV_FRED_SEED", "0") or 0)
        return cls(
            yf_backfill_limit=yf_limit,
            yf_backfill_seed=yf_seed,
            fred_backfill_limit=fred_limit,
            fred_backfill_seed=fred_seed,
        )

    def _hash_rank(self, key: str, *, seed: int) -> int:
        return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16)

    def sample_targets(self, targets: list[dict]) -> list[dict]:
        if not self.yf_backfill_limit or len(targets) <= self.yf_backfill_limit:
            return targets
        keyed = sorted(
            targets,
            key=lambda t: self._hash_rank(
                str(t.get("series_id", t.get("symbol", ""))),
                seed=self.yf_backfill_seed,
            ),
        )
        return keyed[: self.yf_backfill_limit]

    def sample_fred_series(self, series_ids: list[str]) -> list[str]:
        if not self.fred_backfill_limit or len(series_ids) <= self.fred_backfill_limit:
            return series_ids
        keyed = sorted(
            series_ids,
            key=lambda s: self._hash_rank(s.upper(), seed=self.fred_backfill_seed),
        )
        return keyed[: self.fred_backfill_limit]

    def to_dict(self) -> dict:
        return {
            "yf_backfill_limit": self.yf_backfill_limit,
            "yf_backfill_seed": self.yf_backfill_seed,
            "fred_backfill_limit": self.fred_backfill_limit,
            "fred_backfill_seed": self.fred_backfill_seed,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> DevProfile:
        if not d:
            return cls.from_env()
        return cls(
            yf_backfill_limit=d.get("yf_backfill_limit"),
            yf_backfill_seed=int(d.get("yf_backfill_seed") or 0),
            fred_backfill_limit=d.get("fred_backfill_limit"),
            fred_backfill_seed=int(d.get("fred_backfill_seed") or 0),
        )


@dataclass
class MarketsConfig:
    """Runtime settings loaded from root ``.env`` via ``from_env()``."""

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
    fred_vintage_start: date = field(default_factory=lambda: date.fromisoformat(FRED_VINTAGE_START_DEFAULT))
    fred_vintage_min_days: int = FRED_VINTAGE_MIN_DAYS_DEFAULT
    fred_vintage_max_window_days: int = FRED_VINTAGE_MAX_WINDOW_DAYS_DEFAULT
    fred_pace_per_minute: float = FRED_PACE_PER_MINUTE_DEFAULT
    fred_max_in_flight: int = FRED_MAX_IN_FLIGHT_DEFAULT
    fred_series_per_task: int = FRED_SERIES_PER_TASK_DEFAULT
    eod_pace_seconds: float = EOD_PACE_SECONDS_DEFAULT
    eod_max_in_flight: int = EOD_MAX_IN_FLIGHT_DEFAULT
    yf_pace_max_seconds: float = YF_PACE_MAX_SECONDS_DEFAULT
    yf_pace_backoff_factor: float = 2.0
    yf_pace_recovery_step: float = 0.1
    eod_chunk_size: int = EOD_CHUNK_SIZE_DEFAULT
    eod_chunk_size_min: int = EOD_CHUNK_SIZE_MIN_DEFAULT
    eod_start_slop_days: int = EOD_START_SLOP_DAYS_DEFAULT
    eod_start_slop_days_min: int = EOD_START_SLOP_DAYS_MIN_DEFAULT
    yf_wave_jobs: int = YF_WAVE_JOBS_DEFAULT
    discover_min_history_days: int = 252
    minio_max_workers: int = 8
    eod_delay_hours: float = 3.0
    eod_target_finish_hour_et: int = 7
    defer_heavy_before_hour_et: int = 9
    defer_heavy_after_hour_et: int = 23
    supervisor_state_path: str = ""
    lake_prefix: str = ""
    dev: DevProfile = field(default_factory=DevProfile.from_env)
    test: TestProfile = field(default_factory=TestProfile.from_env)

    def lake_key(self, rel: str) -> str:
        rel = rel.lstrip("/")
        if not self.lake_prefix:
            return rel
        prefix = self.lake_prefix if self.lake_prefix.endswith("/") else f"{self.lake_prefix}/"
        return f"{prefix}{rel}"

    def cache_object_key(self, series_id: str) -> str:
        return self.lake_key(cache_object_key(series_id))

    def outputs_prefix(self, job_id: str) -> str:
        return self.lake_key(f"{OUTPUTS_PREFIX}{job_id}/")

    def yf_backfill_limit(self) -> int | None:
        if self.test.enabled:
            return self.test.yf_limit
        return self.dev.yf_backfill_limit

    def sample_yf_targets(self, targets: list[dict]) -> list[dict]:
        if self.test.enabled:
            return self.test.sample_targets(targets)
        return self.dev.sample_targets(targets)

    def resolve_fred_series(self, series_ids: list[str]) -> list[str]:
        if self.test.enabled:
            fixed = list(self.test.fred_series)
            return fixed if fixed else series_ids[: self.test.yf_limit]
        return self.dev.sample_fred_series(series_ids)

    def seed_align_symbols(self, symbols: list[str]) -> list[str]:
        if self.test.enabled:
            allowed = set(self.test.align_symbols)
            picked = [s for s in symbols if s.upper() in allowed]
            return picked if picked else list(self.test.align_symbols)
        limit = self.dev.yf_backfill_limit
        if limit and len(symbols) > limit:
            sampled = self.dev.sample_targets([{"symbol": s, "series_id": s} for s in symbols])
            return [t["symbol"] for t in sampled]
        return symbols

    def eod_resolve_symbols(self) -> tuple[str, ...] | None:
        if not self.test.enabled:
            return None
        symbols = self.test.test_symbols or self.test.align_symbols
        if self.test.eod_yf_limit > len(symbols):
            return None
        return symbols

    def sample_eod_targets(self, targets: list[dict]) -> list[dict]:
        if not self.test.enabled:
            return targets
        sampled = self.test.sample_eod_targets(
            targets,
            symbol_filter=self.eod_resolve_symbols(),
        )
        clip = self.test.eod_backfill_start
        out: list[dict] = []
        for t in sampled:
            row = dict(t)
            if row["start"] < clip:
                row["start"] = clip
            if row["start"] <= row["end"]:
                out.append(row)
        return out

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
        state_path = os.environ.get(
            "MARKETS_SUPERVISOR_STATE",
            str(OUTPUT_DIR / ".supervisor" / "state.db"),
        )
        return cls(
            ray_address=os.environ["RAY_ADDRESS"],
            ray_namespace=os.environ["RAY_NAMESPACE"],
            ray_serve_url=os.environ["RAY_SERVE_URL"],
            s3_endpoint=s3_endpoint,
            s3_access_key=os.environ.get("AWS_ACCESS_KEY_ID") or os.environ["MINIO_ACCESS_KEY"],
            s3_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ["MINIO_SECRET_KEY"],
            s3_bucket=s3_bucket,
            s3_region=os.environ.get("MINIO_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            postgres_url=pg,
            fred_api_key=os.environ["FRED_API_KEY"],
            kaggle_api_token=os.environ["KAGGLE_API_TOKEN"],
            runtime_env=runtime_env,
            eod_pace_seconds=float(
                os.environ.get("EOD_PACE_SECONDS") or EOD_PACE_SECONDS_DEFAULT
            ),
            eod_max_in_flight=int(
                os.environ.get("EOD_MAX_IN_FLIGHT") or EOD_MAX_IN_FLIGHT_DEFAULT
            ),
            yf_pace_max_seconds=float(
                os.environ.get("YF_PACE_MAX_SECONDS") or YF_PACE_MAX_SECONDS_DEFAULT
            ),
            yf_pace_backoff_factor=float(os.environ.get("YF_PACE_BACKOFF_FACTOR") or 2.0),
            yf_pace_recovery_step=float(os.environ.get("YF_PACE_RECOVERY_STEP") or 0.1),
            eod_chunk_size=int(os.environ.get("EOD_CHUNK_SIZE") or EOD_CHUNK_SIZE_DEFAULT),
            eod_chunk_size_min=int(
                os.environ.get("EOD_CHUNK_SIZE_MIN") or EOD_CHUNK_SIZE_MIN_DEFAULT
            ),
            eod_start_slop_days=int(
                os.environ.get("EOD_START_SLOP_DAYS") or EOD_START_SLOP_DAYS_DEFAULT
            ),
            eod_start_slop_days_min=int(
                os.environ.get("EOD_START_SLOP_DAYS_MIN") or EOD_START_SLOP_DAYS_MIN_DEFAULT
            ),
            yf_wave_jobs=int(os.environ.get("YF_WAVE_JOBS") or YF_WAVE_JOBS_DEFAULT),
            discover_min_history_days=int(os.environ.get("DISCOVER_MIN_HISTORY_DAYS") or 252),
            minio_max_workers=int(os.environ.get("MINIO_MAX_WORKERS") or 8),
            eod_delay_hours=float(os.environ.get("MARKETS_EOD_DELAY_HOURS") or 3.0),
            eod_target_finish_hour_et=int(os.environ.get("MARKETS_EOD_TARGET_FINISH_HOUR_ET") or 7),
            defer_heavy_before_hour_et=int(os.environ.get("MARKETS_DEFER_HEAVY_BEFORE_HOUR_ET") or 9),
            defer_heavy_after_hour_et=int(os.environ.get("MARKETS_DEFER_HEAVY_AFTER_HOUR_ET") or 23),
            fred_vintage_min_days=int(
                os.environ.get("FRED_VINTAGE_MIN_DAYS") or FRED_VINTAGE_MIN_DAYS_DEFAULT
            ),
            fred_vintage_max_window_days=int(
                os.environ.get("FRED_VINTAGE_MAX_WINDOW_DAYS")
                or FRED_VINTAGE_MAX_WINDOW_DAYS_DEFAULT
            ),
            fred_vintage_start=date.fromisoformat(
                os.environ.get("FRED_VINTAGE_START") or FRED_VINTAGE_START_DEFAULT
            ),
            fred_pace_per_minute=float(
                os.environ.get("FRED_PACE_PER_MINUTE") or FRED_PACE_PER_MINUTE_DEFAULT
            ),
            fred_max_in_flight=int(
                os.environ.get("FRED_MAX_IN_FLIGHT") or FRED_MAX_IN_FLIGHT_DEFAULT
            ),
            fred_series_per_task=int(
                os.environ.get("FRED_SERIES_PER_TASK") or FRED_SERIES_PER_TASK_DEFAULT
            ),
            supervisor_state_path=state_path,
            lake_prefix=os.environ.get("MARKETS_LAKE_PREFIX", "").strip(),
            dev=DevProfile.from_env(),
            test=TestProfile.from_env(),
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
            "fred_vintage_min_days": self.fred_vintage_min_days,
            "fred_vintage_max_window_days": self.fred_vintage_max_window_days,
            "fred_vintage_start": self.fred_vintage_start.isoformat(),
            "fred_pace_per_minute": self.fred_pace_per_minute,
            "fred_max_in_flight": self.fred_max_in_flight,
            "fred_series_per_task": self.fred_series_per_task,
            "eod_pace_seconds": self.eod_pace_seconds,
            "eod_max_in_flight": self.eod_max_in_flight,
            "yf_pace_max_seconds": self.yf_pace_max_seconds,
            "yf_pace_backoff_factor": self.yf_pace_backoff_factor,
            "yf_pace_recovery_step": self.yf_pace_recovery_step,
            "eod_chunk_size": self.eod_chunk_size,
            "eod_chunk_size_min": self.eod_chunk_size_min,
            "eod_start_slop_days": self.eod_start_slop_days,
            "eod_start_slop_days_min": self.eod_start_slop_days_min,
            "yf_wave_jobs": self.yf_wave_jobs,
            "discover_min_history_days": self.discover_min_history_days,
            "minio_max_workers": self.minio_max_workers,
            "eod_delay_hours": self.eod_delay_hours,
            "eod_target_finish_hour_et": self.eod_target_finish_hour_et,
            "defer_heavy_before_hour_et": self.defer_heavy_before_hour_et,
            "defer_heavy_after_hour_et": self.defer_heavy_after_hour_et,
            "supervisor_state_path": self.supervisor_state_path,
            "lake_prefix": self.lake_prefix,
            "dev": self.dev.to_dict(),
            "test": self.test.to_dict(),
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
            fred_vintage_min_days=int(
                d.get("fred_vintage_min_days", FRED_VINTAGE_MIN_DAYS_DEFAULT)
            ),
            fred_vintage_max_window_days=int(
                d.get(
                    "fred_vintage_max_window_days",
                    FRED_VINTAGE_MAX_WINDOW_DAYS_DEFAULT,
                )
            ),
            fred_vintage_start=date.fromisoformat(
                d.get("fred_vintage_start", FRED_VINTAGE_START_DEFAULT)
            ),
            fred_pace_per_minute=float(
                d.get("fred_pace_per_minute", FRED_PACE_PER_MINUTE_DEFAULT)
            ),
            fred_max_in_flight=int(d.get("fred_max_in_flight", FRED_MAX_IN_FLIGHT_DEFAULT)),
            fred_series_per_task=int(
                d.get("fred_series_per_task", FRED_SERIES_PER_TASK_DEFAULT)
            ),
            eod_pace_seconds=float(d.get("eod_pace_seconds", EOD_PACE_SECONDS_DEFAULT)),
            eod_max_in_flight=int(d.get("eod_max_in_flight", EOD_MAX_IN_FLIGHT_DEFAULT)),
            yf_pace_max_seconds=float(d.get("yf_pace_max_seconds", YF_PACE_MAX_SECONDS_DEFAULT)),
            yf_pace_backoff_factor=float(d.get("yf_pace_backoff_factor", 2.0)),
            yf_pace_recovery_step=float(d.get("yf_pace_recovery_step", 0.1)),
            eod_chunk_size=int(d.get("eod_chunk_size", EOD_CHUNK_SIZE_DEFAULT)),
            eod_chunk_size_min=int(d.get("eod_chunk_size_min", EOD_CHUNK_SIZE_MIN_DEFAULT)),
            eod_start_slop_days=int(d.get("eod_start_slop_days", EOD_START_SLOP_DAYS_DEFAULT)),
            eod_start_slop_days_min=int(
                d.get("eod_start_slop_days_min", EOD_START_SLOP_DAYS_MIN_DEFAULT)
            ),
            yf_wave_jobs=int(d.get("yf_wave_jobs", YF_WAVE_JOBS_DEFAULT)),
            discover_min_history_days=int(d.get("discover_min_history_days", 252)),
            minio_max_workers=int(d.get("minio_max_workers", 8)),
            eod_delay_hours=float(d.get("eod_delay_hours", 3.0)),
            eod_target_finish_hour_et=int(d.get("eod_target_finish_hour_et", 7)),
            defer_heavy_before_hour_et=int(d.get("defer_heavy_before_hour_et", 9)),
            defer_heavy_after_hour_et=int(d.get("defer_heavy_after_hour_et", 23)),
            supervisor_state_path=d.get("supervisor_state_path") or str(OUTPUT_DIR / ".supervisor" / "state.db"),
            lake_prefix=d.get("lake_prefix") or "",
            dev=DevProfile.from_dict(d.get("dev")),
            test=TestProfile.from_dict(d.get("test")),
        )
