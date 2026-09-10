"""Serve request handlers: series query, EOD snapshot, universe list, dataset export."""
from __future__ import annotations

import json
from datetime import date, timedelta
from uuid import uuid4

import pandas as pd
import ray
from fastapi import HTTPException

from lexis_markets.config import MarketsConfig
from lexis_markets.domain.quality import QUALITY_FLAG_COLUMNS, series_quality_window
from lexis_markets.domain.recipes import canonicalize_recipe_mode, parse_nan_policy
from lexis_markets.jobs.clock import eod_target_date
from lexis_markets.lake import LakeStore, PgClient, get_json
from lexis_markets.logging_setup import get_logger
from lexis_markets.registry import resolve_series_ids
from lexis_markets.registry.filters import covers_through_end, effective_last_seen
from lexis_markets.registry.meta import list_series_meta
from lexis_markets.serve.cache import CacheService
from lexis_markets.serve.dedupe import InFlightDedupe
from lexis_markets.serve.export import task_build_dataset
from lexis_markets.serve.revision import parse_revision_mode
from lexis_markets.serve.series_query import task_query_series_bars

logger = get_logger("serve.handlers")

DEFAULT_EOD_LOOKBACK_DAYS = 14
DATASET_MODES = frozenset({"range", "range_panel", "eod_snapshot", "wide_matrix"})


def clip_range(pg: PgClient, series_ids: list[str], start: date, end: date) -> tuple[date, date]:
    if len(series_ids) != 1:
        return start, end
    row = pg.fetchone(
        "SELECT first_seen, last_seen, extras FROM series_meta WHERE series_id = %s",
        (series_ids[0],),
    )
    if not row or not row["first_seen"] or not row["last_seen"]:
        return start, end
    last = effective_last_seen(row) or row["last_seen"]
    return max(start, row["first_seen"]), min(end, last)


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


def _meta_json_row(row: dict) -> dict:
    out = {
        "series_id": row["series_id"],
        "canonical_symbol": row.get("canonical_symbol"),
        "asset_class": row.get("asset_class"),
        "status": row.get("status"),
        "first_seen": row["first_seen"].isoformat() if row.get("first_seen") else None,
        "last_seen": row["last_seen"].isoformat() if row.get("last_seen") else None,
        "gap_count": row.get("gap_count"),
        "suspicious_count": row.get("suspicious_count"),
        "disagreement_count": row.get("disagreement_count"),
        "quality_score": row.get("quality_score"),
        "calendar_id": row.get("calendar_id"),
    }
    for flag in QUALITY_FLAG_COLUMNS:
        key = f"flag_{flag}"
        if key in row:
            out[key] = row.get(key)
    last = effective_last_seen(row)
    if last is not None:
        out["effective_last_seen"] = last.isoformat()
    return out


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


def _query_dedupe_key(
    series_id: str,
    start: date,
    end: date,
    revision_mode: str,
    as_of: date | None,
    granularity: str,
    min_volume: float | None,
    features: list[str] | None,
) -> tuple:
    feat = ",".join(features or [])
    mv = "" if min_volume is None else str(min_volume)
    return (
        series_id,
        start.isoformat(),
        end.isoformat(),
        revision_mode,
        as_of.isoformat() if as_of else "",
        granularity,
        mv,
        feat,
    )


def parse_dataset_mode(raw: str | None) -> str:
    return canonicalize_recipe_mode(raw)


def resolve_eod_snapshot_day(eod_date: str | date | None = None) -> date:
    if eod_date is None or eod_date == "":
        return eod_target_date()
    if isinstance(eod_date, date):
        return eod_date
    return date.fromisoformat(str(eod_date))


def snapshot_last_bars(bars: pd.DataFrame, *, as_of: date) -> pd.DataFrame:
    """Keep the last bar on or before ``as_of`` for each series_id."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    out = bars.copy()
    out["ts"] = pd.to_datetime(out["ts"]).dt.date
    out = out[out["ts"] <= as_of]
    if out.empty:
        return out
    return out.sort_values(["series_id", "ts"]).groupby("series_id", as_index=False).tail(1)


def normalize_query_spec(spec: dict) -> dict:
    """Expand recipe modes into concrete query windows; default revision to L3 ``latest``."""
    out = dict(spec)
    mode = parse_dataset_mode(out.get("mode"))
    out["mode"] = mode
    if out.get("revision_mode") is None:
        out["revision_mode"] = "latest"
    if mode == "wide_matrix":
        out["nan_policy"] = parse_nan_policy(out.get("nan_policy"))
        out.setdefault("value_col", "close")
        out.setdefault("column_key", "series_id")
    if mode in ("range_panel", "wide_matrix"):
        if not out.get("start") or not out.get("end"):
            raise ValueError("start and end are required when mode=range_panel or wide_matrix")
        return out
    if mode != "eod_snapshot":
        raise ValueError(f"unsupported mode {mode!r}")

    as_of = resolve_eod_snapshot_day(out.get("eod_date"))
    raw_lookback = out.get("eod_lookback_days")
    lookback = DEFAULT_EOD_LOOKBACK_DAYS if raw_lookback is None else int(raw_lookback)
    if lookback < 1 or lookback > 365:
        raise ValueError("eod_lookback_days must be between 1 and 365")
    out["start"] = (as_of - timedelta(days=lookback)).isoformat()
    out["end"] = as_of.isoformat()
    out["granularity"] = "daily"
    out["_snapshot_as_of"] = as_of.isoformat()
    return out


class MarketsService:
    def __init__(self, cfg: MarketsConfig, *, dedupe: InFlightDedupe | None = None):
        self.cfg = cfg
        self.pg = PgClient(cfg.postgres_url)
        self.lake = LakeStore(cfg)
        self._dedupe = dedupe or InFlightDedupe()
        self.cache = CacheService(cfg, dedupe=self._dedupe)

    def query_stitched_bars(self, spec: dict) -> pd.DataFrame:
        try:
            spec = normalize_query_spec(spec)
            series_ids = resolve_series_ids(self.pg, spec)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        start = date.fromisoformat(spec["start"])
        end = date.fromisoformat(spec["end"])
        start, end = clip_range(self.pg, series_ids, start, end)
        granularity = spec.get("granularity", "daily")
        revision_mode = parse_revision_mode(spec.get("revision_mode"))
        as_of_raw = spec.get("as_of")
        as_of = date.fromisoformat(as_of_raw) if as_of_raw else None
        min_volume = spec.get("min_volume")
        features = spec.get("features")
        cfg_d = self.cfg.to_dict()
        as_of_iso = as_of.isoformat() if as_of else None

        pending: list[tuple[tuple, ray.ObjectRef]] = []
        for sid in series_ids:
            key = _query_dedupe_key(
                sid, start, end, revision_mode, as_of, granularity, min_volume, features
            )

            def _submit(
                series_id: str = sid,
            ) -> ray.ObjectRef:
                return task_query_series_bars.remote(
                    cfg_d,
                    series_id,
                    start.isoformat(),
                    end.isoformat(),
                    revision_mode,
                    as_of_iso,
                    granularity,
                    min_volume,
                    features,
                )

            pending.append((key, self._dedupe.get_or_submit(key, _submit)))

        try:
            parts = ray.get([ref for _, ref in pending])
        finally:
            for key, ref in pending:
                self._dedupe.release(key, ref)

        frames = [p for p in parts if p is not None and not p.empty]
        if not frames:
            bars = pd.DataFrame()
        else:
            bars = pd.concat(frames, ignore_index=True)

        snap = spec.get("_snapshot_as_of")
        if snap:
            bars = snapshot_last_bars(bars, as_of=date.fromisoformat(snap))
        return bars

    def series_meta_row(self, series_id: str) -> dict | None:
        return self.pg.fetchone(
            """
            SELECT series_id, canonical_symbol, asset_class, calendar_id,
                   gap_count, disagreement_count, suspicious_count,
                   quality_score, first_seen, last_seen, status, extras,
                   flag_linear_ramp, flag_sparse_bridge, flag_flat_close,
                   flag_ohlc_violation, flag_non_positive_close,
                   flag_duplicate_ts, flag_extreme_return
            FROM series_meta WHERE series_id = %s
            """,
            (series_id,),
        )

    def list_universe(
        self,
        *,
        series_ids: list[str] | None = None,
        symbols: list[str] | None = None,
        asset_classes: list[str] | None = None,
        statuses: list[str] | None = None,
        max_gap_count: int | None = None,
        max_suspicious_count: int | None = None,
        include_partial_coverage: bool = True,
        end: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict:
        """Series names / registry rows only (no OHLCV)."""
        body = {
            "series_ids": series_ids,
            "symbols": symbols,
            "asset_classes": asset_classes,
            "statuses": statuses,
            "max_gap_count": max_gap_count,
            "max_suspicious_count": max_suspicious_count,
            "include_partial_coverage": include_partial_coverage,
            "end": end or eod_target_date().isoformat(),
        }
        try:
            rows = list_series_meta(self.pg, body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        total = len(rows)
        if offset < 0:
            raise HTTPException(400, "offset must be >= 0")
        if limit is not None:
            if limit < 1:
                raise HTTPException(400, "limit must be >= 1")
            rows = rows[offset : offset + limit]
        elif offset:
            rows = rows[offset:]

        return {
            "count": len(rows),
            "total": total,
            "offset": offset,
            "limit": limit,
            "series": [_meta_json_row(r) for r in rows],
        }

    def query_series(
        self,
        series_id: str,
        start: str,
        end: str,
        *,
        granularity: str = "daily",
        min_volume: float | None = None,
        revision_mode: str | None = None,
        as_of: str | None = None,
        features: list[str] | None = None,
        statuses: list[str] | None = None,
        max_gap_count: int | None = None,
        max_suspicious_count: int | None = None,
        include_partial_coverage: bool = True,
        include_rows: bool = True,
    ) -> dict:
        meta = self.series_meta_row(series_id)
        if not meta:
            raise HTTPException(404, "series not found")

        body = {
            "mode": "range",
            "series_ids": [series_id],
            "start": start,
            "end": end,
            "granularity": granularity,
            "min_volume": min_volume,
            "revision_mode": revision_mode,
            "as_of": as_of,
            "features": features,
            "statuses": statuses,
            "max_gap_count": max_gap_count,
            "max_suspicious_count": max_suspicious_count,
            "include_partial_coverage": include_partial_coverage,
        }
        try:
            allowed = resolve_series_ids(self.pg, body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if series_id not in allowed:
            raise HTTPException(404, "series excluded by filters")

        try:
            bars = self.query_stitched_bars(body)
        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(400, str(exc)) from exc
        calendar_id = meta.get("calendar_id") or "nyse"
        end_date = date.fromisoformat(end)
        quality = series_quality_window(bars, calendar_id, meta)
        last = effective_last_seen(meta)
        if last is not None:
            quality["effective_last_seen"] = last.isoformat()
            quality["covers_through_end"] = covers_through_end(meta, end_date)
        out = {
            "series_id": series_id,
            "mode": "range",
            "start": start,
            "end": end,
            "granularity": granularity,
            "revision_mode": parse_revision_mode(revision_mode),
            "as_of": as_of,
            "features": features or [],
            "quality": quality,
        }
        if include_rows:
            out["rows"] = _rows_json(bars)
        else:
            out["rows"] = []
            out["row_count"] = 0 if bars is None else int(len(bars))
        return out

    def query_series_eod(
        self,
        series_id: str,
        *,
        eod_date: str | None = None,
        eod_lookback_days: int = DEFAULT_EOD_LOOKBACK_DAYS,
        min_volume: float | None = None,
        revision_mode: str | None = None,
        features: list[str] | None = None,
        statuses: list[str] | None = None,
        max_gap_count: int | None = None,
        max_suspicious_count: int | None = None,
        include_partial_coverage: bool = True,
    ) -> dict:
        """Last available daily bar on/before the EOD snapshot day."""
        meta = self.series_meta_row(series_id)
        if not meta:
            raise HTTPException(404, "series not found")

        as_of = resolve_eod_snapshot_day(eod_date)
        body = {
            "mode": "eod_snapshot",
            "series_ids": [series_id],
            "eod_date": as_of.isoformat(),
            "eod_lookback_days": eod_lookback_days,
            "min_volume": min_volume,
            "revision_mode": revision_mode,
            "features": features,
            "statuses": statuses,
            "max_gap_count": max_gap_count,
            "max_suspicious_count": max_suspicious_count,
            "include_partial_coverage": include_partial_coverage,
            "end": as_of.isoformat(),
        }
        try:
            allowed = resolve_series_ids(self.pg, body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if series_id not in allowed:
            raise HTTPException(404, "series excluded by filters")

        try:
            bars = self.query_stitched_bars(body)
        except (ValueError, NotImplementedError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if bars.empty:
            raise HTTPException(404, f"no EOD bar on or before {as_of.isoformat()}")

        row = _rows_json(bars)[0]
        return {
            "series_id": series_id,
            "mode": "eod_snapshot",
            "eod_date": as_of.isoformat(),
            "eod_lookback_days": eod_lookback_days,
            "revision_mode": parse_revision_mode(revision_mode or "latest"),
            "features": features or [],
            "bar": row,
            "rows": [row],
        }

    def create_dataset(self, spec: dict, *, sync: bool = False) -> dict:
        try:
            spec = normalize_query_spec(spec)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        job_id = str(uuid4())
        prefix = f"s3://{self.lake.bucket}/layer_3/outputs/{job_id}/"
        self.pg.execute(
            """
            INSERT INTO dataset_jobs (job_id, status, spec, s3_prefix)
            VALUES (%s, 'queued', %s::jsonb, %s)
            """,
            (job_id, json.dumps(spec), prefix),
        )
        logger.info("dataset queued job=%s mode=%s", job_id, spec.get("mode"))
        if sync:
            from lexis_markets.serve.export import build_dataset_parquet

            try:
                manifest = build_dataset_parquet(self.cfg, job_id, spec)
            except (ValueError, NotImplementedError) as exc:
                raise HTTPException(400, str(exc)) from exc
            return {"job_id": job_id, "reused": False, "mode": spec.get("mode"), **manifest}
        task_build_dataset.remote(self.cfg.to_dict(), job_id, spec)
        return {
            "job_id": job_id,
            "status": "queued",
            "mode": spec.get("mode"),
            "s3_prefix": prefix,
            "files": ["dataset.parquet"],
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
        spec = row.get("spec")
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except json.JSONDecodeError:
                spec = None
        if isinstance(spec, dict) and spec.get("mode"):
            out["mode"] = spec["mode"]
        if manifest:
            out["files"] = manifest.get("files", [])
        elif row["status"] == "complete":
            out["files"] = ["dataset.parquet"]
        return out
