"""Ray remote tasks: daily EOD via MarketParquet (free window) + yfinance backfill."""
from __future__ import annotations

import io
import time
from collections import defaultdict
from datetime import date, timedelta
from uuid import uuid4

import pandas as pd
import ray
import requests
import yfinance as yf

from lexis_markets.lake import LakeStore, L1Writer, PgClient, get_json, put_json, utcnow
from lexis_markets.registry.nasdaq import fetch_nasdaq_directory
from lexis_markets.registry import (
    EOD_ELIGIBLE_WHERE,
    EOD_SOURCES,
    ensure_eod_aliases,
    patch_eod_registry,
    seed_default_stitch,
)
from lexis_markets.registry.universe import apply_nasdaq_yf_gate, patch_yf_skip_failures, extras_date
from lexis_markets.config import MarketsConfig
from lexis_markets.domain.sources.mappers import map_ohlcv, merge_details
from lexis_markets.logging_setup import get_logger
from lexis_markets.eod.entity import run_eod_gap_scan, run_entity_detect
from lexis_markets.jobs.clock import eod_target_date
from lexis_markets.jobs.scheduler import plan_task_resources, run_batches
from lexis_markets.ray.gate_client import (
    is_yf_rate_limited,
    yf_acquire,
    yf_batch_knobs,
    yf_record_rate_limited,
    yf_record_success,
)

logger = get_logger("eod.ingest")

EOD_MARKER_PREFIX = "ops/markers/eod_l1"
MP_BASE = "https://marketparquet.com/api/data/download"
MP_CACHE_PREFIX = "ops/cache/marketparquet"
MP_COMBINED_PREFIX = f"{MP_CACHE_PREFIX}/combined_daily"
# MarketParquet serves the last N calendar days without a paid key.
MP_FREE_DAYS = 7
MP_HEADERS = {"User-Agent": "Mozilla/5.0"}


def eod_marker_key(target: date | None = None) -> str:
    return f"{EOD_MARKER_PREFIX}/{eod_target_date() if target is None else target}.json"


def mp_cutoff(today: date | None = None) -> date:
    return (today or date.today()) - timedelta(days=MP_FREE_DAYS)


def resolve_eod_targets(
    pg: PgClient,
    *,
    limit: int | None = None,
    symbols: tuple[str, ...] | None = None,
    target_date: date | None = None,
) -> list[dict]:
    target = target_date or eod_target_date()
    symbol_clause = ""
    params: list = []
    if symbols:
        syms = [s.upper() for s in symbols]
        symbol_clause = " AND UPPER(m.canonical_symbol) = ANY(%s)"
        params.append(syms)
    rows = pg.fetchall(
        f"""
        SELECT m.series_id, m.canonical_symbol, m.asset_class, m.series_type,
               m.last_seen, m.extras,
               MAX(CASE WHEN a.source = 'yfinance' THEN a.source_symbol END) AS yf_sym
        FROM series_meta m
        LEFT JOIN symbol_aliases a ON a.series_id = m.series_id
        WHERE m.status = 'ACTIVE' AND m.asset_class IN ('equity', 'etf')
          AND {EOD_ELIGIBLE_WHERE}
          {symbol_clause}
        GROUP BY m.series_id, m.canonical_symbol, m.asset_class, m.series_type, m.last_seen, m.extras
        """,
        params or None,
    )
    out = []
    for r in rows:
        last = r["last_seen"]
        if not last:
            continue
        extras = r.get("extras") or {}
        primary_last = extras_date(extras, "primary_last_seen") or last
        eod_through = extras_date(extras, "eod_filled_through") or primary_last
        start = primary_last + timedelta(days=1)
        if start > target:
            continue
        if eod_through >= target:
            continue
        sym = str(r["canonical_symbol"]).upper()
        out.append(
            {
                "series_id": r["series_id"],
                "symbol": sym,
                "series_type": r["series_type"] or r["asset_class"],
                "primary_last": primary_last,
                "start": start,
                "end": target,
                "yf_sym": r["yf_sym"] or sym,
            }
        )
    if limit is not None:
        out = out[:limit]
    return out


def mp_combined_key(day: date | str) -> str:
    day_iso = day.isoformat() if isinstance(day, date) else day
    return f"{MP_COMBINED_PREFIX}/{day_iso}.parquet"


def fetch_mp_daily(day: date, lake: LakeStore) -> pd.DataFrame:
    combined_key = mp_combined_key(day)
    if lake.exists(combined_key):
        cached = lake.get_df_parquet(combined_key)
        if not cached.empty and "asset_type" not in cached.columns:
            logger.warning("marketparquet: stale combined cache missing asset_type day=%s", day.isoformat())
        else:
            return cached
    frames = []
    for kind in ("stock", "etf"):
        asset_type = "Stock" if kind == "stock" else "ETF"
        cache_key = f"{MP_CACHE_PREFIX}/{kind}_daily/{day.isoformat()}.parquet"
        if lake.exists(cache_key):
            part = lake.get_df_parquet(cache_key)
            if not part.empty:
                part = part.copy()
                part["asset_type"] = asset_type
                frames.append(part)
            continue
        url = f"{MP_BASE}/{kind}_daily/{day.isoformat()}.parquet"
        resp = None
        for attempt in range(6):
            resp = requests.get(url, headers=MP_HEADERS, timeout=120)
            if resp.status_code in (401, 403, 404):
                break
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(15 * (attempt + 1), 90))
                continue
            resp.raise_for_status()
            lake.put_bytes(cache_key, resp.content, content_type="application/octet-stream")
            part = pd.read_parquet(io.BytesIO(resp.content))
            if not part.empty:
                part = part.copy()
                part["asset_type"] = asset_type
                frames.append(part)
            break
        else:
            if resp is not None and resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    "marketparquet: unavailable skip %s day=%s status=%s",
                    kind,
                    day.isoformat(),
                    resp.status_code,
                )
                continue
            if resp is not None:
                resp.raise_for_status()
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    lake.put_df_parquet(combined_key, df)
    return df


def map_marketparquet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "asset_type" not in df.columns:
        return map_ohlcv(
            df, source="marketparquet", symbol_col="symbol", date_col="date", series_type="equity"
        )
    parts = []
    for asset, series_type in (("Stock", "equity"), ("ETF", "etf")):
        sub = df[df["asset_type"] == asset]
        if sub.empty:
            continue
        parts.append(
            map_ohlcv(sub, source="marketparquet", symbol_col="symbol", date_col="date", series_type=series_type)
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _mp_days_needed(targets: list[dict], today: date | None = None) -> dict[date, list[dict]]:
    today = today or date.today()
    cutoff = mp_cutoff(today)
    by_day: dict[date, list[dict]] = defaultdict(list)
    seen: dict[date, set[str]] = defaultdict(set)
    for t in targets:
        d = max(t["start"], cutoff)
        while d <= t["end"]:
            sym = t["symbol"]
            if sym not in seen[d]:
                by_day[d].append(t)
                seen[d].add(sym)
            d += timedelta(days=1)
    return dict(by_day)


def _ingest_mp_day_df(
    cfg: MarketsConfig,
    lake: LakeStore,
    raw: pd.DataFrame,
    targets: list[dict],
    run_id: str,
) -> list[dict]:
    sym_set = {t["symbol"] for t in targets}
    sym_by = {t["symbol"]: t for t in targets}
    if raw.empty:
        return []
    raw = raw.copy()
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw = raw[raw["symbol"].isin(sym_set)]
    mapped = map_marketparquet(raw)
    if mapped.empty:
        return []
    details = []
    for sym, sub in mapped.groupby("source_symbol"):
        t = sym_by[sym]
        meta = L1Writer(lake, run_id=run_id).write_parts(sub, shard=uuid4().hex[:8])
        detail = meta["details"][0] if meta["details"] else {}
        detail["source"] = "marketparquet"
        detail["symbol"] = t["symbol"]
        detail["series_id"] = t["series_id"]
        detail["series_type"] = t["series_type"]
        details.append(detail)
    return details


@ray.remote
def task_ingest_mp_day(cfg_d: dict, day_iso: str, targets: list[dict], run_id: str) -> list[dict]:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    key = mp_combined_key(day_iso)
    if not lake.exists(key):
        return []
    raw = lake.get_df_parquet(key)
    return _ingest_mp_day_df(cfg, lake, raw, targets, run_id)


def _run_mp_days(cfg: MarketsConfig, by_day: dict[date, list[dict]], run_id: str) -> list[dict]:
    if not by_day:
        return []
    lake = LakeStore(cfg)
    cfg_d = cfg.to_dict()
    jobs = []
    for d, tgts in sorted(by_day.items()):
        raw = fetch_mp_daily(d, lake)
        if raw.empty:
            logger.info("marketparquet: skip day=%s (no file)", d.isoformat())
            continue
        jobs.append({"day": d.isoformat(), "targets": tgts})
    shape = plan_task_resources(batch_size=1)
    results = run_batches(
        "marketparquet",
        jobs,
        lambda batch: task_ingest_mp_day.options(max_retries=2).remote(
            cfg_d, batch[0]["day"], batch[0]["targets"], run_id
        ),
        shape,
    )
    return [d for r in results for d in r]


def _yf_jobs(targets: list[dict]) -> list[dict]:
    """Deduped yfinance windows: each target's full ``start..end`` (MP tip overlaps OK)."""
    jobs: list[dict] = []
    seen: set[tuple] = set()
    for t in targets:
        key = (t["series_id"], t["start"], t["end"])
        if key not in seen:
            jobs.append(dict(t))
            seen.add(key)
    return jobs


def _split_yf_bulk(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    if len(tickers) == 1:
        t = tickers[0]
        out = df.reset_index()
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
        out["Symbol"] = t.upper()
        return {t: out}
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            sub = df[t].dropna(how="all")
            if sub.empty:
                continue
            sub = sub.reset_index()
            sub["Symbol"] = t.upper()
            out[t] = sub
        except (KeyError, TypeError):
            pass
    return out


def _download_yf(tickers: list[str], start: date, end: date, yf_gate, backoff: float) -> pd.DataFrame:
    for attempt in range(3):
        yf_acquire(yf_gate)
        try:
            raw = yf.download(
                tickers,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=False,
                threads=False,
                group_by="ticker",
            )
        except Exception as exc:
            err = str(exc)
            if is_yf_rate_limited(err):
                yf_record_rate_limited(yf_gate)
                if attempt < 2:
                    time.sleep(backoff * (attempt + 1))
                continue
            raise
        if not raw.empty:
            yf_record_success(yf_gate)
            return raw
        if attempt < 2:
            time.sleep(backoff * (attempt + 1))
    return pd.DataFrame()


def _batch_yf_jobs(targets: list[dict], chunk_size: int, start_slop_days: int) -> list[dict]:
    if not targets:
        return []
    sorted_t = sorted(targets, key=lambda t: (t["end"], t["start"]))
    jobs: list[dict] = []
    i = 0
    while i < len(sorted_t):
        batch = [sorted_t[i]]
        batch_end = sorted_t[i]["end"]
        i += 1
        while len(batch) < chunk_size and i < len(sorted_t):
            t = sorted_t[i]
            if t["end"] != batch_end:
                break
            starts = [x["start"] for x in batch] + [t["start"]]
            if (max(starts) - min(starts)).days > start_slop_days:
                break
            batch.append(t)
            i += 1
        jobs.append(
            {
                "start": min(t["start"] for t in batch),
                "end": batch_end,
                "targets": batch,
            }
        )
    return jobs


def _ingest_yf_targets(
    cfg: MarketsConfig,
    lake: LakeStore,
    targets: list[dict],
    start: date,
    end: date,
    run_id: str,
    yf_gate,
) -> list[dict]:
    tickers = [t["yf_sym"] for t in targets]
    sym_by_yf = {t["yf_sym"]: t for t in targets}
    raw = _download_yf(tickers, start, end, yf_gate, cfg.eod_pace_seconds)
    frames = _split_yf_bulk(raw, tickers)
    details = []
    for yf_sym, t in sym_by_yf.items():
        try:
            sub = frames.get(yf_sym)
            if sub is None or sub.empty:
                raise RuntimeError("yfinance empty")
            mapped = map_ohlcv(
                sub, source="yfinance", symbol_col="Symbol", date_col="Date", series_type=t["series_type"]
            )
            if "adj_close" in mapped.columns:
                mapped["close"] = mapped["adj_close"].where(mapped["adj_close"].notna(), mapped["close"])
            meta = L1Writer(lake, run_id=run_id).write_parts(mapped, shard=uuid4().hex[:8])
            detail = meta["details"][0] if meta["details"] else {}
            detail["source"] = "yfinance"
            detail["symbol"] = t["symbol"]
            detail["series_id"] = t["series_id"]
            detail["series_type"] = t["series_type"]
            detail["primary_last"] = t["primary_last"].isoformat()
            details.append(detail)
        except Exception as e:
            err = str(e)
            if is_yf_rate_limited(err):
                yf_record_rate_limited(yf_gate)
            details.append(
                {
                    "series_id": t["series_id"],
                    "symbol": t["symbol"],
                    "rows": 0,
                    "source": "eod_failed",
                    "error": err,
                }
            )
    return details


@ray.remote
def task_ingest_eod_chunk(cfg_d: dict, job: dict, run_id: str, yf_gate) -> list[dict]:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    return _ingest_yf_targets(cfg, lake, job["targets"], job["start"], job["end"], run_id, yf_gate)


def _run_yf(
    cfg: MarketsConfig,
    targets: list[dict],
    run_id: str,
    yf_gate,
) -> list[dict]:
    if not targets:
        return []

    def _target_key(t: dict) -> tuple:
        return (t["series_id"], t["start"], t["end"])

    remaining = list(targets)
    details: list[dict] = []
    cfg_d = cfg.to_dict()
    wave_cap = (
        cfg.eod_max_in_flight if cfg.eod_max_in_flight > 0 else max(1, cfg.yf_wave_jobs)
    )
    wave_i = 0
    while remaining:
        knobs = yf_batch_knobs(yf_gate)
        chunk = int(knobs["chunk_size"])
        slop = int(knobs["start_slop_days"])
        jobs = _batch_yf_jobs(remaining, chunk, slop)
        if not jobs:
            break
        wave = jobs[:wave_cap]
        wave_i += 1
        avg_tickers = sum(len(j["targets"]) for j in wave) / len(wave)
        logger.info(
            "eod: yfinance wave=%s remaining=%s jobs=%s/%s avg_tickers=%.1f "
            "chunk=%s slop=%sd pace=%.2fs",
            wave_i,
            len(remaining),
            len(wave),
            len(jobs),
            avg_tickers,
            chunk,
            slop,
            float(knobs["interval"]),
        )
        shape = plan_task_resources(batch_size=1, max_in_flight=wave_cap)
        batch_results = run_batches(
            "yfinance",
            wave,
            lambda batch: task_ingest_eod_chunk.options(max_retries=2).remote(
                cfg_d, batch[0], run_id, yf_gate
            ),
            shape,
        )
        details.extend(d for r in batch_results for d in r)
        done = {_target_key(t) for j in wave for t in j["targets"]}
        remaining = [t for t in remaining if _target_key(t) not in done]
    return details


def _as_of_today(target: date) -> date:
    """Calendar day on which EOD for ``target`` (yesterday) is assumed to run."""
    return target + timedelta(days=1)


def ingest_eod(
    cfg: MarketsConfig,
    lake: LakeStore,
    pg: PgClient,
    *,
    yf_gate,
    limit: int | None = None,
    target_date: date | None = None,
) -> dict:
    t0 = time.perf_counter()
    ensure_eod_aliases(pg)
    target = target_date or eod_target_date()
    as_of = _as_of_today(target)
    full_path = (not cfg.test.enabled) or cfg.test.eod_full_path

    if not full_path:
        detect_out = {"registered": 0, "skipped": "test_profile"}
        gap_out = {"rewound": 0, "skipped": "test_profile"}
        directory = None
    else:
        directory = fetch_nasdaq_directory()
        detect_out = run_entity_detect(
            cfg, lake, pg, directory=directory, yf_gate=yf_gate, today=as_of
        )
        if detect_out.get("registered"):
            ensure_eod_aliases(pg)
        gap_out = run_eod_gap_scan(pg, lake, directory, target_date=target, today=as_of)

    if cfg.test.enabled:
        targets = cfg.sample_eod_targets(
            resolve_eod_targets(pg, symbols=cfg.eod_resolve_symbols(), target_date=target)
        )
    else:
        targets = resolve_eod_targets(pg, limit=limit, target_date=target)

    if not targets:
        marker = eod_marker_key(target)
        if lake.exists(marker):
            out = get_json(lake, marker)
            logger.info(
                "eod: up to date date=%s ok=%s rows=%s entity_new=%s gap_rewound=%s",
                target,
                out.get("symbols"),
                out.get("rows"),
                detect_out.get("registered", 0),
                gap_out.get("rewound", 0),
            )
            return {
                **out,
                "details": [],
                "elapsed_s": 0.0,
                "entity_detect": detect_out,
                "gap_scan": gap_out,
            }
        logger.info("eod: nothing stale gap_rewound=%s", gap_out.get("rewound", 0))
        return {
            "source": "eod",
            "symbols": 0,
            "rows": 0,
            "details": [],
            "entity_detect": detect_out,
            "gap_scan": gap_out,
            "target_date": target.isoformat(),
        }

    if not full_path:
        nasdaq_skip = 0
    else:
        targets, nasdaq_skip = apply_nasdaq_yf_gate(pg, targets, directory.symbols)
    logger.info(
        "eod: nasdaq_listed=%s yf_skip_not_listed=%s targets=%s entity_new=%s target=%s",
        "n/a" if not full_path else len(directory.symbols),
        nasdaq_skip,
        len(targets),
        detect_out.get("registered", 0),
        target.isoformat(),
    )
    if not targets:
        logger.info("eod: nothing left after nasdaq yf gate")
        return {
            "source": "eod",
            "symbols": 0,
            "rows": 0,
            "details": [],
            "entity_detect": detect_out,
            "gap_scan": gap_out,
            "target_date": target.isoformat(),
        }

    run_id = uuid4().hex[:12]
    details: list[dict] = []

    if full_path:
        by_day = _mp_days_needed(targets, today=as_of)
    else:
        by_day = {}
        logger.info("eod: test_profile skip marketparquet days")
    if by_day:
        sym_n = len({t["symbol"] for tgts in by_day.values() for t in tgts})
        logger.info("marketparquet: days=%s symbols=%s free_window=%sd", len(by_day), sym_n, MP_FREE_DAYS)
        details.extend(_run_mp_days(cfg, by_day, run_id))

    yf_targets = _yf_jobs(targets)
    if yf_targets:
        details.extend(_run_yf(cfg, yf_targets, run_id, yf_gate))

    details = merge_details(details)
    yf_skip_n = patch_yf_skip_failures(pg, details)
    if yf_skip_n:
        logger.info("eod: yf_skip_failures=%s", yf_skip_n)
    eod_details = [d for d in details if d.get("source") in EOD_SOURCES]
    reg = patch_eod_registry(pg, eod_details)
    seed_default_stitch(pg)
    rows = sum(int(d.get("rows") or 0) for d in details)
    elapsed = time.perf_counter() - t0
    ok_n = len([d for d in details if int(d.get("rows") or 0) > 0])
    mp_n = len([d for d in details if d.get("source") == "marketparquet" and int(d.get("rows") or 0) > 0])
    out = {
        "source": "eod",
        "symbols": ok_n,
        "rows": rows,
        "details": details,
        "run_id": run_id,
        "elapsed_s": elapsed,
        "targets": len(targets),
        "failed": len(targets) - ok_n,
        "marketparquet_ok": mp_n,
        "target_date": target.isoformat(),
        "entity_detect": detect_out,
        "gap_scan": gap_out,
        "registry_updated": reg.get("updated", 0),
    }
    marker = eod_marker_key(target)
    put_json(lake, marker, {**out, "finished_at": utcnow().isoformat(), "details": None})
    logger.info(
        "eod_total: %.1fs targets=%s ok=%s marketparquet=%s failed=%s rows=%s",
        elapsed,
        len(targets),
        ok_n,
        mp_n,
        out["failed"],
        rows,
    )
    return out
