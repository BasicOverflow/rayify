from __future__ import annotations

import io
import json
import time
from collections import defaultdict
from datetime import date, timedelta
from uuid import uuid4

import pandas as pd
import ray
import requests
import yfinance as yf

from lexis_markets.cluster import plan_task_resources, run_batches
from lexis_markets.config import MarketsConfig
from lexis_markets.ingest import map_ohlcv, merge_details
from lexis_markets.registry import ensure_eod_aliases
from lexis_markets.storage import LakeStore, ObsWriter, PgClient, get_json, put_json, utcnow
from lexis_markets.nasdaq_symbols import fetch_nasdaq_directory
from lexis_markets.universe import EOD_ELIGIBLE_WHERE, apply_nasdaq_yf_gate, patch_yf_skip_failures

EOD_MARKER_PREFIX = "ops/markers/eod_l1"


def eod_marker_key(target: date | None = None) -> str:
    return f"{EOD_MARKER_PREFIX}/{eod_target_date() if target is None else target}.json"
MP_BASE = "https://marketparquet.com/api/data/download"
MP_CACHE_PREFIX = "ops/cache/marketparquet"
MP_COMBINED_PREFIX = f"{MP_CACHE_PREFIX}/combined_daily"
MP_FREE_DAYS = 7
MP_HEADERS = {"User-Agent": "Mozilla/5.0"}
YF_RATE_LIMIT_MARKERS = ("YFRateLimitError", "Too Many Requests", "429")


@ray.remote
class YfRateGate:
    def __init__(self, min_interval: float = 5.0):
        self.min_interval = min_interval
        self._next = 0.0


    def acquire(self):
        now = time.monotonic()
        wait = self._next - now
        if wait > 0:
            time.sleep(wait)
        self._next = time.monotonic() + self.min_interval


def eod_target_date(today: date | None = None) -> date:
    today = today or date.today()
    return today - timedelta(days=1)


def mp_cutoff(today: date | None = None) -> date:
    return (today or date.today()) - timedelta(days=MP_FREE_DAYS)


def _extras_date(extras, key: str) -> date | None:
    if not extras:
        return None
    if isinstance(extras, str):
        extras = json.loads(extras)
    raw = extras.get(key)
    return date.fromisoformat(raw) if raw else None


def resolve_eod_targets(pg: PgClient) -> list[dict]:
    target = eod_target_date()
    rows = pg.fetchall(
        f"""
        SELECT m.series_id, m.canonical_symbol, m.asset_class, m.series_type,
               m.last_seen, m.extras,
               MAX(CASE WHEN a.source = 'yfinance' THEN a.source_symbol END) AS yf_sym
        FROM series_meta m
        LEFT JOIN symbol_aliases a ON a.series_id = m.series_id
        WHERE m.status = 'ACTIVE' AND m.asset_class IN ('equity', 'etf')
          AND {EOD_ELIGIBLE_WHERE}
        GROUP BY m.series_id, m.canonical_symbol, m.asset_class, m.series_type, m.last_seen, m.extras
        """
    )
    out = []
    for r in rows:
        last = r["last_seen"]
        if not last:
            continue
        extras = r.get("extras") or {}
        primary_last = _extras_date(extras, "primary_last_seen") or last
        eod_through = _extras_date(extras, "eod_filled_through") or primary_last
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
    return out


def mp_combined_key(day: date | str) -> str:
    day_iso = day.isoformat() if isinstance(day, date) else day
    return f"{MP_COMBINED_PREFIX}/{day_iso}.parquet"


def fetch_mp_daily(day: date, lake: LakeStore) -> pd.DataFrame:
    combined_key = mp_combined_key(day)
    if lake.exists(combined_key):
        return lake.get_df_parquet(combined_key)
    frames = []
    for kind in ("stock", "etf"):
        cache_key = f"{MP_CACHE_PREFIX}/{kind}_daily/{day.isoformat()}.parquet"
        if lake.exists(cache_key):
            frames.append(lake.get_df_parquet(cache_key))
            continue
        url = f"{MP_BASE}/{kind}_daily/{day.isoformat()}.parquet"
        resp = None
        for attempt in range(6):
            resp = requests.get(url, headers=MP_HEADERS, timeout=120)
            if resp.status_code in (401, 403, 404):
                break
            if resp.status_code == 429:
                time.sleep(min(15 * (attempt + 1), 90))
                continue
            resp.raise_for_status()
            lake.put_bytes(cache_key, resp.content, content_type="application/octet-stream")
            frames.append(pd.read_parquet(io.BytesIO(resp.content)))
            break
        else:
            if resp is not None and resp.status_code == 429:
                print(f"marketparquet: rate limited skip {kind} day={day.isoformat()}", flush=True)
                continue
            if resp is not None:
                resp.raise_for_status()
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    lake.put_df_parquet(combined_key, df)
    return df


def map_marketparquet(df: pd.DataFrame) -> pd.DataFrame:
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
        meta = ObsWriter(lake, run_id=run_id).write_parts(sub, shard=uuid4().hex[:8])
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
            print(f"marketparquet: skip day={d.isoformat()} (no file)", flush=True)
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


def _yf_jobs(targets: list[dict], details: list[dict], today: date | None = None) -> list[dict]:
    today = today or date.today()
    cutoff = mp_cutoff(today)
    mp_ok = {d["series_id"] for d in details if d.get("source") == "marketparquet" and int(d.get("rows") or 0) > 0}
    jobs: list[dict] = []
    seen: set[tuple] = set()
    for t in targets:
        sid = t["series_id"]
        hist_end = min(t["end"], cutoff - timedelta(days=1))
        if t["start"] <= hist_end:
            key = (sid, t["start"], hist_end)
            if key not in seen:
                jobs.append({**t, "end": hist_end})
                seen.add(key)
        if sid not in mp_ok and t["end"] >= cutoff:
            recent_start = max(t["start"], cutoff)
            key = (sid, recent_start, t["end"])
            if key not in seen:
                jobs.append({**t, "start": recent_start})
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


def _is_rate_limited(err: str) -> bool:
    return any(m in err for m in YF_RATE_LIMIT_MARKERS)


def _download_yf(tickers: list[str], start: date, end: date, yf_gate, backoff: float) -> pd.DataFrame:
    for attempt in range(3):
        ray.get(yf_gate.acquire.remote())
        raw = yf.download(
            tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=False,
            threads=False,
            group_by="ticker",
        )
        if not raw.empty:
            return raw
        if attempt < 2:
            time.sleep(backoff * (attempt + 1))
    return pd.DataFrame()


def _batch_yf_jobs(targets: list[dict], chunk_size: int, start_slop_days: int) -> list[dict]:
    """Pack tickers sharing similar [start,end] into one yf.download call."""
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
            meta = ObsWriter(lake, run_id=run_id).write_parts(mapped, shard=uuid4().hex[:8])
            detail = meta["details"][0] if meta["details"] else {}
            detail["source"] = "yfinance"
            detail["symbol"] = t["symbol"]
            detail["series_id"] = t["series_id"]
            detail["series_type"] = t["series_type"]
            detail["primary_last"] = t["primary_last"].isoformat()
            details.append(detail)
        except Exception as e:
            err = str(e)
            if _is_rate_limited(err):
                time.sleep(cfg.eod_pace_seconds * 10)
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
    jobs = _batch_yf_jobs(targets, cfg.eod_chunk_size, cfg.eod_start_slop_days)
    avg_tickers = sum(len(j["targets"]) for j in jobs) / len(jobs)
    print(
        f"eod: yfinance targets={len(targets)} jobs={len(jobs)} avg_tickers={avg_tickers:.1f} "
        f"chunk={cfg.eod_chunk_size} slop={cfg.eod_start_slop_days}d "
        f"pace={cfg.eod_pace_seconds}s",
        flush=True,
    )
    cfg_d = cfg.to_dict()
    shape = plan_task_resources(batch_size=1)
    batch_results = run_batches(
        "yfinance",
        jobs,
        lambda batch: task_ingest_eod_chunk.options(max_retries=2).remote(
            cfg_d, batch[0], run_id, yf_gate
        ),
        shape,
    )
    return [d for r in batch_results for d in r]


def reset_eod_fill_pointers(pg: PgClient) -> int:
    row = pg.fetchone(
        """
        WITH updated AS (
            UPDATE series_meta SET
                extras = COALESCE(extras, '{}'::jsonb)
                    || jsonb_build_object('eod_filled_through', extras->>'primary_last_seen')
            WHERE asset_class IN ('equity', 'etf')
              AND extras->>'primary_last_seen' IS NOT NULL
              AND (
                extras->>'eod_filled_through' IS NULL
                OR (extras->>'eod_filled_through')::date > (extras->>'primary_last_seen')::date + 7
              )
            RETURNING 1
        )
        SELECT COUNT(*) AS n FROM updated
        """
    )
    return int(row["n"]) if row else 0


def ingest_eod(
    cfg: MarketsConfig,
    lake: LakeStore,
    pg: PgClient,
    *,
    yf_gate=None,
) -> dict:
    t0 = time.perf_counter()
    from lexis_markets.entity_detect import run_entity_detect

    ensure_eod_aliases(pg)
    directory = fetch_nasdaq_directory()
    detect_out = run_entity_detect(cfg, lake, pg, directory=directory)
    if detect_out.get("registered"):
        ensure_eod_aliases(pg)

    from lexis_markets.entity_detect import run_eod_gap_scan

    gap_out = run_eod_gap_scan(pg, lake, directory)
    target = eod_target_date()
    targets = resolve_eod_targets(pg)

    if not targets:
        marker = eod_marker_key(target)
        if lake.exists(marker):
            out = get_json(lake, marker)
            print(
                f"eod: up to date date={target} ok={out.get('symbols')} rows={out.get('rows')} "
                f"entity_new={detect_out.get('registered', 0)} gap_rewound={gap_out.get('rewound', 0)}",
                flush=True,
            )
            return {
                **out,
                "details": [],
                "elapsed_s": 0.0,
                "entity_detect": detect_out,
                "gap_scan": gap_out,
            }
        print(
            f"eod: nothing stale gap_rewound={gap_out.get('rewound', 0)}",
            flush=True,
        )
        return {
            "source": "eod",
            "symbols": 0,
            "rows": 0,
            "details": [],
            "entity_detect": detect_out,
            "gap_scan": gap_out,
        }

    targets, nasdaq_skip = apply_nasdaq_yf_gate(pg, targets, directory.symbols)
    print(
        f"eod: nasdaq_listed={len(directory.symbols)} yf_skip_not_listed={nasdaq_skip} "
        f"targets={len(targets)} entity_new={detect_out.get('registered', 0)}",
        flush=True,
    )
    if not targets:
        print("eod: nothing left after nasdaq yf gate")
        return {
            "source": "eod",
            "symbols": 0,
            "rows": 0,
            "details": [],
            "entity_detect": detect_out,
            "gap_scan": gap_out,
        }

    run_id = uuid4().hex[:12]
    details: list[dict] = []

    by_day = _mp_days_needed(targets)
    if by_day:
        sym_n = len({t["symbol"] for tgts in by_day.values() for t in tgts})
        print(f"marketparquet: days={len(by_day)} symbols={sym_n} free_window={MP_FREE_DAYS}d", flush=True)
        details.extend(_run_mp_days(cfg, by_day, run_id))

    yf_targets = _yf_jobs(targets, details)
    if yf_targets:
        if yf_gate is None:
            yf_gate = YfRateGate.remote(cfg.eod_pace_seconds)
        details.extend(_run_yf(cfg, yf_targets, run_id, yf_gate))

    details = merge_details(details)
    yf_skip_n = patch_yf_skip_failures(pg, details)
    if yf_skip_n:
        print(f"eod: yf_skip_failures={yf_skip_n}")
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
    }
    marker = eod_marker_key(target)
    put_json(lake, marker, {**out, "finished_at": utcnow().isoformat(), "details": None})
    print(
        f"eod_total: {elapsed:.1f}s targets={len(targets)} ok={ok_n} "
        f"marketparquet={mp_n} failed={out['failed']} rows={rows}"
    )
    return out
