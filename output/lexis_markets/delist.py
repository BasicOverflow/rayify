from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import ray
import yfinance as yf

from lexis_markets.config import MarketsConfig
from lexis_markets.eod import (
    YfRateGate,
    _batch_yf_jobs,
    _split_yf_bulk,
    eod_target_date,
    resolve_eod_targets,
)
from lexis_markets.storage import PgClient
from lexis_markets.universe import LIVE_L2_WHERE, YF_SKIP_WHERE, enrich_universe_fields, parse_extras


def _download_yf(
    tickers: list[str],
    start: date,
    end: date,
    *,
    gate=None,
    pace_seconds: float = 5.0,
) -> pd.DataFrame:
    for attempt in range(3):
        if gate is not None:
            ray.get(gate.acquire.remote())
        else:
            time.sleep(pace_seconds)
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
            time.sleep(pace_seconds * (attempt + 1))
    return pd.DataFrame()


def enrich_probe_target(row: dict) -> dict:
    extras = parse_extras(row.get("extras"))
    sym = str(row["symbol"] if "symbol" in row else row["canonical_symbol"]).upper()
    fields = enrich_universe_fields(row)
    return {
        "series_id": row["series_id"],
        "symbol": sym,
        "series_type": row.get("series_type") or "equity",
        "primary_last": row["primary_last"],
        "yf_sym": str(row.get("yf_sym") or sym).upper(),
        "extras": extras,
        "in_jacksoncrow": fields["in_jacksoncrow"],
        "us_listed": fields["us_listed"],
        "live_l2": fields["live_l2"],
    }


def prepare_probe_targets(
    targets: list[dict],
    *,
    min_span_days: int,
    recent_max_days: int,
) -> list[dict]:
    end = eod_target_date()
    out = []
    for t in targets:
        if not t.get("live_l2"):
            continue
        primary_last = t["primary_last"]
        if (end - primary_last).days <= recent_max_days:
            continue
        start = max(primary_last + timedelta(days=1), end - timedelta(days=min_span_days))
        if (end - start).days < min_span_days:
            continue
        out.append({**t, "start": start, "end": end})
    return out


def _probe_batch(
    targets: list[dict],
    start: date,
    end: date,
    *,
    gate=None,
    pace_seconds: float = 5.0,
) -> tuple[dict[str, str], dict[str, str]]:
    tickers = [t["yf_sym"] for t in targets]
    sym_by_yf = {t["yf_sym"]: t for t in targets}
    raw = _download_yf(tickers, start, end, gate=gate, pace_seconds=pace_seconds)
    frames = _split_yf_bulk(raw, tickers)
    delisted: dict[str, str] = {}
    unknown: dict[str, str] = {}
    for yf_sym, t in sym_by_yf.items():
        sym = t["symbol"]
        sub = frames.get(yf_sym)
        if sub is not None and not sub.empty:
            continue
        note = _classify_empty(yf_sym)
        if note == "empty_download":
            continue
        if note == "no_timezone":
            unknown[sym] = note
        else:
            delisted[sym] = note
    return delisted, unknown


def _classify_empty(yf_sym: str) -> str:
    try:
        ticker = yf.Ticker(yf_sym)
        hist = ticker.history(period="5d", auto_adjust=False)
        if hist is not None and not hist.empty:
            return "empty_download"
        info = ticker.info or {}
        if str(info.get("quoteType") or "").upper() == "NONE":
            return "no_price_data"
    except Exception as e:
        err = str(e).lower()
        if "delisted" in err:
            return "delisted"
        if "timezone" in err:
            return "no_timezone"
    return "no_price_data"


def probe_delisted(
    targets: list[dict],
    *,
    min_span_days: int = 180,
    recent_max_days: int = 14,
    chunk_size: int = 25,
    start_slop_days: int = 90,
    pace_seconds: float = 5.0,
    gate=None,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """Probe live jacksoncrow US tickers; empty yfinance => delisted hint."""
    live = [t for t in targets if t.get("live_l2")]
    jobs = _batch_yf_jobs(
        prepare_probe_targets(
            live,
            min_span_days=min_span_days,
            recent_max_days=recent_max_days,
        ),
        chunk_size,
        start_slop_days,
    )
    delisted: dict[str, str] = {}
    unknown: dict[str, str] = {}
    hits: dict[str, int] = {}
    for job in jobs:
        d, u = _probe_batch(
            job["targets"],
            job["start"],
            job["end"],
            gate=gate,
            pace_seconds=pace_seconds,
        )
        for sym, note in d.items():
            delisted[sym] = note
            hits[sym] = hits.get(sym, 0) + 1
        for sym, note in u.items():
            if sym not in delisted:
                unknown[sym] = note
                hits[sym] = hits.get(sym, 0) + 1
    return delisted, unknown, hits


@ray.remote
def task_probe_delist_chunk(cfg_d: dict, job: dict) -> tuple[dict, dict]:
    cfg = MarketsConfig.from_dict(cfg_d)
    gate = YfRateGate.remote(cfg.eod_pace_seconds)
    return _probe_batch(
        job["targets"],
        job["start"],
        job["end"],
        gate=gate,
        pace_seconds=cfg.eod_pace_seconds,
    )


def probe_delisted_ray(
    cfg: MarketsConfig,
    targets: list[dict],
    *,
    min_span_days: int = 180,
    recent_max_days: int = 14,
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    from lexis_markets.cluster import plan_task_resources, run_batches

    live = [t for t in targets if t.get("live_l2")]
    prepared = prepare_probe_targets(
        live,
        min_span_days=min_span_days,
        recent_max_days=recent_max_days,
    )
    if not prepared:
        return {}, {}, {}
    jobs = _batch_yf_jobs(prepared, cfg.eod_chunk_size, cfg.eod_start_slop_days)
    cfg_d = cfg.to_dict()
    shape = plan_task_resources(batch_size=1)
    results = run_batches(
        "yfinance_delist",
        jobs,
        lambda batch: task_probe_delist_chunk.options(max_retries=2).remote(cfg_d, batch[0]),
        shape,
    )
    delisted: dict[str, str] = {}
    unknown: dict[str, str] = {}
    hits: dict[str, int] = {}
    for d, u in results:
        for sym, note in d.items():
            delisted[sym] = note
            hits[sym] = hits.get(sym, 0) + 1
        for sym, note in u.items():
            if sym not in delisted:
                unknown[sym] = note
                hits[sym] = hits.get(sym, 0) + 1
    return delisted, unknown, hits


def resolve_delist_candidates(pg: PgClient, *, mode: str = "stale", limit: int | None = None) -> list[dict]:
    if mode == "stale":
        rows = resolve_eod_targets(pg)
        if not rows:
            return []
        ids = [r["series_id"] for r in rows]
        meta = {
            r["series_id"]: r
            for r in pg.fetchall(
                """
                SELECT m.series_id, m.extras,
                       BOOL_OR(a.source = 'jacksoncrow') AS in_jacksoncrow
                FROM series_meta m
                LEFT JOIN symbol_aliases a ON a.series_id = m.series_id
                WHERE m.series_id = ANY(%s)
                GROUP BY m.series_id, m.extras
                """,
                (ids,),
            )
        }
        out = []
        for r in rows:
            m = meta.get(r["series_id"], {})
            out.append(
                enrich_probe_target(
                    {
                        **r,
                        "extras": m.get("extras"),
                        "in_jacksoncrow": m.get("in_jacksoncrow"),
                    }
                )
            )
        return out[: limit or None]
    rows = pg.fetchall(
        f"""
        SELECT m.series_id, m.canonical_symbol, m.series_type, m.last_seen, m.extras,
               COALESCE(MAX(CASE WHEN a.source = 'yfinance' THEN a.source_symbol END), m.canonical_symbol) AS yf_sym,
               BOOL_OR(a.source = 'jacksoncrow') AS in_jacksoncrow
        FROM series_meta m
        LEFT JOIN symbol_aliases a ON a.series_id = m.series_id
        WHERE m.status = 'ACTIVE' AND m.asset_class IN ('equity', 'etf')
          AND {LIVE_L2_WHERE}
          AND {YF_SKIP_WHERE}
          AND m.last_seen IS NOT NULL
        GROUP BY m.series_id, m.canonical_symbol, m.series_type, m.last_seen, m.extras
        ORDER BY m.series_id
        """
    )
    out = [enrich_probe_target({**r, "primary_last": _primary_last(r)}) for r in rows]
    return out[: limit or None]


def _primary_last(row: dict) -> date:
    extras = parse_extras(row.get("extras"))
    raw = extras.get("primary_last_seen")
    if raw:
        return date.fromisoformat(raw)
    return row["last_seen"]


def reclassify_false_delisted(pg: PgClient) -> int:
    """Move non-live-L2 yfinance DELISTED marks to UNSUPPORTED."""
    rows = pg.fetchall(
        """
        SELECT m.series_id, m.canonical_symbol, m.extras,
               BOOL_OR(a.source = 'jacksoncrow') AS in_jacksoncrow
        FROM series_meta m
        LEFT JOIN symbol_aliases a ON a.series_id = m.series_id
        WHERE m.status = 'DELISTED'
          AND m.asset_class IN ('equity', 'etf')
          AND m.extras->>'status_source' = 'yfinance_eod'
        GROUP BY m.series_id, m.canonical_symbol, m.extras
        """
    )
    n = 0
    for r in rows:
        fields = enrich_universe_fields(r)
        if fields["live_l2"]:
            continue
        pg.execute(
            """
            UPDATE series_meta SET status = 'UNSUPPORTED',
                extras = COALESCE(extras, '{}'::jsonb)
                    || jsonb_build_object('status_source', 'universe', 'status_note', 'reclassify_from_delisted')
            WHERE series_id = %s AND status = 'DELISTED'
            """,
            (r["series_id"],),
        )
        n += 1
    return n
