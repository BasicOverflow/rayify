"""Enrich L2 from jacksoncrow meta + yfinance EOD delist probes."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from lexis_markets.cluster import init_ray
from lexis_markets.config import MarketsConfig
from lexis_markets.delist import (
    probe_delisted,
    probe_delisted_ray,
    reclassify_false_delisted,
    resolve_delist_candidates,
)
from lexis_markets.ingest import JC_CACHE_KEY
from lexis_markets.storage import LakeStore, PgClient
from lexis_markets.universe import EXCHANGE_MAP, sync_live_universe


def load_meta_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    return df


def load_meta_from_lake(cfg: MarketsConfig) -> pd.DataFrame | None:
    lake = LakeStore(cfg)
    if not lake.exists(JC_CACHE_KEY):
        return None
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zpath = Path(tmp.name)
    lake.download_file(JC_CACHE_KEY, zpath)
    try:
        with zipfile.ZipFile(zpath) as zf:
            name = next((n for n in zf.namelist() if n.endswith("symbols_valid_meta.csv")), None)
            if not name:
                return None
            return pd.read_csv(BytesIO(zf.read(name)))
    finally:
        zpath.unlink(missing_ok=True)


def resolve_meta_path(cfg: MarketsConfig, explicit: Path | None) -> pd.DataFrame | None:
    if explicit:
        return load_meta_csv(explicit)
    candidates = [
        Path(__file__).resolve().parents[1] / "input" / "jacksoncrow_kaggle" / "symbols_valid_meta.csv",
        Path(__file__).resolve().parents[1] / "input" / "jacksoncrow_kaggle" / "stock-market-dataset" / "symbols_valid_meta.csv",
    ]
    for p in candidates:
        if p.exists():
            return load_meta_csv(p)
    return load_meta_from_lake(cfg)


def _series_id(asset_class: str, symbol: str) -> str:
    return f"{asset_class}:{symbol.upper()}"


def apply_meta(pg: PgClient, meta: pd.DataFrame) -> dict:
    updated = 0
    for _, r in meta.iterrows():
        sym = str(r["Symbol"]).upper()
        is_etf = str(r.get("ETF", "N")).upper() == "Y"
        asset = "etf" if is_etf else "equity"
        sid = _series_id(asset, sym)
        row = pg.fetchone("SELECT series_id FROM series_meta WHERE series_id = %s", (sid,))
        if not row:
            continue
        exch_code = str(r.get("Listing Exchange", "") or "").strip()
        extras = {
            "security_name": str(r.get("Security Name", "") or ""),
            "listing_exchange": exch_code,
            "market_category": str(r.get("Market Category", "") or ""),
            "financial_status": str(r.get("Financial Status", "") or ""),
            "meta_source": "jacksoncrow_symbols_valid_meta",
        }
        exchange = EXCHANGE_MAP.get(exch_code, exch_code or None)
        pg.execute(
            """
            UPDATE series_meta SET
                exchange = COALESCE(%s, exchange),
                country = 'US',
                series_type = %s,
                extras = COALESCE(extras, '{}'::jsonb) || %s::jsonb
            WHERE series_id = %s
            """,
            (exchange, asset, json.dumps(extras), sid),
        )
        updated += 1
    return {"meta_updated": updated}


def apply_status(pg: PgClient, by_symbol: dict[str, str], status: str, source: str) -> int:
    n = 0
    for sym, note in by_symbol.items():
        rows = pg.fetchall(
            """
            SELECT series_id FROM series_meta
            WHERE UPPER(canonical_symbol) = %s AND asset_class IN ('equity', 'etf')
            """,
            (sym,),
        )
        for row in rows:
            sid = row["series_id"]
            pg.execute(
                """
                UPDATE series_meta SET status = %s,
                    extras = COALESCE(extras, '{}'::jsonb)
                        || jsonb_build_object('status_source', %s::text, 'status_note', %s::text)
                WHERE series_id = %s AND status = 'ACTIVE'
                """,
                (status, source, note, sid),
            )
            pg.execute(
                """
                INSERT INTO series_links (series_id, link_type, related_series_id, effective_date, note)
                SELECT %s, %s, NULL, last_seen, %s FROM series_meta WHERE series_id = %s
                AND NOT EXISTS (
                    SELECT 1 FROM series_links
                    WHERE series_id = %s AND link_type = %s
                )
                """,
                (sid, status, f"{source}: {note}", sid, sid, status),
            )
            n += 1
    return n


def main():
    p = argparse.ArgumentParser(description="Sync L2 status from jacksoncrow meta + yfinance delist probes")
    p.add_argument("--meta", type=Path, help="symbols_valid_meta.csv (default: input/ or MinIO cache)")
    p.add_argument("--skip-meta", action="store_true")
    p.add_argument("--skip-universe", action="store_true", help="skip jakewright-only / non-US demotion")
    p.add_argument("--skip-yf", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ray", action="store_true", help="probe via Ray cluster (same path as EOD)")
    p.add_argument("--candidates", choices=("stale", "all-active"), default="stale")
    p.add_argument("--limit", type=int, help="cap probe candidates (smoke test)")
    p.add_argument("--min-span-days", type=int, default=180, help="min yfinance request span")
    p.add_argument("--recent-max-days", type=int, default=14, help="skip if primary_last within this many days")
    p.add_argument("--pace", type=float, default=5.0, help="local probe seconds between yfinance calls")
    p.add_argument(
        "--reclassify-delisted",
        action="store_true",
        help="move false DELISTED (non-live-L2) to UNSUPPORTED",
    )
    args = p.parse_args()

    cfg = MarketsConfig.from_env()
    pg = PgClient(cfg.postgres_url)

    if args.reclassify_delisted:
        n = reclassify_false_delisted(pg)
        print(f"reclassify: DELISTED → UNSUPPORTED n={n}")

    if not args.skip_meta:
        meta = resolve_meta_path(cfg, args.meta)
        if meta is None:
            print("meta: not found (pass --meta or download jacksoncrow to input/)")
        else:
            print(f"meta: {len(meta)} rows")
            if args.dry_run:
                print(f"meta: would update up to {len(meta)} series")
            else:
                out = apply_meta(pg, meta)
                print(f"meta: updated {out['meta_updated']} series")

    if not args.skip_universe:
        if args.dry_run:
            jw = pg.fetchone(
                """
                SELECT COUNT(*) n FROM series_meta m
                WHERE m.asset_class IN ('equity','etf') AND m.status = 'ACTIVE'
                  AND NOT EXISTS (
                      SELECT 1 FROM symbol_aliases a
                      WHERE a.series_id = m.series_id AND a.source = 'jacksoncrow'
                  )
                """
            )
            non_us = pg.fetchone(
                """
                SELECT COUNT(*) n FROM series_meta m
                WHERE m.asset_class IN ('equity','etf') AND m.status = 'ACTIVE'
                  AND EXISTS (
                      SELECT 1 FROM symbol_aliases a
                      WHERE a.series_id = m.series_id AND a.source = 'jacksoncrow'
                  )
                  AND COALESCE(m.extras->>'listing_exchange', '') <> ''
                  AND UPPER(COALESCE(m.extras->>'listing_exchange', '')) NOT IN ('Q','N','P','Z','A')
                """
            )
            print(
                f"universe dry-run: jakewright_only={jw['n']} non_us_listing={non_us['n']}"
            )
        else:
            out = sync_live_universe(pg)
            print(
                f"universe: UNSUPPORTED jakewright_only={out['jakewright_only']} "
                f"non_us_listing={out['non_us_listing']}"
            )

    if not args.skip_yf:
        candidates = resolve_delist_candidates(pg, mode=args.candidates, limit=args.limit)
        live_n = sum(1 for c in candidates if c.get("live_l2"))
        print(f"yf_probe: candidates={len(candidates)} live_l2={live_n} mode={args.candidates}")
        if not candidates:
            print("yf_probe: nothing to probe")
        elif args.ray:
            init_ray(cfg)
            delisted, unknown, hits = probe_delisted_ray(
                cfg,
                candidates,
                min_span_days=args.min_span_days,
                recent_max_days=args.recent_max_days,
            )
        else:
            delisted, unknown, hits = probe_delisted(
                candidates,
                min_span_days=args.min_span_days,
                recent_max_days=args.recent_max_days,
                pace_seconds=args.pace,
            )
        print(
            f"yf_probe: probed={len(hits)} delisted={len(delisted)} "
            f"unknown={len(unknown)} (min_span={args.min_span_days}d skip_recent<={args.recent_max_days}d)"
        )
        if delisted:
            print("delisted sample:", list(delisted.items())[:8])
        if unknown:
            print("unknown sample:", list(unknown.items())[:5])
        if args.dry_run:
            print("dry-run: no L2 writes")
        else:
            d_n = apply_status(pg, delisted, "DELISTED", "yfinance_eod")
            u_n = apply_status(pg, unknown, "UNKNOWN", "yfinance_eod")
            print(f"status: marked DELISTED={d_n} UNKNOWN={u_n}")

    by_status = pg.fetchall(
        """
        SELECT status, COUNT(*) n FROM series_meta
        WHERE asset_class IN ('equity','etf') GROUP BY status ORDER BY status
        """
    )
    active = pg.fetchone(
        "SELECT COUNT(*) n FROM series_meta WHERE asset_class IN ('equity','etf') AND status='ACTIVE'"
    )
    print("equity/etf by status:", {r["status"]: r["n"] for r in by_status}, f"active={active['n']}")


if __name__ == "__main__":
    main()
