"""CLI: hash-sample the universe, plot those series, write a quality report.

One Serve GET per name (rows + quality). Same sample feeds PNGs and the report.
"""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from threading import Lock

from lexis_markets.cli._runner import base_parser
from lexis_markets.cli.visuals import plot_window, rows_to_df, write_plot
from lexis_markets.config import MarketsConfig
from lexis_markets.lake import PgClient
from lexis_markets.logging_setup import configure_logging, get_logger
from lexis_markets.registry import EOD_ELIGIBLE_WHERE
from lexis_markets.serve.client import (
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    fetch_series,
    hash_sample,
    serve_base,
    serve_health,
)

logger = get_logger("cli.qa")

DEFAULT_CLASSES = ("equity", "etf")


def _parse_classes(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_CLASSES)
    out = [c.strip().lower() for c in raw.split(",") if c.strip()]
    bad = [c for c in out if c not in ("equity", "etf", "macro")]
    if bad:
        raise ValueError(f"unknown asset-class: {bad}")
    return out or list(DEFAULT_CLASSES)


def list_pool(
    pg: PgClient,
    asset_class: str,
    *,
    status: str | None = "ACTIVE",
    flagged: bool = False,
    eod_eligible: bool = True,
) -> list[dict]:
    extra = ""
    params: list = [asset_class]
    if status:
        extra += " AND m.status = %s"
        params.append(status)
    if flagged:
        extra += " AND COALESCE(m.suspicious_count, 0) > 0"
    if eod_eligible and asset_class in ("equity", "etf"):
        extra += f" AND ({EOD_ELIGIBLE_WHERE.strip()})"
    return pg.fetchall(
        f"""
        SELECT m.series_id, m.canonical_symbol, m.first_seen, m.last_seen,
               m.extras, m.asset_class, m.status, m.suspicious_count
        FROM series_meta m
        WHERE m.asset_class = %s
          AND m.first_seen IS NOT NULL
          AND m.last_seen IS NOT NULL
          {extra}
        ORDER BY m.series_id
        """,
        tuple(params),
    )


def _load_series_rows(pg: PgClient, series_ids: list[str]) -> list[dict]:
    return pg.fetchall(
        """
        SELECT m.series_id, m.canonical_symbol, m.first_seen, m.last_seen,
               m.extras, m.asset_class, m.status, m.suspicious_count
        FROM series_meta m
        WHERE m.series_id = ANY(%s)
        """,
        (series_ids,),
    )


def _pick_rows(
    pg: PgClient,
    *,
    series: list[str] | None,
    asset_classes: list[str],
    n: int | None,
    equity_n: int | None,
    etf_n: int | None,
    macro_n: int | None,
    seed: int,
    status: str | None,
    flagged: bool,
) -> list[dict]:
    if series:
        return _load_series_rows(pg, series)
    per = {
        "equity": equity_n,
        "etf": etf_n,
        "macro": macro_n,
    }
    use_per = any(v is not None for v in per.values())
    picks: list[dict] = []
    if use_per:
        for i, cls in enumerate(asset_classes):
            count = per.get(cls)
            if count is None or count <= 0:
                continue
            pool = list_pool(
                pg,
                cls,
                status=None if cls == "macro" else status,
                flagged=flagged,
                eod_eligible=cls != "macro",
            )
            picks.extend(hash_sample(pool, count, seed=seed + i))
        return picks
    pools: list[dict] = []
    for cls in asset_classes:
        pools.extend(
            list_pool(
                pg,
                cls,
                status=None if cls == "macro" else status,
                flagged=flagged,
                eod_eligible=cls != "macro",
            )
        )
    take = n if n is not None else 50
    return hash_sample(pools, take, seed=seed)


def _quality_flags(quality: dict | None) -> list[str]:
    if not quality:
        return []
    flags = quality.get("flags") or []
    if isinstance(flags, list):
        return [str(f) for f in flags if f]
    return []


def _write_report(out_dir: Path, rows: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps({"meta": meta, "series": rows}, indent=2, default=str) + "\n", encoding="utf-8")

    by_class = Counter(r.get("asset_class") or "?" for r in rows)
    flag_hist: Counter[str] = Counter()
    flagged = []
    for r in rows:
        qflags = r.get("flags") or []
        flag_hist.update(qflags)
        if r.get("suspicious_count") or qflags:
            flagged.append(r)
    flagged.sort(key=lambda r: int(r.get("suspicious_count") or 0), reverse=True)

    lines = [
        "# Sample quality report",
        "",
        f"sampled={len(rows)} written={meta.get('written')} skipped={meta.get('skipped')} seed={meta.get('seed')}",
        "",
        "## By class",
        "",
    ]
    for cls, n in sorted(by_class.items()):
        lines.append(f"- {cls}: {n}")
    lines.extend(["", "## Flag histogram", ""])
    if flag_hist:
        for name, n in flag_hist.most_common():
            lines.append(f"- {name}: {n}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Flagged first", ""])
    if not flagged:
        lines.append("- (none)")
    else:
        for r in flagged:
            png = r.get("png") or ""
            lines.append(
                f"- `{r.get('series_id')}` suspicious={r.get('suspicious_count')} "
                f"flags={','.join(r.get('flags') or []) or '-'} [plot]({png})"
            )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qa(
    cfg: MarketsConfig,
    *,
    out_dir: Path,
    series: list[str] | None,
    asset_classes: list[str],
    n: int | None,
    equity_n: int | None,
    etf_n: int | None,
    macro_n: int | None,
    seed: int,
    years: int | None,
    start: date | None,
    end: date | None,
    status: str | None,
    flagged: bool,
    workers: int,
    timeout: float,
    revision_mode_macro: str,
    revision_mode_price: str,
) -> dict:
    base = serve_base(cfg)
    if not serve_health(base):
        raise RuntimeError(
            f"Serve health failed at {base}/health — start the supervisor or deploy Serve first"
        )
    pg = PgClient(cfg.postgres_url)
    picks = _pick_rows(
        pg,
        series=series,
        asset_classes=asset_classes,
        n=n,
        equity_n=equity_n,
        etf_n=etf_n,
        macro_n=macro_n,
        seed=seed,
        status=status,
        flagged=flagged,
    )
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    workers = len(picks) if int(workers) <= 0 else max(1, int(workers))
    lock = Lock()
    written = 0
    skipped = 0
    report_rows: list[dict] = []

    def _one(row: dict) -> dict:
        sid = str(row["series_id"])
        asset_class = str(row.get("asset_class") or "equity")
        sym = str(row.get("canonical_symbol") or sid).replace("/", "_").replace("\\", "_")
        if start is not None and end is not None:
            pstart, pend = start, end
        else:
            pstart, pend = plot_window(row, years)
        rev = revision_mode_macro if asset_class == "macro" else revision_mode_price
        as_of = pend if rev == "as_of" else None
        safe_id = sid.replace(":", "_")
        rel_png = f"plots/{asset_class}/{sym}_{safe_id}.png"
        out_path = out_dir / rel_png
        rec = {
            "series_id": sid,
            "asset_class": asset_class,
            "symbol": sym,
            "start": pstart.isoformat(),
            "end": pend.isoformat(),
            "png": rel_png,
        }
        try:
            body = fetch_series(
                base,
                sid,
                pstart,
                pend,
                revision_mode=rev,
                include_rows=True,
                as_of=as_of,
                timeout=timeout,
            )
        except Exception as exc:
            rec["error"] = str(exc)
            rec["rows"] = 0
            rec["flags"] = []
            rec["suspicious_count"] = None
            return rec
        df = rows_to_df(body.get("rows") or [])
        quality = body.get("quality") or {}
        rec["rows"] = len(df)
        rec["quality"] = quality
        rec["flags"] = _quality_flags(quality)
        rec["suspicious_count"] = quality.get("suspicious_count")
        rec["quality_score"] = quality.get("quality_score")
        if df.empty:
            rec["error"] = "empty"
            return rec
        write_plot(df, out_path, sid, sym)
        rec["png"] = rel_png
        return rec

    work = list(picks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, row) for row in work]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                report_rows.append(rec)
                if rec.get("error"):
                    skipped += 1
                    logger.warning("qa skip %s (%s)", rec.get("series_id"), rec.get("error"))
                else:
                    written += 1
                    logger.info("qa %s -> %s", rec.get("series_id"), rec.get("png"))

    report_rows.sort(key=lambda r: str(r.get("series_id") or ""))
    meta = {
        "written": written,
        "skipped": skipped,
        "seed": seed,
        "workers": workers,
        "asset_classes": asset_classes,
        "n": n,
    }
    _write_report(out_dir, report_rows, meta)
    return {
        "written": written,
        "skipped": skipped,
        "sampled": len(report_rows),
        "out": str(out_dir),
        "report": str(out_dir / "report.json"),
    }


def main() -> None:
    p = base_parser("Hash-sample universe: plots + quality report via Serve")
    p.add_argument("--out", type=Path, default=Path("plot_samples"))
    p.add_argument("--n", type=int, default=None, help="total hash sample across --asset-class (default 50)")
    p.add_argument("--equity", type=int, default=None, help="per-class count (overrides --n for equity)")
    p.add_argument("--etf", type=int, default=None)
    p.add_argument("--macro", type=int, default=None)
    p.add_argument("--asset-class", default="equity,etf", help="comma list: equity,etf,macro")
    p.add_argument("--series", default="", help="comma series_ids; skip sampling")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--years", type=int, default=0, help="plot window years; 0 = first_seen..last_seen")
    p.add_argument("--start", default=None, help="inclusive YYYY-MM-DD (with --end)")
    p.add_argument("--end", default=None, help="inclusive YYYY-MM-DD (with --start)")
    p.add_argument("--status", default="ACTIVE", help="series_meta.status; empty = any")
    p.add_argument("--flagged", action="store_true", help="only suspicious_count > 0")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--macro-revision", default="latest", choices=("as_of", "latest"))
    p.add_argument("--price-revision", default="latest", choices=("as_of", "latest"))
    args = p.parse_args()

    configure_logging()
    cfg = MarketsConfig.from_env()
    series = [s.strip() for s in args.series.split(",") if s.strip()] or None
    years = args.years if args.years > 0 else None
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    if (start is None) ^ (end is None):
        raise SystemExit("--start and --end must be used together")
    status = args.status.strip() or None
    try:
        out = run_qa(
            cfg,
            out_dir=args.out,
            series=series,
            asset_classes=_parse_classes(args.asset_class),
            n=args.n,
            equity_n=args.equity,
            etf_n=args.etf,
            macro_n=args.macro,
            seed=args.seed,
            years=years,
            start=start,
            end=end,
            status=status,
            flagged=bool(args.flagged),
            workers=args.workers,
            timeout=args.timeout,
            revision_mode_macro=args.macro_revision,
            revision_mode_price=args.price_revision,
        )
        logger.info("done %s", out)
    except Exception:
        logger.exception("failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
