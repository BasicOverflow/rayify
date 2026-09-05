"""CLI: plot + warm L3 via Serve HTTP (ACTIVE EOD-eligible equity/ETF + macros).

Fetches series from ``GET {RAY_SERVE_URL}/v1/series/{id}`` (parallel by default),
writes PNGs under ``--out/plots/``. Equity/ETF ``latest`` requests warm L3 on miss.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from threading import Lock
from urllib.parse import quote

import requests

from lexis_markets.cli._runner import base_parser
from lexis_markets.cli.visuals import plot_window, rows_to_df, write_plot
from lexis_markets.config import MarketsConfig
from lexis_markets.lake import PgClient
from lexis_markets.logging_setup import configure_logging, get_logger
from lexis_markets.registry import EOD_ELIGIBLE_WHERE

logger = get_logger("cli.plot_warm")

# Default concurrency for Serve warm/plot (ceiling is Serve max_ongoing_requests).
DEFAULT_WORKERS = 32


def _stable_sample(rows: list[dict], n: int, *, seed: int) -> list[dict]:
    if n <= 0 or not rows:
        return []
    if len(rows) <= n:
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda r: int(
            hashlib.sha256(f"{seed}:{r['series_id']}".encode()).hexdigest(),
            16,
        ),
    )
    return ranked[:n]


def _list_class(pg: PgClient, asset_class: str, *, eod_eligible_only: bool = False) -> list[dict]:
    """List plottable series. Equity/ETF default to the live EOD universe only."""
    extra = ""
    if eod_eligible_only:
        extra = f" AND m.status = 'ACTIVE' AND ({EOD_ELIGIBLE_WHERE.strip()})"
    return pg.fetchall(
        f"""
        SELECT m.series_id, m.canonical_symbol, m.first_seen, m.last_seen, m.extras, m.asset_class
        FROM series_meta m
        WHERE m.asset_class = %s
          AND m.first_seen IS NOT NULL
          AND m.last_seen IS NOT NULL
          {extra}
        ORDER BY m.series_id
        """,
        (asset_class,),
    )


def _serve_base(cfg: MarketsConfig) -> str:
    return cfg.ray_serve_url.rstrip("/")


def _health_ok(base: str, *, timeout: float = 5.0) -> bool:
    try:
        r = requests.get(f"{base}/health", timeout=timeout)
        return r.ok and bool(r.json().get("ok"))
    except Exception:
        return False


def fetch_series_api(
    base: str,
    series_id: str,
    start: date,
    end: date,
    *,
    revision_mode: str,
    as_of: date | None,
    timeout: float,
) -> dict:
    path = quote(series_id, safe="")
    params: dict[str, str] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "revision_mode": revision_mode,
    }
    if as_of is not None:
        params["as_of"] = as_of.isoformat()
    url = f"{base}/v1/series/{path}"
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def _plot_one(
    *,
    base: str,
    asset_class: str,
    row: dict,
    rev_mode: str,
    plot_dir: Path,
    timeout: float,
    index: int,
    total: int,
) -> tuple[str, str | None]:
    sid = str(row["series_id"])
    sym = str(row["canonical_symbol"]).replace("/", "_").replace("\\", "_")
    start, end = plot_window(row, plot_years=None)
    as_of = end if rev_mode == "as_of" else None
    safe_id = sid.replace(":", "_")
    out_path = plot_dir / asset_class / f"{sym}_{safe_id}.png"
    try:
        body = fetch_series_api(
            base,
            sid,
            start,
            end,
            revision_mode=rev_mode,
            as_of=as_of,
            timeout=timeout,
        )
        df = rows_to_df(body.get("rows") or [])
    except Exception as exc:
        logger.warning("skip %s (%s)", sid, exc)
        return "skip", str(exc)
    if df.empty:
        logger.warning("skip %s (empty)", sid)
        return "skip", "empty"
    write_plot(df, out_path, sid, sym)
    logger.info(
        "[%s/%s] %s/%s days=%s -> %s",
        index,
        total,
        asset_class,
        sym,
        len(df),
        out_path.name,
    )
    return "ok", out_path.name


def run_plot_warm(
    cfg: MarketsConfig,
    *,
    out_dir: Path,
    equity_n: int,
    etf_n: int,
    seed: int,
    timeout: float,
    workers: int,
    revision_mode_macro: str,
    revision_mode_price: str,
) -> dict:
    base = _serve_base(cfg)
    if not _health_ok(base):
        raise RuntimeError(
            f"Serve health failed at {base}/health — start the supervisor or deploy Serve first"
        )

    pg = PgClient(cfg.postgres_url)
    picks: list[tuple[str, dict, str]] = []
    for row in _list_class(pg, "macro"):
        picks.append(("macro", row, revision_mode_macro))
    equity_pool = _list_class(pg, "equity", eod_eligible_only=True)
    etf_pool = _list_class(pg, "etf", eod_eligible_only=True)
    logger.info(
        "plot pools equity_eligible=%s etf_eligible=%s",
        len(equity_pool),
        len(etf_pool),
    )
    for row in _stable_sample(equity_pool, equity_n, seed=seed):
        picks.append(("equity", row, revision_mode_price))
    for row in _stable_sample(etf_pool, etf_n, seed=seed + 1):
        picks.append(("etf", row, revision_mode_price))

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    workers = len(picks) if int(workers) <= 0 else max(1, int(workers))
    written = 0
    skipped = 0
    lock = Lock()

    logger.info(
        "plot_warm base=%s workers=%s macro=%s equity=%s etf=%s total=%s",
        base,
        workers,
        sum(1 for a, _, _ in picks if a == "macro"),
        sum(1 for a, _, _ in picks if a == "equity"),
        sum(1 for a, _, _ in picks if a == "etf"),
        len(picks),
    )

    def _job(item: tuple[int, str, dict, str]) -> tuple[str, str | None]:
        index, asset_class, row, rev_mode = item
        return _plot_one(
            base=base,
            asset_class=asset_class,
            row=row,
            rev_mode=rev_mode,
            plot_dir=plot_dir,
            timeout=timeout,
            index=index,
            total=len(picks),
        )

    work = [(i, a, r, m) for i, (a, r, m) in enumerate(picks, start=1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, item) for item in work]
        for fut in as_completed(futures):
            status, _ = fut.result()
            with lock:
                if status == "ok":
                    written += 1
                else:
                    skipped += 1

    summary = out_dir / "summary.txt"
    summary.write_text(
        f"written={written} skipped={skipped} out={plot_dir} workers={workers}\n"
        f"macro_revision={revision_mode_macro} price_revision={revision_mode_price}\n",
        encoding="utf-8",
    )
    return {"written": written, "skipped": skipped, "workers": workers, "out": str(plot_dir)}


def main() -> None:
    p = base_parser("Plot + warm L3 via Serve (ACTIVE EOD-eligible equity/ETF + macros)")
    p.add_argument("--out", type=Path, default=Path("plot_samples"))
    p.add_argument("--equity", type=int, default=50, help="equity sample size")
    p.add_argument("--etf", type=int, default=50, help="ETF sample size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=300.0, help="per-series HTTP timeout seconds")
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="parallel Serve API requests (default 32; 0 = one per series)",
    )
    p.add_argument(
        "--macro-revision",
        default="as_of",
        choices=("as_of", "latest"),
        help="FRED vintage collapse (default as_of = leakage-proof through end)",
    )
    p.add_argument(
        "--price-revision",
        default="latest",
        choices=("as_of", "latest"),
        help="equity/ETF revision_mode (default latest uses L3 cache)",
    )
    args = p.parse_args()

    configure_logging()
    cfg = MarketsConfig.from_env()
    try:
        out = run_plot_warm(
            cfg,
            out_dir=args.out,
            equity_n=args.equity,
            etf_n=args.etf,
            seed=args.seed,
            timeout=args.timeout,
            workers=args.workers,
            revision_mode_macro=args.macro_revision,
            revision_mode_price=args.price_revision,
        )
        logger.info("done %s", out)
    except Exception:
        logger.exception("failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
