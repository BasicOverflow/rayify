"""CLI entry: read L3 cache and print gap/disagreement metrics for one series."""
from __future__ import annotations

import argparse
from datetime import date

from lexis_markets.lake import LakeStore, PgClient
from lexis_markets.cli._runner import base_parser, cli_main
from lexis_markets.domain.quality import series_quality_window
from lexis_markets.serve.cache import read_series_cache


def main():
    p = base_parser("Inspect one cached series quality")
    p.add_argument("--series", default="equity:AAPL")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2020-12-31")
    args = p.parse_args()

    def run(cfg):
        lake = LakeStore(cfg)
        pg = PgClient(cfg.postgres_url)
        meta = pg.fetchone(
            "SELECT calendar_id FROM series_meta WHERE series_id = %s",
            (args.series,),
        )
        bars = read_series_cache(
            lake,
            args.series,
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
        )
        cal = (meta or {}).get("calendar_id") or "nyse"
        q = series_quality_window(bars, cal)
        return {
            "series_id": args.series,
            "rows": len(bars),
            **{k: q[k] for k in ("gap_count", "disagreement_count", "suspicious_count", "flags", "quality_score")},
        }

    cli_main("inspect series", run)


if __name__ == "__main__":
    main()
