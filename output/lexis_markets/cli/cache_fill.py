"""CLI entry: build L3 cache for one series/date range via Ray tasks."""
from __future__ import annotations

import argparse
from datetime import date

from lexis_markets.serve.cache import CacheService
from lexis_markets.cli._runner import base_parser, cli_main


def main():
    p = base_parser("Bulk cache fill")
    p.add_argument("--series", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()

    def run(cfg):
        svc = CacheService(cfg)
        return svc.ensure_series_cached(
            args.series,
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
        )

    cli_main("cache fill", run)


if __name__ == "__main__":
    main()
