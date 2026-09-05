"""CLI entry: FRED / ALFRED vintage backfill."""
from __future__ import annotations

from datetime import date

from lexis_markets.cli._runner import base_parser, cli_main
from lexis_markets.fred.backfill import run_fred_backfill


def main():
    parser = base_parser("FRED ALFRED vintage backfill")
    parser.add_argument("--force", action="store_true", help="re-fetch full history from FRED_VINTAGE_START")
    parser.add_argument("--through", metavar="YYYY-MM-DD", help="vintage end date (default: yesterday ET)")
    parser.add_argument("--limit", type=int, metavar="N", help="cap Ray jobs (dev smoke)")
    args = parser.parse_args()

    def run(cfg):
        vintage_end = date.fromisoformat(args.through) if args.through else None
        return run_fred_backfill(
            cfg,
            vintage_end=vintage_end,
            force=args.force,
            job_limit=args.limit,
        )

    cli_main("FRED backfill", run)


if __name__ == "__main__":
    main()
