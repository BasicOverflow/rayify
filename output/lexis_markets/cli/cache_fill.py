"""CLI: L3 cache fill via Serve HTTP (one series or a universe)."""
from __future__ import annotations

from datetime import date

from lexis_markets.cli._runner import base_parser, cli_main
from lexis_markets.serve.cache_fill import fill_active_universe
from lexis_markets.serve.client import warm_series


def main():
    p = base_parser("L3 cache fill via Serve (single series or universe)")
    p.add_argument("--series", help="Single series_id (omit with --universe active)")
    p.add_argument("--start", help="Inclusive start YYYY-MM-DD (single series)")
    p.add_argument("--end", help="Inclusive end YYYY-MM-DD (single series)")
    p.add_argument(
        "--universe",
        choices=["active", "delisted", "all_tradeable"],
        help="Fill universe: active | delisted | all_tradeable (ACTIVE+DELISTED)",
    )
    p.add_argument(
        "--status",
        action="append",
        dest="statuses",
        help="Repeatable status filter (overrides --universe defaults)",
    )
    p.add_argument(
        "--tip-days",
        type=int,
        default=None,
        help="Tip window days before last_seen (default CACHE_FILL_TIP_DAYS or 60)",
    )
    p.add_argument(
        "--tip-only",
        action="store_true",
        help="With --universe: only tip window, skip deepen",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap series count")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore lake progress and redo tip/deep for the universe",
    )
    p.add_argument(
        "--asset-class",
        action="append",
        dest="asset_classes",
        help="Restrict universe (repeatable): equity, etf, macro",
    )
    args = p.parse_args()

    if args.universe or args.statuses:

        def run(cfg):
            if args.statuses:
                statuses = args.statuses
            elif args.universe == "delisted":
                statuses = ["DELISTED"]
            elif args.universe == "all_tradeable":
                statuses = ["ACTIVE", "DELISTED"]
            else:
                statuses = ["ACTIVE"]
            return fill_active_universe(
                cfg,
                tip_days=args.tip_days,
                deepen=not args.tip_only,
                asset_classes=args.asset_classes,
                limit=args.limit,
                statuses=statuses,
                fresh=bool(args.fresh),
            )

        cli_main("cache fill universe", run)
        return

    if not args.series or not args.start or not args.end:
        p.error("--series/--start/--end required unless --universe / --status")

    def run(cfg):
        return warm_series(
            cfg,
            [(args.series, date.fromisoformat(args.start), date.fromisoformat(args.end))],
            include_rows=False,
        )

    cli_main("cache fill", run)


if __name__ == "__main__":
    main()
