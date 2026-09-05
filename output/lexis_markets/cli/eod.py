"""CLI entry: daily EOD backfill."""
from __future__ import annotations

from lexis_markets.cli._runner import cli_main
from lexis_markets.eod.pipeline import run_eod


def main():
    cli_main("EOD pipeline", lambda cfg: run_eod(cfg))


if __name__ == "__main__":
    main()
