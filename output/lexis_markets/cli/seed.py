"""CLI entry: full seed pipeline (``MARKETS_RESET=1`` truncates L2 first)."""
from __future__ import annotations

import os

from lexis_markets.cli._runner import cli_main
from lexis_markets.kaggle.seed import run_seed


def main():
    reset = os.environ.get("MARKETS_RESET") == "1"
    cli_main("Full seed pipeline", lambda cfg: run_seed(cfg, reset=reset))


if __name__ == "__main__":
    main()
