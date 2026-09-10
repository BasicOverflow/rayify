"""CLI entry: full seed pipeline."""
from __future__ import annotations

from lexis_markets.cli._runner import cli_main
from lexis_markets.kaggle.seed import run_seed


def main():
    cli_main("Full seed pipeline", lambda cfg: run_seed(cfg))


if __name__ == "__main__":
    main()
