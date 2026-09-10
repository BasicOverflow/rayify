"""Shared CLI bootstrap: logging, Ray init, Postgres schema ensure."""
from __future__ import annotations

import argparse
import sys

from lexis_markets.config import MarketsConfig
from lexis_markets.logging_setup import configure_logging, get_logger
from lexis_markets.ray.runtime import init_ray


def cli_main(description: str, run_fn) -> None:
    configure_logging()
    logger = get_logger("cli")
    cfg = MarketsConfig.from_env()
    init_ray(cfg)
    from lexis_markets.lake import PgClient, ensure_schema

    pg = PgClient(cfg.postgres_url)
    ensure_schema(pg)
    try:
        out = run_fn(cfg)
        logger.info("done %s", out)
    except Exception:
        logger.exception("failed")
        sys.exit(1)


def base_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)
