"""Stdlib logging for the ``lexis_markets`` package tree.

Controlled by ``LOG_LEVEL`` (default INFO). Call ``configure_logging()`` once at process entry.
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger("lexis_markets")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    if name.startswith("lexis_markets."):
        return logging.getLogger(name)
    return logging.getLogger(f"lexis_markets.{name}")
