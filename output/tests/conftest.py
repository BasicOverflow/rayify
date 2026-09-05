"""Root pytest hooks — load ``.env.test`` when present (weeksim owns wipe/Ray)."""
from __future__ import annotations

from tests.support.config import load_test_env


def pytest_configure(config):
    try:
        load_test_env()
    except FileNotFoundError:
        pass
