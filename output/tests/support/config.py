"""Load test environment and build MarketsConfig."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from lexis_markets.config import MarketsConfig, OUTPUT_DIR

TEST_ENV = OUTPUT_DIR / ".env.test"
PROD_NAMESPACE = "lexis-markets"


def load_test_env() -> None:
    """Load ``.env.test`` when present; compose may inject the same keys via env_file."""
    if TEST_ENV.is_file():
        load_dotenv(TEST_ENV, override=True)
    elif os.environ.get("MARKETS_TEST_PROFILE") != "1":
        raise FileNotFoundError(
            f"missing {TEST_ENV} — copy .env.test.example and fill credentials "
            "(or set MARKETS_TEST_PROFILE=1 via compose env_file)"
        )
    load_dotenv(OUTPUT_DIR.parent / ".env", override=False)


def for_tests() -> MarketsConfig:
    load_test_env()
    cfg = MarketsConfig.from_env()
    assert cfg.ray_namespace != PROD_NAMESPACE, "refusing to run tests against prod RAY_NAMESPACE"
    assert cfg.lake_prefix, "MARKETS_LAKE_PREFIX must be set for tests"
    assert "lexis_markets_test" in cfg.postgres_url, "tests must use lexis_markets_test database"
    assert cfg.test.enabled, "MARKETS_TEST_PROFILE=1 required"
    return cfg
