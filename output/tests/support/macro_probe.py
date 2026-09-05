"""Probe macro series via live Serve API."""
from __future__ import annotations

from lexis_markets.lake import PgClient
from lexis_markets.config import DEFAULT_FRED_SERIES, MarketsConfig

from tests.support.serve_client import get_series


async def probe_macro_series(
    cfg: MarketsConfig,
    start: str,
    end: str,
    *,
    max_tries: int = 15,
) -> str:
    pg = PgClient(cfg.postgres_url)
    candidates = pg.fetchall(
        """
        SELECT series_id FROM series_meta
        WHERE asset_class = 'macro' AND first_seen IS NOT NULL AND last_seen IS NOT NULL
        ORDER BY series_id
        LIMIT 50
        """
    )
    ids = [r["series_id"] for r in candidates]
    preferred = [f"macro:{s}" for s in DEFAULT_FRED_SERIES if f"macro:{s}" in ids]
    for sid in preferred + [i for i in ids if i not in preferred]:
        try:
            body = await get_series(cfg, sid, start, end, revision_mode="latest")
            if body.get("rows"):
                return sid
        except RuntimeError:
            continue
        max_tries -= 1
        if max_tries <= 0:
            break
    raise RuntimeError("no macro series returned data from Serve")
