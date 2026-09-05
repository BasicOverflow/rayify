"""Goal-state assertions for weeksim (markers, fill-through, Serve)."""
from __future__ import annotations

from datetime import date

from lexis_markets.lake import LakeStore, PgClient
from lexis_markets.eod.ingest import EOD_MARKER_PREFIX, eod_marker_key
from lexis_markets.kaggle.ingest import JC_MARKER, JW_MARKER
from tests.support import api_assertions
from tests.weeksim.universe import WeekUniverse


def assert_seed_markers(lake: LakeStore) -> None:
    assert lake.exists(JW_MARKER), "missing jakewright L1 marker"
    assert lake.exists(JC_MARKER), "missing jacksoncrow L1 marker"
    assert lake.exists("ops/markers/seed_complete.json"), "missing seed_complete"


def assert_pool_in_registry(pg: PgClient, universe: WeekUniverse) -> None:
    for sym in universe.equity_symbols:
        row = pg.fetchone(
            """
            SELECT series_id FROM series_meta
            WHERE UPPER(canonical_symbol) = %s AND asset_class = 'equity'
            """,
            (sym,),
        )
        assert row, f"missing equity series for {sym}"
    for sid in universe.fred_series:
        row = pg.fetchone(
            "SELECT series_id FROM series_meta WHERE series_id = %s",
            (f"macro:{sid}",),
        )
        assert row, f"missing macro series macro:{sid}"


def assert_eod_marker(lake: LakeStore, target: date) -> None:
    key = eod_marker_key(target)
    assert lake.exists(key), f"missing EOD marker {key}"


def assert_weekend_eod_noop(out: dict, target: date) -> None:
    """Weekend targets may be empty; do not treat missing work as a green marker."""
    assert out.get("target_date") == target.isoformat()
    # Empty / short-circuit is OK; inventing success rows is not.
    if int(out.get("symbols") or 0) == 0 and int(out.get("rows") or 0) == 0:
        return
    # If data landed anyway (rare holiday calendars), require an honest marker key in payload.
    assert out.get("source") == "eod"


def assert_eod_fill_through(pg: PgClient, universe: WeekUniverse, target: date) -> None:
    for sym in universe.equity_symbols:
        row = pg.fetchone(
            """
            SELECT extras->>'eod_filled_through' AS through
            FROM series_meta
            WHERE UPPER(canonical_symbol) = %s AND asset_class = 'equity'
            """,
            (sym,),
        )
        assert row and row.get("through"), f"{sym} missing eod_filled_through after EOD"
        through = date.fromisoformat(row["through"])
        assert through >= target, f"{sym} eod_filled_through={through} < {target}"


def assert_fred_vintage_through(pg: PgClient, universe: WeekUniverse, target: date) -> None:
    for sid in universe.fred_series:
        row = pg.fetchone(
            """
            SELECT extras->>'fred_vintage_through' AS through
            FROM series_meta WHERE series_id = %s
            """,
            (f"macro:{sid}",),
        )
        assert row and row.get("through"), f"{sid} missing fred_vintage_through after EOD"
        through = date.fromisoformat(row["through"])
        assert through >= target, f"{sid} fred_vintage_through={through} < {target}"


def list_eod_markers(lake: LakeStore) -> list[str]:
    return lake.list_keys(f"{EOD_MARKER_PREFIX}/")


def assert_stitch_source_mix(pg: PgClient, symbol: str = "AAPL") -> None:
    """Stitched equity should expose Kaggle history plus EOD fill sources."""
    rows = pg.fetchall(
        """
        SELECT DISTINCT s.source
        FROM stitch_segments s
        JOIN series_meta m ON m.series_id = s.series_id
        WHERE UPPER(m.canonical_symbol) = %s AND m.asset_class = 'equity'
        """,
        (symbol.upper(),),
    )
    sources = {r["source"] for r in rows}
    kaggle = sources & {"jakewright", "jacksoncrow"}
    eod = sources & {"marketparquet", "yfinance"}
    assert kaggle, f"{symbol} missing kaggle stitch sources, got {sources}"
    assert eod, f"{symbol} missing EOD stitch sources, got {sources}"


async def assert_serve_equity_range(cfg, series_id: str, start: str, end: str) -> None:
    from tests.support.serve_client import get_series

    body = await get_series(cfg, series_id, start, end)
    df = api_assertions.parse_rows(body)
    api_assertions.assert_range(df, start, end)
    api_assertions.assert_ohlcv_sanity(df)
    api_assertions.assert_quality(df, "us_equity", min_score=0.2)
    quality = body.get("quality") or {}
    bad = set(quality.get("flags") or []) & {
        "linear_ramp",
        "sparse_bridge",
        "flat_close",
        "ohlc_violation",
        "non_positive_close",
        "duplicate_ts",
        "extreme_return",
    }
    assert not bad, f"{series_id} quality flags on sim window: {bad} ({quality})"


def assert_pool_quality_registry(pg: PgClient, universe: WeekUniverse) -> None:
    """After quality job: pool equities have metrics; liquid names must stay clean."""
    for sym in universe.all_equity_etf:
        row = pg.fetchone(
            """
            SELECT series_id, gap_count, disagreement_count, suspicious_count, quality_score
            FROM series_meta
            WHERE UPPER(canonical_symbol) = %s AND asset_class IN ('equity', 'etf')
            """,
            (sym,),
        )
        assert row, f"missing meta for {sym}"
        assert row.get("quality_score") is not None, f"{sym} missing quality_score"
        assert row.get("suspicious_count") is not None, f"{sym} missing suspicious_count"
        # AAPL/MSFT/SPY must not look like ITMR sparse bridges or linear fills.
        assert int(row["suspicious_count"] or 0) == 0, (
            f"{sym} suspicious_count={row['suspicious_count']} (expected 0 for liquid pool)"
        )
