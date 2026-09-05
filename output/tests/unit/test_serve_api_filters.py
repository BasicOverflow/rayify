"""Unit tests: Serve universe filters / resolve_series_ids (deterministic past dates)."""
from __future__ import annotations

from datetime import date

import pytest

from lexis_markets.registry.filters import (
    append_series_filters,
    covers_through_end,
    effective_last_seen,
    normalize_statuses,
)
from lexis_markets.registry.meta import resolve_series_ids
from tests.unit.api_fixtures import (
    ERA_END,
    RECENT_END,
    FilterPg,
    build_api_test_universe,
    effective_last,
)


@pytest.fixture
def universe():
    return build_api_test_universe(100)


@pytest.fixture
def pg(universe):
    return FilterPg(universe)


@pytest.mark.unit
def test_universe_has_all_asset_classes(universe):
    classes = {r["asset_class"] for r in universe}
    assert classes == {"equity", "etf", "macro"}
    assert len(universe) == 100


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ([], None),
        (["active", "Delisted"], ["ACTIVE", "DELISTED"]),
        (["ACTIVE"], ["ACTIVE"]),
    ],
)
def test_normalize_statuses_ok(raw, expected):
    assert normalize_statuses(raw) == expected


@pytest.mark.unit
def test_normalize_statuses_rejects_unknown():
    with pytest.raises(ValueError, match="invalid statuses"):
        normalize_statuses(["ACTIVE", "BOGUS"])


@pytest.mark.unit
def test_append_series_filters_sql_and_params():
    sql, params = append_series_filters(
        "SELECT m.series_id FROM series_meta m WHERE 1=1",
        [],
        {
            "statuses": ["active"],
            "max_gap_count": 5,
            "max_suspicious_count": 2,
            "include_partial_coverage": False,
            "end": "2020-12-31",
        },
    )
    assert "m.status = ANY(%s)" in sql
    assert "m.gap_count <= %s" in sql
    assert "m.suspicious_count <= %s" in sql
    assert ">= %s::date" in sql
    assert params[0] == ["ACTIVE"]
    assert params[1] == 5
    assert params[2] == 2
    assert params[3] == "2020-12-31"


@pytest.mark.unit
def test_resolve_by_series_ids(pg, universe):
    pick = [universe[0]["series_id"], universe[1]["series_id"]]
    got = resolve_series_ids(pg, {"series_ids": pick, "start": "2010-01-01", "end": "2020-01-01"})
    assert set(got) == set(pick)


@pytest.mark.unit
def test_resolve_by_symbols(pg, universe):
    sym = universe[3]["canonical_symbol"]
    got = resolve_series_ids(pg, {"symbols": [sym.lower()], "start": "2010-01-01", "end": "2020-01-01"})
    assert any(r["series_id"].endswith(f":{sym}") for r in universe if r["series_id"] in got)
    assert got


@pytest.mark.unit
def test_resolve_asset_classes(pg):
    got = resolve_series_ids(
        pg, {"asset_classes": ["macro"], "statuses": ["ACTIVE"], "start": "2000-01-01", "end": "2020-01-01"}
    )
    assert got
    assert all(s.startswith("macro:") for s in got)


@pytest.mark.unit
def test_resolve_statuses_active_only(pg, universe):
    got = resolve_series_ids(pg, {"statuses": ["ACTIVE"], "start": "2000-01-01", "end": "2020-01-01"})
    active = {r["series_id"] for r in universe if r["status"] == "ACTIVE"}
    assert set(got) == active


@pytest.mark.unit
def test_resolve_max_gap_count(pg, universe):
    got = resolve_series_ids(pg, {"max_gap_count": 5, "start": "2000-01-01", "end": "2020-01-01"})
    allowed = {r["series_id"] for r in universe if r["gap_count"] <= 5}
    assert set(got) == allowed


@pytest.mark.unit
def test_resolve_max_suspicious_count(pg, universe):
    got = resolve_series_ids(pg, {"max_suspicious_count": 0, "start": "2000-01-01", "end": "2020-01-01"})
    allowed = {r["series_id"] for r in universe if r["suspicious_count"] <= 0}
    assert set(got) == allowed


@pytest.mark.unit
def test_resolve_include_partial_coverage_false_excludes_short_series(pg, universe):
    end = ERA_END.isoformat()
    got = resolve_series_ids(
        pg,
        {
            "statuses": ["ACTIVE"],
            "include_partial_coverage": False,
            "end": end,
            "start": "2019-01-01",
        },
    )
    for sid in got:
        row = next(r for r in universe if r["series_id"] == sid)
        assert effective_last(row) >= ERA_END
    # At least one ACTIVE row ends at RECENT_END and must be excluded.
    short = [r for r in universe if r["status"] == "ACTIVE" and effective_last(r) == RECENT_END]
    assert short
    assert all(r["series_id"] not in got for r in short)


@pytest.mark.unit
def test_resolve_include_partial_coverage_true_keeps_short_series(pg, universe):
    end = ERA_END.isoformat()
    got = resolve_series_ids(
        pg,
        {
            "statuses": ["ACTIVE"],
            "include_partial_coverage": True,
            "end": end,
            "start": "2019-01-01",
        },
    )
    short_ids = {
        r["series_id"]
        for r in universe
        if r["status"] == "ACTIVE" and effective_last(r) == RECENT_END
    }
    assert short_ids & set(got)


@pytest.mark.unit
def test_contradictory_asset_class_and_series_ids_yields_empty(pg, universe):
    macro = next(r for r in universe if r["asset_class"] == "macro")
    got = resolve_series_ids(
        pg,
        {
            "series_ids": [macro["series_id"]],
            "asset_classes": ["equity"],
            "start": "2010-01-01",
            "end": "2020-01-01",
        },
    )
    assert got == []


@pytest.mark.unit
def test_contradictory_status_filter_and_series_ids_yields_empty(pg, universe):
    delisted = next(r for r in universe if r["status"] == "DELISTED")
    got = resolve_series_ids(
        pg,
        {
            "series_ids": [delisted["series_id"]],
            "statuses": ["ACTIVE"],
            "start": "2010-01-01",
            "end": "2020-01-01",
        },
    )
    assert got == []


@pytest.mark.unit
def test_combo_status_gap_suspicious_asset_class(pg, universe):
    spec = {
        "asset_classes": ["equity", "etf"],
        "statuses": ["ACTIVE"],
        "max_gap_count": 20,
        "max_suspicious_count": 5,
        "include_partial_coverage": False,
        "start": "2015-01-01",
        "end": ERA_END.isoformat(),
    }
    got = set(resolve_series_ids(pg, spec))
    for r in universe:
        ok = (
            r["asset_class"] in ("equity", "etf")
            and r["status"] == "ACTIVE"
            and r["gap_count"] <= 20
            and r["suspicious_count"] <= 5
            and effective_last(r) >= ERA_END
        )
        assert (r["series_id"] in got) == ok


@pytest.mark.unit
def test_resolve_invalid_statuses_raises(pg):
    with pytest.raises(ValueError, match="invalid statuses"):
        resolve_series_ids(pg, {"statuses": ["NOPE"], "start": "2010-01-01", "end": "2020-01-01"})


@pytest.mark.unit
def test_effective_last_seen_and_covers_through_end():
    row = {
        "last_seen": date(2020, 6, 1),
        "extras": {"eod_filled_through": "2020-12-31"},
    }
    assert effective_last_seen(row) == date(2020, 12, 31)
    assert covers_through_end(row, date(2020, 12, 31)) is True
    assert covers_through_end(row, date(2021, 1, 1)) is False
