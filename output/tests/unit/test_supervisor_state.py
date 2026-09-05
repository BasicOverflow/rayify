"""Unit tests: supervisor SQLite state."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lexis_markets.jobs.queue import SupervisorState


@pytest.mark.unit
def test_enqueue_and_complete():
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "state.db")
        st = SupervisorState(db)
        item_id = st.enqueue_pending("seed", {"reset": False})
        st.mark_running(item_id)
        st.mark_complete(item_id, {"rows": 1})
        pending = st.fetch_pending()
        assert not any(p["id"] == item_id for p in pending)


@pytest.mark.unit
def test_dedupe_pending_seed():
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "state.db")
        st = SupervisorState(db)
        a = st.enqueue_pending("seed", {})
        b = st.enqueue_pending("seed", {})
        dropped = st.dedupe_pending_jobs("seed")
        assert dropped == 1
        assert a != b


@pytest.mark.unit
def test_recover_interrupted_requeues_eod():
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "state.db")
        st = SupervisorState(db)
        item_id = st.enqueue_pending("eod", {"target_date": "2024-01-01"})
        st.mark_running(item_id)
        out = st.recover_interrupted_jobs(seed_complete=False)
        assert out["requeued"] == 1


@pytest.mark.unit
def test_recover_interrupted_seed_complete_when_marker_exists():
    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "state.db")
        st = SupervisorState(db)
        item_id = st.enqueue_pending("seed", {})
        st.mark_running(item_id)
        out = st.recover_interrupted_jobs(seed_complete=True)
        assert out["completed"] == 1
