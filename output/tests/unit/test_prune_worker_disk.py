"""Worker disk prune must tolerate one unschedulable node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_prune_ray_worker_disk_skips_unschedulable_nodes():
    from lexis_markets import cleanup as cleanup_mod

    good = {"node_ip": "10.0.0.1", "freed_mb": 1.5}
    refs = [MagicMock(name="ok_ref"), MagicMock(name="dead_ref")]

    def fake_get(ref):
        if ref is refs[1]:
            raise RuntimeError("TaskUnschedulableError: node gone")
        return good

    nodes = [
        {"Alive": True, "NodeID": "a" * 56},
        {"Alive": True, "NodeID": "b" * 56},
        {"Alive": False, "NodeID": "c" * 56},
    ]
    remote_fn = MagicMock()
    remote_fn.options.return_value.remote.side_effect = refs

    with (
        patch.object(cleanup_mod.ray, "nodes", return_value=nodes),
        patch.object(cleanup_mod.ray, "get", side_effect=fake_get),
        patch.object(cleanup_mod, "_prune_worker_tmp", remote_fn),
    ):
        out = cleanup_mod.prune_ray_worker_disk()

    assert out == [good]
    assert remote_fn.options.call_count == 2


@pytest.mark.unit
def test_prune_runtime_resources_never_deletes_packages(tmp_path):
    """Regression: age-pruning pip/working_dir broke Serve workers mid-flight."""
    from lexis_markets.cleanup import _prune_runtime_resources

    session = tmp_path / "session_x"
    wdf = session / "runtime_resources" / "working_dir_files" / "_ray_pkg_deadbeef"
    pip = session / "runtime_resources" / "pip" / "abc123"
    wdf.mkdir(parents=True)
    pip.mkdir(parents=True)
    (wdf / "keep.txt").write_text("x", encoding="utf-8")
    (pip / "keep.txt").write_text("x", encoding="utf-8")

    removed, freed = _prune_runtime_resources(str(session), cutoff=10**18)
    assert removed == 0
    assert freed == 0
    assert (wdf / "keep.txt").exists()
    assert (pip / "keep.txt").exists()


@pytest.mark.unit
def test_prune_orphan_tmp_dirs_skips_live_and_deletes_dead(tmp_path):
    from lexis_markets.scratch import LIVE_MARKER, mark_live, prune_orphan_tmp_dirs

    live = tmp_path / "lexis-jw-stage-live"
    live.mkdir()
    (live / "dataset.zip").write_bytes(b"x" * 100)
    mark_live(live)

    dead = tmp_path / "lexis-jc-stage-dead"
    dead.mkdir()
    (dead / "dataset.zip").write_bytes(b"y" * 200)
    (dead / LIVE_MARKER).write_text("999999999", encoding="utf-8")

    empty_seed = tmp_path / "lexis-seed-0-orph"
    empty_seed.mkdir()

    other = tmp_path / "not-ours"
    other.mkdir()
    (other / "keep.txt").write_text("ok", encoding="utf-8")

    removed, freed = prune_orphan_tmp_dirs(str(tmp_path))
    assert removed == 2
    assert freed >= 200
    assert live.exists()
    assert (live / "dataset.zip").exists()
    assert not dead.exists()
    assert not empty_seed.exists()
    assert (other / "keep.txt").exists()


@pytest.mark.unit
def test_truncate_large_logs_uses_lower_trigger(tmp_path):
    from lexis_markets.cleanup import _truncate_large_logs

    logs = tmp_path / "logs"
    logs.mkdir()
    path = logs / "raylet.out"
    path.write_bytes(b"a" * 80 + b"TAILKEEP")
    trimmed, freed = _truncate_large_logs(
        str(tmp_path),
        trigger_bytes=50,
        keep_bytes=8,
    )
    assert trimmed == 1
    assert freed > 0
    data = path.read_bytes()
    assert data.endswith(b"TAILKEEP")
    assert b"...truncated..." in data
    assert len(data) < 80 + 8
