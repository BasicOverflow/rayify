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
