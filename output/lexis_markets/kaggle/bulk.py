"""Parallel jakewright + jacksoncrow L1 bulk ingest remotes."""
from __future__ import annotations

import ray

from lexis_markets.lake import LakeStore
from lexis_markets.config import MarketsConfig
from lexis_markets.kaggle.ingest import ingest_jacksoncrow, ingest_jakewright


@ray.remote
def _remote_ingest_jakewright(cfg_d: dict) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    return ingest_jakewright(cfg, LakeStore(cfg))


@ray.remote
def _remote_ingest_jacksoncrow(cfg_d: dict) -> dict:
    cfg = MarketsConfig.from_dict(cfg_d)
    return ingest_jacksoncrow(cfg, LakeStore(cfg))
