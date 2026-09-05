"""Read lake + Postgres markers to report what is loaded and what remains."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lexis_markets.config import MarketsConfig
from lexis_markets.eod.ingest import eod_marker_key
from lexis_markets.jobs.clock import eod_target_date
from lexis_markets.kaggle.ingest import JC_MARKER, JW_MARKER
from lexis_markets.kaggle.seed import (
    SEED_COMPLETE_MARKER,
    SEED_PROGRESS_MARKER,
    is_seed_complete,
)
from lexis_markets.lake import LakeStore, PgClient, get_json


@dataclass(frozen=True)
class SourceStatus:
    done: bool
    rows: int | None = None
    symbols: int | None = None


@dataclass(frozen=True)
class MarketsPipelineStatus:
    seed_complete: bool
    seed_stage: str | None
    jakewright: SourceStatus
    jacksoncrow: SourceStatus
    registry_series: int
    registry_stitch: int
    macro_series: int
    last_eod_date: str | None
    resume: str

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_status(lake: LakeStore, marker_key: str) -> SourceStatus:
    if not lake.exists(marker_key):
        return SourceStatus(done=False)
    body = get_json(lake, marker_key)
    return SourceStatus(
        done=True,
        rows=int(body.get("rows") or 0) or None,
        symbols=int(body.get("symbols") or 0) or None,
    )


def _seed_stage(lake: LakeStore) -> str | None:
    if lake.exists(SEED_COMPLETE_MARKER):
        body = get_json(lake, SEED_COMPLETE_MARKER)
        return body.get("stage") or "complete"
    if lake.exists(SEED_PROGRESS_MARKER):
        return get_json(lake, SEED_PROGRESS_MARKER).get("stage")
    return None


def _resume_action(*, seed_complete: bool, jw: SourceStatus, jc: SourceStatus) -> str:
    if seed_complete:
        return "idle"
    if jw.done and jc.done:
        return "resume_seed"
    return "run_seed"


def assess_markets_state(cfg: MarketsConfig) -> MarketsPipelineStatus:
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)

    jw = _source_status(lake, JW_MARKER)
    jc = _source_status(lake, JC_MARKER)
    seed_complete = is_seed_complete(cfg)
    seed_stage = _seed_stage(lake)

    series_row = pg.fetchone("SELECT COUNT(*)::int AS n FROM series_meta") or {"n": 0}
    stitch_row = pg.fetchone("SELECT COUNT(*)::int AS n FROM stitch_segments") or {"n": 0}
    macro_row = pg.fetchone(
        "SELECT COUNT(*)::int AS n FROM series_meta WHERE asset_class = 'macro'"
    ) or {"n": 0}

    target = eod_target_date()
    last_eod = target.isoformat() if lake.exists(eod_marker_key(target)) else None

    return MarketsPipelineStatus(
        seed_complete=seed_complete,
        seed_stage=seed_stage,
        jakewright=jw,
        jacksoncrow=jc,
        registry_series=int(series_row["n"]),
        registry_stitch=int(stitch_row["n"]),
        macro_series=int(macro_row["n"]),
        last_eod_date=last_eod,
        resume=_resume_action(seed_complete=seed_complete, jw=jw, jc=jc),
    )
