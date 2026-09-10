"""Restore macro registry windows + L3 after bad dense_window_trim.

``fred_native`` series must not use equity-style dense trim: month gaps and
historical missing stretches make the densest run end years ago (e.g. DFF→2011),
which then rewrites L3 and leaves Serve empty for recent plot windows.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from lexis_markets.config import MarketsConfig
from lexis_markets.jobs.clock import as_date, eod_target_date
from lexis_markets.lake import LakeStore, PgClient, utcnow
from lexis_markets.logging_setup import get_logger
from lexis_markets.registry import merge_spans, update_cache_meta
from lexis_markets.serve.cache import build_cache_span, invalidate_series_l3, l1_fingerprint
from lexis_markets.serve.quality_persist import persist_l3_quality
from lexis_markets.serve.stitch import stitch_series

logger = get_logger("fred.repair")

# Series that QA found empty under ACTIVE + recent windows.
DEFAULT_REPAIR_IDS = (
    "macro:VIXCLS",
    "macro:DFF",
    "macro:M2SL",
    "macro:PERMIT",
    "macro:BUSLOANS",
    "macro:STLFSI4",
    "macro:T10YFF",
)


def _restore_trade_window(pg: PgClient, series_id: str, first_seen: date, last_seen: date) -> None:
    pg.execute(
        """
        UPDATE series_meta
        SET first_seen = %s,
            last_seen = %s,
            extras = (COALESCE(extras, '{}'::jsonb)
                - 'window_trim'
                - 'window_trim_at')
                || jsonb_build_object(
                    'window_restored', 'fred_native',
                    'window_restored_at', %s::text
                )
        WHERE series_id = %s
        """,
        (first_seen, last_seen, utcnow().isoformat(), series_id),
    )


def repair_macro_l3(
    cfg: MarketsConfig,
    series_ids: list[str] | None = None,
    *,
    through: date | None = None,
) -> list[dict]:
    """Stitch latest FRED bars, restore registry window, rebuild L3."""
    pg = PgClient(cfg.postgres_url)
    end = through or eod_target_date()
    ids = list(series_ids or DEFAULT_REPAIR_IDS)
    out: list[dict] = []
    for sid in ids:
        meta = pg.fetchone(
            """
            SELECT series_id, first_seen, last_seen, calendar_id, status
            FROM series_meta WHERE series_id = %s
            """,
            (sid,),
        )
        if not meta:
            out.append({"series_id": sid, "ok": False, "reason": "missing_meta"})
            continue
        start = as_date(meta.get("first_seen")) or date(1950, 1, 1)
        # Pull from well before registry first in case trim also raised first_seen.
        stitch_start = min(start, date(1950, 1, 1))
        bars = stitch_series(
            LakeStore(cfg),
            pg,
            sid,
            stitch_start,
            end,
            revision_mode="latest",
            as_of=None,
        )
        if bars is None or bars.empty:
            out.append({"series_id": sid, "ok": False, "reason": "stitch_empty"})
            continue
        ts = pd.to_datetime(bars["ts"]).dt.date
        first, last = ts.min(), ts.max()
        old_first, old_last = as_date(meta.get("first_seen")), as_date(meta.get("last_seen"))
        _restore_trade_window(pg, sid, first, last)
        invalidate_series_l3(cfg, sid)
        built = build_cache_span(cfg, sid, first, last)
        merge_spans(pg, sid, first, last)
        update_cache_meta(pg, sid, int(built.get("rows") or 0), l1_fingerprint(pg, first, last))
        persist_l3_quality(cfg, sid, first, last)
        rec = {
            "series_id": sid,
            "ok": True,
            "rows": int(built.get("rows") or 0),
            "old_window": f"{old_first}..{old_last}",
            "new_window": f"{first}..{last}",
        }
        out.append(rec)
        logger.info(
            "repair_macro %s rows=%s window=%s..%s (was %s..%s)",
            sid,
            rec["rows"],
            first,
            last,
            old_first,
            old_last,
        )
    return out
