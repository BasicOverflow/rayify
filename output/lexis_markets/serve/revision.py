"""FRED / ALFRED revision semantics for stitch and API queries."""
from __future__ import annotations

from datetime import date
from typing import Literal

RevisionMode = Literal["as_of", "latest"]

DEFAULT_REVISION_MODE: RevisionMode = "as_of"


def parse_revision_mode(raw: str | None) -> RevisionMode:
    mode = (raw or DEFAULT_REVISION_MODE).strip().lower()
    if mode not in ("as_of", "latest"):
        raise ValueError(f"revision_mode must be 'as_of' or 'latest', got {raw!r}")
    return mode  # type: ignore[return-value]


def resolve_collapse_as_of(
    *,
    revision_mode: RevisionMode,
    request_end: date,
    as_of: date | None,
) -> date | None:
    """Map API revision_mode to ``collapse_fred_vintages``'s ``as_of`` argument.

    ``latest`` → None (newest ``realtime_end`` per observation in L1).
    ``as_of`` → explicit ``as_of`` or the request end date when omitted.
    """
    if revision_mode == "latest":
        return None
    return as_of or request_end


def uses_l3_cache(revision_mode: RevisionMode) -> bool:
    """L3 cache is built with ``latest`` vintage collapse only."""
    return revision_mode == "latest"
