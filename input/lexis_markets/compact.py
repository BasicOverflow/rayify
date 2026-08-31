from __future__ import annotations

from datetime import date
from uuid import uuid4

import pandas as pd
import ray

from lexis_markets.cluster import plan_task_resources, run_batches
from lexis_markets.config import MarketsConfig
from lexis_markets.storage import (
    LakeStore,
    OBS_COLUMNS,
    PgClient,
    month_prefix,
    normalize_obs,
    utcnow,
    write_parquet_lake,
)


def compacted_key(year: int, month: int, run_id: str) -> str:
    return f"{month_prefix(year, month)}compacted-{run_id}.parquet"


def _month_keys(lake: LakeStore, year: int, month: int) -> tuple[list[str], list[str]]:
    prefix = month_prefix(year, month)
    keys = lake.list_keys(prefix)
    parts = [k for k in keys if "/part-" in k and k.endswith(".parquet")]
    compacted = sorted(k for k in keys if "/compacted-" in k and k.endswith(".parquet"))
    return parts, compacted


def compact_month(lake: LakeStore, pg: PgClient, year: int, month: int) -> dict:
    parts, compacted = _month_keys(lake, year, month)
    if not parts:
        return {"year": year, "month": month, "rows": 0, "skipped": True}
    read_keys = parts + compacted[-1:]

    frames = lake.get_dfs_parquet_parallel(read_keys)
    df = normalize_obs(pd.concat(frames, ignore_index=True))
    df = df.sort_values(["source_symbol", "source", "ts", "fetched_at"])
    df = df.drop_duplicates(subset=["source", "source_symbol", "ts"], keep="last")

    run_id = uuid4().hex[:12]
    out_key = compacted_key(year, month, run_id)
    write_parquet_lake(
        lake,
        out_key,
        df[OBS_COLUMNS],
        sort_by=["source_symbol", "source", "ts", "fetched_at"],
    )

    delete_keys = parts + compacted
    if delete_keys:
        lake.delete_keys(delete_keys)

    cov_rows = []
    sym_df = df[["source", "source_symbol"]].drop_duplicates()
    for r in sym_df.itertuples(index=False):
        cov_rows.append((r.source, str(r.source_symbol).upper(), year, month))
    pg.executemany(
        """
        INSERT INTO symbol_month_coverage (source, source_symbol, year, month)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source, source_symbol, year, month) DO NOTHING
        """,
        cov_rows,
    )
    pg.execute(
        """
        INSERT INTO l1_month_manifest (year, month, compacted_key, row_count, symbol_count, compacted_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (year, month) DO UPDATE SET
            compacted_key = EXCLUDED.compacted_key,
            row_count = EXCLUDED.row_count,
            symbol_count = EXCLUDED.symbol_count,
            compacted_at = EXCLUDED.compacted_at
        """,
        (year, month, out_key, len(df), len(sym_df), utcnow()),
    )
    return {"year": year, "month": month, "rows": len(df), "key": out_key, "deleted": len(delete_keys)}


@ray.remote
def task_compact_month_batch(cfg_d: dict, months: list[tuple[int, int]]) -> list[dict]:
    cfg = MarketsConfig.from_dict(cfg_d)
    lake = LakeStore(cfg)
    pg = PgClient(cfg.postgres_url)
    return [compact_month(lake, pg, y, m) for y, m in months]


def compact_months(cfg: MarketsConfig, months: list[tuple[int, int]]) -> list[dict]:
    months = sorted(set(months))
    if not months:
        return []
    cfg_d = cfg.to_dict()
    shape = plan_task_resources(batch_size=4)
    batches = run_batches(
        "compact",
        months,
        lambda batch: task_compact_month_batch.remote(cfg_d, batch),
        shape,
    )
    return [row for batch in batches for row in batch]


def months_from_df(df: pd.DataFrame) -> list[tuple[int, int]]:
    if df.empty:
        return []
    ts = pd.to_datetime(df["ts"])
    return sorted({(int(y), int(m)) for y, m in zip(ts.dt.year, ts.dt.month)})


def months_from_details(details: list[dict]) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for d in details:
        if int(d.get("rows") or 0) <= 0 or not d.get("first") or not d.get("last"):
            continue
        y, m = date.fromisoformat(d["first"]).year, date.fromisoformat(d["first"]).month
        end_y, end_m = date.fromisoformat(d["last"]).year, date.fromisoformat(d["last"]).month
        while (y, m) <= (end_y, end_m):
            out.add((y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1
    return sorted(out)


def discover_l1_months(lake: LakeStore) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for yp in lake.list_common_prefixes("layer_1/"):
        for mp in lake.list_common_prefixes(yp):
            tail = mp.rstrip("/").split("/")[-1]
            if tail.startswith("month="):
                yp_tail = yp.rstrip("/").split("/")[-1]
                if yp_tail.startswith("year="):
                    out.append((int(yp_tail.split("=")[1]), int(tail.split("=")[1])))
    return sorted(out)


def compact_all_l1(cfg: MarketsConfig, lake: LakeStore) -> list[dict]:
    return compact_months(cfg, discover_l1_months(lake))
