# Lexis Markets — current state

Rayified implementation lives in `output/`. Notebooks in `input/` are playground/exploration only.

**Scope:** US equities/ETFs + macro via free sources (Kaggle jakewright/jacksoncrow, MarketParquet, yfinance, FRED). EOD only — no tick feed, no paid vendors.

---

## Architecture

```text
SOURCES
  jakewright | jacksoncrow | marketparquet | yfinance | fred
                    │
                    ▼
  L1 — raw observations (MinIO Parquet, year/month partitions)
                    │
                    ▼
  L2 — registry (PostgreSQL): series_meta, aliases, stitch segments
                    │
                    ▼
  L3 — stitched bars (MinIO monthly cache + on-demand dataset snapshots)
```

| Layer | Store | Holds |
| --- | --- | --- |
| L1 | MinIO | Immutable multi-source OHLCV/macro rows |
| L2 | Postgres | Identity, stitch recipe, quality rollups, job rows |
| L3 | MinIO | `layer_3/year=…/month=…/stitched.parquet` cache; `layer_3/datasets/{job_id}/` snapshots |

Postgres is the map; MinIO is the lake. L2 is not exported to S3.

---

## Sources (implemented)

| Source | Role |
| --- | --- |
| `jakewright` | Kaggle seed — main equity breadth (~1962–2024) |
| `jacksoncrow` | Kaggle seed — fast history through ~Apr 2020 |
| `marketparquet` | Live EOD primary (7-day free window) |
| `yfinance` | EOD gap-fill / fallback |
| `fred` | Macro/rates (ALFRED as-of ingest; `DEFAULT_FRED_SERIES` + H.15 release bulk) |

**Stitch priority:** `jakewright` → `jacksoncrow` → `marketparquet` → `yfinance` → `fred` (`RULE_VERSION = exchange_priority_v1` in code).

**NASDAQ symbol directory** — used for entity discovery and US-listing filters (`nasdaq_symbols.py`, `entity_detect.py`). Not an OHLCV source.

**Not used:** Stooq, exchange-file OHLCV ingest, paid APIs.

---

## S3 layout

```text
s3://{MINIO_FAST_BUCKET}/
  layer_1/year=YYYY/month=MM/
    part-{run_id}-{shard}.parquet    # ingest micro-files
    compacted-{run_id}.parquet       # after compaction; part-* deleted
  layer_3/year=YYYY/month=MM/stitched.parquet
  layer_3/datasets/{job_id}/
    manifest.json
    part-000.parquet
  ops/markers/          # ingest + EOD markers
  ops/cache/            # Kaggle zips, MarketParquet daily pulls
```

L1 partitions: `year`, `month` only. Identity in-file: `(source, source_symbol, ts)`.

---

## L1 observation schema

Columns: `source`, `source_symbol`, `series_type`, `ts`, OHLCV, `adj_close`, `dividend`, `split`, `currency`, `fetched_at`, `realtime_start`, `realtime_end`, `extras`.

- Kaggle/equity rows: `realtime_*` null.
- FRED rows: ALFRED as-of window in `realtime_start` / `realtime_end`; value in `close`.
- L1 is append-only; overlaps stay as separate rows.

---

## PostgreSQL (L2)

**Used tables:** `calendars`, `series_meta`, `symbol_aliases`, `stitch_segments`, `stitch_decisions`, `dataset_jobs`, `l1_month_manifest`, `l3_month_manifest`, `symbol_month_coverage`.

**Sparse / minimal:**
- `series_links` — schema exists; only written by `sync_series_status.py` on delist/unknown probes (not full rename/merge lifecycle from notebook 3).

**Not in schema:** `ingestion_runs`, `ingestion_run_symbols`, `quality_flags`, `layer2_exports`, `universe_version` / `rule_version` columns (dropped; rule is a code constant).

**Quality:** integer rollups on `series_meta` (`gap_count`, `disagreement_count`, `suspicious_count`, `quality_score`). Deep inspection via `inspect_l1.py` → `output/l1_inspect/`.

**Calendars:** `nyse`, `fred_native` seeded. Gap = missing session day per series calendar.

---

## L3 stitch cache

Monthly parquet at `layer_3/year=…/month=…/stitched.parquet`, tracked in `l3_month_manifest`.

Rebuild skips a month when `l1_fp` (L1 `compacted_at`) and `stitch_rule` are unchanged. When a month does rebuild, the whole month is restitched (no day-level incremental patch).

`revision_mode=as_of` (default) reads the cache; `revision_mode=latest` stitches live from L1 in-process.

---

## API (Ray Serve)

Deployed by `seed.py` → `deploy_markets_api`. Route prefix: `/markets` (full base from `RAY_SERVE_URL`).

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | |
| GET | `/v1/cache/status` | L1/L3 month counts, stale months |
| POST | `/v1/cache/warm` | Build missing stitched months for date range |
| POST | `/v1/datasets` | Enqueue Ray task (`?sync=true` builds in-request) |
| GET | `/v1/datasets/{job_id}` | Job status + manifest |
| GET | `/v1/series/{series_id}` | Sync single-series JSON |

Dataset jobs: `spec_hash` + L1 fingerprint reuse identical completed builds. Default `POST /v1/datasets` returns `queued` and runs `task_build_dataset_snapshot` on the cluster.

**Not implemented:** `POST/GET /v1/layer2/exports`.

Stitch engine: pandas/pyarrow over MinIO — not DuckDB, not a hosted query service.

---

## Entry scripts (`output/`)

| Script | Purpose |
| --- | --- |
| `seed.py` | Full wipe + jakewright/jacksoncrow ingest + EOD backfill + FRED + compact L1 + L3 cache + deploy API |
| `eod_job.py` | Daily EOD: gap scan, entity detect, MP+yfinance ingest, compact, L3 rebuild, registry patch |
| `entity_detect.py` | Standalone MP + NASDAQ directory scan |
| `sync_series_status.py` | Delist/universe status from jacksoncrow meta + yfinance probes |
| `inspect_l1.py` | L1 quality scan + sample plots |
| `run_full_pipeline.py` | `seed.py` then `inspect_l1.py` |
| `build_stitch_cache.py` | Rebuild L3 months |
| `eod_backfill.py` | Historical EOD gap fill |
| `wipe_all.py` / `drop_pg_tables.py` | Teardown helpers |

Run from `output/` with repo-root `.env` loaded (`RAY_ADDRESS`, `RAY_NAMESPACE`, MinIO Fast, Postgres Fast, `FRED_API_KEY`, `KAGGLE_API_TOKEN`).

**Scheduling:** no Prefect — run `eod_job.py` via host cron or Task Scheduler after US close.

**Packaging:** no `Dockerfile` / `docker-compose.yml` for this project (run scripts directly against the Ray cluster).

---

## EOD flow

```text
1. Resolve stale ACTIVE equity/ETF series (jacksoncrow-listed or MP-discovered, US listing)
2. MarketParquet combined daily for recent window; yfinance for gaps / older tail
3. Write L1 part files → compact touched months
4. Rebuild changed L3 months (skip if L1 fingerprint unchanged)
5. Patch series_meta last_seen / eod_filled_through
```

Entity detect (in `eod_job` / `entity_detect.py`) registers new listings seen in MP free window + NASDAQ directory.

---

## Playground notebooks

| Notebook | Explores |
| --- | --- |
| `1_explore_ticker_universe.ipynb` | Ticker universe, Crow∥Wright gaps, yfinance fill experiments |
| `2_unified_schema.ipynb` | L1 obs + L2 meta/aliases POC |
| `3_stitch_l3_quality_fred.ipynb` | Stitch, calendars, quality flags, identity demos (POC parquet under `poc/`) |

Notebook 3 concepts like `quality_flags` and rich `series_links` were not carried into production schema usage.

---

## Non-goals

- Stooq ingest
- Exchange-file OHLCV
- `ingestion_runs` / per-fetch Postgres logs
- `quality_flags` table
- L2 S3 export API
- Standing canonical L3 universe refresh
- L1 partition by ticker
- Docker job container for this workload
- Prefect / built-in scheduler
