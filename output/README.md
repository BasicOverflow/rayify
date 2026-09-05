# Lexis Markets v4

Ray cluster datalake + Serve API. Namespace: `lexis-markets` (from root `.env`).

## Layout

- `lexis_markets/` — package (`lake/`, `registry/`, `kaggle/`, `eod/`, `fred/`, `jobs/`, `serve/`, `supervisor/`, `cli/`, `ray/` gates)
- `.supervisor/` — SQLite state (bind mount)
- L1: MinIO `layer_1/`
- L3 cache: `layer_3/cache/{series_id}.parquet` (e.g. `equity__AAPL.parquet`)
- L3 outputs: `layer_3/outputs/{job_id}/`

## Run (production supervisor)

```bash
cd output
docker compose up --build supervisor
```

The supervisor runs the full pipeline on startup: **seed** (once), then daily **EOD** and **quality** cron. Quality scan writes ``gap_count``, ``disagreement_count``, ``suspicious_count``, and ``quality_score``. Flags include ``linear_ramp`` / ``sparse_bridge`` (ITMR-style plot diagonals), plus bar integrity checks (``flat_close``, ``ohlc_violation``, ``non_positive_close``, ``duplicate_ts``, ``extreme_return``). Serve responses include ``flags``; filter with ``max_suspicious_count``. Plot samples mark flagged stretches and log them in ``summary.txt``. Set ``MARKETS_RESET=1`` to truncate L2 and re-seed once on the next successful seed run (does not loop while the env var stays set).

If the Ray cluster is offline, the supervisor still talks to NAS (MinIO/Postgres) and keeps cron scheduling into SQLite. Catch-up EOD/FRED jobs stay pending and their targets refresh as days pass; ``submit_loop`` drains them when Ray reconnects.

On boot the supervisor reads lake markers + Postgres and logs what is loaded (jakewright/jacksoncrow L1, registry counts, seed stage). Seed completion is the ``ops/markers/seed_complete.json`` marker only. If L1 ingest markers exist but seed is incomplete, the next seed job skips finished ingest stages and continues (registry, align, EOD backfill, compact). Interrupted SQLite jobs are deduped and reconciled against lake state.

Requires root `.env` with `RAY_ADDRESS`, `RAY_NAMESPACE`, `RAY_SERVE_URL`, MinIO Fast, Postgres Fast, `FRED_API_KEY`, `KAGGLE_API_TOKEN`.

## Tests (local pytest)

See [`tests/README.md`](tests/README.md). **Default is weeksim** — run on the host against the cluster (not via Docker).

```powershell
cd output
pip install -r requirements-dev.txt
$env:PYTHONPATH = (Get-Location).Path
pytest                    # weeksim (default)
pytest tests/unit -m unit # opt-in offline units
```

Copy `.env.test.example` to `.env.test` (gitignored) with test namespace and `lexis_markets_test` Postgres.

## Run (Windows dev — manual CLIs)

```powershell
cd output
$env:PYTHONPATH = (Get-Location).Path
py -3 -m lexis_markets.cli.seed
py -3 -m lexis_markets.cli.eod
py -3 -m lexis_markets.cli.cache_fill --series equity:AAPL --start 2020-01-01 --end 2020-12-31
py -3 -m lexis_markets.cli.plot_warm --out plot_samples --equity 50 --etf 50
py -3 -m lexis_markets.cli.wipe --fred-only   # FRED/macro only; omit flag for full wipe
```

`plot_warm` calls Serve (`RAY_SERVE_URL`) for full stitched history with **32 parallel HTTP workers by default** (`--workers N`; Serve `max_ongoing_requests=10000` is the ceiling): **all macro/FRED** (`revision_mode=as_of`) plus N equity/ETF (`revision_mode=latest`, warms L3). PNGs land under `--out/plots/{macro,equity,etf}/`.
## API (Ray Serve `/markets`)

Swagger: `{RAY_SERVE_URL}/docs`

- `GET /health`
- `GET /v1/universe` — series names / registry rows (status, asset_class, gap/suspicious, partial coverage, limit/offset)
- `GET /v1/series/{series_id}?start=&end=&granularity=daily|weekly|monthly` — history; cache-first when `revision_mode=latest`; default `as_of` for FRED vintages
- `GET /v1/eod/{series_id}` — last EOD bar on/before `eod_date` (default yesterday)
- `GET /v1/series/...&features=sma_20&features=ema_50` — SMA/EMA of `close`
- `POST /v1/datasets` — `mode=range` (needs start/end) or `mode=eod_snapshot` (one row per series at last EOD); always one `dataset.parquet` under `layer_3/outputs/{job_id}/`
- `GET /v1/datasets/{job_id}`

Serve deploys on supervisor startup and seed.

## Env (optional)

| Key | Default | Purpose |
|-----|---------|---------|
| `EOD_PACE_SECONDS` | 2.5 | Floor spacing (seconds) between yfinance calls; gate starts here and backs off on 429 |
| `YF_PACE_MAX_SECONDS` | 30.0 | Ceiling spacing after rate-limit backoff |
| `YF_PACE_BACKOFF_FACTOR` | 2.0 | Multiply current interval on 429 |
| `YF_PACE_RECOVERY_STEP` | 0.1 | Seconds shaved off interval after each successful download |
| `EOD_CHUNK_SIZE` | 150 | Opening symbols per bulk ``yf.download``; halves on 429 down to `EOD_CHUNK_SIZE_MIN` |
| `EOD_CHUNK_SIZE_MIN` | 30 | Floor chunk size after rate-limit shrink |
| `EOD_START_SLOP_DAYS` | 150 | Opening start-date slop for packing jobs; halves on 429 |
| `EOD_START_SLOP_DAYS_MIN` | 30 | Floor start slop after rate-limit shrink |
| `YF_WAVE_JOBS` | 8 | Jobs per adaptive re-batch wave when `EOD_MAX_IN_FLIGHT` is 0 |
| `MARKETS_RESET` | — | Truncate L2 and re-seed once (supervisor cron or manual seed CLI) |
| `MARKETS_DEV_YF_LIMIT` | — | Hash-sample jakewright symbols (dev only; leave unset for full universe) |
| `MARKETS_DEV_FRED_LIMIT` | — | Hash-sample FRED series in seed backfill, macro EOD, and fred_backfill CLI |
| `MARKETS_DEV_FRED_SEED` | 0 | Stable hash seed for FRED dev sampling |
| `MARKETS_EOD_DELAY_HOURS` | 3 | Hours after 16:00 ET before EOD cron |
| `MARKETS_SUPERVISOR_STATE` | `output/.supervisor/state.db` | SQLite path |
| `FRED_PACE_PER_MINUTE` | 120 | Cluster-wide FRED HTTP cap (requests/min via gate actor) |
| `FRED_MAX_IN_FLIGHT` | 28 | Max concurrent FRED ingest Ray tasks (one series each) |
| `FRED_SERIES_PER_TASK` | 1 | Series processed serially inside each FRED Ray task |
| `FRED_VINTAGE_START` | 1970-01-01 | Earliest vintage/observation date for macro backfill |
| `FRED_VINTAGE_MIN_DAYS` | 31 | Adaptive ALFRED bisect / window-sizer floor |
| `FRED_VINTAGE_MAX_WINDOW_DAYS` | 1825 | Opening pack size (~5y); shrinks on gateway bisects, grows after clean fetches |
| `LOG_LEVEL` | INFO | Python logging |

Default FRED macro set: **117 series** in `config.DEFAULT_FRED_SERIES` (rates, credit, inflation, labor, FX, equities/vol, energy, commodities). No H.15 release bulk — list is explicit.

## Verification

- Import: `PYTHONPATH=output py -3 -c "import lexis_markets"`
- Build: `cd output && docker compose build supervisor`
- Cluster smoke: test CLIs above against live `.env` (not run in CI here)
