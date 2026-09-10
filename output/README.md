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

The supervisor runs seed once (no `seed_complete` marker), then daily EOD after the post-close window. Catch-up EOD/FRED fill missing session days as soon as seed is done.

Seed is an initial phase on Ray workers: Kaggle L1 (JW ∥ JC) in parallel with FRED, compact JW/JC, registry, yfinance waves (align JW as YF lands; no Serve warm), compact remaining months, stitch+quality into L3 on the workers. Compacted L1 + cleaned L3 flush to MinIO once. Daily EOD after `seed_complete` writes incremental L1 to MinIO and Serve-tip-warms the missing window (`last_seen - 60d`).

Wipe is CLI-only: `py -3 -m lexis_markets.cli.wipe --confirm`. Empty lake makes the supervisor enqueue seed again.

Serve prefers L3 (`revision_mode=latest`). Stitch applies halt collapse, JW demote, calibration, OHLC bounds. `QUALITY_INLINE_REPAIR=0` is an emergency score-only kill-switch on those L3 writes.

`cache_fill` checkpoints done series under `ops/progress/{job}/{scope}.json`.

Dataset recipes: `range_panel` (alias `range`), `eod_snapshot`, `wide_matrix`
(NaN policy: `keep` / `drop_rows` / `ffill`). Features: `sma_*`, `ema_*`,
`returns`, `log_price`, `gap_mask`, `is_suspicious`.

If the Ray cluster is offline, the supervisor still talks to NAS (MinIO/Postgres) and keeps cron scheduling into SQLite. Catch-up EOD/FRED jobs stay pending and their targets refresh as days pass; `submit_loop` drains them when Ray reconnects.

On boot the supervisor reads lake markers + Postgres and logs what is loaded (jakewright/jacksoncrow L1, registry counts, seed stage). Seed completion is the `ops/markers/seed_complete.json` marker only.

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
py -3 -m lexis_markets.cli.cache_fill --universe active --tip-days 60
# tip only: add --tip-only; cap: --limit 500 --asset-class equity
py -3 -m lexis_markets.cli.qa --out plot_samples --n 50 --asset-class equity,etf --seed 42
py -3 -m lexis_markets.cli.wipe --confirm
py -3 -m lexis_markets.cli.wipe --fred-only --confirm
```

`cli.qa` hash-samples the universe, GETs Serve once per name, writes PNGs under `--out/plots/` plus `report.json` / `report.md` for that same sample. Filters: `--asset-class`, `--n` / `--equity/--etf/--macro`, `--years` / `--start/--end`, `--flagged`, `--series`.

`cache_fill` is HTTP to Serve (`include_rows=false`). L3 persist is a side effect of writing spans.

## API (Ray Serve `/markets`)

Swagger: `{RAY_SERVE_URL}/docs`

- `GET /health`
- `GET /v1/universe` — series names / registry rows (status, asset_class, gap/suspicious, partial coverage, limit/offset)
- `GET /v1/series/{series_id}?start=&end=&granularity=daily|weekly|monthly` — history; default `revision_mode=latest` (L3); pass `as_of` for FRED vintages; `include_rows=false` omits OHLCV (ingest/cache_fill)
- `GET /v1/eod/{series_id}` — last EOD bar on/before `eod_date` (default yesterday)
- `GET /v1/series/...&features=sma_20&features=returns&features=is_suspicious`
- `POST /v1/datasets` — `mode=range_panel` | `eod_snapshot` | `wide_matrix` (`nan_policy`); always one `dataset.parquet` under `layer_3/outputs/{job_id}/`
- `GET /v1/datasets/{job_id}`

Serve deploys on supervisor startup and seed.

## Env (optional)

| Key | Default | Purpose |
|-----|---------|---------|
| `EOD_PACE_SECONDS` | 2.5 | Floor spacing (seconds) between yfinance calls; gate starts here and backs off on 429 |
| `YF_PACE_MAX_SECONDS` | 30.0 | Ceiling spacing after rate-limit backoff |
| `YF_PACE_BACKOFF_FACTOR` | 2.0 | Multiply current interval on 429 |
| `YF_PACE_RECOVERY_STEP` | 0.1 | Seconds shaved off interval after each successful download |
| `EOD_CHUNK_SIZE` | 400 | Opening symbols per bulk `yf.download`; halves on 429 down to `EOD_CHUNK_SIZE_MIN` |
| `EOD_CHUNK_SIZE_MIN` | 30 | Floor chunk size after rate-limit shrink |
| `EOD_START_SLOP_DAYS` | 365 | Opening start-date slop for packing jobs; halves on 429 |
| `EOD_START_SLOP_DAYS_MIN` | 30 | Floor start slop after rate-limit shrink |
| `YF_WAVE_JOBS` | 8 | Jobs per adaptive re-batch wave when `EOD_MAX_IN_FLIGHT` is 0 |
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
| `QUALITY_INLINE_REPAIR` | 1 | L3 write applies trim/OHLC/quarantine/halt rewrite (0 = score-only) |

Default FRED macro set: **117 series** in `config.DEFAULT_FRED_SERIES` (rates, credit, inflation, labor, FX, equities/vol, energy, commodities). No H.15 release bulk — list is explicit.

## Verification

- Import: `PYTHONPATH=output py -3 -c "import lexis_markets"`
- Build: `cd output && docker compose build supervisor`
- Cluster smoke: test CLIs above against live `.env` (not run in CI here)
