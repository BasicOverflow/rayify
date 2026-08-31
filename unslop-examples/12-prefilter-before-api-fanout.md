# Pre-filter before API fan-out: metadata beats bombardment

## Slop tells

- Historical backfill API call for every stale symbol, including names dead for years
- Using yfinance to *discover* delists when a free symbol directory already exists
- 10k+ download jobs when half the universe is jakewright-only foreign tickers
- Parallel MP/yfinance workers → 429 before real work starts

---

## AI-slop: yfinance as discovery + backfill in one pass

```python
def resolve_eod_targets(pg):
    # every ACTIVE equity/etf, including jakewright-only ADRs
    rows = pg.fetchall("SELECT ... FROM series_meta WHERE status = 'ACTIVE'")
    return rows   # ~13k series

def run_eod(targets):
    for target in targets:
        if target.last < mp_cutoff:
            # 6-year historical window for everyone stale
            jobs.append({"start": date(2020, 4, 2), "end": mp_cutoff, "symbol": target.symbol})

    # 442 parallel yf.download batches, no NASDAQ gate
    run_batches("eod_yf", jobs, lambda b: task_ingest_eod_chunk.remote(b), ...)
```

Logs fill with `possibly delisted; no timezone found` — yfinance probing symbols that were never US-listed.

---

## Unslopified: shrink the target set before any HTTP

**1. Universe policy** — jakewright-only foreign names → `UNSUPPORTED` (not in live EOD path):

```python
# sync_live_universe: jacksoncrow US equities/ETFs only for L2
```

**2. NASDAQ directory** — free text files, pull once per run:

```python
directory = fetch_nasdaq_directory()
targets, nasdaq_skip = apply_nasdaq_yf_gate(pg, targets, directory.symbols)
print(f"eod: nasdaq_listed={len(directory.symbols)} yf_skip_not_listed={nasdaq_skip} targets={len(targets)}")
# e.g. 10909 → 5167 targets (~47% fewer yfinance jobs)
```

**3. Rate gate + batch downloads** — parallelism without hammering Yahoo:

```python
@ray.remote
class YfRateGate:
    def acquire(self):
        # min_interval between yf.download calls cluster-wide

jobs = _batch_yf_jobs(targets, chunk_size=cfg.eod_chunk_size, ...)  # pack tickers per call
```

**4. MarketParquet** — driver caches to MinIO first; workers read keys (not parallel blind fetch → 429):

```python
def fetch_mp_daily(day, lake):
    combined_key = mp_combined_key(day)
    if lake.exists(combined_key):
        return lake.get_df_parquet(combined_key)
    # download once, put_bytes cache_key, then serve from lake
```

---

## Rule

**Filter with cheap local metadata before fan-out.** Symbol directories, registry flags, and universe rules eliminate work; rate gates and batch APIs protect what remains. Never use a backfill API as a discovery loop.
