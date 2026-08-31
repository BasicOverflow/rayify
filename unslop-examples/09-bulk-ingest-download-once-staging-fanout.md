# Bulk ingest: re-download per task → download once, fan out staging parts

## Slop tells

- Every Ray task pulls the same multi-GB Kaggle zip or parquet
- Parallel workers multiply bandwidth (16 workers × same blob = MinIO meltdown)
- “Distributed” ingest that is really N identical full-file reads
- No staging marker — every rerun re-downloads from Kaggle

---

## AI-slop: full blob inside each batch task

```python
JW_DATA_KEY = "ops/cache/jakewright.parquet"

@ray.remote
def task_ingest_jakewright_batch(cfg_d: dict, symbols: list[str], run_id: str) -> list[dict]:
    lake = LakeStore(MarketsConfig.from_dict(cfg_d))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        lake.download_file(JW_DATA_KEY, path)          # full dataset, every task
        df = pd.read_parquet(path)
        df = df[df["Ticker"].isin(symbols)]
        # map + write L1 ...
    return details

# ~9000 symbols / 50 per batch ≈ 180 tasks → 180 full downloads
shape.max_in_flight = min(16, cluster_slots)           # still 16 concurrent full pulls
```

Jacksoncrow was the same pattern with the zip:

```python
@ray.remote
def task_ingest_jacksoncrow_batch(cfg_d: dict, csv_names: list[str], run_id: str):
    lake.download_file(JC_CACHE_KEY, zpath)            # whole zip again
    with zipfile.ZipFile(zpath) as zf:
        for name in csv_names:
            ...
```

---

## Unslopified: cache once → stage once → many small reads

**1. Kaggle → MinIO once** (`cache_kaggle_zip` skips if key exists):

```python
def cache_kaggle_zip(lake, dataset, token, cache_key) -> str:
    if lake.exists(cache_key):
        return cache_key
    # stream download from Kaggle API → lake.put_file(cache_key, ...)
```

**2. One prepare task splits into staging parts** (marker skips re-staging):

```python
@ray.remote
def prepare_jakewright_staging(cfg_d: dict) -> int:
    lake = LakeStore(MarketsConfig.from_dict(cfg_d))
    if lake.exists(JW_STAGING_MARKER):
        return int(get_json(lake, JW_STAGING_MARKER)["parts"])
    cache_kaggle_zip(lake, JAKEWRIGHT, cfg.kaggle_api_token, JW_CACHE_KEY)
    lake.download_file(JW_CACHE_KEY, zpath)            # once per staging build
    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_ROWS)):
        lake.put_file(f"{JW_STAGING}part-{i:05d}.parquet", part)
    put_json(lake, JW_STAGING_MARKER, {"parts": parts})
    return parts
```

**3. Fan out ingest across staging keys** (each worker reads one part, not the whole corpus):

```python
n_parts = int(ray.get(prepare_jakewright_staging.remote(cfg_d)))
keys = sorted(lake.list_keys(JW_STAGING))
results = run_batches(
    "jakewright",
    keys,
    lambda batch: ingest_staging_parts.remote(cfg_d, batch, run_id, "jakewright"),
    plan_task_resources(batch_size=1),
)
```

**4. Tear down staging after L1 is written** so worker/object-store scratch does not grow forever (`clear_kaggle_source_build`).

---

## Rule

**Materialize bulk inputs once.** Fan out over **partition keys** (staging parts, shards, cache paths), not over repeated full downloads. Ray parallelism belongs on the work units, not on re-fetching the same blob.
