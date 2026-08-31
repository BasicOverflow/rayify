# MinIO write thrash: per-symbol PUTs → year/month fat parts

## Slop tells

- One S3 PUT per symbol per month
- `list_keys` / `present_months` before every write to dedupe
- Hundreds of thousands of tiny objects; CPUs idle while MinIO handles metadata
- “Distributed ingest” that spends all wall-clock on object-store chatter

---

## AI-slop: per-symbol month files

```python
def write_symbol_months(lake, df: pd.DataFrame, symbol: str, run_id: str):
    sym_prefix = f"layer_1/obs/symbol={symbol}/"
    for ts, chunk in df.groupby(df["ts"].dt.to_period("M")):
        month_key = f"{sym_prefix}year={ts.year}/month={ts.month}/data.parquet"
        if month_key in lake.present_months(sym_prefix):   # LIST per symbol
            continue
        lake.put_df_parquet(month_key, chunk)            # tiny PUT
```

Seed ingest for ~9k symbols × decades of history → **tens of thousands of LIST + PUT pairs**.

---

## Unslopified: groupby year/month, one fat part per group

```python
class ObsWriter:
    def write_parts(self, df: pd.DataFrame, *, shard: str) -> dict:
        df = normalize_obs(df)
        work = df.copy()
        work["_y"] = pd.to_datetime(work["ts"]).dt.year
        work["_m"] = pd.to_datetime(work["ts"]).dt.month
        for (y, m), part in work.groupby(["_y", "_m"], sort=False):
            key = part_key(int(y), int(m), self.run_id, shard)
            self.lake.put_df_parquet(key, part.drop(columns=["_y", "_m"]))
```

Layout: `layer_1/obs/year=YYYY/month=MM/part-{run_id}-{shard}.parquet` — many symbols per file, zstd, row groups sized for scan.

Each ingest task writes **a handful of parts** per staging chunk, not thousands of micro-objects.

---

## Rule

**Batch remote writes.** Object stores are fast at large sequential PUTs and slow at huge key counts. Group output the same way you group Ray tasks — by partition (year/month), not by row or symbol.
