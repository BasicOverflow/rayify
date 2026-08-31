# Ephemeral artifacts: return bytes from workers, not MinIO round-trips

## Slop tells

- Worker produces PNG/JSON → upload to durable lake → driver downloads same bytes
- MinIO used for throwaway inspect output that never serves another client
- “Keep task return values small” taken to mean “always spill to S3”
- Windows driver pays egress twice (upload + download) for a one-off plot run

---

## AI-slop: render on worker, persist on MinIO, re-fetch on driver

```python
@ray.remote
def task_render_plot(cfg_d, series_id, start, end):
    df = query_stitched_bars(...)
    png = render_png(df)
    key = f"ops/plots/{series_id}.png"
    lake.put_bytes(key, png, "image/png")              # durable store for ephemeral art
    return {"series_id": series_id, "out_key": key}    # tiny ref

# driver
for ref in refs:
    meta = ray.get(ref)
    lake.download_file(meta["out_key"], local_path)      # same bytes back off MinIO
```

Every plot: **worker PUT + driver GET** — MinIO busy, driver slow, object store cluttered.

---

## Unslopified: PNG in task result (inspect-scale payloads only)

```python
@ray.remote
def task_render_plot_batch(cfg_d, jobs, via_api=True) -> list[dict]:
    out = []
    for series_id, start, end, fname in jobs:
        bars = _fetch_bars(cfg, series_id, start, end, via_api=via_api)
        out.append({
            "series_id": series_id,
            "fname": fname,
            "png": _render_png(bars, series_id) if not bars.empty else None,
            "rows": len(bars),
        })
    return out

# driver — write locally once
for r in results:
    if r.get("png"):
        path.write_bytes(r["png"])
```

**When to spill to MinIO instead:** artifacts other services or later runs must read (L1/L2/L3 parquet, dataset job outputs, shared cache keys). **When to return bytes:** one-off inspect plots, small reports, driver-owned output on a single machine.

For large payloads (multi-GB frames), use `ray.put` / object refs or lake keys — not this pattern.

---

## Rule

**Durable store for durable artifacts.** MinIO is the lake, not a shuttle for bytes that only cross worker → driver once. Return modest results in the task output; reserve S3 for data that must outlive the job.
