# Ray vs MinIO concurrency: one cap does not fit both

## Slop tells

- Single `max_in_flight=16` on everything because “MinIO was saturating”
- Or the opposite: uncap MinIO thread pools because “use the whole cluster”
- `num_cpus` / `memory` on every remote when the scheduler already knows the cluster
- Treating S3 latency and CPU work as the same bottleneck

---

## AI-slop: one knob for compute and I/O

```python
def plan_task_resources(...):
    slots = int(cluster_cpus * target_frac / task_cpus)
    max_in_flight = min(slots, cfg.eod_max_in_flight, 16)   # always 16 for ingest too

@ray.remote(num_cpus=0.25, memory=1_000_000_000)
def task_ingest_eod_chunk(...):
    ...

# L1 scan: unlimited ThreadPoolExecutor — 200 workers hammer MinIO
with ThreadPoolExecutor(max_workers=200) as ex:
    ex.map(lake.get_df_parquet, all_keys)
```

Result: Ray dashboard shows **5% CPU** and **Sent/Received** pegged — workers wait on MinIO, or MinIO returns 429/slowdown while CPUs sit idle.

---

## Unslopified: unlimited Ray fan-out, throttled MinIO reads

**Ray side** — submit all batches; let the cluster scheduler place work (`max_in_flight=0` = no artificial cap):

```python
@dataclass
class TaskShape:
    batch_size: int
    max_in_flight: int = 0

def plan_task_resources(*, batch_size: int = 50, **_) -> TaskShape:
    return TaskShape(batch_size=batch_size)

def run_batches(...):
    cap = shape.max_in_flight if shape.max_in_flight > 0 else len(batches)
    # ray.wait loop — no num_cpus/memory on remotes unless user approved
```

**MinIO side** — separate pool for boto GET/PUT parallelism:

```python
# config.py
minio_max_workers: int = 8   # MINIO_MAX_WORKERS env

# inspect_l1.py
p.add_argument("--scan-workers", default=0, help="0 = MINIO_MAX_WORKERS")
p.add_argument("--file-workers", default=0, help="0 = MINIO_MAX_WORKERS")

lake.get_dfs_parquet_parallel(keys, max_workers=cfg.minio_max_workers)
```

EOD yfinance stays **serialized** via `YfRateGate` — external API limits are a third knob, not Ray’s problem.

---

## Rule

**Compute parallelism ≠ I/O parallelism.** Fan out Ray tasks for CPU/transform work; cap client-side S3/MinIO threads separately; serialize or gate external APIs. Three bottlenecks, three controls.
