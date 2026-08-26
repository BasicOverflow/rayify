# AGENTS.md - Ray Conversion Guide

Convert Python scripts in `input/` into Ray-optimized, cluster-ready code in `output/`. Connect to an existing Ray cluster (never start local Ray unless explicitly required).

**Docs:** open listed [ray-resources/](ray-resources/) files for the workload ([index](ray-resources/README.md)), then **always cross-check that Ray practice against current online docs** (using web search) before implementing. Repo docs can be stale — if online and local disagree, prefer online and note the mismatch.

**Style:** open [unslop-examples/](unslop-examples/) ([README](unslop-examples/README.md) + matching pattern files) before non-trivial structure.

**Prior rayify examples:** list all branches (`git branch -a`); if other branches hold approved real rayified projects, switch/read them when relevant (how connect, actors, tests, layout). Return to the working branch when done. Do not invent from scratch if a branch already solved a similar case.

**Long-running / Prefect:** open [orchestration-patterns/prefect-long-running.md](orchestration-patterns/prefect-long-running.md), then re-verify Prefect online.

## Hard rules

- Cluster-first: `ray.init` via env vars, not implicit local Ray
- Required: `RAY_ADDRESS`, `RAY_NAMESPACE` from repo-root [`.env`](.env) (key names in [`.env.example`](.env.example)). Optional: `RAY_RUNTIME_ENV` (JSON)
- **Root `.env` required** — if missing, stop; tell the user to add a filled copy. Never invent hosts/creds; never hardcode secrets
- **One namespace per project** — unique `RAY_NAMESPACE`; document it for that project
- Read from `input/`; write to `output/`; confirm expected output before converting
- **Backends from `.env`:** treat listed services (Ray, MinIO/S3, Postgres, Mongo, Neo4j, SearXNG, Gitea, …) as a shared toolbox — use any of them for **any project-appropriate role** (not only “S3=blobs / SQL=tables”); keys in [`.env.example`](.env.example) are endpoints/creds, not fixed recipes. Prefer fitting what’s already there; if the job needs persistence or a new dedicated DB/bucket/instance, ask once (reuse vs dedicated), recommend, wait for yes, then update root `.env`. Wire only via env; propagate needed vars to workers via `runtime_env`. **If something needs manual setup the agent can’t do** (new Postgres/Mongo/Neo4j DB or instance, MinIO bucket, Gitea package index/org, registry image, etc.), **stop and point it out to the dev as a requirement** before coding that path — don’t pretend it’s already provisioned
- Resource hints on remotes when non-trivial (after user approves)
- **Run mode — confirm with dev** (do not auto-classify): propose short one-shot vs long-running / fault-tolerant from the script shape, then **wait for an explicit choice**. See [Run mode](#run-mode-confirm-with-dev)

## Run mode (confirm with dev)

### Short / one-shot (after dev confirms)

- Convert to Ray as usual; run from the driver/host with root `.env`.
- Output: `<script>.py`, `smoke_test.py`, `requirements.txt` as needed.
- No Prefect; no job container/compose unless the dev later asks to package it.

### Long-running / fault-tolerant (after dev confirms)

- Always: Prefect-wrapped entry, Alpine-first `Dockerfile`, `docker-compose.yml` (Prefect server UI + job). Crib: [orchestration-patterns/prefect-long-running.md](orchestration-patterns/prefect-long-running.md).
- Ask once which **remote** store holds workload progress/checkpoints (MinIO/S3, Postgres, Mongo, Neo4j — reuse vs dedicated from the toolbox). Never host bind-mount recovery for that state.
- If training/ML or heavy experiment tracking: **ask** whether to wire Weights & Biases (or similar). Only if yes, create **project-local** env keys under `output/` from scratch — never blank Prefect/W&B slots in root [`.env.example`](.env.example).
- Shared infra: pass root `.env` into the job service. Prefect client→server URL is compose-internal glue only (see crib), not a root template key.
- Durable orchestration = Prefect (flows, tasks, retries, UI logs). Workload artifacts = chosen remote backend. Ray Train/Tune: their checkpoints into that store when those primitives apply. Do **not** invent a repo-local checkpoint base-class package.
- Actors are disposable memory; reload progress from remote on restart.
- No Prometheus/Grafana for this path unless the dev later asks (not default).
- Prefer `python:3.x-alpine` for the job image; switch only if a dep forces a different slim base.

## Ray Hive (LLM serving)

Ray Hive is a distributed LLM serving SDK for a Ray cluster on k3s. It deploys vLLM models across heterogeneous GPUs with VRAM-aware scheduling, then exposes a simple Python API for inference.

- **When:** project needs LLM inference (especially high-throughput serving), embeddings, multimodal, or any vLLM-supported Hugging Face model
- **Permission:** if that need is detected, **ask the user before using Ray Hive**; do not deploy models until approved
- **How:** web-search / open [ray-hive README](https://github.com/BasicOverflow/ray-hive/blob/main/README.md) for current API and features — do not rely on memory alone
- **Examples (required):** before implementing Hive usage, open and crib from the matching scripts under [ray-hive/examples](https://github.com/BasicOverflow/ray-hive/tree/main/examples) (inference, multimodal, embeddings, allocation, tensor parallel, shared GPU, idle sleep, etc.) — pick the closest example(s) for the workload; do not invent API usage from scratch
- **Install** (LAN Gitea PyPI — fill from root `.env` `GITEA_USER` / `GITEA_TOKEN` / `GITEA_PYPI_SIMPLE`):

```bash
pip install ray-hive \
  --index-url http://<GITEA_USER>:<GITEA_TOKEN>@<gitea-host>:<port>/api/packages/<owner>/pypi/simple/ \
  --extra-index-url https://pypi.org/simple
```

Connect with `RAY_ADDRESS` from `.env`.

## Optional: cudf.pandas (GPU pandas)

[cudf.pandas](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/) is NVIDIA’s drop-in GPU accelerator for pandas (`python -m cudf.pandas` / `cudf.pandas.install()` before `import pandas`). Useful when the input does **heavy** DataFrame work (groupby, joins, large CSV/Parquet) that fits on a GPU worker.

- **Ask first** — do not wire RAPIDS/cuDF unless the user agrees; needs CUDA GPUs + cuDF on those workers
- **Not the default** — prefer plain pandas or [Ray Data](ray-resources/data/examples.md) for distributed/out-of-core; cudf.pandas is single-process acceleration (fine inside a GPU actor/task), not a cluster dataframe layer
- Skip for tiny frames or non-pandas pipelines

## Code style (unslop + clear OOP)

Match the cleaned half of [unslop-examples/](unslop-examples/). Index: [unslop-examples/README.md](unslop-examples/README.md).

- **Simple, readable OOP when it helps** — name types for real concepts (config, planner, model replica), keep methods short, prefer classes over spaghetti free functions when state or related ops cluster together. See [08-attention-specs-vram-planning.md](unslop-examples/08-attention-specs-vram-planning.md) for split: pure planning objects vs Ray/Serve actors.
- **Not OOP theater** — no empty plugin stubs, unused bookkeeping, or metrics nobody reads ([06](unslop-examples/06-scaffolds-magic-env.md), [07](unslop-examples/07-router-overbuild.md)).
- **Efficiency still matters** — don’t add layers that thrash the object store, serialize huge payloads, or spawn extra remotes “for structure.” Prefer a clear class *or* a plain function when the function is shorter and just as clear; never trade a big runtime cost for ceremony.
- **No defensive soup** — let bad state raise; no silent `except`, stacked `hasattr`, soft-fail timeouts, or invent-a-fallback deploys ([01](unslop-examples/01-wait-helpers-and-fallbacks.md), [04](unslop-examples/04-inference-defensive-soup.md)).
- **No print theater / dead paths** — skip section banners, emoji status systems, legacy dual branches, double-checks ([02](unslop-examples/02-legacy-paths-and-double-checks.md), [03](unslop-examples/03-examples-print-theater.md)).
- **Don’t paper over bad data** — no regex JSON recovery if you control the schema; fail or use real structured output ([05](unslop-examples/05-json-parse-slop.md)).
- **No compliance / process comments** — do not litter output code with comments that narrate following AGENTS.md, Ray docs, unslop rules, “as required,” “per instructions,” checklist ticks, etc. Comments only when they clarify non-obvious *code* behavior.

## Connect

```python
import os
import ray

# RAY_NAMESPACE must be unique per project (do not reuse another project's namespace)
ray.init(
    address=os.environ["RAY_ADDRESS"],
    namespace=os.environ["RAY_NAMESPACE"],
)
```

## Workflow

1. **Analyze** – Confirm expected output; resources; project `RAY_NAMESPACE`. Stop if no root `.env`. **Propose short vs long-running, wait for explicit confirmation.** Then: persistence reuse/dedicated, Ray Hive, and (if long + training) W&B / extra UIs as needed.
2. **Examples** – Prior rayify branches if useful; if long-running, open [orchestration-patterns/prefect-long-running.md](orchestration-patterns/prefect-long-running.md).
3. **Choose primitive** – Table below; local doc then online Ray docs.
4. **Implement** – Ray + unslop; backends only via confirmed env keys. If long: wrap with Prefect (flows/tasks, retries, progress logged in Prefect UI, workload state on remote store).
5. **Deps** – `runtime_env` / `RAY_RUNTIME_ENV` as needed; long mode also `requirements.txt` with `prefect` (and backend clients).
6. **Verify** – Tiny real-cluster smoke; print a real result. Long: same path through Prefect + **remote** store proof.
7. **Package** – Long only: Alpine-first `Dockerfile` + `docker-compose.yml` (Prefect server UI + job). Short: Docker only if the dev asks.

## Smoke tests (not pytest scaffolding)

Do **not** set up elaborate pytest suites, fixtures, or mocking. Ship plain scripts the user can run, e.g.:

```bash
python output/smoke_test.py
# or: python output/tests/run_smoke.py
```

Requirements:
- **Same real Ray cluster and infra** as the production script (`RAY_ADDRESS`, project `RAY_NAMESPACE`, same DBs/NFS/endpoints, same connect path). No fake/local-only Ray, no mocked backends.
- **Tiny runtime only** — fraction of full load (few items, 1 epoch/step, short timeout, subsample of data). Same code paths, less work.
- **Print proof it worked** — real result from the cluster/infra (sample output, count, actor reply, write confirmation, etc.), not a silent exit or empty “ok”
- Fail loudly on errors (raise / non-zero exit); no soft-fail soup
- **Long mode:** exercise the Prefect entry (or the same tasks) plus a small remote-store read/write on real MinIO/DB, and print proof of both

## Which docs to open

Always: local path first, then verify against [docs.ray.io](https://docs.ray.io/) for the same feature.

| Workload | Open first (local) |
|---|---|
| Stateless parallel (tasks) | [tasks.md](ray-resources/ray-core/tasks.md) |
| Stateful / shared model (actors) | [actors.md](ray-resources/ray-core/actors.md) |
| Shared large data / object store | [objects.md](ray-resources/ray-core/objects.md) |
| Scheduling / placement / resources | [scheduling.md](ray-resources/ray-core/scheduling.md) |
| Fault tolerance | [fault-tolerance.md](ray-resources/ray-core/fault-tolerance.md) |
| Deps / runtime_env | [handling-dependencies.md](ray-resources/ray-core/handling-dependencies.md), [runtime-env-apis.md](ray-resources/ray-core/api/runtime-env-apis.md) |
| Configure / core APIs | [configure.md](ray-resources/ray-core/configure.md), [core-apis.md](ray-resources/ray-core/api/core-apis.md) |
| Getting started | [getting-started.md](ray-resources/ray-core/getting-started.md) |
| Job submission | [job-submission.md](ray-resources/cluster/running-applications/job-submission.md) |
| Batch / data / inference | [data/examples.md](ray-resources/data/examples.md) |
| Distributed training | [train/examples.md](ray-resources/train/examples.md) |
| Hyperparameter tuning | [tune.md](ray-resources/tune.md), [tune/examples.md](ray-resources/tune/examples.md) |
| Model serving | [serve.md](ray-resources/serve.md), [serve/examples.md](ray-resources/serve/examples.md) |
| Reinforcement learning | [rllib.md](ray-resources/rllib.md) |
| LLM batch | [ray-resources/llm/](ray-resources/llm/) |
| Long-running / Prefect | [orchestration-patterns/prefect-long-running.md](orchestration-patterns/prefect-long-running.md) then Prefect online docs |
| Everything else | [ray-resources/README.md](ray-resources/README.md) |

```
Is your workload...
├─ Confirmed long-running / FT? → orchestration-patterns/prefect-long-running.md  (+ Ray primitive below)
├─ Data / batch inference?     → ray-resources/data/
├─ Distributed training?       → ray-resources/train/
├─ Hyperparameter tuning?      → ray-resources/tune/
├─ Model serving?              → ray-resources/serve/
├─ RL?                         → ray-resources/rllib.md
└─ Generic distributed?
   ├─ Stateful?  → actors  (ray-resources/ray-core/actors.md)
   └─ Stateless? → tasks   (ray-resources/ray-core/tasks.md)
```

## Patterns (must follow)

- [Limit running tasks](ray-resources/ray-core/patterns/limit-running-tasks.md) – control concurrency with resources
- [Limit pending tasks](ray-resources/ray-core/patterns/limit-pending-tasks.md) – backpressure with `ray.wait`
- [Pass large arg by value](ray-resources/ray-core/patterns/pass-large-arg-by-value.md) – use `ray.put` + object refs
- [Closure capture of large objects](ray-resources/ray-core/patterns/closure-capture-large-objects.md) – pass large data via args/refs, not closures

## Output layout

**Short / one-shot**

```
output/
├── <script>.py           # rayified script
├── smoke_test.py         # same cluster/infra as prod; tiny run; prints real result
└── requirements.txt      # as needed
```

**Long-running** (after dev confirms)

```
output/
├── flow.py               # Prefect entry; calls Ray work
├── <ray_work>.py         # optional split
├── smoke_test.py         # Prefect path + remote store + Ray; tiny; prints proof
├── Dockerfile            # Alpine Python preferred
├── docker-compose.yml    # prefect-server (UI :4200) + job
├── .env                  # only project-local keys created for this job if needed
└── requirements.txt      # prefect + Ray + backend clients used
```
