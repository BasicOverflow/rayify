# AGENTS.md - Ray Conversion Guide

Convert Python scripts in `input/` into Ray-optimized, cluster-ready code in `output/`. Connect to an existing Ray cluster (never start local Ray unless explicitly required).

**Docs:** open listed [ray-resources/](ray-resources/) files for the workload ([index](ray-resources/README.md)), then **always cross-check that Ray practice against current online docs** (using web search) before implementing. Repo docs can be stale — if online and local disagree, prefer online and note the mismatch.

**Style:** open [unslop-examples/](unslop-examples/) ([README](unslop-examples/README.md) + matching pattern files) before non-trivial structure.

**Prior rayify examples:** list all branches (`git branch -a`); if other branches hold approved real rayified projects, switch/read them when relevant (how connect, actors, tests, layout). Return to the working branch when done. Do not invent from scratch if a branch already solved a similar case.

## Hard rules

- Cluster-first: `ray.init` via env vars, not implicit local Ray
- Required env: `RAY_ADDRESS`, `RAY_NAMESPACE`. Optional: `RAY_RUNTIME_ENV` (JSON)
- **One namespace per project** — every rayified project gets its own `RAY_NAMESPACE` (unique name for that project). Do not share a namespace across unrelated projects/jobs/actors. Document the chosen namespace (e.g. in output or `.env` notes for that project)
- Read original from `input/`; write converted script(s) to `output/`
- Confirm expected output with the user before converting
- Specify resource minimum needs on remotes (`num_cpus`, `num_gpus`, `memory`, etc.) if needed or when non-trivial (assuming approval received from dev after making a case for the need of defining resource needs)

## Code style (unslop + clear OOP)

Match the cleaned half of [unslop-examples/](unslop-examples/). Index: [unslop-examples/README.md](unslop-examples/README.md).

- **Simple, readable OOP when it helps** — name types for real concepts (config, planner, model replica), keep methods short, prefer classes over spaghetti free functions when state or related ops cluster together. See [08-attention-specs-vram-planning.md](unslop-examples/08-attention-specs-vram-planning.md) for split: pure planning objects vs Ray/Serve actors.
- **Not OOP theater** — no empty plugin stubs, unused bookkeeping, or metrics nobody reads ([06](unslop-examples/06-scaffolds-magic-env.md), [07](unslop-examples/07-router-overbuild.md)).
- **Efficiency still matters** — don’t add layers that thrash the object store, serialize huge payloads, or spawn extra remotes “for structure.” Prefer a clear class *or* a plain function when the function is shorter and just as clear; never trade a big runtime cost for ceremony.
- **No defensive soup** — let bad state raise; no silent `except`, stacked `hasattr`, soft-fail timeouts, or invent-a-fallback deploys ([01](unslop-examples/01-wait-helpers-and-fallbacks.md), [04](unslop-examples/04-inference-defensive-soup.md)).
- **No print theater / dead paths** — skip section banners, emoji status systems, legacy dual branches, double-checks ([02](unslop-examples/02-legacy-paths-and-double-checks.md), [03](unslop-examples/03-examples-print-theater.md)).
- **Don’t paper over bad data** — no regex JSON recovery if you control the schema; fail or use real structured output ([05](unslop-examples/05-json-parse-slop.md)).

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

1. **Analyze** – What should the script produce? Confirm with the user. Note parallelizable work, stateful vs stateless, data sharing, resource needs. Pick a **dedicated project namespace**.
2. **Examples** – Scan other branches for relevant prior rayify projects if they exist; read those when useful.
3. **Choose primitive** – Use the table below; open the linked local doc, then confirm vs online Ray docs.
4. **Implement** – Ray patterns + unslop style (clear OOP where natural). Shared data via `ray.put`, resource hints, concurrency limits.
5. **Deps / infra** – `runtime_env` / `RAY_RUNTIME_ENV` as needed. Hosts/ports/endpoints via local infra notes or notes MCP.
6. **Verify** – Smoke script(s) on real cluster/infra, short run, print real results.
7. **Docker** – Only for long-running scripts: `Dockerfile`, `docker-compose.yml`, `requirements.txt` in `output/` with `RAY_ADDRESS` / `RAY_NAMESPACE`.

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
| Everything else | [ray-resources/README.md](ray-resources/README.md) |

```
Is your workload...
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

```
output/
├── <script>.py           # rayified script
├── smoke_test.py         # same cluster/infra as prod; tiny run; prints real result
├── Dockerfile            # optional, long-running
├── docker-compose.yml
└── requirements.txt
```
