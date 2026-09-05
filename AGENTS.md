# AGENTS.md — Ray Conversion Guide

Convert Python scripts in `input/` into Ray-optimized, cluster-ready code in `output/`. Connect to an existing Ray cluster (never start local Ray unless explicitly required).

---

## How to work (high-level mindset)

**This section is non-negotiable.**

A rayify conversion is not a sketch. It is a complete, runnable artifact: Ray code that connects to the real cluster, durable state on the chosen remote backend, Alpine-first Docker packaging, and verification against the infra the user actually has.

Search before building. Read `input/` before writing `output/`. Cross-check Ray APIs online before implementing. Ship the whole conversion, not a plan to convert it later.

When the user asks for a rayification, the answer is finished code under `output/`, not a bullet list of next steps. If something needs manual infra setup the agent cannot provision, stop and name it as a requirement before coding around it.

You can outsource the typing. You cannot outsource the understanding. Before calling a conversion done, be able to explain why the Ray primitive fits, where object-store or actor state would break, and what happens on worker restart. "It looks right" is not done.

---

## Task sizing — triage before spending tokens

**This section is non-negotiable.** It gates verification depth, parallel exploration, and how much packaging you touch. "Do the whole thing" means the whole thing the conversion actually needs — not a full cluster integration test for a one-line comment fix.

**Every conversion task starts with a printed triage block, before any work.** Four lines:

```
Size: small | medium | large — why
Verify: local lint/import | docker compose build | cluster smoke run — why
Agents: solo | fan-out (how many, on what) | variant tournament — why
Scope: <input file(s)> → <output path(s)>
```

This block is mandatory. A wrong mode is only correctable if the choice is visible. Never skip it, never bury it mid-report.

**The sizes:**

- **small** — config tweak, env key rename, comment fix, one-file mechanical edit in `output/` with no behavior change. Solo. Lint/import check only. No new Docker layer unless the edit requires it. No cluster smoke run unless behavior changed.
- **medium** — localized behavior change or bug fix inside one `output/` project or module. Solo by default; fan out only if the work splits into truly independent units. One cold critic pass on the repro. `docker compose build` for the affected job. Cluster smoke run if Ray scheduling, runtime_env, or remote I/O changed.
- **large** — new conversion from `input/`, new Ray primitive choice, cross-module architecture, persistence wiring, resource placement, or anything judgment-heavy (tasks vs actors vs Ray Data, checkpoint store, fan-out design). Full workflow + fan-out/critic loop (see below): prior branches, docs index, packaging, remote verification plan.

**Deciding rules:**

- When torn between two sizes, pick the smaller one and say so in the triage block. Escalating mid-task is cheap; burning a large-protocol run on a typo is not.
- Escalate the moment the change turns out bigger than triaged (touches persistence contracts, spreads across modules, needs new infra). Print an updated triage block with what changed the call.
- Verification follows blast radius: if the diff cannot reach Ray init, remote I/O, or container entrypoint, you do not need a cluster run to prove completeness.
- The final report restates what was actually verified so the triage call can be judged after the fact.

---

## Repo layout — boundaries

**This section is non-negotiable and must never be removed.**

| Path | Role |
|---|---|
| `input/` | **Read only.** Original scripts notebooks, requirements. Never modify unless the user explicitly asks. |
| `output/` | **Write here.** Rayified entry scripts, `lexis_markets/`-style packages, `Dockerfile`, `docker-compose.yml`, project-local `.env` when approved. |
| Root `.env` | Shared cluster + infra creds. **Required.** If missing, stop and tell the user to copy from [`.env.example`](.env.example). Never invent hosts or secrets. |
| [`ray-docs-index.md`](ray-docs-index.md) | Curated links to [docs.ray.io](https://docs.ray.io/). Pick a section, then web-search or fetch the linked page. |
| [`unslop-examples/`](unslop-examples/) | Code style bible. Open before non-trivial structure. |
| [`orchestration-patterns/`](orchestration-patterns/) | Job-container packaging crib. Open before writing Docker/compose. |

**One namespace per project:** set a unique `RAY_NAMESPACE` per conversion. Document it in the output README or entry script docstring. Never reuse another project's namespace.

**Prior rayify examples:** `git branch -a` — if other branches hold approved rayified projects, read them for connect patterns, actors, layout, Docker. Return to the working branch when done. Do not invent from scratch when a branch already solved a similar case.

**Git:** do not commit unless the user asks. When they do, never commit secrets; verify `.gitignore` covers `.env`.

---

## The two machine spaces — read this before converting

Every piece of work belongs to one of two spaces. Picking the wrong one is the most common way agents produce bad Ray code.

**Latent space = LLM work.** Judgment calls: tasks vs actors vs Ray Data, where to checkpoint, how to partition work, API shape, error semantics. Cost: tokens. Variability: high.

**Deterministic space = code.** Data transforms, file I/O, SQL queries, Parquet reads, hash computations, progress counters, validation scripts. Cost: one-time write. Variability: zero. Inspectability: total.

**The rule:** if the same input always produces the same correct output, do not do it in latent space. Write a `@ray.remote` task or a plain function and test it. If you find yourself reasoning through row counts, date math, or CSV parsing in prose, stop and write code.

**The Ray meta-loop:** the agent designs the deterministic Ray graph; the graph constrains all future runs. Object refs instead of repeated serialization, `ray.wait` for backpressure, remote checkpoints instead of driver memory — each pattern removes a class of latent-space mistakes.

Every conversion starts with: what must run on the cluster (deterministic), and what requires architectural judgment (latent)? Split accordingly.

---

## The context window is the lever

Curate what you load. A bloated context produces bloated Ray wrappers.

**Load for every large conversion:**

- The `input/` script(s) and expected output behavior
- [ray-docs-index.md](ray-docs-index.md) section matching the chosen primitive
- Matching [unslop-examples/](unslop-examples/) pattern files
- [orchestration-patterns/docker-compose.md](orchestration-patterns/docker-compose.md) when packaging
- Relevant prior branch code if one exists

**Fetch online before implementing:** the specific [docs.ray.io](https://docs.ray.io/) pages from the index. Ray APIs change between releases; the index is a map, not the API.

**Do not load:** entire unrelated modules, stale local doc mirrors, or speculative abstractions the input script does not need.

When a conversion goes sideways, the first question is "what was in the window," not "was the model dumb."

---

## Non-negotiable rules

### Cluster-first

```python
import os
import ray

# RAY_NAMESPACE must be unique per project (do not reuse another project's namespace)
ray.init(
    address=os.environ["RAY_ADDRESS"],
    namespace=os.environ["RAY_NAMESPACE"],
)
```

- Connect via env vars. Never start implicit local Ray unless the user explicitly requires it.
- Required: `RAY_ADDRESS`, `RAY_NAMESPACE` from root [`.env`](.env). Optional: `RAY_RUNTIME_ENV` (JSON).
- Resource hints on remotes when non-trivial — after user approves CPU/GPU/custom resource counts.

### Backends from `.env`

Treat listed services (Ray, MinIO/S3, Postgres, Mongo, Neo4j, SearXNG, Gitea, …) as a shared toolbox — use any of them for any project-appropriate role (not only "S3=blobs / SQL=tables"). Keys in [`.env.example`](.env.example) are endpoints/creds, not fixed recipes.

- Prefer what is already provisioned. If the job needs a new dedicated DB/bucket/instance, **ask once** (reuse vs dedicated), recommend, wait for yes, then update root `.env`.
- Wire only via env; propagate needed vars to workers through `runtime_env`.
- **If something needs manual setup the agent cannot do** (new Postgres/Mongo/Neo4j DB, MinIO bucket, Gitea package index, registry image, …), stop and point it out before coding that path.

### Durable state

- Ask once which **remote** store holds progress/checkpoints when the job needs durable state (MinIO/S3, Postgres, Mongo, Neo4j — reuse vs dedicated).
- Never host bind-mount recovery for that state.
- Workload artifacts and progress live on the chosen remote backend. Ray Train/Tune checkpoints go to that store when those primitives apply.
- Actors are disposable memory; reload progress from remote on restart.
- Do **not** invent a repo-local checkpoint base-class package.

### Packaging — every conversion ships

- Alpine-first `Dockerfile` + `docker-compose.yml` (job container only — no orchestration UI).
- Crib: [orchestration-patterns/docker-compose.md](orchestration-patterns/docker-compose.md).
- Shared infra: pass root `.env` into the job service via compose.
- Prefer `python:3.x-alpine` for the job image; switch only if a dep forces a different slim base.
- Project-local keys (e.g. W&B) live under `output/.env` **only after user approval** — never blank W&B slots in root [`.env.example`](.env.example).

### Verify before calling done

- **`docker compose build`** — image builds for medium and large changes.
- **Cluster smoke run** — when Ray init, runtime_env, remote I/O, or scheduling changed; run from `output/` against the user's cluster when available.
- **Copy-paste commands** — every command in output docs gets run or marked unverified with what would settle it.
- Behavior changes ship evidence: log excerpt, test output, or explicit "cluster run not performed because …".

### Search before building

Three layers, in order:

1. **Prior art in this repo** — other branches, existing `output/` patterns.
2. **Official Ray docs** — [ray-docs-index.md](ray-docs-index.md) → fetch the linked page.
3. **First-principles** — custom Ray design only when the input workload genuinely does not fit library primitives; document why.

### Optional capabilities — ask first

| Capability | When | Rule |
|---|---|---|
| **Ray Hive** | High-throughput LLM serving (vLLM), embeddings, multimodal | Ask before using. Web-search [ray-hive README](https://github.com/BasicOverflow/ray-hive/blob/main/README.md) and crib [examples](https://github.com/BasicOverflow/ray-hive/tree/main/examples). Install via Gitea PyPI keys from `.env`. |
| **cudf.pandas** | Heavy single-node DataFrame work on GPU workers | Ask first. Not the default — prefer plain pandas or [Ray Data](https://docs.ray.io/en/latest/data/data.html) for distributed/out-of-core. |
| **W&B or similar** | Training/ML experiment tracking | Ask first. Project-local env keys under `output/` only. |

**Ray Hive install** (fill from root `.env` `GITEA_USER` / `GITEA_TOKEN` / `GITEA_PYPI_SIMPLE`):

```bash
pip install ray-hive \
  --index-url http://<GITEA_USER>:<GITEA_TOKEN>@<gitea-host>:<port>/api/packages/<owner>/pypi/simple/ \
  --extra-index-url https://pypi.org/simple
```

Connect with `RAY_ADDRESS` from `.env`.

---

## Conversion workflow

For **large** tasks, follow all steps. For **medium**, skip steps that do not apply. For **small**, steps 1 and 7 only unless scope demands more.

1. **Analyze** — Confirm expected output from `input/`. Stop if no root `.env`. Set unique `RAY_NAMESPACE`. Resolve: persistence reuse/dedicated, Ray Hive, W&B.
2. **Prior art** — Check branches and existing `output/` for similar conversions.
3. **Choose primitive** — Use the picker below; open [ray-docs-index.md](ray-docs-index.md); fetch the linked docs.ray.io page.
4. **Style pass** — Open matching [unslop-examples/](unslop-examples/) before writing non-trivial structure.
5. **Implement** — Write to `output/` only. Ray + unslop. Backends via confirmed env keys. Persist progress/artifacts remotely.
6. **Deps** — `runtime_env` / `RAY_RUNTIME_ENV` as needed; `requirements.txt` with Ray + backend clients used.
7. **Verify** — Build container; cluster smoke run when blast radius requires it.
8. **Package** — Alpine `Dockerfile` + `docker-compose.yml` (job service only).

**Fan out when the size calls for it.** Whether to fan out is decided in the triage block. Large work runs parallel; small never does. How to fan out and the critic loop are in "Fan-out + harsh critic" below.

---

## Fan-out + harsh critic — for large work

**This section is non-negotiable.**

Explicit opt-in to multi-agent orchestration (Task tool / parallel subagents) for every task triaged **large**, and for **medium** tasks that split into truly independent units. Small tasks never fan out. When this loop runs, say so in the triage block and final report; when skipped, say that too and why.

**Step 0 — name the reference before building.** The critic is only as good as what it judges against. Every task in this loop (and every medium task getting its one cold critic pass) writes down its reference first:

1. **The real thing** (parity conversion): the `input/` script's behavior — same outputs, same semantics, now on Ray. Side-by-side against the original.
2. **Best-in-class analog** (new Ray design): a named prior branch conversion or official Ray example closest to this workload. Judged side-by-side even when not copying it.
3. **A frozen rubric** (nothing comparable): concrete acceptance criteria plus measurable outcome (throughput, checkpoint recovery, container build, cluster connect) written **before** building starts. Frozen once building begins; the builder cannot negotiate it down.

No reference, no build. If you cannot write what "done well" means, that is a Confusion Protocol stop.

**The loop, for every task triaged large:**

1. **Decompose and fan out.** Independent units, one builder subagent per unit, run in parallel. Serial work on parallelizable units is wasted wall-clock. Examples of splittable units: read/analyze separate `input/` modules; wire persistence vs Ray graph vs Docker packaging; explore competing primitive choices. Every **judgment-heavy** unit gets a **variant tournament**: 2–3 competing builders on the **same** unit (e.g. tasks vs actors layout, partition strategy, checkpoint design) so the critic has variants to compare blind. Parallel builders that write the same files must not share one working tree — use isolated worktrees or non-overlapping file scopes.
2. **Builder never grades its own work.** Output goes to a separate critic subagent that did not build it and never sees the builder's reasoning. Deliverable plus reference only.
3. **The critic is harsh by default; its job is to reject.** Blind wherever comparison exists: outputs labeled A/B in random order. Verdict must be concrete: which is better and exactly why. "Pretty good" is FAIL. "Acceptable" is FAIL. Pass only when the critic would pick ours (or cannot tell) in blind comparison. **Replace-vs-layer is a default reject on every critic pass.** Walk the diff for a second path, wrapper, `_v2` / `_fixed` / `patch_*` / `fix_*` fork, or leftover old body. If the old implementation is still reachable, FAIL. Name the layered symbol and the original it should have replaced.
4. **Loop until pass.** Builder revises against named findings. Fresh critic re-judges cold each round. Pass requires the critic's explicit verdict, never the builder's claim.
5. **Stall rule.** If 3 consecutive rounds show no improvement on the critic's named criteria, report **BLOCKED** with the last verdict, evidence, and what is missing (cluster access, infra decision, user input). Do not silently lower the bar.
6. **Evidence or it didn't happen.** Every critic verdict ships with artifacts: diffs, build logs, cluster smoke output, anti-pattern checklist, A/B comparison. Keep under `/tmp/<task>/critique/` and reference exact paths in the final report. Never commit critique artifacts to the repo.

**The critic per work type** (pattern constant, weapon changes):

- **Ray conversion / parity:** `input/` behavior preserved; cluster connect works; object-store patterns from [ray-docs-index.md](ray-docs-index.md) followed; no unslop violations.
- **New Ray architecture:** frozen rubric plus best-in-class analog; variant tournament on primitive/partition choice; critic reads cold like a maintainer seeing the code for the first time.
- **Bug fix:** reference is the repro. Critic is an attacker: re-break the fix, probe worker restart, verify regression evidence fails with the bug present. The fix must live in the broken function, not around it (`patch_*` / `fix_*` wrappers FAIL).
- **Performance / resources:** numeric budget stated before work (task count, object-store churn, GPU allocation); critic reads only the numbers and logs.
- **Docker / packaging:** critic builds the image and checks compose wiring against [orchestration-patterns/docker-compose.md](orchestration-patterns/docker-compose.md).
- **Security / code quality:** adversarial review — env leakage, secrets in output, global mutable state across tasks, out-of-band ObjectRefs.

**Solo (no fan-out) is the rule for:** small tasks, most medium tasks, and investigation that fits one context. Medium bug fixes still get one cold critic pass, not a tournament. When torn between medium and large, triage says pick medium; when a large task splits cleanly, fan out.

---

## Choose your Ray primitive

Full index: **[ray-docs-index.md](ray-docs-index.md)** (patterns, anti-patterns, workload tree).

| Workload | Doc |
|---|---|
| Stateless parallel (tasks) | [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html) |
| Stateful / shared model (actors) | [Actors](https://docs.ray.io/en/latest/ray-core/actors.html) |
| Shared large data / object store | [Objects](https://docs.ray.io/en/latest/ray-core/objects.html) |
| Scheduling / placement / resources | [Scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/index.html) |
| Fault tolerance | [Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault-tolerance.html) |
| Deps / runtime_env | [Handling Dependencies](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html) |
| Configure / core APIs | [Configuring Ray](https://docs.ray.io/en/latest/ray-core/configure.html), [Core API](https://docs.ray.io/en/latest/ray-core/api/core.html) |
| Getting started | [Getting Started](https://docs.ray.io/en/latest/ray-core/getting-started.html) |
| Job submission | [Job Submission](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) |
| Batch / data / inference | [Ray Data](https://docs.ray.io/en/latest/data/data.html) |
| Distributed training | [Ray Train](https://docs.ray.io/en/latest/train/train.html) |
| Hyperparameter tuning | [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) |
| Model serving | [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) |
| Reinforcement learning | [RLlib](https://docs.ray.io/en/latest/rllib/index.html) |
| LLM batch | [Ray LLM](https://docs.ray.io/en/latest/ray-llm/index.html) |
| Job container packaging | [orchestration-patterns/docker-compose.md](orchestration-patterns/docker-compose.md) |

```
Is your workload...
├─ Batch data / ETL / inference?  → Ray Data
├─ Distributed training?          → Ray Train
├─ Hyperparameter search?         → Ray Tune
├─ Online model serving?          → Ray Serve
├─ Reinforcement learning?        → RLlib
├─ Batch LLM (vLLM)?              → Ray LLM
└─ Generic distributed Python?
   ├─ Stateful / shared?          → Actors
   └─ Stateless / parallel?       → Tasks
```

---

## Ray patterns (must follow)

See [ray-docs-index.md — Design patterns](ray-docs-index.md#design-patterns-follow-these) and [Anti-patterns](ray-docs-index.md#anti-patterns-avoid-these). Minimum set:

- [Limit running tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html) — control concurrency with resources
- [Limit pending tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html) — backpressure with `ray.wait`
- [Pass large arg by value](https://docs.ray.io/en/latest/ray-core/patterns/pass-large-arg-by-value.html) — use `ray.put` + object refs
- [Closure capture of large objects](https://docs.ray.io/en/latest/ray-core/patterns/closure-capturing-large-objects.html) — pass large data via args/refs, not closures

Before shipping any code that calls `ray.get`, reads the object store, or spawns many remotes: scan the anti-pattern list in [ray-docs-index.md](ray-docs-index.md).

---

## Code style (unslop + clear OOP)

Match the cleaned half of [unslop-examples/](unslop-examples/). Index: [unslop-examples/README.md](unslop-examples/README.md).

- **Simple, readable OOP when it helps** — name types for real concepts (config, planner, model replica), keep methods short, prefer classes over spaghetti free functions when state or related ops cluster together. See [08-attention-specs-vram-planning.md](unslop-examples/08-attention-specs-vram-planning.md) for split: pure planning objects vs Ray/Serve actors.
- **Not OOP theater** — no empty plugin stubs, unused bookkeeping, or metrics nobody reads ([06](unslop-examples/06-scaffolds-magic-env.md), [07](unslop-examples/07-router-overbuild.md)).
- **Efficiency still matters** — do not add layers that thrash the object store, serialize huge payloads, or spawn extra remotes "for structure." Prefer a clear class *or* a plain function when the function is shorter and just as clear.
- **No defensive soup** — let bad state raise; no silent `except`, stacked `hasattr`, soft-fail timeouts, or invent-a-fallback deploys ([01](unslop-examples/01-wait-helpers-and-fallbacks.md), [04](unslop-examples/04-inference-defensive-soup.md)).
- **No print theater / dead paths** — skip section banners, emoji status systems, legacy dual branches, double-checks ([02](unslop-examples/02-legacy-paths-and-double-checks.md), [03](unslop-examples/03-examples-print-theater.md)).
- **Do not paper over bad data** — no regex JSON recovery if you control the schema; fail or use real structured output ([05](unslop-examples/05-json-parse-slop.md)).
- **No compliance / process comments** — do not litter output code with comments that narrate following AGENTS.md, Ray docs, unslop rules, "as required," checklist ticks, etc. Comments only when they clarify non-obvious *code* behavior.
- **Replace, do not layer** — when fixing, patching, or upgrading: edit the original until it is right. Callers keep the same name. The old body is gone. Do not wrap, fork, or add a second path over the first. Layering is a FAIL: `foo_v2` / `foo_fixed` / `foo_new` beside `foo`; `patch_*` or `fix_*` helpers; a wrapper that calls the old function then patches the result; `if use_new: ... else: <old body>`; a post-step that cleans bad output instead of fixing the producer; a new file that reimplements the old one while the old file stays live. A one-line in-place edit is replacement. Leaving the broken path is not. Exceptions (ask if unsure): the user asked for a compatibility window; a public API must keep an old signature while a new one ships.

---

## Output layout

```
output/
├── <script>.py           # rayified entry (or split below)
├── <ray_work>.py         # optional split when entry gets large
├── Dockerfile            # Alpine Python preferred
├── docker-compose.yml    # job service → Ray cluster + root .env
├── .env                  # only project-local keys created for this job if needed
└── requirements.txt      # Ray + backend clients used
```

Run from `output/`:

```bash
docker compose up --build job
```

---

## Confusion protocol

When you hit high-stakes ambiguity, **stop**. Name it in one sentence. Present 2–3 options with real trade-offs. Ask the user. Do not guess.

Triggers specific to rayify:

- Tasks vs actors vs Ray Data for the same workload
- Which remote store holds checkpoints/progress (reuse vs dedicated)
- Whether to wire Ray Hive, cudf.pandas, or W&B
- Infra that requires manual provisioning (new DB, bucket, registry image)
- Resource placement (GPU count, custom resources) without user input
- Destructive cluster or data operations

Does **not** apply to routine coding inside an agreed architecture.

---

## Self-rating — proud or loop

Reporting a completion status is not the end of the task. Before the final report, rate the work. Scale with triage size: a **small** task gets one line (score + yes/no from a fresh read of the diff) and no loop; **medium** and **large** get the full protocol:

- Score the finished work 1–10 and print the score. Rate from a fresh read of the deliverable (the diff, `output/`, build/cluster evidence), not from memory of building it.
- Then answer honestly: am I proud of this work? Yes or no.
- The bar is **How to work**, not "it compiles": complete conversion, verified, packaged, understood, cluster-ready. A 7 with a shrug is no.
- If no, do not stop. Name exactly what falls short, fix it, re-rate. Loop until the honest answer is yes. Each pass states what changed since the last rating.
- If no cannot be fixed here (blocked on user, cluster access, missing infra), report **DONE_WITH_CONCERNS** or **BLOCKED** with the gap named. Never inflate the score or fake yes to exit.
- Anchor the score. Every point below 10 names a specific gap against the task's reference or rubric (Fan-out + harsh critic, Step 0). A score with no named gaps is a guess.
- **Drift guard.** If the rating loop reaches a third pass, hand rating to a fresh critic subagent (clean context, deliverable plus reference only); its score replaces the self-score from then on.
- Rating comes before commit (when the user asked for one), so fixes from the loop land in the same commit.
- Rating is not the review. Where a critic pass applies (medium and large), rate only after every unit passed critic; proud yes never substitutes for critic pass, and critic pass never skips rating.

---

## Completion status protocol

At the end of every task, report one of:

- **DONE** — Conversion complete. Critic pass(es) passed when triage required them. Self-rating yes. Verification matches triage size. Packaging present when scope requires it. Ready for user review.
- **DONE_WITH_CONCERNS** — Completed, but issues the user should know about. List each concern with severity and proposed follow-up.
- **BLOCKED** — Cannot proceed. State what is blocking and what was already tried.
- **NEEDS_CONTEXT** — Missing information required to continue. State exactly what is needed (`.env` values, cluster access, persistence choice, expected output).

"Partially done" is not a status.

---

## After every task

1. **Self-rate** — per "Self-rating" above (before this report when medium/large).
2. **Report how to run it.** Give the exact commands: `cd output/`, `docker compose up --build job`, env vars to set, cluster prerequisites. If nothing needs running, say so.
3. **Report what was verified.** Match the triage block: lint, build, cluster run, critic artifacts, or explicit skip with reason.
4. **Commit only when asked.** Do not commit unless the user requests it. When they do: no secrets, no `--no-verify`, clear message focused on why.

---

## Long-running Ray jobs

Backfills, ingest pipelines, and cluster batch jobs that run for minutes or hours:

- **Checkpoint to remote store** — never rely on driver memory or bind mounts for recovery.
- **Monitor, do not fire-and-forget** — progress updates with rows/tasks done, error count, rate. Surface anomalies plainly.
- **Idempotent stages** — design so a restarted actor/task can resume from remote state.
- **Report on completion** — verdict (worked / partial / failed), what changed, where artifacts landed (bucket prefix, table, path).

Progress counts and rates are deterministic: read them from logs, checkpoint files, or remote store queries — do not estimate in prose.

---

## Safety

- Never commit secrets. If `.env` is touched, verify `.gitignore` before any commit.
- Never run `rm -rf`, `git reset --hard`, `git push --force`, `DROP TABLE`, or similar destructive ops without explicit user confirmation.
- Never skip pre-commit hooks with `--no-verify`. If a hook fails, fix the underlying issue.
- Never commit binaries, compiled outputs, or model weights. Use object store with pointers.
- Before any action that touches production data or a shared cluster, state what you are about to do and wait for confirmation when scope is unclear.

---

## How the user wants to be talked to

- Direct. Short. Concrete. No preamble.
- Specific file names, function names, line numbers. Not "there's an issue in the ingest module" — it's `output/lexis_markets/ingest.py:142`.
- No em dashes. No AI vocabulary (delve, crucial, robust, comprehensive, nuanced, multifaceted, furthermore, moreover, pivotal, landscape, tapestry, underscore, foster, showcase, intricate, vibrant, fundamental, significant, interplay).
- No banned phrases: "here's the kicker", "here's the thing", "plot twist", "let me break this down", "the bottom line", "make no mistake".
- If something is broken, say so plainly.
- End responses with the next action, not a recap of what was just done.

When the user asks for a rayification, the answer is the finished product under `output/`. Verification included. Packaging included.
