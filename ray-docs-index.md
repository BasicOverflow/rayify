# Ray Documentation Index

Curated links to [docs.ray.io](https://docs.ray.io/en/latest/) for coding agents. **Do not treat this file as authoritative API reference** — always web-search or fetch the linked page for the feature you are implementing; Ray APIs and best practices change between releases.

## How to use

1. Pick the section that matches your workload (decision tree at bottom).
2. Open the linked doc page (or search `site:docs.ray.io <topic>`).
3. Cross-check patterns/anti-patterns before shipping concurrency or object-store code.

---

## Ray Core

Foundation for distributed Python: tasks, actors, objects, scheduling, fault tolerance.

| Topic | Description | Doc |
|---|---|---|
| Getting started | Install, `ray.init`, first remote task | [Ray Core — Getting Started](https://docs.ray.io/en/latest/ray-core/getting-started.html) |
| Tasks | Stateless `@ray.remote` functions; `.remote()`, refs, parallelism | [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html) |
| Actors | Stateful `@ray.remote` classes; handles, concurrency, lifetimes | [Actors](https://docs.ray.io/en/latest/ray-core/actors.html) |
| Objects & object store | `ray.put`, `ray.get`, refs, serialization, plasma | [Objects](https://docs.ray.io/en/latest/ray-core/objects.html) |
| Scheduling & resources | CPUs/GPUs, custom resources, strategies, placement | [Scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/index.html) |
| Placement groups | Reserve bundles of resources for gangs of tasks/actors | [Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) |
| Fault tolerance | Retries, reconstruction, actor/task failure handling | [Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault-tolerance.html) |
| Dependencies & runtime env | `pip`/`conda` packages, env vars, working dir on workers | [Handling Dependencies](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html) |
| Runtime env API | `runtime_env` dict, `ray.init(runtime_env=…)` | [Runtime Env API](https://docs.ray.io/en/latest/ray-core/api/runtime-env.html) |
| Namespaces | Isolate actors/tasks per logical application | [Namespaces](https://docs.ray.io/en/latest/ray-core/namespaces.html) |
| Configure Ray | Logging, memory, object spilling, system config | [Configuring Ray](https://docs.ray.io/en/latest/ray-core/configure.html) |
| Compiled DAGs | Low-latency pipelined actor/task graphs (advanced) | [Ray Compiled Graph](https://docs.ray.io/en/latest/ray-core/compiled-graph/ray-compiled-graph.html) |
| Cross-language | Java/C++ interop with Python | [Cross-Language](https://docs.ray.io/en/latest/ray-core/cross-language.html) |
| Core API reference | `ray.init`, `ray.remote`, `ray.get`, `ray.wait`, etc. | [Core API](https://docs.ray.io/en/latest/ray-core/api/core.html) |

---

## Ray Libraries

Higher-level APIs for data, training, tuning, serving, RL, and LLM workloads.

| Library | Description | Doc |
|---|---|---|
| Ray Data | Distributed datasets, batch inference, ETL, streaming | [Ray Data](https://docs.ray.io/en/latest/data/data.html) |
| Ray Data — batch inference | `map_batches`, actors, GPU inference patterns | [Batch Inference](https://docs.ray.io/en/latest/data/batch_inference.html) |
| Ray Train | Distributed training (PyTorch, TF, Lightning, HF, …) | [Ray Train](https://docs.ray.io/en/latest/train/train.html) |
| Ray Train — PyTorch | `TorchTrainer`, workers, checkpoints | [PyTorch Guide](https://docs.ray.io/en/latest/train/getting-started-pytorch.html) |
| Ray Tune | Hyperparameter search, schedulers, trial storage | [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) |
| Ray Serve | Model serving, deployments, autoscaling, FastAPI ingress | [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) |
| Ray Serve — production | Multi-app, K8s, observability | [Production Guide](https://docs.ray.io/en/latest/serve/production-guide/index.html) |
| RLlib | RL algorithms, envs, distributed rollouts | [RLlib](https://docs.ray.io/en/latest/rllib/index.html) |
| Ray LLM | Batch LLM inference, vLLM integration | [Ray LLM](https://docs.ray.io/en/latest/ray-llm/index.html) |
| Ray LLM — batch | Offline/batch LLM pipelines | [Batch LLM](https://docs.ray.io/en/latest/ray-llm/examples/batch/vllm-with-lora.html) |

---

## Cluster & Jobs

Connect workers, submit jobs, deploy on K8s or VMs.

| Topic | Description | Doc |
|---|---|---|
| Ray Clusters overview | Head/worker nodes, scaling | [Ray Clusters](https://docs.ray.io/en/latest/cluster/getting-started.html) |
| Job submission | Submit scripts to a running cluster | [Job Submission](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) |
| Kubernetes | KubeRay operator, Helm, in-cluster Ray | [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) |
| VM clusters | AWS/GCP/Azure manual cluster setup | [VM Clusters](https://docs.ray.io/en/latest/cluster/vms/index.html) |
| Monitoring | Dashboard, metrics, logging | [Monitoring](https://docs.ray.io/en/latest/ray-observability/index.html) |

---

## Design patterns (follow these)

Recommended patterns from Ray Core docs. Read the full page before applying.

| Pattern | Description | Doc |
|---|---|---|
| Limit running tasks | Cap concurrency with custom `resources` on `@ray.remote` | [Using resources to limit concurrent tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html) |
| Limit pending tasks | Backpressure with `ray.wait` instead of unbounded `.remote()` | [Using ray.wait to limit pending tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html) |
| Pass large arg by value | Use `ray.put` + object refs, not repeated serialization | [Passing large argument by value repeatedly](https://docs.ray.io/en/latest/ray-core/patterns/pass-large-arg-by-value.html) |
| Closure capture | Don't close over large objects; pass refs as args | [Closure capturing large objects](https://docs.ray.io/en/latest/ray-core/patterns/closure-capturing-large-objects.html) |
| Nested parallelism | Subtasks inside tasks for hierarchical work | [Nested tasks](https://docs.ray.io/en/latest/ray-core/patterns/nested-tasks.html) |
| Generators | Stream results to reduce driver memory | [Generators](https://docs.ray.io/en/latest/ray-core/patterns/generators.html) |
| Async actor methods | Concurrent actor calls with `async def` | [asyncio in actors](https://docs.ray.io/en/latest/ray-core/patterns/concurrent-operations-asyncio.html) |
| Actor synchronization | One actor coordinates other tasks/actors | [Actor synchronization](https://docs.ray.io/en/latest/ray-core/patterns/actor-sync.html) |
| Pipelining | Overlap stages for throughput | [Pipelining](https://docs.ray.io/en/latest/ray-core/patterns/pipelining.html) |
| **Full pattern index** | All official patterns | [Patterns index](https://docs.ray.io/en/latest/ray-core/patterns/index.html) |

---

## Anti-patterns (avoid these)

Common mistakes that hurt performance, fault tolerance, or correctness.

| Anti-pattern | Description | Doc |
|---|---|---|
| `ray.get` in a loop | Blocks parallelism; batch gets or use refs | [ray.get in a loop](https://docs.ray.io/en/latest/ray-core/patterns/ray-get-loop.html) |
| Unnecessary `ray.get` | Pulls data to driver when refs suffice | [Unnecessary ray.get](https://docs.ray.io/en/latest/ray-core/patterns/unnecessary-ray-get.html) |
| Submission-order `ray.get` | Wait in completion order with `ray.wait` | [Submission order ray.get](https://docs.ray.io/en/latest/ray-core/patterns/ray-get-submission-order.html) |
| Too many objects at once | OOM/timeouts from huge `ray.get` lists | [Too many objects ray.get](https://docs.ray.io/en/latest/ray-core/patterns/ray-get-too-many-objects.html) |
| `ray.get` on task args | Redundant fetch inside nested tasks | [ray.get on task arguments](https://docs.ray.io/en/latest/ray-core/patterns/nested-ray-get.html) |
| Return `ray.put` refs | Breaks lineage and fault tolerance | [Return ray.put ObjectRefs](https://docs.ray.io/en/latest/ray-core/patterns/return-ray-put.html) |
| Fine-grained tasks | Task overhead dominates; batch work | [Too fine-grained tasks](https://docs.ray.io/en/latest/ray-core/patterns/too-fine-grained-tasks.html) |
| Redefine remote functions | Re-registering hurts performance | [Redefining remote functions](https://docs.ray.io/en/latest/ray-core/patterns/redefine-ray-func-class.html) |
| Global shared state | Race conditions across tasks/actors | [Global variables](https://docs.ray.io/en/latest/ray-core/patterns/global-variables.html) |
| Out-of-band ObjectRef | Don't pickle/send refs outside Ray | [Out-of-band ObjectRef](https://docs.ray.io/en/latest/ray-core/patterns/out-of-band-object-ref-serialization.html) |
| Fork in application code | `fork()` after Ray init causes hangs | [Fork new processes](https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html) |
| **Full anti-pattern index** | All official anti-patterns | [Patterns index](https://docs.ray.io/en/latest/ray-core/patterns/index.html) |

---

## Workload picker

```
Is your workload...
├─ Batch data / ETL / inference?  → Ray Data
├─ Distributed training?          → Ray Train
├─ Hyperparameter search?         → Ray Tune
├─ Online model serving?          → Ray Serve
├─ Reinforcement learning?        → RLlib
├─ Batch LLM (vLLM)?              → Ray LLM
└─ Generic distributed Python?
   ├─ Stateful / shared?          → Actors (+ patterns above)
   └─ Stateless / parallel?      → Tasks (+ patterns above)
```

---

## External references

| Resource | Description | Link |
|---|---|---|
| Official docs home | Latest stable docs | [docs.ray.io](https://docs.ray.io/en/latest/) |
| GitHub | Source, issues, examples | [ray-project/ray](https://github.com/ray-project/ray) |
| Example gallery | End-to-end samples | [Ray Examples](https://docs.ray.io/en/latest/ray-overview/examples.html) |
