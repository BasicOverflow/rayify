# AGENTS.md - Ray Conversion Guide

## Purpose

This guide helps coding agents convert Python scripts into Ray-optimized, cluster-ready code that connects to and runs on remote Ray clusters. All conversions assume a **cluster-first approach** - scripts will connect to an existing Ray cluster rather than starting a local Ray instance.

## How to Use This Guide

1. **Start with Cluster Connection** (Section 2) - Always begin by setting up cluster connection via environment variables
2. **Follow the Conversion Workflow** (Section 3) - Step-by-step process to convert your script
3. **Reference Use Case Sections** (Section 4) - Find patterns specific to your workload type
4. **Check Patterns & Anti-Patterns** (Section 5) - Avoid common mistakes
5. **Use Quick Reference Sections** (Sections 6-9) - Fast lookup for APIs, examples, and configuration

## Prerequisites

- Python 3.7+
- Ray installed: `pip install "ray[default]"`
- Access to a Ray cluster (cluster address)
- `RAY_ADDRESS` environment variable set

## Project Structure and Workflow

### Input/Output Directories

This repository uses a simple workflow for script conversion:

- **`input/`**: Place the original Python script(s) to be converted here
- **`output/`**: Write the converted Ray-optimized script(s) here

**Workflow:**
1. Original script is placed in `input/` directory
2. Agent reads script from `input/` and converts it using this guide
3. Converted script is written to `output/` directory
4. Converted script connects to Ray cluster via `RAY_ADDRESS` environment variable

**Example:**
- Input: `input/my_script.py` (original Python script)
- Output: `output/my_script.py` (Ray-optimized version)

## Resource Directory Structure

This guide references documentation in the `resources/` directory:

- `resources/ray-core/` - Core Ray concepts (tasks, actors, objects, scheduling)
- `resources/ray-core/patterns/` - Design patterns and anti-patterns
- `resources/ray-core/api/` - API reference documentation
- `resources/data/` - Ray Data for batch processing
- `resources/train/` - Ray Train for distributed training
- `resources/tune/` - Ray Tune for hyperparameter tuning
- `resources/serve/` - Ray Serve for model serving
- `resources/rllib.md` - Ray RLlib for reinforcement learning
- `resources/cluster/` - Cluster deployment and job submission

---

## 2. Cluster Connection Setup (CRITICAL FIRST STEP)

**IMPORTANT**: All Ray scripts must connect to a cluster. Never assume local Ray initialization unless explicitly required.

### Environment Variables for Cluster Connection

Configure these environment variables before running your script:

#### Required Variables

- **`RAY_ADDRESS`**: Cluster address
  - Format: `ray://head-node-ip:port` (e.g., `ray://192.168.1.100:10001`)
  - Or: `http://head-node-ip:8265` (for Ray Jobs API)
  - Or: `auto` (connects to existing local cluster)
  - Example: `export RAY_ADDRESS="ray://cluster.example.com:10001"`

#### Optional Variables

- **`RAY_NAMESPACE`**: Logical grouping of jobs and actors
  - Example: `export RAY_NAMESPACE="production"`

- **`RAY_RUNTIME_ENV`**: JSON string for runtime environment
  - Example: `export RAY_RUNTIME_ENV='{"pip": ["numpy", "pandas"], "env_vars": {"MY_VAR": "value"}}'`

- **`RAY_JOB_CONFIG`**: JSON string for job configuration
  - Example: `export RAY_JOB_CONFIG='{"runtime_env": {"pip": ["requests"]}}'`

#### Other Ray Environment Variables

- `RAY_TASK_MAX_RETRIES`: Maximum task retries (default: 3)
- `RAY_gcs_rpc_server_reconnect_timeout_s`: GCS reconnection timeout
- `RAY_PICKLE_VERBOSE_DEBUG`: Enable pickle debugging (values: 0, 1, 2)

See [Environment Variables Reference](#9-environment-variables-reference) for complete list.

### Standard Connection Pattern

Minimal connection pattern:

```python
import os
import ray

# Connect to cluster using address from environment variable
cluster_address = os.getenv("RAY_ADDRESS", "auto")
ray.init(address=cluster_address)

# Verify connection
if not ray.is_initialized():
    raise RuntimeError("Failed to connect to Ray cluster")
```

With optional namespace:

```python
import os
import ray

cluster_address = os.getenv("RAY_ADDRESS", "auto")
namespace = os.getenv("RAY_NAMESPACE")

init_kwargs = {"address": cluster_address}
if namespace:
    init_kwargs["namespace"] = namespace

ray.init(**init_kwargs)
```

### Error Handling for Cluster Connection

```python
import os
import ray
import sys

def connect_to_cluster():
    cluster_address = os.getenv("RAY_ADDRESS")
    
    if not cluster_address:
        raise ValueError(
            "RAY_ADDRESS environment variable not set. "
            "Set it to your cluster address (e.g., ray://head-node:10001)"
        )
    
    try:
        ray.init(address=cluster_address)
        if not ray.is_initialized():
            raise RuntimeError("Ray initialization failed")
        return True
    except Exception as e:
        print(f"Failed to connect to cluster at {cluster_address}: {e}", file=sys.stderr)
        sys.exit(1)

# Use in your script
connect_to_cluster()
```

### Connection Verification

After connecting, verify the connection and check available resources:

```python
import ray

# Check if connected
if not ray.is_initialized():
    raise RuntimeError("Not connected to Ray cluster")

# Get cluster information
cluster_resources = ray.cluster_resources()
available_resources = ray.available_resources()

print(f"Cluster resources: {cluster_resources}")
print(f"Available resources: {available_resources}")

# Get GCS address
from ray._private.services import get_node_ip_address
node_ip = get_node_ip_address()
print(f"Connected to node: {node_ip}")
```

**References:**
- [Configuring Ray](resources/ray-core/configure.md)
- [Core APIs - ray.init()](resources/ray-core/api/core-apis.md)

---

## 3. Conversion Workflow (Step-by-Step)

### Step 1: Analyze the Script

Before converting, analyze your script to identify:

1. **Parallelizable Operations**
   - Loops that can run independently
   - Data processing that can be split
   - Independent function calls
   - Batch operations

2. **Stateful vs Stateless Components**
   - Stateless: Functions that don't modify external state → Use `@ray.remote` tasks
   - Stateful: Classes/objects that maintain state → Use `@ray.remote` actors

3. **Resource Requirements**
   - CPU-intensive operations → Specify `num_cpus`
   - GPU operations → Specify `num_gpus`
   - Memory requirements → Specify `memory` resource
   - Custom resources needed

4. **Data Dependencies and Flow**
   - Data that needs to be shared → Use `ray.put()` and object refs
   - Large data passed to multiple tasks → Use object refs to avoid duplication
   - Data locality requirements

**Note**: All operations will run on the cluster, not locally. Consider network latency and data transfer costs.

**Example Analysis:**

```python
# Original script
def process_file(filename):
    data = load_file(filename)
    result = expensive_computation(data)
    return result

results = []
for filename in file_list:
    results.append(process_file(filename))  # Can be parallelized
```

**Analysis:**
- ✅ Parallelizable: Each `process_file` call is independent
- ✅ Stateless: Function doesn't maintain state
- ✅ Resource needs: May need CPU/memory per task
- ✅ Data: Each file is independent

### Step 2: Choose Ray Primitive

Use this decision tree to choose the right Ray primitive:

```
Is your workload...
├─ Data processing/batch inference?
│  └─ Use Ray Data → resources/data/
├─ Distributed training?
│  └─ Use Ray Train → resources/train/
├─ Hyperparameter tuning?
│  └─ Use Ray Tune → resources/tune/
├─ Model serving?
│  └─ Use Ray Serve → resources/serve/
├─ Reinforcement learning?
│  └─ Use Ray RLlib → resources/rllib.md
└─ Generic distributed computing?
   ├─ Need to maintain state?
   │  └─ Use @ray.remote class (Actor) → resources/ray-core/actors.md
   └─ Stateless operations?
      └─ Use @ray.remote function (Task) → resources/ray-core/tasks.md
```

#### When to Use Tasks (`@ray.remote` functions)

- Stateless operations
- Independent function calls
- Embarrassingly parallel workloads
- One-time computations

```python
import ray
import os

ray.init(address=os.getenv("RAY_ADDRESS"))

@ray.remote
def process_item(item):
    # Stateless processing
    return item * 2

items = [1, 2, 3, 4, 5]
results = ray.get([process_item.remote(i) for i in items])
```

**References:**
- [Getting Started](resources/ray-core/getting-started.md)
- [Tasks](resources/ray-core/tasks.md)

#### When to Use Actors (`@ray.remote` classes)

- Stateful operations
- Services that maintain state
- Shared resources (models, connections)
- Sequential operations on shared state

```python
import ray
import os

ray.init(address=os.getenv("RAY_ADDRESS"))

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value

counter = Counter.remote()
results = [counter.increment.remote() for _ in range(10)]
```

**References:**
- [Actors](resources/ray-core/actors.md)

#### When to Use Ray Libraries

- **Ray Data**: Batch data processing, ETL pipelines
- **Ray Train**: Distributed ML training (PyTorch, TensorFlow, etc.)
- **Ray Tune**: Hyperparameter optimization
- **Ray Serve**: Model serving and APIs
- **Ray RLlib**: Reinforcement learning

See [Reference Sections by Use Case](#4-reference-sections-by-use-case) for details.

### Step 3: Apply Ray Patterns

Apply these patterns for optimal cluster performance:

#### Resource Specification Patterns

Always specify resource requirements for cluster scheduling:

```python
@ray.remote(num_cpus=2, num_gpus=1)
def gpu_task(data):
    # GPU computation
    pass

@ray.remote(num_cpus=4, memory=8 * 1024 * 1024 * 1024)  # 8GB
def memory_intensive_task(data):
    # Memory-intensive operation
    pass
```

**References:**
- [Scheduling](resources/ray-core/scheduling.md)
- [Pattern: Limit Running Tasks](resources/ray-core/patterns/limit-running-tasks.md)

#### Object Reference Patterns

Use `ray.put()` for large objects shared across tasks:

```python
import ray
import numpy as np

# Large shared data
large_data = np.random.rand(10000, 10000)

# Put in object store once
data_ref = ray.put(large_data)

# Pass reference to multiple tasks (avoids duplication)
@ray.remote
def process(data_ref, index):
    data = ray.get(data_ref)  # Fetch from object store
    return data[index].sum()

results = [process.remote(data_ref, i) for i in range(100)]
```

**References:**
- [Objects](resources/ray-core/objects.md)
- [Pattern: Pass Large Arg By Value](resources/ray-core/patterns/pass-large-arg-by-value.md)
- [Anti-Pattern: Closure Capture](resources/ray-core/patterns/closure-capture-large-objects.md)

#### Scheduling and Placement Patterns

Use placement groups and scheduling strategies for cluster-aware placement:

```python
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Create placement group
pg = placement_group([{"CPU": 4}, {"GPU": 1}])
ray.get(pg.ready())

@ray.remote(
    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
)
def task_with_placement():
    pass
```

**References:**
- [Scheduling](resources/ray-core/scheduling.md)

#### Fault Tolerance Patterns

Handle failures gracefully:

```python
@ray.remote(max_retries=3)
def unreliable_task():
    # May fail, will retry up to 3 times
    pass

# Check for exceptions
try:
    result = ray.get(unreliable_task.remote())
except Exception as e:
    print(f"Task failed: {e}")
```

**References:**
- [Fault Tolerance](resources/ray-core/fault-tolerance.md)

**All Patterns:**
- [Pattern: Limit Running Tasks](resources/ray-core/patterns/limit-running-tasks.md)
- [Pattern: Limit Pending Tasks](resources/ray-core/patterns/limit-pending-tasks.md)
- [Anti-Pattern: Pass Large Arg By Value](resources/ray-core/patterns/pass-large-arg-by-value.md)
- [Anti-Pattern: Closure Capture](resources/ray-core/patterns/closure-capture-large-objects.md)

### Step 4: Optimize for Scale

Optimize your code for cluster-scale execution:

#### Memory Management

- Use `ray.put()` for large shared objects
- Avoid closure capture of large objects
- Use object refs instead of passing large values

```python
# ❌ BAD: Large object in closure
large_data = np.random.rand(1000000)

@ray.remote
def task():
    return len(large_data)  # large_data serialized with task!

# ✅ GOOD: Pass via object ref
large_data_ref = ray.put(np.random.rand(1000000))

@ray.remote
def task(data_ref):
    data = ray.get(data_ref)
    return len(data)
```

**References:**
- [Anti-Pattern: Closure Capture](resources/ray-core/patterns/closure-capture-large-objects.md)
- [Anti-Pattern: Pass Large Arg By Value](resources/ray-core/patterns/pass-large-arg-by-value.md)

#### Concurrency Limits

Limit concurrent tasks to prevent OOM:

```python
# Limit running tasks using resources
@ray.remote(memory=2 * 1024 * 1024 * 1024)  # 2GB per task
def memory_intensive_task(data):
    pass

# Limit pending tasks using ray.wait
MAX_PENDING = 100
pending = []
for item in items:
    if len(pending) >= MAX_PENDING:
        ready, pending = ray.wait(pending, num_returns=1)
        ray.get(ready)
    pending.append(memory_intensive_task.remote(item))
```

**References:**
- [Pattern: Limit Running Tasks](resources/ray-core/patterns/limit-running-tasks.md)
- [Pattern: Limit Pending Tasks](resources/ray-core/patterns/limit-pending-tasks.md)

#### Resource Allocation Strategies

Request appropriate cluster resources:

```python
# CPU-bound tasks
@ray.remote(num_cpus=4)
def cpu_task():
    pass

# GPU tasks
@ray.remote(num_gpus=1)
def gpu_task():
    pass

# Custom resources
@ray.remote(resources={"custom_resource": 1})
def custom_task():
    pass
```

**References:**
- All pattern files in [resources/ray-core/patterns/](resources/ray-core/patterns/)

### Step 5: Handle Dependencies & Runtime

Configure runtime environment for cluster workers:

#### Runtime Environment via Environment Variables

```python
import os
import json
import ray

# Parse runtime environment from env var
runtime_env_json = os.getenv("RAY_RUNTIME_ENV")
if runtime_env_json:
    runtime_env = json.loads(runtime_env_json)
else:
    runtime_env = {
        "pip": ["numpy", "pandas"],
        "env_vars": {"MY_VAR": "value"}
    }

ray.init(
    address=os.getenv("RAY_ADDRESS"),
    runtime_env=runtime_env
)
```

#### Dependency Management

Specify dependencies that cluster workers need:

```python
runtime_env = {
    "pip": ["torch", "transformers", "datasets"],
    "conda": ["pytorch"],
    "env_vars": {
        "HF_HOME": "/tmp/huggingface",
        "TRANSFORMERS_CACHE": "/tmp/transformers"
    },
    "working_dir": "https://github.com/user/repo/archive/main.zip"
}

ray.init(address=os.getenv("RAY_ADDRESS"), runtime_env=runtime_env)
```

**References:**
- [Handling Dependencies](resources/ray-core/handling-dependencies.md)
- [Runtime Environment APIs](resources/ray-core/api/runtime-env-apis.md)

### Step 6: Cluster Deployment & Job Submission

#### Using Ray Jobs API

Submit jobs to cluster using environment variables:

```python
import os
from ray.job_submission import JobSubmissionClient

# Get cluster address from environment
cluster_address = os.getenv("RAY_ADDRESS", "http://localhost:8265")

# Parse runtime environment
import json
runtime_env = json.loads(os.getenv("RAY_RUNTIME_ENV", "{}"))

# Submit job
client = JobSubmissionClient(cluster_address)
job_id = client.submit_job(
    entrypoint="python my_script.py",
    runtime_env=runtime_env
)

print(f"Submitted job {job_id}")
```

#### Job Submission with Environment Variables

Set environment variables before submitting:

```bash
export RAY_ADDRESS="http://cluster-head:8265"
export RAY_RUNTIME_ENV='{"pip": ["numpy"], "env_vars": {"MY_VAR": "value"}}'
python submit_job.py
```

**References:**
- [Job Submission](resources/cluster/running-applications/job-submission.md)
- [Kubernetes Deployment](resources/cluster/kubernetes.md)

---

## 4. Reference Sections by Use Case

### 4.1 Data Processing & Batch Inference

Use Ray Data for scalable data processing and batch inference.

**Cluster Connection Example:**

```python
import os
import ray
import ray.data

ray.init(address=os.getenv("RAY_ADDRESS"))

# Create dataset
ds = ray.data.range(1000)

# Process in parallel on cluster
ds = ds.map(lambda x: x * 2)

# Batch inference
@ray.remote
class Model:
    def __init__(self):
        # Load model
        pass
    
    def predict(self, batch):
        # Inference
        return batch

predictions = ds.map_batches(
    Model,
    compute=ray.data.ActorPoolStrategy(size=4)
)
```

**References:**
- [Data Examples](resources/data/examples.md)
- [Batch Inference Examples](resources/data/examples/batch_inference_object_detection.md)
- [PyTorch Batch Prediction](resources/data/examples/pytorch_resnet_batch_prediction.md)
- [HuggingFace Batch Prediction](resources/data/examples/huggingface_vit_batch_prediction.md)

### 4.2 Distributed Training

Use Ray Train for distributed ML training across cluster nodes.

**Cluster Connection Example:**

```python
import os
import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

ray.init(address=os.getenv("RAY_ADDRESS"))

def train_func():
    # Training logic
    pass

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(
        num_workers=int(os.getenv("NUM_WORKERS", "4")),
        use_gpu=os.getenv("USE_GPU", "false").lower() == "true"
    )
)

result = trainer.fit()
```

**References:**
- [Train Examples](resources/train/examples.md)
- [PyTorch Examples](resources/train/examples/pytorch/)
- [Lightning Examples](resources/train/examples/lightning/)
- [Transformers Examples](resources/train/examples/transformers/)
- [DeepSpeed Examples](resources/train/examples/deepspeed/)

### 4.3 Hyperparameter Tuning

Use Ray Tune for distributed hyperparameter optimization.

**Cluster Connection Example:**

```python
import os
import ray
from ray import tune
from ray.train import ScalingConfig

ray.init(address=os.getenv("RAY_ADDRESS"))

def trainable(config):
    # Training with hyperparameters
    return {"loss": config["lr"] ** 2}

tuner = tune.Tuner(
    trainable,
    param_space={"lr": tune.grid_search([0.1, 0.01, 0.001])},
    tune_config=tune.TuneConfig(
        num_samples=int(os.getenv("NUM_SAMPLES", "10"))
    )
)

results = tuner.fit()
```

**References:**
- [Tune Guide](resources/tune.md)
- [Tune Examples](resources/tune/examples.md)
- [PBT Guide](resources/tune/examples/pbt_guide.md)
- [Integration Examples](resources/tune/examples/) (W&B, MLflow, etc.)

### 4.4 Model Serving

Use Ray Serve for scalable model serving on clusters.

**Cluster Connection Example:**

```python
import os
from ray import serve
from fastapi import FastAPI

ray.init(address=os.getenv("RAY_ADDRESS"))

app = FastAPI()

@serve.deployment(
    num_replicas=int(os.getenv("NUM_REPLICAS", "2")),
    ray_actor_options={"num_gpus": 1 if os.getenv("USE_GPU") == "true" else 0}
)
@serve.ingress(app)
class ModelDeployment:
    def __init__(self):
        # Load model
        pass
    
    @app.post("/predict")
    def predict(self, data):
        # Inference
        return {"result": "prediction"}

serve.run(ModelDeployment.bind())
```

**References:**
- [Serve Guide](resources/serve.md)
- [Serve Examples](resources/serve/examples.md)

### 4.5 Reinforcement Learning

Use Ray RLlib for distributed reinforcement learning.

**Cluster Connection Example:**

```python
import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig

ray.init(address=os.getenv("RAY_ADDRESS"))

config = (
    PPOConfig()
    .environment(env="CartPole-v1")
    .rollouts(num_rollout_workers=int(os.getenv("NUM_WORKERS", "4")))
)

algo = config.build()
algo.train()
```

**References:**
- [RLlib Guide](resources/rllib.md)

### 4.6 Generic Distributed Computing

Use Ray Core primitives (tasks, actors, objects) for custom distributed workloads.

**Cluster Connection Example:**

```python
import os
import ray

ray.init(address=os.getenv("RAY_ADDRESS"))

@ray.remote(num_cpus=2)
def compute_task(data):
    # Distributed computation
    return result

results = ray.get([compute_task.remote(i) for i in range(100)])
```

**References:**
- [Getting Started](resources/ray-core/getting-started.md)
- [Tasks](resources/ray-core/tasks.md)
- [Actors](resources/ray-core/actors.md)
- [Objects](resources/ray-core/objects.md)
- [Scheduling](resources/ray-core/scheduling.md)
- [Fault Tolerance](resources/ray-core/fault-tolerance.md)

---

## 5. Design Patterns & Anti-Patterns

### Patterns (Best Practices)

1. **Limit Running Tasks** - Use resources to control concurrency
   - [Pattern: Limit Running Tasks](resources/ray-core/patterns/limit-running-tasks.md)
   - Use when: Tasks are memory-intensive or cause interference

2. **Limit Pending Tasks** - Use `ray.wait()` for backpressure
   - [Pattern: Limit Pending Tasks](resources/ray-core/patterns/limit-pending-tasks.md)
   - Use when: Submitting tasks faster than they complete

### Anti-Patterns (Common Mistakes)

1. **Pass Large Arguments By Value** - Use `ray.put()` instead
   - [Anti-Pattern: Pass Large Arg By Value](resources/ray-core/patterns/pass-large-arg-by-value.md)
   - Problem: Duplicates large data in object store
   - Solution: Use `ray.put()` and pass object refs

2. **Closure Capture of Large Objects** - Pass via arguments
   - [Anti-Pattern: Closure Capture](resources/ray-core/patterns/closure-capture-large-objects.md)
   - Problem: Large objects serialized with task definition
   - Solution: Pass large objects via `ray.put()` and arguments

### Cluster-Specific Considerations

- **Network Latency**: Minimize data transfer between nodes
- **Resource Availability**: Check cluster resources before requesting
- **Fault Tolerance**: Handle node failures gracefully
- **Data Locality**: Consider data placement for performance

**All Pattern Files:**
- [resources/ray-core/patterns/](resources/ray-core/patterns/)

---

## 6. API Reference Quick Links

### Core APIs

- [Core APIs](resources/ray-core/api/core-apis.md) - `ray.init()`, `ray.get()`, `ray.put()`, `ray.wait()`
- [Actor APIs](resources/ray-core/api/actor-apis.md) - Actor management
- [Utility APIs](resources/ray-core/api/utility-apis.md) - Utility functions

### Runtime & Configuration

- [Runtime Environment APIs](resources/ray-core/api/runtime-env-apis.md) - Runtime environment configuration
- [Runtime Context APIs](resources/ray-core/api/runtime-context-apis.md) - Runtime context access
- [Job Config APIs](resources/ray-core/api/job-config-apis.md) - Job configuration
- [Logging APIs](resources/ray-core/api/logging-apis.md) - Logging configuration

### Advanced Features

- [Placement Group APIs](resources/ray-core/api/placement-group-apis.md) - Resource placement
- [Cross-Language APIs](resources/ray-core/api/cross-language-apis.md) - Java/other languages
- [Compiled Graph API](resources/ray-core/compiled-graph/api-reference.md) - Ray DAG compilation
- [Ray DAG](resources/ray-core/ray-dag.md) - Dynamic task graphs

### Complete API Reference

- [API Reference](resources/ray-core/api-reference.md) - Complete API documentation

---

## 7. Examples Index

### Ray Core Examples

- [Getting Started Examples](resources/ray-core/getting-started.md)
- [Task Examples](resources/ray-core/tasks.md)
- [Actor Examples](resources/ray-core/actors.md)
- [Object Examples](resources/ray-core/objects.md)

### Ray Data Examples

- [Data Examples Overview](resources/data/examples.md)
- [Batch Inference - Object Detection](resources/data/examples/batch_inference_object_detection.md)
- [PyTorch ResNet Batch Prediction](resources/data/examples/pytorch_resnet_batch_prediction.md)
- [HuggingFace ViT Batch Prediction](resources/data/examples/huggingface_vit_batch_prediction.md)

### Ray Train Examples

- [Train Examples Overview](resources/train/examples.md)
- [PyTorch Examples](resources/train/examples/pytorch/)
- [Lightning Examples](resources/train/examples/lightning/)
- [Transformers Examples](resources/train/examples/transformers/)
- [DeepSpeed Examples](resources/train/examples/deepspeed/)
- [Horovod Examples](resources/train/examples/horovod/)
- [TensorFlow Examples](resources/train/examples/tf/)

### Ray Tune Examples

- [Tune Examples Overview](resources/tune/examples.md)
- [PBT Guide](resources/tune/examples/pbt_guide.md)
- [Integration Examples](resources/tune/examples/) (W&B, MLflow, Comet, Aim)
- [Framework Examples](resources/tune/examples/) (PyTorch, Keras, XGBoost)

### Ray Serve Examples

- [Serve Examples](resources/serve/examples.md)

### LLM Examples

- [VLLM with LoRA](resources/llm/examples/batch/vllm-with-lora.md)
- [VLLM with Structural Output](resources/llm/examples/batch/vllm-with-structural-output.md)

### Cluster Examples

- [Kubernetes Deployment](resources/cluster/kubernetes.md)
- [Job Submission](resources/cluster/running-applications/job-submission.md)

---

## 8. Optimization Checklist

### Pre-Conversion Analysis

- [ ] Identify all parallelizable operations
- [ ] Determine stateful vs stateless components
- [ ] Map resource requirements (CPU/GPU/memory)
- [ ] Identify data dependencies and sharing needs
- [ ] Check cluster resource availability
- [ ] Verify cluster connection (RAY_ADDRESS set)

### Conversion Steps

- [ ] Set up cluster connection via `RAY_ADDRESS`
- [ ] Choose appropriate Ray primitive (Task/Actor/Library)
- [ ] Apply resource specifications
- [ ] Use object refs for large shared data
- [ ] Configure runtime environment
- [ ] Add error handling and retries

### Post-Conversion Optimization

- [ ] Verify cluster connection works
- [ ] Check resource utilization
- [ ] Optimize memory usage (use `ray.put()` for large objects)
- [ ] Set concurrency limits if needed
- [ ] Add monitoring/logging
- [ ] Test fault tolerance
- [ ] Profile and optimize hot paths

### Performance Tuning

- [ ] Monitor cluster resource usage
- [ ] Adjust resource requests based on actual usage
- [ ] Optimize data transfer (minimize object store transfers)
- [ ] Use placement groups for data locality
- [ ] Tune concurrency limits
- [ ] Profile task execution times

### Cluster Resource Optimization

- [ ] Request appropriate resources (not too much, not too little)
- [ ] Use custom resources for specialized hardware
- [ ] Consider data locality when scheduling
- [ ] Monitor cluster-wide resource utilization
- [ ] Scale cluster if needed

---

## 9. Environment Variables Reference

### Cluster Connection Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RAY_ADDRESS` | Cluster address (required) | `ray://head-node:10001` or `http://head-node:8265` |
| `RAY_NAMESPACE` | Logical grouping namespace (optional) | `production` or `development` |

### Runtime Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RAY_RUNTIME_ENV` | JSON string for runtime environment | `{"pip": ["numpy"], "env_vars": {"VAR": "value"}}` |
| `RAY_JOB_CONFIG` | JSON string for job configuration | `{"runtime_env": {"pip": ["requests"]}}` |

### Fault Tolerance Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RAY_gcs_rpc_server_reconnect_timeout_s` | GCS reconnection timeout (seconds) | `60` |

### Task Configuration Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RAY_TASK_MAX_RETRIES` | Maximum task retries | `3` (default) or `0` to disable |

### Debugging Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RAY_PICKLE_VERBOSE_DEBUG` | Enable pickle debugging | `0` (off), `1`, or `2` |

### Other Ray Environment Variables

Ray may use additional environment variables for configuration. Check Ray documentation for the complete list.

**References:**
- [Configuring Ray](resources/ray-core/configure.md)
- [Core APIs](resources/ray-core/api/core-apis.md)
- [Runtime Environment APIs](resources/ray-core/api/runtime-env-apis.md)

---

## Quick Reference: Conversion Template

Use this template as a starting point for converting scripts:

```python
import os
import ray

# 1. Connect to cluster
cluster_address = os.getenv("RAY_ADDRESS", "auto")
ray.init(address=cluster_address)

# 2. Verify connection
if not ray.is_initialized():
    raise RuntimeError("Failed to connect to Ray cluster")

# 3. Convert your functions to Ray tasks/actors
@ray.remote(num_cpus=1)
def your_function(arg):
    # Your logic here
    return result

# 4. Use Ray primitives
results = ray.get([your_function.remote(arg) for arg in args])

# 5. Cleanup
ray.shutdown()
```

---

## Additional Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ray Core Patterns](https://docs.ray.io/en/latest/ray-core/patterns/index.html)
- [Ray GitHub](https://github.com/ray-project/ray)

---

**Remember**: Always connect to a cluster via `RAY_ADDRESS`. Never assume local Ray unless explicitly required.

