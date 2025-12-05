# Scheduling

Ray decides how to schedule tasks and actors to nodes based on resource requirements, scheduling strategies, and data locality.

## Resources

Each task or actor has specified resource requirements. Given that, a node can be in one of the following states:

- **Feasible**: the node has the required resources to run the task or actor
  - **Available**: the node has the required resources and they are free now
  - **Unavailable**: the node has the required resources but they are currently being used
- **Infeasible**: the node doesn't have the required resources (e.g., a CPU-only node is infeasible for a GPU task)

Resource requirements are **hard** requirements meaning that only feasible nodes are eligible. If there are feasible nodes, Ray will either choose an available node or wait until an unavailable node becomes available. If all nodes are infeasible, the task or actor cannot be scheduled until feasible nodes are added to the cluster.

### Specifying Resource Requirements

You can specify resource requirements for tasks and actors:

```python
# Specify the default resource requirements for this remote function.
@ray.remote(num_cpus=2, num_gpus=2, resources={"special_hardware": 1})
def func():
    return 1

# You can override the default resource requirements.
func.options(num_cpus=3, num_gpus=1, resources={"special_hardware": 0}).remote()

@ray.remote(num_cpus=0, num_gpus=1)
class Actor:
    pass

# You can override the default resource requirements for actors as well.
actor = Actor.options(num_cpus=1, num_gpus=0).remote()
```python

### Fractional Resource Requirements

Ray supports fractional resource requirements. For example, if your task or actor is IO bound and has low CPU usage, you can specify fractional CPU `num_cpus=0.5` or even zero CPU `num_cpus=0`:

```python
@ray.remote(num_cpus=0.5)
def io_bound_task():
    import time
    time.sleep(1)
    return 2

io_bound_task.remote()
```

**Note:** GPU, TPU, and neuron_cores resource requirements that are greater than 1 need to be whole numbers. For example, `num_gpus=1.5` is invalid.

### Physical vs Logical Resources

Ray resources are **logical** and don't need to have 1-to-1 mapping with physical resources. They are mainly used for admission control during scheduling.

**Important implications:**
- Resource requirements of tasks or actors do NOT impose limits on actual physical resource usage
- Ray doesn't provide CPU isolation for tasks or actors
- Ray does provide GPU isolation in the form of visible devices by automatically setting the `CUDA_VISIBLE_DEVICES` environment variable

Ray sets the environment variable `OMP_NUM_THREADS=<num_cpus>` if `num_cpus` is set on the task/actor. Ray sets `OMP_NUM_THREADS=1` if `num_cpus` is not specified.

## Labels

Labels provide a simplified solution for controlling scheduling for tasks, actors, and placement group bundles using default and custom labels.

**Default node labels:**
- `ray.io/node-id`: A unique ID generated for the node
- `ray.io/accelerator-type`: The accelerator type of the node (e.g., `L4`, `H100`)

**Specifying label selectors:**

```python
# Specify label_selector in task's @ray.remote annotation
@ray.remote(label_selector={"label_name":"label_value"})
def f():
    pass

# Specify label_selector in actor's @ray.remote annotation
@ray.remote(label_selector={"ray.io/accelerator-type": "H100"})
class Actor:
    pass

# Specify label_selector in task's options
@ray.remote
def test_task_label_in_options():
    pass

test_task_label_in_options.options(label_selector={"test-label-key": "test-label-value"}).remote()
```python

**Label selector operators:**
- Equals: `{"key": "value"}`
- Not equal: `{"key": "!value"}`
- In: `{"key": "in(val1,val2)"}`
- Not in: `{"key": "!in(val1,val2)"}`

## Scheduling Strategies

Tasks or actors support a `scheduling_strategy` option to specify the strategy used to decide the best node among feasible nodes.

### DEFAULT

`"DEFAULT"` is the default strategy used by Ray. Ray schedules tasks or actors onto a group of the top k nodes. Specifically, the nodes are sorted to first favor those that already have tasks or actors scheduled (for locality), then to favor those that have low resource utilization (for load balancing).

```python
@ray.remote
def func():
    return 1

@ray.remote(num_cpus=1)
class Actor:
    pass

# If unspecified, "DEFAULT" scheduling strategy is used.
func.remote()
actor = Actor.remote()

# Explicitly set scheduling strategy to "DEFAULT".
func.options(scheduling_strategy="DEFAULT").remote()
actor = Actor.options(scheduling_strategy="DEFAULT").remote()
```python

### SPREAD

`"SPREAD"` strategy will try to spread the tasks or actors among available nodes:

```python
@ray.remote(scheduling_strategy="SPREAD")
def spread_func():
    return 2

@ray.remote(num_cpus=1)
class SpreadActor:
    pass

# Spread tasks across the cluster.
[spread_func.remote() for _ in range(10)]
# Spread actors across the cluster.
actors = [SpreadActor.options(scheduling_strategy="SPREAD").remote() for _ in range(10)]
```python

### PlacementGroupSchedulingStrategy

`PlacementGroupSchedulingStrategy` will schedule the task or actor to where the placement group is located. This is useful for actor gang scheduling. See [Placement Groups](#placement-groups) for more details.

### NodeAffinitySchedulingStrategy

`NodeAffinitySchedulingStrategy` is a low-level strategy that allows a task or actor to be scheduled onto a particular node specified by its node id:

```python
@ray.remote
def node_affinity_func():
    return ray.get_runtime_context().get_node_id()

# Only run the task on the local node.
node_affinity_func.options(
    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=ray.get_runtime_context().get_node_id(),
        soft=False,
    )
).remote()
```python

**Note:** This strategy should only be used if other high level scheduling strategies cannot give the desired task or actor placements.

## Locality-Aware Scheduling

By default, Ray prefers available nodes that have large task arguments local to avoid transferring data over the network. If there are multiple large task arguments, the node with most object bytes local is preferred.

**Note:** Locality-aware scheduling is only for tasks, not actors.

```python
@ray.remote
def large_object_func():
    # Large object is stored in the local object store
    return [1] * (1024 * 1024)

@ray.remote
def consume_func(data):
    return len(data)

large_object = large_object_func.remote()

# Ray will try to run consume_func on the same node
# where large_object_func runs.
consume_func.remote(large_object)
```python

## Placement Groups

Placement groups allow users to atomically reserve groups of resources across multiple nodes (i.e., gang scheduling). They can then be used to schedule Ray tasks and actors packed together for locality (PACK), or spread apart (SPREAD).

### Key Concepts

**Bundles:** A bundle is a collection of resources. It could be a single resource, `{"CPU": 1}`, or a group of resources, `{"CPU": 1, "GPU": 4}`. A bundle must be able to fit on a single node.

**Placement Group:** A placement group reserves the resources from the cluster. The reserved resources can only be used by tasks or actors that use the PlacementGroupSchedulingStrategy.

### Creating a Placement Group

```python
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

ray.init(num_cpus=2, num_gpus=2)

# Reserve a placement group of 1 bundle that reserves 1 CPU and 1 GPU.
pg = placement_group([{"CPU": 1, "GPU": 1}])

# Wait until placement group is created.
ray.get(pg.ready(), timeout=10)
```python

### Placement Strategies

Ray supports four placement group strategies:

- **STRICT_PACK**: All bundles must be placed into a single node
- **PACK**: All provided bundles are packed onto a single node on a best-effort basis
- **STRICT_SPREAD**: Each bundle must be scheduled in a separate node
- **SPREAD**: Each bundle is spread onto separate nodes on a best-effort basis

```python
# Reserve a placement group of 2 bundles that have to be packed on the same node.
pg = placement_group([{"CPU": 1}, {"GPU": 1}], strategy="PACK")
```python

### Scheduling to Placement Groups

```python
@ray.remote(num_cpus=1)
class Actor:
    def __init__(self):
        pass
    def ready(self):
        pass

# Create an actor to a placement group.
actor = Actor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
    )
).remote()

# Verify the actor is scheduled.
ray.get(actor.ready.remote(), timeout=10)
```python

### Removing Placement Groups

```python
from ray.util.placement_group import remove_placement_group

# This API is asynchronous.
remove_placement_group(pg)
```

**Note:** When you remove the placement group, actors or tasks that still use the reserved resources are forcefully killed.

## Accelerator Support

Ray Core natively supports many accelerators as pre-defined resource types. The accelerators natively supported are:

- **NVIDIA GPU** (GPU) - Fully tested
- **AMD GPU** (GPU) - Experimental
- **Intel GPU** (GPU) - Experimental
- **AWS Neuron Core** (neuron_cores) - Experimental
- **Google TPU** (TPU) - Experimental
- **Intel Gaudi** (HPU) - Experimental
- **Huawei Ascend** (NPU) - Experimental
- **Rebellions RBLN** (RBLN) - Experimental

### Using Accelerators

If a task or actor requires accelerators, you can specify the corresponding resource requirements:

```python
import os
import ray

ray.init(num_gpus=2)

@ray.remote(num_gpus=1)
class GPUActor:
    def ping(self):
        print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

@ray.remote(num_gpus=1)
def gpu_task():
    print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

gpu_actor = GPUActor.remote()
ray.get(gpu_actor.ping.remote())
ray.get(gpu_task.remote())
```python

Ray automatically sets the corresponding environment variable (e.g., `CUDA_VISIBLE_DEVICES`) before running the task or actor code.

### Fractional Accelerators

Ray supports fractional resource requirements so multiple tasks and actors can share the same accelerator:

```python
ray.init(num_cpus=4, num_gpus=1)

@ray.remote(num_gpus=0.25)
def f():
    import time
    time.sleep(1)

# The four tasks created here can execute concurrently
# and share the same GPU.
ray.get([f.remote() for _ in range(4)])
```python

**Note:** It is the user's responsibility to make sure that the individual tasks don't use more than their share of the accelerator memory.

### Accelerator Types

Ray supports resource specific accelerator types. The `accelerator_type` option can be used to force a task or actor to run on a node with a specific type of accelerator:

```python
from ray.util.accelerators import NVIDIA_TESLA_V100

@ray.remote(num_gpus=1, accelerator_type=NVIDIA_TESLA_V100)
def train(data):
    return "This function was run on a node with a Tesla V100 GPU"

ray.get(train.remote(1))
```
