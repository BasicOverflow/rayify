# Utility APIs

Helper utilities for working with Ray tasks and scheduling.

## Task Utilities

### ray.util.as_completed()

Yield results as they become available.

```python
refs = [task.remote(i) for i in range(10)]
for result in ray.util.as_completed(refs, chunk_size=2):
    print(result)
```

**Parameters:**
- `refs`: List of Ray object refs
- `chunk_size`: Number of tasks to wait for in each iteration (default: 10)
- `yield_obj_refs`: If True, return ObjectRefs instead of results
- `timeout`: Maximum time to wait
- `fetch_local`: Whether to fetch to local store

**Note:** Use this instead of calling `ray.get()` in a loop.

### ray.util.map_unordered()

Apply a remote function to items and yield results as they complete.

```python
@ray.remote
def process(item):
    return item * 2

for result in ray.util.map_unordered(process, [1, 2, 3, 4, 5]):
    print(result)
```

**Parameters:**
- `fn`: Remote function to apply
- `items`: Iterable of items
- `backpressure_size`: Max in-flight tasks before blocking (default: 100)
- `chunk_size`: Number of tasks to wait for (default: 10)
- `yield_obj_refs`: If True, return ObjectRefs instead of results

Applies backpressure to limit pending tasks.

## Scheduling Strategies

### PlacementGroupSchedulingStrategy

Schedule tasks/actors using a placement group.

```python
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

task.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
        placement_group_capture_child_tasks=True
    )
).remote()
```

### NodeAffinitySchedulingStrategy

Schedule tasks/actors on a specific node.

```python
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

task.options(
    scheduling_strategy=NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=True  # Allow scheduling elsewhere if node unavailable
    )
).remote()
```

**Parameters:**
- `node_id`: Hex ID of target node
- `soft`: If True, schedule elsewhere if node unavailable; if False, fail

## Resource Utilities

### ray.get_gpu_ids()

Get IDs of GPUs available to the worker.

```python
@ray.remote(num_gpus=1)
def task():
    gpu_ids = ray.get_gpu_ids()
    return gpu_ids[0]  # Use first GPU
```

**Note:** Only call inside a task or actor, not in driver.

Returns list of GPU IDs (integers or strings depending on CUDA_VISIBLE_DEVICES).

