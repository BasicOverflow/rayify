# Runtime Context APIs

APIs for accessing runtime context information about the current driver, task, or actor.

## Getting Runtime Context

### ray.runtime_context.get_runtime_context()

Get the runtime context of the current driver/worker.

```python
import ray
from ray import runtime_context

ctx = runtime_context.get_runtime_context()
job_id = ctx.get_job_id()
```

## Context Information

### Job and Worker Information

- `get_job_id()`: Get current job ID
- `get_worker_id()`: Get current worker ID
- `get_node_id()`: Get the ID of the node this process is running on
- `get_node_labels()`: Get node labels of the current worker

### Task Information

- `get_task_id()`: Get current task ID
- `get_task_name()`: Get current task name
- `get_task_function_name()`: Get current task function name

### Actor Information

- `get_actor_id()`: Get current actor ID (only in actor context)
- `get_actor_name()`: Get current actor name (only in actor context)
- `current_actor`: Get the current actor handle (only in actor context)
- `was_current_actor_reconstructed()`: Check if actor has been restarted

### Resource Information

- `get_assigned_resources()`: Get resources assigned to this worker
- `get_resource_ids()`: Get resource IDs
- `get_accelerator_ids()`: Get visible accelerator IDs
- `get_placement_group_id()`: Get current placement group ID
- `current_placement_group_id`: Get current placement group ID (property)

### Environment Information

- `namespace`: Get current namespace
- `runtime_env`: Get runtime environment used for current driver/worker
- `get_runtime_env_string()`: Get runtime environment string
- `gcs_address`: Get GCS address of the Ray cluster

### Utility Methods

- `get()`: Get a dictionary of the current context
- `should_capture_child_tasks_in_placement_group()`: Check if child tasks should capture parent's placement group

## Example

```python
import ray
from ray import runtime_context

@ray.remote
def task():
    ctx = runtime_context.get_runtime_context()
    return {
        "job_id": ctx.get_job_id(),
        "task_id": ctx.get_task_id(),
        "node_id": ctx.get_node_id()
    }

@ray.remote
class Actor:
    def get_context(self):
        ctx = runtime_context.get_runtime_context()
        return {
            "actor_id": ctx.get_actor_id(),
            "job_id": ctx.get_job_id(),
            "was_reconstructed": ctx.was_current_actor_reconstructed()
        }
```

