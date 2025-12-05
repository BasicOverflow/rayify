# API Reference

This document provides a reference to Ray Core APIs, organized by category. For detailed documentation, see the consolidated API files in the `api/` directory.

## Core APIs

Essential APIs for initialization, object management, and task control.

See [Core APIs](api/core-apis.md) for details on:
- `ray.init()`, `ray.shutdown()`, `ray.is_initialized()`
- `ray.get()`, `ray.put()`, `ray.wait()`
- `ray.cancel()`, `ray.get_actor()`, `ray.kill()`

## Actor APIs

APIs for creating and managing Ray actors.

See [Actor APIs](api/actor-apis.md) for details on:
- `@ray.remote` decorator
- `ActorClass.remote()`, `ActorClass.options()`
- `@ray.method`, `ActorMethod.options()`
- `ActorHandle`, `ray.get_actor()`, `ray.kill()`

## Runtime Context APIs

APIs for accessing runtime context information.

See [Runtime Context APIs](api/runtime-context-apis.md) for details on:
- `ray.runtime_context.get_runtime_context()`
- `RuntimeContext.get_job_id()`, `get_task_id()`, `get_actor_id()`
- `get_node_id()`, `get_assigned_resources()`, etc.

## Runtime Environment APIs

APIs for configuring runtime environments.

See [Runtime Environment APIs](api/runtime-env-apis.md) for details on:
- `RuntimeEnv` class
- `RuntimeEnvConfig` class

## Job Config APIs

APIs for configuring Ray jobs.

See [Job Config APIs](api/job-config-apis.md) for details on:
- `JobConfig` class and its methods

## Logging APIs

APIs for configuring logging.

See [Logging APIs](api/logging-apis.md) for details on:
- `LoggingConfig` class

## Placement Group APIs

APIs for creating and managing placement groups.

See [Placement Group APIs](api/placement-group-apis.md) for details on:
- `ray.util.placement_group()`
- `PlacementGroup` methods
- Placement group utilities

## Cross Language APIs

APIs for calling Java code from Python.

See [Cross Language APIs](api/cross-language-apis.md) for details on:
- `ray.cross_language.java_function()`
- `ray.cross_language.java_actor_class()`

## Utility APIs

Helper utilities for tasks and scheduling.

See [Utility APIs](api/utility-apis.md) for details on:
- `ray.util.as_completed()`, `ray.util.map_unordered()`
- Scheduling strategies
- `ray.get_gpu_ids()`

## Common Patterns

### Creating a Remote Function

```python
@ray.remote
def my_function(x):
    return x * 2

# Call the function
result = ray.get(my_function.remote(5))
```

### Creating an Actor

```python
@ray.remote
class MyActor:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value

# Create an actor
actor = MyActor.remote()
# Call actor method
result = ray.get(actor.increment.remote())
```

### Working with Object References

```python
# Put an object in the object store
obj_ref = ray.put([1, 2, 3])

# Get the object
value = ray.get(obj_ref)

# Wait for multiple objects
ready, remaining = ray.wait([ref1, ref2, ref3], num_returns=2)
```

### Resource Requirements

```python
# Specify resources for a task
@ray.remote(num_cpus=2, num_gpus=1)
def gpu_task():
    pass

# Override resources
gpu_task.options(num_cpus=4).remote()
```

### Fault Tolerance

```python
# Retry failed tasks
@ray.remote(max_retries=3, retry_exceptions=[ValueError])
def task():
    pass

# Restart failed actors
@ray.remote(max_restarts=3, max_task_retries=2)
class Actor:
    pass
```

## See Also

- [Tasks](tasks.md) - Detailed guide on using tasks
- [Actors](actors.md) - Detailed guide on using actors
- [Objects](objects.md) - Detailed guide on working with objects
- [Scheduling](scheduling.md) - Detailed guide on scheduling
- [Fault Tolerance](fault-tolerance.md) - Detailed guide on fault tolerance
