# Actor APIs

APIs for creating and managing Ray actors (stateful remote classes).

## Creating Actors

### @ray.remote

Decorator to create an actor class.

```python
@ray.remote
class MyActor:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value

actor = MyActor.remote(42)
```

### ActorClass.remote()

Create an actor instance.

```python
actor = MyActor.remote(*args, **kwargs)
```

Returns an `ActorHandle` to the newly created actor.

### ActorClass.options()

Configure actor instantiation parameters.

```python
Actor = MyActor.options(
    num_cpus=2,
    num_gpus=1,
    resources={"CustomResource": 1},
    max_restarts=3,
    name="my_actor",
    namespace="my_namespace",
    lifetime="detached"
)
actor = Actor.remote()
```

**Key Parameters:**
- `num_cpus`, `num_gpus`: Resource requirements
- `resources`: Custom resource dictionary
- `max_restarts`: Maximum restart attempts (0=no restart, -1=infinite)
- `max_task_retries`: Retry failed tasks (0=no retry, -1=infinite)
- `name`: Globally unique actor name for retrieval
- `namespace`: Actor namespace
- `lifetime`: "detached" for independent lifetime, None for fate-sharing
- `runtime_env`: Runtime environment configuration
- `scheduling_strategy`: Placement strategy

## Actor Methods

### @ray.method

Annotate an actor method with options.

```python
@ray.remote
class Actor:
    @ray.method(num_returns=2, max_task_retries=3)
    def method(self):
        return 1, 2
```

**Parameters:**
- `num_returns`: Number of return values (default: 1, use "streaming" for generators)
- `max_task_retries`: Retry count for failed tasks
- `retry_exceptions`: Retry on Python exceptions (bool or list of exceptions)
- `concurrency_group`: Concurrency group name for the method

### ActorMethod.options()

Configure method call options.

```python
result = actor.method.options(num_returns=2).remote()
```

## Actor Handles

### ActorHandle

A handle to an actor, returned by `ActorClass.remote()`.

```python
actor = MyActor.remote()
result = actor.method.remote()
value = ray.get(result)
```

Actor handles can be passed to other tasks or actors.

### ray.get_actor()

Get a handle to a named actor.

```python
actor = ray.get_actor("actor_name", namespace=None)
```

The actor must have been created with `Actor.options(name="actor_name")`.

### ray.kill()

Kill an actor forcefully.

```python
ray.kill(actor, no_restart=True)
```

Interrupts running tasks immediately. Use `actor.__ray_terminate__.remote()` to let pending tasks finish.

### ray.actor.exit_actor()

Exit the current actor from within the actor.

```python
@ray.remote
class Actor:
    def shutdown(self):
        ray.actor.exit_actor()
```

Can only be called from inside an actor.

