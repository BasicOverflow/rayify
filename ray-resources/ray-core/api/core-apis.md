# Core APIs

Essential Ray Core APIs for initialization, object management, and task control.

## Initialization

### ray.init()

Connect to an existing Ray cluster or start one and connect to it.

```python
ray.init()
ray.init(address="auto")
ray.init(address="ray://123.45.67.89:10001")
```

**Key Parameters:**
- `address`: Cluster address. Use "auto" to connect to existing, "local" to start new, or "ray://host:port" for remote
- `num_cpus`: Number of CPUs to assign
- `num_gpus`: Number of GPUs to assign
- `resources`: Dictionary of custom resources
- `runtime_env`: Runtime environment configuration
- `namespace`: Logical grouping of jobs and named actors

### ray.shutdown()

Disconnect the worker and terminate processes started by ray.init().

```python
ray.shutdown()
```

### ray.is_initialized()

Check if ray.init has been called.

```python
if ray.is_initialized():
    # Ray is ready
```

## Object Management

### ray.get()

Get a remote object or list of remote objects from the object store. Blocks until the object is available.

```python
result = ray.get(object_ref)
results = ray.get([ref1, ref2, ref3])
```

**Parameters:**
- `object_refs`: ObjectRef or list of ObjectRefs
- `timeout`: Maximum time in seconds to wait (None blocks indefinitely)

**Note:** In async context, use `await object_ref` instead.

### ray.put()

Store an object in the object store.

```python
ref = ray.put(value)
```

**Parameters:**
- `value`: The Python object to store

The object won't be evicted while a reference exists.

### ray.wait()

Return lists of ready and unready object refs.

```python
ready, unready = ray.wait(object_refs, num_returns=1, timeout=None)
```

**Parameters:**
- `ray_waitables`: List of ObjectRef or ObjectRefGenerator
- `num_returns`: Number of objects to wait for (default: 1)
- `timeout`: Maximum time to wait (None waits indefinitely)
- `fetch_local`: Whether to fetch objects to local store (default: True)

**Returns:** Tuple of (ready_list, unready_list)

**Note:** In async context, use `await asyncio.wait(ray_waitables)` instead.

## Task Control

### ray.cancel()

Cancel a task.

```python
ray.cancel(object_ref, force=False, recursive=True)
```

**Parameters:**
- `ray_waitable`: ObjectRef or ObjectRefGenerator to cancel
- `force`: Force-kill running task (default: False)
- `recursive`: Cancel child tasks (default: True)

**Behavior:**
- Pending tasks: Cancelled immediately
- Running tasks: KeyboardInterrupt if force=False, immediate exit if force=True
- Actor tasks: Only async actor tasks can be interrupted

## Actor Management

### ray.get_actor()

Get a handle to a named actor.

```python
actor = ray.get_actor("actor_name", namespace=None)
```

**Parameters:**
- `name`: Name of the actor (must be created with `Actor.options(name="name")`)
- `namespace`: Namespace of the actor (None uses current namespace)

### ray.kill()

Kill an actor forcefully.

```python
ray.kill(actor, no_restart=True)
```

**Parameters:**
- `actor`: ActorHandle to kill
- `no_restart`: Prevent restart if actor is restartable (default: True)

Interrupts running tasks immediately. Use `actor.__ray_terminate__.remote()` to let pending tasks finish.

