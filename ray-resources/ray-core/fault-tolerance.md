# Fault Tolerance

Ray is a distributed system, and failures can happen. Ray classifies failures into two classes:

1. **Application-level failures**: Bugs in user-level code or external system failures
2. **System-level failures**: Node failures, network failures, or bugs in Ray

## How to Write Fault Tolerant Ray Applications

### 1. Catch Exceptions Manually

If Ray's fault tolerance mechanisms don't work for you, you can always catch exceptions caused by failures and recover manually:

```python
@ray.remote
class Actor:
    def read_only(self):
        import sys
        import random
        rand = random.random()
        if rand < 0.2:
            return 2 / 0
        elif rand < 0.3:
            sys.exit(1)
        return 2

actor = Actor.remote()
# Manually retry the actor task.
while True:
    try:
        print(ray.get(actor.read_only.remote()))
        break
    except ZeroDivisionError:
        pass
    except ray.exceptions.RayActorError:
        # Manually restart the actor
        actor = Actor.remote()
```python

### 2. Avoid Outliving Object Owners

Avoid letting an `ObjectRef` outlive its owner task or actor. As long as there are still references to an object, the owner worker keeps running even after the corresponding task or actor finishes. If the owner worker fails, Ray cannot recover the object automatically.

**Non-fault tolerant version:**
```python
@ray.remote
def a():
    x_ref = ray.put(1)
    return x_ref

x_ref = ray.get(a.remote())
# Object x outlives its owner task A.
try:
    # If owner of x (i.e. the worker process running task A) dies,
    # the application can no longer get value of x.
    print(ray.get(x_ref))
except ray.exceptions.OwnerDiedError:
    pass
```python

**Fault tolerant version:**
```python
# Fault tolerant version:
@ray.remote
def a():
    # Here we return the value directly instead of calling ray.put() first.
    return 1

# The owner of x is the driver
# so x is accessible and can be auto recovered
# during the entire lifetime of the driver.
x_ref = a.remote()
print(ray.get(x_ref))
```python

### 3. Avoid Node-Specific Resource Requirements

Avoid using custom resource requirements that only particular nodes can satisfy. If that particular node fails, Ray won't retry the running tasks or actors.

Instead, use `NodeAffinitySchedulingStrategy` with `soft=True`:

```python
# Prefer running on the particular node specified by node id
# but can also run on other nodes if the target node fails.
b.options(
    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=ray.get_runtime_context().get_node_id(), soft=True
    )
).remote()
```python

## Task Fault Tolerance

### Catching Application-Level Failures

Ray surfaces application-level failures as Python-level exceptions. When a task fails due to a Python-level exception, Ray wraps the original exception in a `RayTaskError`:

```python
import ray

@ray.remote
def f():
    raise Exception("the real error")

try:
    ray.get(f.remote())
except ray.exceptions.RayTaskError as e:
    print(e)
    # ray::f() (pid=71867, ip=XXX.XX.XXX.XX)
    #   File "errors.py", line 5, in f
    #     raise Exception("the real error")
    # Exception: the real error
```python

If the user's exception type can be subclassed, the raised exception is an instance of both `RayTaskError` and the user's exception type. Otherwise, the actual user's exception type can be accessed via the `cause` field of the `RayTaskError`.

### Retrying Failed Tasks

When a worker dies unexpectedly, Ray will rerun the task until either the task succeeds or the maximum number of retries is exceeded. The default number of retries is 3 and can be overridden:

```python
@ray.remote(max_retries=1)
def potentially_fail(failure_probability):
    import time
    import os
    import numpy as np
    time.sleep(0.2)
    if np.random.random() < failure_probability:
        os._exit(0)
    return 0
```python

- `max_retries=-1`: Infinite retries
- `max_retries=0`: Disables retries
- `max_retries=N`: Retry up to N times

### Retrying on Application Exceptions

By default, Ray will **not** retry tasks upon exceptions thrown by application code. However, you can control whether application-level errors are retried via the `retry_exceptions` argument:

```python
@ray.remote(max_retries=1, retry_exceptions=True)
def potentially_fail(failure_probability):
    if np.random.random() < failure_probability:
        raise RandomError("Failed!")
    return 0

# Or specify specific exceptions to retry:
@ray.remote(max_retries=1, retry_exceptions=[RandomError])
def potentially_fail(failure_probability):
    # ...
```python

### Cancelling Misbehaving Tasks

If a task is hanging, you can cancel it by calling `ray.cancel` on an `ObjectRef`:

```python
obj_ref = long_running_task.remote()
ray.cancel(obj_ref)
```python

By default, this will send a KeyboardInterrupt to the task's worker. Passing `force=True` will force-exit the worker.

**Note:** Ray will not automatically retry tasks that have been cancelled.

## Actor Fault Tolerance

### Actor Process Failure

Ray can automatically restart actors that crash unexpectedly. This behavior is controlled using `max_restarts`:

```python
@ray.remote(max_restarts=4, max_task_retries=-1)
class Actor:
    def __init__(self):
        self.counter = 0

    def increment_and_possibly_fail(self):
        import os
        # Exit after every 10 tasks.
        if self.counter == 10:
            os._exit(0)
        self.counter += 1
        return self.counter

actor = Actor.remote()
# The actor will be reconstructed up to 4 times
for _ in range(50):
    counter = ray.get(actor.increment_and_possibly_fail.remote())
    print(counter)  # Prints the sequence 1-10 5 times.
```python

- `max_restarts=0`: Actor won't be restarted (default)
- `max_restarts=-1`: Actor will be restarted infinitely
- `max_restarts=N`: Actor will be restarted up to N times

### Actor Task Retries

By default, actor tasks execute with at-most-once semantics (`max_task_retries=0`). Ray also offers at-least-once execution semantics:

```python
@ray.remote(max_restarts=4, max_task_retries=-1)
class Actor:
    # ...
```python

- `max_task_retries=0`: At-most-once semantics (default)
- `max_task_retries=-1`: Infinite retries (at-least-once)
- `max_task_retries=N`: Retry up to N times

**Note:** For at-least-once actors, the system will still guarantee execution ordering according to the initial submission order.

### Actor Checkpointing

`max_restarts` automatically restarts the crashed actor, but it doesn't automatically restore application level state. You should manually checkpoint your actor's state and recover upon actor restart:

```python
@ray.remote(max_restarts=-1, max_task_retries=-1)
class ImmortalActor:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        import os
        import json
        
        if os.path.exists(self.checkpoint_file):
            # Restore from a checkpoint
            with open(self.checkpoint_file, "r") as f:
                self.state = json.load(f)
        else:
            self.state = {}

    def update(self, key, value):
        import sys
        import random
        if random.randrange(10) < 5:
            sys.exit(1)
        
        self.state[key] = value
        
        # Checkpoint the latest state
        with open(self.checkpoint_file, "w") as f:
            import json
            json.dump(self.state, f)

    def get(self, key):
        return self.state[key]
```python

**Important:** If the checkpoint is saved to external storage, make sure it's accessible to the entire cluster since the actor can be restarted on a different node.

### Actor Creator Failure

For non-detached actors, the owner of an actor is the worker that created it. If the owner dies, then the actor will also fate-share with the owner. Ray will not automatically recover an actor whose owner is dead, even if it has a nonzero `max_restarts`.

Since detached actors do not have an owner, they will still be restarted by Ray even if their original creator dies:

```python
@ray.remote(max_restarts=-1)
class Actor:
    def ping(self):
        return "hello"

# Detached actor survives owner death
detached_actor = Actor.options(name="actor", lifetime="detached").remote()
```python

### Actor Method Exceptions

You can retry when an actor method raises exceptions using `max_task_retries` with `retry_exceptions`:

```python
@ray.remote(max_task_retries=5)
class Actor:
    @ray.method(retry_exceptions=True)
    def method(self):
        # This method will be retried on any exception
        # ...
```python

- `retry_exceptions=False` (default): No retries for user exceptions
- `retry_exceptions=True`: Ray retries a method on any user exception
- `retry_exceptions=[Exception1, Exception2]`: Ray retries only on specific exceptions

## Object Fault Tolerance

A Ray object has both data (the value) and metadata (e.g., the location). Data is stored in the Ray object store while the metadata is stored at the object's **owner**. The owner of an object is the worker process that creates the original `ObjectRef`.

### Recovering from Data Loss

When an object value is lost from the object store, Ray will use **lineage reconstruction** to recover the object. Ray will first automatically attempt to recover the value by looking for copies on other nodes. If none are found, then Ray will automatically recover the value by re-executing the task that previously created the value.

**Limitations:**
- Objects created by `ray.put` are not recoverable
- Tasks are assumed to be deterministic and idempotent
- By default, objects created by actor tasks are not reconstructable (set `max_task_retries` to allow reconstruction)
- Tasks will only be re-executed up to their maximum number of retries
- The owner of the object must still be alive

To disable lineage reconstruction entirely, set the environment variable `RAY_TASK_MAX_RETRIES=0`. With this setting, if there are no copies of an object left, an `ObjectLostError` will be raised.

### Recovering from Owner Failure

Currently, **Ray does not support recovery from owner failure**. In this case, Ray will clean up any remaining copies of the object's value. Any workers that subsequently try to get the object's value will receive an `OwnerDiedError` exception.

### Understanding ObjectLostErrors

Ray throws an `ObjectLostError` when an object cannot be retrieved. Different error types indicate different root causes:

- **`OwnerDiedError`**: The owner of an object has died. The owner stores critical object metadata and an object cannot be retrieved if this process is lost.
- **`ObjectReconstructionFailedError`**: An object, or another object that this object depends on, cannot be reconstructed due to limitations.
- **`ReferenceCountingAssertionError`**: The object has already been deleted.
- **`ObjectFetchTimedOutError`**: A node timed out while trying to retrieve a copy of the object from a remote node.
- **`ObjectLostError`**: The object was successfully created, but no copy is reachable.

## GCS Fault Tolerance

The Global Control Service, or GCS, manages cluster-level metadata. It also provides a handful of cluster-level operations including actors, placement groups and node management. By default, the GCS isn't fault tolerant because it stores all data in memory. If it fails, the entire Ray cluster fails. To enable GCS fault tolerance, you need a highly available Redis instance, known as HA Redis. Then, when the GCS restarts, it loads all the data from the Redis instance and resumes regular functions.

During the recovery period, the following functions aren't available:

  * Actor creation, deletion and reconstruction.
  * Placement group creation, deletion and reconstruction.
  * Resource management.
  * Worker node registration.
  * Worker process creation.

However, running Ray tasks and actors remain alive, and any existing objects stay available.

### Setting up Redis

**KubeRay (officially supported)**

If you are using KubeRay, refer to KubeRay docs on GCS Fault Tolerance.

**ray start**

If you are using ray start to start the Ray head node, set the OS environment `RAY_REDIS_ADDRESS` to the Redis address, and supply the `--redis-password` flag with the password when calling `ray start`:

```bash
RAY_REDIS_ADDRESS=redis_ip:port ray start --head --redis-password PASSWORD --redis-username default
```

**ray up**

If you are using ray up to start the Ray cluster, change head_start_ray_commands field to add `RAY_REDIS_ADDRESS` and `--redis-password` to the `ray start` command:

```yaml
head_start_ray_commands:
  - ray stop
  - ulimit -n 65536; RAY_REDIS_ADDRESS=redis_ip:port ray start --head --redis-password PASSWORD --redis-username default --port=6379 --object-manager-port=8076 --autoscaling-config=~/ray_bootstrap_config.yaml --dashboard-host=0.0.0.0
```

After you back the GCS with Redis, it recovers its state from Redis when it restarts. While the GCS recovers, each raylet tries to reconnect to it. If a raylet can't reconnect for more than 60 seconds, that raylet exits and the corresponding node fails. Set this timeout threshold with the OS environment variable `RAY_gcs_rpc_server_reconnect_timeout_s`.

If the GCS IP address might change after restarts, use a qualified domain name and pass it to all raylets at start time. Each raylet resolves the domain name and connects to the correct GCS. You need to ensure that at any time, only one GCS is alive.

**Note:** GCS fault tolerance with external Redis is officially supported only if you are using KubeRay for Ray serve fault tolerance. For other cases, you can use it at your own risk and you need to implement additional mechanisms to detect the failure of GCS or the head node and restart it.

**Note:** You can also enable GCS fault tolerance when running Ray on Anyscale. See the Anyscale documentation for instructions.

## Node Fault Tolerance

A Ray cluster consists of one or more worker nodes, each of which consists of worker processes and system processes (e.g. raylet). One of the worker nodes is designated as the head node and has extra processes like the GCS.

Here, we describe node failures and their impact on tasks, actors, and objects.

### Worker node failure

When a worker node fails, all the running tasks and actors will fail and all the objects owned by worker processes of this node will be lost. In this case, the tasks, actors, objects fault tolerance mechanisms will kick in and try to recover the failures using other worker nodes.

### Head node failure

When a head node fails, the entire Ray cluster fails. To tolerate head node failures, we need to make GCS fault tolerant so that when we start a new head node we still have all the cluster-level data.

### Raylet failure

When a raylet process fails, the corresponding node will be marked as dead and is treated the same as a node failure. Each raylet is associated with a unique id, so even if the raylet restarts on the same physical machine, it'll be treated as a new raylet/node to the Ray cluster.
