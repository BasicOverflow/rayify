# Actors

Actors extend the Ray API from functions (tasks) to classes. An actor is essentially a stateful worker (or a service). When you instantiate a new actor, Ray creates a new worker and schedules methods of the actor on that specific worker. The methods can access and mutate the state of that worker.

## Basic Usage

The `ray.remote` decorator indicates that instances of a class are actors. Each actor runs in its own Python process.

**Example:**

```python
import ray

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value

# Create an actor from this class.
counter = Counter.remote()

# Call the actor.
obj_ref = counter.increment.remote()
print(ray.get(obj_ref))
# -> 1
```python

## Key Behaviors

**Methods on different actors execute in parallel:**

```python
# Create ten Counter actors.
counters = [Counter.remote() for _ in range(10)]

# Increment each Counter once. These tasks all happen in parallel.
results = ray.get([c.increment.remote() for c in counters])
print(results)
# -> [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```python

**Methods on the same actor execute serially and share state:**

```python
# Increment the first Counter five times. These tasks are executed serially
# and share state.
results = ray.get([counters[0].increment.remote() for _ in range(5)])
print(results)
# -> [2, 3, 4, 5, 6]
```python

## Specifying Resource Requirements

Specify resource requirements for actors:

```python
# Specify required resources for an actor.
@ray.remote(num_cpus=2, num_gpus=0.5)
class Actor:
    pass
```python

## Passing Actor Handles

You can pass actor handles into other tasks and actor methods:

```python
import time

@ray.remote
def f(counter):
    for _ in range(10):
        time.sleep(0.1)
        counter.increment.remote()

counter = Counter.remote()

# Start some tasks that use the actor.
[f.remote(counter) for _ in range(3)]

# Print the counter value.
for _ in range(10):
    time.sleep(0.1)
    print(ray.get(counter.get_counter.remote()))
```python

## Named Actors

An actor can be given a unique name within their namespace. This allows you to retrieve the actor from any job in the Ray cluster.

**Creating a named actor:**

```python
# Create an actor with a name
counter = Counter.options(name="some_name").remote()

# Retrieve the actor later somewhere
counter = ray.get_actor("some_name")
```python

**Get-or-create pattern:**

```python
@ray.remote
class Greeter:
    def __init__(self, value):
        self.value = value

    def say_hello(self):
        return self.value

# Actor `g1` doesn't yet exist, so it is created with the given args.
a = Greeter.options(name="g1", get_if_exists=True).remote("Old Greeting")
assert ray.get(a.say_hello.remote()) == "Old Greeting"

# Actor `g1` already exists, so it is returned (new args are ignored).
b = Greeter.options(name="g1", get_if_exists=True).remote("New Greeting")
assert ray.get(b.say_hello.remote()) == "Old Greeting"
```python

**Detached actors:**

Actors can be decoupled from the job, allowing them to persist even after the driver process exits:

```python
counter = Counter.options(name="CounterActor", lifetime="detached").remote()
```python

The `CounterActor` will be kept alive even after the driver exits. You can retrieve it from a different driver:

```python
counter = ray.get_actor("CounterActor")
```python

**Note:** Named actors are scoped by namespace. If no namespace is assigned, they will be placed in an anonymous namespace by default.

## Terminating Actors

Actor processes will be terminated automatically when all copies of the actor handle have gone out of scope, or if the original creator process dies.

### Manual Termination via Actor Handle

You can forcefully terminate an actor using `ray.kill()`:

```python
actor_handle = Actor.remote()
ray.kill(actor_handle)
# Force kill: the actor exits immediately without cleanup.
# This will NOT call __ray_shutdown__() or atexit handlers.
```python

This causes the actor to immediately exit its process, causing any current, pending, and future tasks to fail with a `RayActorError`.

### Manual Termination Within the Actor

You can terminate an actor from within one of its methods:

```python
@ray.remote
class Actor:
    def exit(self):
        ray.actor.exit_actor()

actor = Actor.remote()
actor.exit.remote()
```python

This approach waits until any previously submitted tasks finish executing and then exits the process gracefully.

### Actor Cleanup with `__ray_shutdown__`

When an actor terminates gracefully, Ray calls the `__ray_shutdown__()` method if it exists, allowing cleanup of resources:

```python
import ray
import tempfile
import os

@ray.remote
class FileProcessorActor:
    def __init__(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b"processing data")
        self.temp_file.flush()

    def __ray_shutdown__(self):
        # Clean up temporary file
        if hasattr(self, 'temp_file'):
            self.temp_file.close()
            os.unlink(self.temp_file.name)

    def process(self):
        return "done"

actor = FileProcessorActor.remote()
ray.get(actor.process.remote())
del actor  # __ray_shutdown__() is called automatically
```python

**When `__ray_shutdown__()` is called:**
- Automatic termination: When all actor handles go out of scope
- Manual graceful termination: When you call `actor.__ray_terminate__.remote()`

**When `__ray_shutdown__()` is NOT called:**
- Force kill: When you use `ray.kill(actor)`
- Unexpected termination: When the actor process crashes

**Important notes:**
- `__ray_shutdown__()` runs after all actor tasks complete
- Ray waits 30 seconds for graceful shutdown by default
- Exceptions in `__ray_shutdown__()` are caught and logged
- `__ray_shutdown__()` must be a synchronous method

## AsyncIO for Actors

Ray natively integrates with asyncio. You can use Ray alongside popular async frameworks like aiohttp, aioredis, etc.

**Defining an async actor:**

```python
import ray
import asyncio

@ray.remote
class AsyncActor:
    def __init__(self, expected_num_tasks: int):
        self._event = asyncio.Event()
        self._curr_num_tasks = 0
        self._expected_num_tasks = expected_num_tasks

    # Multiple invocations of this method can run concurrently on the same event loop.
    async def run_concurrent(self):
        self._curr_num_tasks += 1
        if self._curr_num_tasks == self._expected_num_tasks:
            print("All coroutines are executing concurrently, unblocking.")
            self._event.set()
        else:
            print("Waiting for other coroutines to start.")
            await self._event.wait()
        print("All coroutines ran concurrently.")

actor = AsyncActor.remote(4)
refs = [actor.run_concurrent.remote() for _ in range(4)]
ray.get(refs)
```python

**ObjectRefs as asyncio.Futures:**

ObjectRefs can be translated to asyncio.Futures, making it possible to `await` on ray futures:

```python
import ray
import asyncio

@ray.remote
def some_task():
    return 1

async def await_obj_ref():
    await some_task.remote()
    await asyncio.wait([some_task.remote()])

asyncio.run(await_obj_ref())
```python

**Setting concurrency in async actors:**

You can set the number of concurrent tasks using the `max_concurrency` flag (default is 1000):

```python
@ray.remote
class AsyncActor:
    def __init__(self, batch_size: int):
        self._event = asyncio.Event()
        self._curr_tasks = 0
        self._batch_size = batch_size

    async def run_task(self):
        print("Started task")
        self._curr_tasks += 1
        if self._curr_tasks == self._batch_size:
            self._event.set()
        else:
            await self._event.wait()
            self._event.clear()
            self._curr_tasks = 0
        print("Finished task")

actor = AsyncActor.options(max_concurrency=2).remote(2)
# Only 2 tasks will run concurrently.
ray.get([actor.run_task.remote() for _ in range(8)])
```python

**Important:** Running blocking `ray.get` or `ray.wait` inside async actor methods is not allowed, because they will block the execution of the event loop.

## Threaded Actors

Sometimes, asyncio is not ideal for your actor. For example, you may have blocking computation that doesn't give up control via `await`. Instead, you can use `max_concurrency` without any async methods to achieve threaded concurrency (like a thread pool).

**Warning:** When there is at least one `async def` method in actor definition, Ray will recognize the actor as AsyncActor instead of ThreadedActor.

```python
@ray.remote
class ThreadedActor:
    def task_1(self): 
        print("I'm running in a thread!")
    def task_2(self): 
        print("I'm running in another thread!")

a = ThreadedActor.options(max_concurrency=2).remote()
ray.get([a.task_1.remote(), a.task_2.remote()])
```python

Each invocation of the threaded actor will be running in a thread pool. The size of the threadpool is limited by the `max_concurrency` value.

**Note:** Python's Global Interpreter Lock (GIL) will only allow one thread of Python code running at once. If you're just parallelizing Python code, you won't get true parallelism. However, libraries like Numpy, Cython, Tensorflow, or PyTorch will release the GIL when calling into C/C++ functions.

## Concurrency Groups

Besides setting the max concurrency overall for an actor, Ray allows methods to be separated into **concurrency groups**, each with its own thread(s). This allows you to limit the concurrency per-method.

**Example:**

```python
import ray

@ray.remote(concurrency_groups={"io": 2, "compute": 4})
class AsyncIOActor:
    def __init__(self):
        pass

    @ray.method(concurrency_group="io")
    async def f1(self):
        pass

    @ray.method(concurrency_group="io")
    async def f2(self):
        pass

    @ray.method(concurrency_group="compute")
    async def f3(self):
        pass

    @ray.method(concurrency_group="compute")
    async def f4(self):
        pass

    async def f5(self):
        pass  # Executed in the default group

a = AsyncIOActor.remote()
a.f1.remote()  # executed in the "io" group
a.f2.remote()  # executed in the "io" group
a.f3.remote()  # executed in the "compute" group
a.f4.remote()  # executed in the "compute" group
a.f5.remote()  # executed in the default group
```python

**Default concurrency group:**

By default, methods are placed in a default concurrency group which has a concurrency limit of 1000 for AsyncIO actors and 1 otherwise. The concurrency of the default group can be changed by setting the `max_concurrency` actor option:

```python
@ray.remote(concurrency_groups={"io": 2})
class AsyncIOActor:
    async def f1(self):
        pass

actor = AsyncIOActor.options(max_concurrency=10).remote()
```

**Setting concurrency group at runtime:**

You can dispatch actor methods into a specific concurrency group at runtime:

```python
# Executed in the "io" group (as defined in the actor class).
a.f2.options().remote()

# Executed in the "compute" group.
a.f2.options(concurrency_group="compute").remote()
```python

## Task Execution Order

### Synchronous, Single-Threaded Actor

For tasks received from the same submitter, a synchronous, single-threaded actor executes them in the order they were submitted:

```python
@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, addition):
        self.value += addition
        return self.value

counter = Counter.remote()

# For tasks from the same submitter, they are executed according to submission order.
value0 = counter.add.remote(1)
value1 = counter.add.remote(2)

# Output: 1. The first submitted task is executed first.
print(ray.get(value0))
# Output: 3. The later submitted task is executed later.
print(ray.get(value1))
```python

However, the actor does not guarantee the execution order of tasks from different submitters.

### Asynchronous or Threaded Actor

Asynchronous or threaded actors do not guarantee the task execution order. This means the system might execute a task even though previously submitted tasks are pending execution.

## Type Hints and Static Typing

Ray supports Python type hints for both remote functions and actors. To get the best type inference:

- **Prefer** `ray.remote(MyClass)` **over** `@ray.remote` **for actors**
- **Use** `@ray.method` **for actor methods**
- **Use the** `ActorClass` **and** `ActorProxy` **types**

**Example:**

```python
import ray
from ray.actor import ActorClass, ActorProxy

class Counter:
    def __init__(self):
        self.value = 0

    @ray.method
    def increment(self) -> int:
        self.value += 1
        return self.value

CounterActor: ActorClass[Counter] = ray.remote(Counter)
counter: ActorProxy[Counter] = CounterActor.remote()

# Type checkers and IDEs will now provide type hints for remote methods
obj_ref: ray.ObjectRef[int] = counter.increment.remote()
print(ray.get(obj_ref))
```python

## Generators

Ray is compatible with Python generator syntax. Generator tasks stream outputs back to the caller before the task finishes. See [Ray Generators](ray-generator.md) for more details.

## Cancelling Actor Tasks

Cancel actor tasks by calling `ray.cancel()` on the returned `ObjectRef`:

```python
import ray
import asyncio
import time

@ray.remote
class Actor:
    async def f(self):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            print("Actor task canceled.")

actor = Actor.remote()
ref = actor.f.remote()

# Wait until task is scheduled.
time.sleep(1)
ray.cancel(ref)

try:
    ray.get(ref)
except ray.exceptions.RayTaskError:
    print("Object reference was cancelled.")
```

**Task cancellation behavior:**
- **Unscheduled tasks**: Ray attempts to cancel the scheduling
- **Running actor tasks (regular/threaded)**: No mechanism for interruption
- **Running async actor tasks**: Ray seeks to cancel the associated `asyncio.Task`
- **Cancellation guarantee**: Ray attempts to cancel on a best-effort basis

## Scheduling

For each actor, Ray chooses a node to run it on based on:
- The actor's resource requirements
- The specified scheduling strategy

See [Scheduling](scheduling.md) for more details.

## Fault Tolerance

By default, Ray actors won't be restarted and actor tasks won't be retried when actors crash unexpectedly. You can change this behavior:

```python
@ray.remote(max_restarts=3, max_task_retries=2)
class Actor:
    pass
```

See [Fault Tolerance](fault-tolerance.md) for more details.

## FAQ: Actors, Workers and Resources

**What's the difference between a worker and an actor?**

Each "Ray worker" is a Python process.

- **Tasks**: When Ray starts on a machine, a number of Ray workers start automatically (1 per CPU by default). Ray uses them to execute tasks (like a process pool).
- **Actor**: A Ray Actor is also a "Ray worker" but you instantiate it at runtime with `actor_cls.remote()`. All of its methods run on the same process, using the same resources Ray designates when you define the Actor. Unlike tasks, Ray doesn't reuse the Python processes that run Ray Actors. Ray terminates them when you delete the Actor.

To maximally utilize your resources, you want to maximize the time that your workers work. You also want to allocate enough cluster resources so Ray can run all of your needed actors and any other tasks you define. This also implies that Ray schedules tasks more flexibly, and that if you don't need the stateful part of an actor, it's better to use tasks.

## Task Events

By default, Ray traces the execution of actor tasks, reporting task status events and profiling events that Ray Dashboard uses. You can disable this to reduce overhead:

```python
@ray.remote(enable_task_events=False)
class Actor:
    pass
```python

You can also disable task event reporting for specific actor methods:

```python
@ray.remote
class FooActor:
    # Disable task events reporting for this method.
    @ray.method(enable_task_events=False)
    def foo(self):
        pass
```
