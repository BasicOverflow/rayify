# Tasks

Ray enables arbitrary functions to be executed asynchronously on separate worker processes. Such functions are called **Ray remote functions** and their asynchronous invocations are called **Ray tasks**.

## Basic Usage

To create a task:

1. Decorate your function with `@ray.remote` to indicate it should run remotely
2. Call the function with `.remote()` instead of a normal function call
3. Use `ray.get()` to retrieve the result from the returned future (Ray _object reference_)

**Example:**

```python
import ray

# A regular Python function.
def normal_function():
    return 1

# By adding the `@ray.remote` decorator, a regular Python function
# becomes a Ray remote function.
@ray.remote
def my_function():
    return 1

# To invoke this remote function, use the `remote` method.
# This will immediately return an object ref (a future) and then create
# a task that will be executed on a worker process.
obj_ref = my_function.remote()

# The result can be retrieved with ray.get.
assert ray.get(obj_ref) == 1

@ray.remote
def slow_function():
    time.sleep(10)
    return 1

# Ray tasks are executed in parallel.
# All computation is performed in the background, driven by Ray's internal event loop.
for _ in range(4):
    # This doesn't block.
    slow_function.remote()
```python

## Specifying Resource Requirements

You can specify resource requirements in tasks:

```python
# Specify required resources.
@ray.remote(num_cpus=4, num_gpus=2)
def my_function():
    return 1

# Override the default resource requirements.
my_function.options(num_cpus=3).remote()
```python

## Passing Object References to Tasks

In addition to values, object refs can also be passed into remote functions. When the task gets executed, inside the function body **the argument will be the underlying value**.

```python
@ray.remote
def function_with_an_argument(value):
    return value + 1

obj_ref1 = my_function.remote()
assert ray.get(obj_ref1) == 1

# You can pass an object ref as an argument to another Ray task.
obj_ref2 = function_with_an_argument.remote(obj_ref1)
assert ray.get(obj_ref2) == 2
```python

**Important behaviors:**

- As the second task depends on the output of the first task, Ray will not execute the second task until the first task has finished.
- If the two tasks are scheduled on different machines, the output of the first task will be sent over the network to the machine where the second task is scheduled.

## Waiting for Partial Results

Calling `ray.get()` on Ray task results will block until the task finished execution. After launching a number of tasks, you may want to know which ones have finished executing without blocking on all of them. This can be achieved with `ray.wait()`:

```python
object_refs = [slow_function.remote() for _ in range(2)]
# Return as soon as one of the tasks finished execution.
ready_refs, remaining_refs = ray.wait(object_refs, num_returns=1, timeout=None)
```python

## Multiple Returns

By default, a Ray task only returns a single Object Ref. However, you can configure Ray tasks to return multiple Object Refs by setting the `num_returns` option:

```python
# By default, a Ray task only returns a single Object Ref.
@ray.remote
def return_single():
    return 0, 1, 2

object_ref = return_single.remote()
assert ray.get(object_ref) == (0, 1, 2)

# However, you can configure Ray tasks to return multiple Object Refs.
@ray.remote(num_returns=3)
def return_multiple():
    return 0, 1, 2

object_ref0, object_ref1, object_ref2 = return_multiple.remote()
assert ray.get(object_ref0) == 0
assert ray.get(object_ref1) == 1
assert ray.get(object_ref2) == 2
```python

## Generators

Ray is compatible with Python generator syntax. Generator tasks stream outputs back to the caller before the task finishes, which is useful for:

- Reducing heap memory or object store memory usage by yielding and garbage collecting output before the task finishes
- Streaming use cases where you want to process results as they become available

**Example:**

```python
import ray
import time

@ray.remote
def task():
    for i in range(5):
        time.sleep(5)
        yield i

# The generator yields output every 5 seconds
for obj_ref in task.remote():
    print(ray.get(obj_ref))
```python

With a normal Ray task, you have to wait 25 seconds to access the output. With a Ray generator, the caller can access the object reference before the task finishes.

## Cancelling Tasks

Ray tasks can be canceled by calling `ray.cancel()` on the returned Object ref:

```python
@ray.remote
def blocking_operation():
    time.sleep(10e6)

obj_ref = blocking_operation.remote()
ray.cancel(obj_ref)

try:
    ray.get(obj_ref)
except ray.exceptions.TaskCancelledError:
    print("Object reference was cancelled.")
```python

## Complete Example: Monte Carlo Pi Estimation

This example demonstrates how to combine tasks and actors to solve a real problem - estimating π using a Monte Carlo method:

```python
import ray
import math
import time
import random

ray.init()

# Define a Ray actor that tracks progress
@ray.remote
class ProgressActor:
    def __init__(self, total_num_samples: int):
        self.total_num_samples = total_num_samples
        self.num_samples_completed_per_task = {}

    def report_progress(self, task_id: int, num_samples_completed: int) -> None:
        self.num_samples_completed_per_task[task_id] = num_samples_completed

    def get_progress(self) -> float:
        return (
            sum(self.num_samples_completed_per_task.values()) / self.total_num_samples
        )

# Define a Ray task that does the sampling
@ray.remote
def sampling_task(num_samples: int, task_id: int,
                  progress_actor: ray.actor.ActorHandle) -> int:
    num_inside = 0
    for i in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        if math.hypot(x, y) <= 1:
            num_inside += 1

        # Report progress every 1 million samples.
        if (i + 1) % 1_000_000 == 0:
            progress_actor.report_progress.remote(task_id, i + 1)

    # Report the final progress.
    progress_actor.report_progress.remote(task_id, num_samples)
    return num_inside

# Configuration
NUM_SAMPLING_TASKS = 10
NUM_SAMPLES_PER_TASK = 10_000_000
TOTAL_NUM_SAMPLES = NUM_SAMPLING_TASKS * NUM_SAMPLES_PER_TASK

# Create the progress actor
progress_actor = ProgressActor.remote(TOTAL_NUM_SAMPLES)

# Create and execute all sampling tasks in parallel
results = [
    sampling_task.remote(NUM_SAMPLES_PER_TASK, i, progress_actor)
    for i in range(NUM_SAMPLING_TASKS)
]

# Query progress periodically
while True:
    progress = ray.get(progress_actor.get_progress.remote())
    print(f"Progress: {int(progress * 100)}%")

    if progress == 1:
        break

    time.sleep(1)

# Get all the sampling tasks results and calculate π
total_num_inside = sum(ray.get(results))
pi = (total_num_inside * 4) / TOTAL_NUM_SAMPLES
print(f"Estimated value of π is: {pi}")
```python

This example shows:
- How tasks execute in parallel
- How to pass actor handles to tasks
- How to retrieve results from multiple tasks
- How to monitor progress of distributed work

## Best Practices and Patterns

### Pattern: Using Nested Tasks for Nested Parallelism

A remote task can dynamically call other remote tasks (including itself) for nested parallelism. This is useful when sub-tasks can be parallelized.

**Example - Quick Sort:**

```python
import ray
import time
from numpy import random

def partition(collection):
    pivot = collection.pop()
    greater, lesser = [], []
    for element in collection:
        if element > pivot:
            greater.append(element)
        else:
            lesser.append(element)
    return lesser, pivot, greater

@ray.remote
def quick_sort_distributed(collection):
    # Use a threshold to avoid over-parallelizing
    if len(collection) <= 200000:
        return sorted(collection)
    else:
        lesser, pivot, greater = partition(collection)
        lesser = quick_sort_distributed.remote(lesser)
        greater = quick_sort_distributed.remote(greater)
        return ray.get(lesser) + [pivot] + ray.get(greater)
```python

**Important:** Nested tasks come with overhead. Make sure each nested task does significant work (at least 1 second of execution time).

### Anti-pattern: Calling ray.get() in a Loop

**Problem:** Calling `ray.get()` in a loop blocks until each result is available, eliminating parallelism.

**Anti-pattern:**
```python
# No parallelism - waits for each task sequentially
sequential_returns = []
for i in range(100):
    sequential_returns.append(ray.get(f.remote(i)))
```python

**Solution:** Schedule all tasks first, then get results:
```python
# Better: all tasks execute in parallel
refs = []
for i in range(100):
    refs.append(f.remote(i))

parallel_returns = ray.get(refs)
```python

### Anti-pattern: Calling ray.get() Unnecessarily

**Problem:** Calling `ray.get()` forces objects to be transferred to the caller. If you don't need to manipulate the object, pass the reference instead.

**Anti-pattern:**
```python
# Downloads result, then reuploads it
rollout = ray.get(generate_rollout.remote())
reduced = ray.get(reduce.remote(rollout))
```python

**Solution:** Pass object references directly:
```python
# Pass reference - object stays in object store
rollout_obj_ref = generate_rollout.remote()
reduced = ray.get(reduce.remote(rollout_obj_ref))
```python

### Anti-pattern: Calling ray.get() on Task Arguments

**Problem:** Calling `ray.get()` inside a task can cause deadlocks and performance issues.

**Anti-pattern:**
```python
@ray.remote
def pass_via_nested_ref(refs):
    print(sum(ray.get(refs)))  # Blocks inside task
```python

**Solution:** Pass refs as direct arguments:
```python
@ray.remote
def pass_via_direct_arg(*args):
    print(sum(args))  # Ray dereferences automatically

ray.get(pass_via_direct_arg.remote(*[f.remote() for _ in range(3)]))
```python

### Anti-pattern: Processing Results in Submission Order

**Problem:** Processing results in submission order may waste time waiting for slower tasks that were submitted earlier.

**Anti-pattern:**
```python
refs = [f.remote(i) for i in range(100)]
for ref in refs:
    result = ray.get(ref)  # Blocks waiting for each in order
    process(result)
```python

**Solution:** Process results in completion order using `ray.wait()`:
```python
refs = [f.remote(i) for i in range(100)]
unfinished = refs
while unfinished:
    finished, unfinished = ray.wait(unfinished, num_returns=1)
    result = ray.get(finished[0])
    process(result)
```python

### Anti-pattern: Fetching Too Many Objects at Once

**Problem:** Calling `ray.get()` on too many objects can cause heap out-of-memory or object store out-of-space.

**Anti-pattern:**
```python
object_refs = [return_big_object.remote() for _ in range(1000)]
results = ray.get(object_refs)  # May fail with OOM
```python

**Solution:** Process results in batches:
```python
BATCH_SIZE = 100
while object_refs:
    ready_object_refs, object_refs = ray.wait(object_refs, num_returns=BATCH_SIZE)
    results = ray.get(ready_object_refs)
    process_results(results)
```python

### Anti-pattern: Over-parallelizing with Too Fine-grained Tasks

**Problem:** Parallelizing very small tasks has higher overhead than the actual work, making it slower than sequential execution.

**Anti-pattern:**
```python
@ray.remote
def remote_double(number):
    return number * 2

# Overhead dominates - slower than sequential
doubled_number_refs = [remote_double.remote(number) for number in range(10000)]
```python

**Solution:** Use batching to make tasks do meaningful work:
```python
@ray.remote
def remote_double_batch(numbers):
    return [number * 2 for number in numbers]

BATCH_SIZE = 1000
doubled_batch_refs = []
for i in range(0, len(numbers), BATCH_SIZE):
    batch = numbers[i : i + BATCH_SIZE]
    doubled_batch_refs.append(remote_double_batch.remote(batch))
```python

**Rule of thumb:** Each task should take at least 1 second to execute.

## Scheduling

For each task, Ray chooses a node to run it based on:
- The task's resource requirements
- The specified scheduling strategy
- Locations of task arguments (for data locality)

See [Scheduling](scheduling.md) for more details.

## Fault Tolerance

By default, Ray will retry failed tasks due to system failures and specified application-level failures. You can change this behavior by setting `max_retries` and `retry_exceptions` options:

```python
@ray.remote(max_retries=3, retry_exceptions=[ValueError])
def my_task():
    # ...
```python

See [Fault Tolerance](fault-tolerance.md) for more details.

## Task Events

By default, Ray traces the execution of tasks, reporting task status events and profiling events that the Ray Dashboard uses. You can disable this to reduce overhead:

```python
@ray.remote(enable_task_events=False)
def my_task():
    # ...
```
