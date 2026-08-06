# Getting Started with Ray Core

Ray Core is a powerful distributed computing framework that provides a small set of essential primitives (tasks, actors, and objects) for building and scaling distributed applications.

## Installation

To get started, install Ray using:

```python
pip install -U ray
```python

## Initialization

The first step is to import and initialize Ray:

```python
import ray

ray.init()
```python

**Note:** Unless you explicitly call `ray.init()`, the first use of a Ray remote API call will implicitly call `ray.init()` with no arguments.

## Core Concepts

Ray provides three essential primitives that work together to enable flexible distributed applications:

### Tasks

Ray enables arbitrary functions to execute asynchronously on separate worker processes. These asynchronous Ray functions are called **tasks**. Tasks can specify their resource requirements in terms of CPUs, GPUs, and custom resources. The cluster scheduler uses these resource requests to distribute tasks across the cluster for parallelized execution.

**Example:**

```python
# Define the square task.
@ray.remote
def square(x):
    return x * x

# Launch four parallel square tasks.
futures = [square.remote(i) for i in range(4)]

# Retrieve results.
print(ray.get(futures))
# -> [0, 1, 4, 9]
```python

To create a task:
1. Decorate your function with `@ray.remote` to indicate it should run remotely
2. Call the function with `.remote()` instead of a normal function call
3. Use `ray.get()` to retrieve the result from the returned future (Ray _object reference_)

### Actors

Actors extend the Ray API from functions (tasks) to classes. An **actor** is essentially a stateful worker (or a service). When you instantiate a new actor, Ray creates a new worker and schedules methods of the actor on that specific worker. The methods can access and mutate the state of that worker. Like tasks, actors support CPU, GPU, and custom resource requirements.

**Example:**

```python
# Define the Counter actor.
@ray.remote
class Counter:
    def __init__(self):
        self.i = 0

    def get(self):
        return self.i

    def incr(self, value):
        self.i += value

# Create a Counter actor.
c = Counter.remote()

# Submit calls to the actor. These calls run asynchronously but in
# submission order on the remote actor process.
for _ in range(10):
    c.incr.remote(1)

# Retrieve final actor state.
print(ray.get(c.get.remote()))
# -> 10
```python

When you instantiate a Ray actor:
1. Ray starts a dedicated worker process somewhere in your cluster
2. The actor's methods run on that specific worker and can access and modify its state
3. The actor executes method calls serially in the order it receives them, preserving consistency

### Objects

Tasks and actors create objects and compute on objects. You can refer to these objects as **remote objects** because Ray stores them anywhere in a Ray cluster, and you use **object refs** to refer to them. Ray caches remote objects in its distributed shared-memory **object store** and creates one object store per node in the cluster. In the cluster setting, a remote object can live on one or many nodes, independent of who holds the object ref.

An **object ref** is essentially a pointer or a unique ID that can be used to refer to a remote object without seeing its value. If you're familiar with futures, Ray object refs are conceptually similar.

**Example:**

```python
import numpy as np

# Define a task that sums the values in a matrix.
@ray.remote
def sum_matrix(matrix):
    return np.sum(matrix)

# Call the task with a literal argument value.
print(ray.get(sum_matrix.remote(np.ones((100, 100)))))
# -> 10000.0

# Put a large array into the object store.
matrix_ref = ray.put(np.ones((1000, 1000)))

# Call the task with the object reference as an argument.
print(ray.get(sum_matrix.remote(matrix_ref)))
# -> 1000000.0
```python

There are three main ways to work with objects in Ray:
1. **Implicit creation**: When tasks and actors return values, they are automatically stored in Ray's distributed object store, returning object references that can be later retrieved.
2. **Explicit creation**: Use `ray.put()` to directly place objects in the store.
3. **Passing references**: You can pass object references to other tasks and actors, avoiding unnecessary data copying and enabling lazy execution.

### Placement Groups

Placement groups allow users to atomically reserve groups of resources across multiple nodes. You can use them to schedule Ray tasks and actors packed as close as possible for locality (PACK), or spread apart (SPREAD). A common use case is gang-scheduling actors or tasks.

### Environment Dependencies

When Ray executes tasks and actors on remote machines, their environment dependencies, such as Python packages, local files, and environment variables, must be available on the remote machines. To address this problem, you can:
1. Prepare your dependencies on the cluster in advance using the Ray Cluster Launcher
2. Use Ray's runtime environments to install them on the fly

## Next Steps

You can combine Ray's simple primitives in powerful ways to express virtually any distributed computation pattern. To dive deeper, explore these guides:

- [Tasks](tasks.md) - Using remote functions
- [Actors](actors.md) - Using remote classes
- [Objects](objects.md) - Working with Ray objects
- [Scheduling](scheduling.md) - Resource management and scheduling
- [Fault Tolerance](fault-tolerance.md) - Handling failures
- [API Reference](api-reference.md) - Complete API documentation

