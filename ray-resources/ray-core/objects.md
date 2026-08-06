# Objects

In Ray, tasks and actors create and compute on objects. We refer to these objects as **remote objects** because they can be stored anywhere in a Ray cluster, and we use **object refs** to refer to them. Remote objects are cached in Ray's distributed shared-memory **object store**, and there is one object store per node in the cluster.

An **object ref** is essentially a pointer or a unique ID that can be used to refer to a remote object without seeing its value. If you're familiar with futures, Ray object refs are conceptually similar.

## Creating Object References

Object refs can be created in two ways:

1. They are returned by remote function calls
2. They are returned by `ray.put()`

**Example:**

```python
import ray

# Put an object in Ray's object store.
y = 1
object_ref = ray.put(y)
```python

**Important:** Remote objects are immutable. That is, their values cannot be changed after creation. This allows remote objects to be replicated in multiple object stores without needing to synchronize the copies.

## Fetching Object Data

You can use the `ray.get()` method to fetch the result of a remote object from an object ref. If the current node's object store does not contain the object, the object is downloaded.

**Example:**

```python
import ray
import time

# Get the value of one object ref.
obj_ref = ray.put(1)
assert ray.get(obj_ref) == 1

# Get the values of multiple object refs in parallel.
assert ray.get([ray.put(i) for i in range(3)]) == [0, 1, 2]

# You can also set a timeout to return early from a get that's blocking for too long.
from ray.exceptions import GetTimeoutError

@ray.remote
def long_running_function():
    time.sleep(8)

obj_ref = long_running_function.remote()
try:
    ray.get(obj_ref, timeout=4)
except GetTimeoutError:
    print("`get` timed out.")
```python

**Note:** If the object is a numpy array or a collection of numpy arrays, the `get` call is zero-copy and returns arrays backed by shared object store memory. Otherwise, we deserialize the object data into a Python object.

## Passing Object Arguments

Ray object references can be freely passed around a Ray application. They can be passed as arguments to tasks, actor methods, and even stored in other objects. Objects are tracked via distributed reference counting, and their data is automatically freed once all references to the object are deleted.

### Top-Level Arguments (By Value)

When an object is passed directly as a top-level argument to a task, Ray will de-reference the object. This means that Ray will fetch the underlying data for all top-level object reference arguments, not executing the task until the object data becomes fully available.

```python
@ray.remote
def echo(a: int, b: int, c: int):
    """This function prints its input values to stdout."""
    print(a, b, c)

# Passing the literal values (1, 2, 3) to `echo`.
echo.remote(1, 2, 3)
# -> prints "1 2 3"

# Put the values (1, 2, 3) into Ray's object store.
a, b, c = ray.put(1), ray.put(2), ray.put(3)

# Passing an object as a top-level argument to `echo`. Ray will de-reference top-level
# arguments, so `echo` will see the literal values (1, 2, 3) in this case as well.
echo.remote(a, b, c)
# -> prints "1 2 3"
```python

### Nested Arguments (By Reference)

When an object is passed within a nested object, for example, within a Python list, Ray will **not** de-reference it. This means that the task will need to call `ray.get()` on the reference to fetch the concrete value. However, if the task never calls `ray.get()`, then the object value never needs to be transferred to the machine the task is running on.

```python
@ray.remote
def echo_and_get(x_list):  # List[ObjectRef]
    """This function prints its input values to stdout."""
    print("args:", x_list)
    print("values:", ray.get(x_list))

# Put the values (1, 2, 3) into Ray's object store.
a, b, c = ray.put(1), ray.put(2), ray.put(3)

# Passing an object as a nested argument to `echo_and_get`. Ray does not
# de-reference nested args, so `echo_and_get` sees the references.
echo_and_get.remote([a, b, c])
# -> prints args: [ObjectRef(...), ObjectRef(...), ObjectRef(...)]
#           values: [1, 2, 3]
```python

**Recommendation:** Pass objects as top-level arguments where possible, but nested arguments can be useful for passing objects on to other tasks without needing to see the data.

The top-level vs nested passing convention also applies to actor constructors and actor method calls:

```python
@ray.remote
class Actor:
    def __init__(self, arg):
        pass

    def method(self, arg):
        pass

obj = ray.put(2)

# Examples of passing objects to actor constructors.
actor_handle = Actor.remote(obj)  # by-value
actor_handle = Actor.remote([obj])  # by-reference

# Examples of passing objects to actor method calls.
actor_handle.method.remote(obj)  # by-value
actor_handle.method.remote([obj])  # by-reference
```python

## Closure Capture of Objects

You can also pass objects to tasks via closure-capture. This can be convenient when you have a large object that you want to share verbatim between many tasks or actors, and don't want to pass it repeatedly as an argument.

**Warning:** Defining a task that closes over an object ref will pin the object via reference-counting, so the object will not be evicted until the job completes.

```python
import ray

# Put the values (1, 2, 3) into Ray's object store.
a, b, c = ray.put(1), ray.put(2), ray.put(3)

@ray.remote
def print_via_capture():
    """This function prints the values of (a, b, c) to stdout."""
    print(ray.get([a, b, c]))

# Passing object references via closure-capture.
print_via_capture.remote()
# -> prints [1, 2, 3]
```python

## Nested Objects

Ray also supports nested object references. This allows you to build composite objects that themselves hold references to further sub-objects.

```python
# Objects can be nested within each other. Ray will keep the inner object
# alive via reference counting until all outer object references are deleted.
object_ref_2 = ray.put([object_ref])
```

## Serialization

Since Ray processes do not share memory space, data transferred between workers and nodes needs to be **serialized** and **deserialized**. Ray uses the Plasma object store to efficiently transfer objects across different processes and different nodes.

### Overview

Ray uses a customized Pickle protocol version 5 backport to replace the original PyArrow serializer. This gets rid of several previous limitations (e.g., cannot serialize recursive objects).

Ray is currently compatible with Pickle protocol version 5, while Ray supports serialization of a wider range of objects (e.g., lambda & nested functions, dynamic classes) with the help of cloudpickle.

### Plasma Object Store

Plasma is an in-memory object store. It is used to efficiently transfer objects across different processes and different nodes. All objects in Plasma object store are **immutable** and held in shared memory. This is so that they can be accessed efficiently by many workers on the same node.

Each node has its own object store. When data is put into the object store, it does not get automatically broadcasted to other nodes. Data remains local to the writer until requested by another task or actor on another node.

### Numpy Arrays

Ray optimizes for numpy arrays by using Pickle protocol 5 with out-of-band data. The numpy array is stored as a read-only object, and all Ray workers on the same node can read the numpy array in the object store without copying (zero-copy reads). Each numpy array object in the worker process holds a pointer to the relevant array held in shared memory.

**Tip:** You can often avoid serialization issues by using only native types (e.g., numpy arrays or lists/dicts of numpy arrays and other primitive types), or by using Actors to hold objects that cannot be serialized.

### Fixing "assignment destination is read-only"

Because Ray puts numpy arrays in the object store, when deserialized as arguments in remote functions they will become read-only. For example, the following code will crash:

```python
import ray
import numpy as np

@ray.remote
def f(arr):
    arr[0] = 1  # This will fail!

try:
    ray.get(f.remote(np.zeros(100)))
except ray.exceptions.RayTaskError as e:
    print(e)
    # ValueError: assignment destination is read-only
```python

To avoid this issue, you can manually copy the array at the destination if you need to mutate it:

```python
@ray.remote
def f(arr):
    arr = arr.copy()  # Adding a copy will fix the error.
    arr[0] = 1
```python

Note that this is effectively like disabling the zero-copy deserialization feature provided by Ray.

### Serialization Notes

- Ray is currently using Pickle protocol version 5. The default pickle protocol used by most python distributions is protocol 3. Protocol 4 & 5 are more efficient than protocol 3 for larger objects.

- For non-native objects, Ray will always keep a single copy even if it is referred multiple times in an object:

```python
import ray
import numpy as np

obj = [np.zeros(42)] * 99
l = ray.get(ray.put(obj))
assert l[0] is l[1]  # no problem!
```python

- Whenever possible, use numpy arrays or Python collections of numpy arrays for maximum performance.

- Lock objects are mostly unserializable, because copying a lock is meaningless and could cause serious concurrency problems. You may have to come up with a workaround if your object contains a lock.

### Customized Serialization

Sometimes you may want to customize your serialization process because the default serializer used by Ray (pickle5 + cloudpickle) does not work for you.

**Method 1: Define `__reduce__` in your class**

If you want to customize the serialization of a type of objects, and you have access to the code, you can define `__reduce__` function inside the corresponding class:

```python
import ray
import sqlite3

class DBConnection:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)

    # without '__reduce__', the instance is unserializable.
    def __reduce__(self):
        deserializer = DBConnection
        serialized_data = (self.path,)
        return deserializer, serialized_data

original = DBConnection("/tmp/db")
copied = ray.get(ray.put(original))
```python

**Method 2: Register a custom serializer**

If you want to customize the serialization of a type of objects, but you cannot access or modify the corresponding class, you can register the class with the serializer:

```python
import ray
import threading

class A:
    def __init__(self, x):
        self.x = x
        self.lock = threading.Lock()  # could not be serialized!

try:
    ray.get(ray.put(A(1)))  # fail!
except TypeError:
    pass

def custom_serializer(a):
    return a.x

def custom_deserializer(b):
    return A(b)

# Register serializer and deserializer for class A:
ray.util.register_serializer(
    A, serializer=custom_serializer, deserializer=custom_deserializer)
ray.get(ray.put(A(1)))  # success!

# You can deregister the serializer at any time.
ray.util.deregister_serializer(A)
```python

**Note:** Serializers are managed locally for each Ray worker. So for every Ray worker, if you want to use the serializer, you need to register the serializer.

### Custom Serializers for Exceptions

When Ray tasks raise exceptions that cannot be serialized with the default pickle mechanism, you can register custom serializers to handle them (Note: the serializer must be registered in the driver and all workers):

```python
import ray
import threading

class CustomError(Exception):
    def __init__(self, message, data):
        self.message = message
        self.data = data
        self.lock = threading.Lock()  # Cannot be serialized

def custom_serializer(exc):
    return {"message": exc.message, "data": str(exc.data)}

def custom_deserializer(state):
    return CustomError(state["message"], state["data"])

# Register in the driver
ray.util.register_serializer(
    CustomError,
    serializer=custom_serializer,
    deserializer=custom_deserializer
)

@ray.remote
def task_that_registers_serializer_and_raises():
    # Register the custom serializer in the worker
    ray.util.register_serializer(
        CustomError,
        serializer=custom_serializer,
        deserializer=custom_deserializer
    )
    raise CustomError("Something went wrong", {"complex": "data"})

# The custom exception will be properly serialized across worker boundaries
try:
    ray.get(task_that_registers_serializer_and_raises.remote())
except ray.exceptions.RayTaskError as e:
    print(f"Caught exception: {e.cause}")  # This will be our CustomError
```python

When a custom exception is raised in a remote task, Ray will:
1. Serialize the exception using your custom serializer
2. Wrap it in a `RayTaskError`
3. The deserialized exception will be available as `ray_task_error.cause`

### Troubleshooting Serialization

Use `ray.util.inspect_serializability` to identify tricky pickling issues. This function can be used to trace a potential non-serializable object within any Python object:

```python
from ray.util import inspect_serializability
import threading

lock = threading.Lock()

def test():
    print(lock)

inspect_serializability(test, name="test")
```python

This will output detailed information about what cannot be serialized and why.

For even more detailed information, set environmental variable `RAY_PICKLE_VERBOSE_DEBUG='2'` before importing Ray. This enables serialization with python-based backend instead of C-Pickle, so you can debug into python code at the middle of serialization. However, this would make serialization much slower.

### Serializing ObjectRefs

Explicitly serializing `ObjectRefs` using `ray.cloudpickle` should be used as a last resort. Passing `ObjectRefs` through Ray task arguments and return values is the recommended approach.

Ray `ObjectRefs` can be serialized using `ray.cloudpickle`. The `ObjectRef` can then be deserialized and accessed with `ray.get()`. Note that `ray.cloudpickle` must be used; other pickle tools are not guaranteed to work. Additionally, the process that deserializes the `ObjectRef` must be part of the same Ray cluster that serialized it.

**Warning:** `ray._private.internal_api.free(obj_ref)` is a private API and may be changed in future Ray versions.

```python
import ray
from ray import cloudpickle

FILE = "external_store.pickle"

ray.init()

my_dict = {"hello": "world"}
obj_ref = ray.put(my_dict)

with open(FILE, "wb+") as f:
    cloudpickle.dump(obj_ref, f)

# ObjectRef remains pinned in memory because it was serialized with ray.cloudpickle.
del obj_ref

with open(FILE, "rb") as f:
    new_obj_ref = cloudpickle.load(f)

# The deserialized ObjectRef works as expected.
assert ray.get(new_obj_ref) == my_dict

# Explicitly free the object.
ray._private.internal_api.free(new_obj_ref)
```python

## Fault Tolerance

Ray can automatically recover from object data loss via lineage reconstruction but not owner failure. See [Fault Tolerance](fault-tolerance.md) for more details.
