# Compiled Graph API Reference

This document provides API reference for Ray Compiled Graph functionality.

## ray.actor.ActorMethod.bind

# ray.actor.ActorMethod.bind

ActorMethod.bind(_* args_, _** kwargs_)[[source]](../../../_modules/ray/actor.html#ActorMethod.bind)#
    

Bind arguments to the actor method for Ray DAG building.

This method generates and returns an intermediate representation (IR) node that indicates the actor method will be called with the given arguments at execution time.

This method is used in both [Ray DAG](../../ray-dag.md#ray-dag-guide) and [Ray Compiled Graph](../ray-compiled-graph.md#ray-compiled-graph) for building a DAG.

**DeveloperAPI:** This API may change across minor Ray releases.

---

## ray.dag.compiled_dag_node.CompiledDAG.visualize

# ray.dag.compiled_dag_node.CompiledDAG.visualize

CompiledDAG.visualize(_filename ='compiled_graph'_, _format ='png'_, _view =False_, _channel_details =False_) → [str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)")[[source]](../../../_modules/ray/dag/compiled_dag_node.html#CompiledDAG.visualize)#
    

Visualize the compiled graph by showing tasks and their dependencies. This method should be called **after** the graph has been compiled using `experimental_compile()`.

Parameters:
    

  * **format** – The format of the output file (e.g., ‘png’, ‘pdf’, ‘ascii’).

  * **channel_details** – If True, adds channel details to edges.

Returns:
    

The string representation of the compiled graph. For Graphviz-based formats (e.g., ‘png’, ‘pdf’, ‘jpeg’), returns the Graphviz DOT string representation of the compiled graph. For ASCII format, returns the ASCII string representation of the compiled graph.

Raises:
    

  * [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "\(in Python v3.14\)") – If the graph is empty or not properly compiled.

  * [**ImportError**](https://docs.python.org/3/library/exceptions.html#ImportError "\(in Python v3.14\)") – If the `graphviz` package is not installed.

---

## ray.dag.DAGNode.experimental_compile

# ray.dag.DAGNode.experimental_compile

DAGNode.experimental_compile(__submit_timeout : [float](https://docs.python.org/3/library/functions.html#float "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, __buffer_size_bytes : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, _enable_asyncio : [bool](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") = False_, __max_inflight_executions : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, __max_buffered_results : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, __overlap_gpu_communication : [bool](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_, __default_communicator : [str](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.14\)") | Communicator | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = 'create'_) → ray.dag.CompiledDAG[[source]](../../../_modules/ray/dag/dag_node.html#DAGNode.experimental_compile)#
    

Compile an accelerated execution path for this DAG.

Parameters:
    

  * **_submit_timeout** – The maximum time in seconds to wait for execute() calls. None means using default timeout, 0 means immediate timeout (immediate success or timeout without blocking), -1 means infinite timeout (block indefinitely).

  * **_buffer_size_bytes** – The initial buffer size in bytes for messages that can be passed between tasks in the DAG. The buffers will be automatically resized if larger messages are written to the channel.

  * **enable_asyncio** – Whether to enable asyncio for this DAG.

  * **_max_inflight_executions** – The maximum number of in-flight executions that can be submitted via `execute` or `execute_async` before consuming the output using `ray.get()`. If the caller submits more executions, `RayCgraphCapacityExceeded` is raised.

  * **_max_buffered_results** – The maximum number of results that can be buffered at the driver. If more than this number of results are buffered, `RayCgraphCapacityExceeded` is raised. Note that when result corresponding to an execution is retrieved (by calling `ray.get()` on a `CompiledDAGRef` or `CompiledDAGRef` or await on a `CompiledDAGFuture`), results corresponding to earlier executions that have not been retrieved yet are buffered.

  * **_overlap_gpu_communication** – (experimental) Whether to overlap GPU communication with computation during DAG execution. If True, the communication and computation can be overlapped, which can improve the performance of the DAG execution. If None, the default value will be used.

  * **_default_communicator** – The default communicator to use to transfer tensors. Three types of values are valid. (1) Communicator: For p2p operations, this is the default communicator to use for nodes annotated with `with_tensor_transport()` and when shared memory is not the desired option (e.g., when transport=”nccl”, or when transport=”auto” for communication between two different GPUs). For collective operations, this is the default communicator to use when a custom communicator is not specified. (2) “create”: for each collective operation without a custom communicator specified, a communicator is created and initialized on its involved actors, or an already created communicator is reused if the set of actors is the same. For all p2p operations without a custom communicator specified, it reuses an already created collective communicator if the p2p actors are a subset. Otherwise, a new communicator is created. (3) None: a ValueError will be thrown if a custom communicator is not specified.

Returns:
    

A compiled DAG.

---

## ray.experimental.compiled_dag_ref.CompiledDAGRef.__init__

# ray.experimental.compiled_dag_ref.CompiledDAGRef.__init__

CompiledDAGRef.__init__(_dag : ray.experimental.CompiledDAG_, _execution_index : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")_, _channel_index : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_)[[source]](../../../_modules/ray/experimental/compiled_dag_ref.html#CompiledDAGRef.__init__)#
    

Parameters:
    

  * **dag** – The compiled DAG that generated this CompiledDAGRef.

  * **execution_index** – The index of the execution for the DAG. A DAG can be executed multiple times, and execution index indicates which execution this CompiledDAGRef corresponds to.

  * **actor_execution_loop_refs** – The actor execution loop refs that are used to execute the DAG. This can be used internally to check the task execution errors in case of exceptions.

  * **channel_index** – The index of the DAG’s output channel to fetch the result from. A DAG can have multiple output channels, and channel index indicates which channel this CompiledDAGRef corresponds to. If channel index is not provided, this CompiledDAGRef wraps the results from all output channels.

---

## ray.experimental.compiled_dag_ref.CompiledDAGRef

# ray.experimental.compiled_dag_ref.CompiledDAGRef

_class _ray.experimental.compiled_dag_ref.CompiledDAGRef(_dag : ray.experimental.CompiledDAG_, _execution_index : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)")_, _channel_index : [int](https://docs.python.org/3/library/functions.html#int "\(in Python v3.14\)") | [None](https://docs.python.org/3/library/constants.html#None "\(in Python v3.14\)") = None_)[[source]](../../../_modules/ray/experimental/compiled_dag_ref.html#CompiledDAGRef)#
    

A reference to a compiled DAG execution result.

This is a subclass of ObjectRef and resembles ObjectRef. For example, similar to ObjectRef, ray.get() can be called on it to retrieve the result. However, there are several major differences: 1\. ray.get() can only be called once per CompiledDAGRef. 2\. ray.wait() is not supported. 3\. CompiledDAGRef cannot be copied, deep copied, or pickled. 4\. CompiledDAGRef cannot be passed as an argument to another task.

**PublicAPI (alpha):** This API is in alpha and may change before becoming stable.

Methods

[`__init__`](ray.experimental.compiled_dag_ref.CompiledDAGRef.__init__.md#ray.experimental.compiled_dag_ref.CompiledDAGRef.__init__ "ray.experimental.compiled_dag_ref.CompiledDAGRef.__init__") | 

param dag:
    The compiled DAG that generated this CompiledDAGRef.  

---

