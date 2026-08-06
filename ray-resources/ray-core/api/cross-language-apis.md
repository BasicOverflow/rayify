# Cross Language APIs

APIs for calling Java functions and actors from Python.

## Java Functions

### ray.cross_language.java_function()

Define a Java function to call from Python.

```python
java_func = ray.cross_language.java_function(
    class_name="com.example.MyClass",
    function_name="myMethod"
)

result = java_func.remote(arg1, arg2)
```

**Parameters:**
- `class_name`: Fully qualified Java class name
- `function_name`: Java method name

**Returns:** Remote function that can be called with `.remote()`

## Java Actors

### ray.cross_language.java_actor_class()

Define a Java actor class to use from Python.

```python
JavaActor = ray.cross_language.java_actor_class(
    class_name="com.example.MyActor"
)

actor = JavaActor.remote(arg1, arg2)
result = actor.method.remote()
```

**Parameters:**
- `class_name`: Fully qualified Java actor class name

**Returns:** Actor class that can be instantiated with `.remote()`

**Note:** Requires Java code to be available in the code search path specified in JobConfig.

