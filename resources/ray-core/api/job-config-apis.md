# Job Config APIs

APIs for configuring Ray jobs.

## JobConfig

Class to store job configurations.

```python
from ray.job_config import JobConfig

ray.init(job_config=JobConfig(
    default_actor_lifetime="detached",
    ray_namespace="my_namespace",
    runtime_env={"pip": ["numpy"]},
    metadata={"key": "value"}
))
```

**Key Parameters:**
- `jvm_options`: JVM options for Java workers
- `code_search_path`: List of directories/jar files for code search path
- `runtime_env`: Runtime environment dictionary
- `metadata`: Opaque metadata dictionary
- `ray_namespace`: Namespace for logical grouping
- `default_actor_lifetime`: Default actor lifetime ("detached" or "non_detached")

**Methods:**
- `set_default_actor_lifetime(lifetime)`: Set default actor lifetime
- `set_metadata(key, value)`: Add metadata key-value pair
- `set_py_logging_config(config)`: Set logging configuration
- `set_ray_namespace(namespace)`: Set Ray namespace
- `set_runtime_env(runtime_env)`: Modify runtime environment
- `from_json(json_str)`: Create JobConfig from JSON

