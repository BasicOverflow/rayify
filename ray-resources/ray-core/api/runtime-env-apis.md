# Runtime Environment APIs

APIs for configuring runtime environments for jobs, tasks, and actors.

## RuntimeEnv

Class to define a runtime environment. Can be used interchangeably with a dictionary.

```python
from ray.runtime_env import RuntimeEnv

# For entire job
ray.init(runtime_env=RuntimeEnv(
    pip=["numpy", "pandas"],
    working_dir="./my_code",
    env_vars={"MY_VAR": "value"}
))

# Per-task or per-actor
@ray.remote(runtime_env=RuntimeEnv(pip=["requests"]))
def task():
    import requests
    return requests.get("https://example.com")

actor = MyActor.options(runtime_env=RuntimeEnv(pip=["torch"])).remote()
```

**Key Parameters:**
- `pip`: List of pip packages to install
- `conda`: Conda environment specification
- `working_dir`: Directory to use as working directory
- `py_modules`: List of Python module paths
- `env_vars`: Dictionary of environment variables
- `container`: Container configuration
- `image_uri`: Docker image URI

## RuntimeEnvConfig

Configuration options for runtime environment setup.

```python
from ray.runtime_env import RuntimeEnv, RuntimeEnvConfig

runtime_env = RuntimeEnv(
    pip=["numpy"],
    config=RuntimeEnvConfig(
        setup_timeout_seconds=300,
        eager_install=True
    )
)
```

**Parameters:**
- `setup_timeout_seconds`: Timeout for runtime environment creation (default: 600, -1 to disable)
- `eager_install`: Install at ray.init() time before workers are leased (default: True)

