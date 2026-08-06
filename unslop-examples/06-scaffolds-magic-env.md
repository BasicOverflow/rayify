# Scaffold stubs, dead client state, magic package `.env`

## Slop tells

- Empty “compat layer” files that are 100% TODO comments (looks productionized, does nothing)
- Client tracks state that Serve already owns
- Libraries `load_dotenv` at import time so side effects spawn by existing
- Silent `RAY_ADDRESS` default / copy `.env.example` errors from deep package code

---

## 1. Fake plugins that never ship

### AI-slop — “LangChain compatibility”

```python
"""LangChain compatibility layer for Ray Hive.

TODO: Implement LangChain LLM wrapper
- Create LangChain LLM class that wraps RayHive
- Implement __call__ method for text generation
- Implement generate method for batch generation
- Implement stream method for streaming
- Support LangChain callbacks
- Add to __init__.py exports when ready
"""

# TODO: Implement LangChain compatibility
# from langchain.llms.base import LLM
# from typing import Optional, List, Any
#
# class RayHiveLLM(LLM):
#     """LangChain LLM wrapper for RayHive."""
#
#     model_id: str
#     ...
#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         from .inference import inference
#         return inference(prompt, self.model_id, ...)
```

### Same vibe — “OpenAI API compatibility”

```python
"""OpenAI API compatibility layer for Ray Hive.

TODO: Investigate Ray Serve response formats and OpenAI API compatibility
- Check if Ray Serve can return OpenAI-compatible responses
- Create OpenAI-compatible endpoint wrapper
...
"""
# TODO: Implement OpenAI API compatibility
# from fastapi import FastAPI
# ...
# @app.post("/v1/chat/completions")
# async def chat_completions(request: Dict[str, Any]):
#     pass
```

### Unslopified

Deleted. Real OpenAI support showed up later as a *working* gateway — not as a commented FastAPI TODO that `import ray_hive` advertised.

---

## 2. Client mirrors deployment state it doesn’t own

### AI-slop

```python
class RayHive:
    def __init__(...):
        self.suppress_logging = suppress_logging
        self._deployed_models: Dict[str, Dict] = {}
        init_ray(...)

    def deploy_model(...):
        ...
        self._deployed_models[model_id] = {
            "model_name": model_name,
            "vram_weights_gb": vram_weights_gb,
            "replicas": replicas,
            ...
        }

    def shutdown(self, model_id=None):
        if model_id is None:
            shutdown_all()
            self._deployed_models.clear()
        else:
            shutdown_model(model_id)
            self._deployed_models.pop(model_id, None)
```

### Unslopified

```python
def __init__(...):
    suppress_ray_warnings(suppress_logging)
    init_ray(suppress_logging=suppress_logging, **kwargs)
    get_vram_allocator()

def shutdown(self, model_id=None):
    if model_id is None:
        shutdown_all()
    else:
        shutdown_model(model_id)
```

Serve is the source of truth. Local dict was fanfic.

---

## 3. Package-import `.env` magic

### AI-slop (inside package utils)

```python
from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_env():
    """Load .env from project root if present."""
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

load_env()  # side effect on import

def init_ray(address: str = None, suppress_logging: bool = True, **kwargs):
    ...
    if address is None:
        address = os.getenv("RAY_ADDRESS")
        if not address:
            raise RuntimeError(
                "RAY_ADDRESS not set. Copy .env.example to .env and set your cluster address."
            )
```

And inference would also auto-init:

```python
def _ensure_connected():
    """Ensure Ray is connected to cluster."""
    if not ray.is_initialized():
        load_env()
        address = getenv("RAY_ADDRESS")
        if not address:
            raise RuntimeError("RAY_ADDRESS not set. Copy .env.example to .env ...")
        ray.init(address=address, ignore_reinit_error=True, log_to_driver=False)
```

### Unslopified

```python
# ray_utils — no dotenv; caller supplies address
def init_ray(address: str, suppress_logging: bool = True, **kwargs):
    ...
    ray.init(address=address, ...)

# inference
def _ensure_connected():
    """Require an existing Ray connection (e.g. via RayHive(address=...))."""
    if not ray.is_initialized():
        raise RuntimeError("Ray is not connected. Call RayHive(address=...) first.")

# hive
def __init__(self, address: str, suppress_logging: bool = True, **kwargs):
    init_ray(address, suppress_logging=suppress_logging, **kwargs)
```

`.env` stays in **examples/tests** only. Hardcoded defaults like `"ray://10.0.1.53:10001"` in older snapshots are the same disease: package code guessing your lab network.
