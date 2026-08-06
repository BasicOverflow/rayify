# Inference layer: defensive soup + fake batch “optimizer”

## Slop tells

- `hasattr` + `.get` dual access “just in case” object shape
- Collecting lists when you only need the first match
- Auto batch-size helper that returns magic `32` on any error
- Duplicated sync/async bodies with multi-paragraph Args: blocks
- NotImplemented streaming kept around to look complete

---

## AI-slop: handle lookup

```python
def _get_handle(model_id: str):
    """Get a handle to the model application. Ray Serve handles load balancing automatically."""
    _ensure_connected()
    status = serve.status()

    # Try to find the model as a single application first
    if model_id in status.applications:
        app = status.applications[model_id]
        deployments = app.deployments if hasattr(app, 'deployments') else app.get('deployments', {})
        if deployments:
            # Get handle to first deployment - Ray Serve automatically load balances
            # across all deployments in the application
            deployment_name = list(deployments.keys())[0]
            return serve.get_deployment_handle(deployment_name, app_name=model_id)

    # Fallback: look for deployments in apps with model_id prefix (legacy support)
    # Format: {model_id}-{gpu_name}
    matching_deployments = []
    for app_name, app in status.applications.items():
        if app_name.startswith(f"{model_id}-"):
            deployments = app.deployments if hasattr(app, 'deployments') else app.get('deployments', {})
            for deployment_name in deployments.keys():
                if deployment_name.startswith(f"{model_id}-"):
                    matching_deployments.append((deployment_name, app_name))

    if not matching_deployments:
        available_apps = list(status.applications.keys())
        raise RuntimeError(f"Model '{model_id}' not found. Available: {available_apps}")

    # Get handle to first matching deployment - Ray Serve handles load balancing
    deployment_name, app_name = matching_deployments[0]
    return serve.get_deployment_handle(deployment_name, app_name=app_name)
```

## Unslopified

```python
def _get_handle(model_id: str):
    _ensure_connected()
    status = serve.status()

    if model_id in status.applications:
        app = status.applications[model_id]
        deployments = app.deployments if hasattr(app, 'deployments') else {}
        if deployments:
            return serve.get_deployment_handle(list(deployments.keys())[0], app_name=model_id)

    for app_name, app in status.applications.items():
        if app_name.startswith(f"{model_id}-"):
            deployments = app.deployments if hasattr(app, 'deployments') else {}
            for deployment_name in deployments.keys():
                if deployment_name.startswith(f"{model_id}-"):
                    return serve.get_deployment_handle(deployment_name, app_name=app_name)

    raise RuntimeError(f"Model '{model_id}' not found")
```

(Still intermediate — later reworks make model lookup less fragile — but the cut is: less talk, no “Available: …”, no full list collect-then-index-0.)

---

## AI-slop: silent max_num_seqs + reinvent batching outside vLLM

```python
def _get_max_num_seqs(model_id: str) -> int:
    """Query the model's max_num_seqs value for optimal batch sizing."""
    handle = _get_handle(model_id)
    try:
        max_num_seqs = handle.get_max_num_seqs.remote().result()
        if max_num_seqs is None or max_num_seqs <= 0:
            return 32  # Default fallback
        return max_num_seqs
    except Exception as e:
        # Fallback if query fails
        return 32


def inference_batch(... batch_size: Optional[int] = None, ...):
    """...
    Batch size is automatically calculated based on the model's max_num_seqs for optimal performance.
    Prompts are automatically split into optimal batches if they exceed max_num_seqs.
    ...
    """
    # Auto-calculate batch size based on model's max_num_seqs if not specified
    if batch_size is None:
        max_num_seqs = _get_max_num_seqs(model_id)
        batch_size = max_num_seqs

    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])

    requests = []
    for batch in batches:
        requests.append(handle.remote({"prompts": batch, **request_template}))

    batch_results = [req.result() for req in requests]
    # flatten with dual list/non-list branches...
```

## Unslopified

```python
def inference_batch(prompts, model_id, structured_output=None, max_tokens=None, **kwargs):
    """Run batch inference on a deployed model. vLLM handles batching internally.

    All prompts are sent in a single request. vLLM's internal batching mechanism
    handles optimal batching based on max_num_seqs and max_num_batched_tokens.
    """
    handle = _get_handle(model_id)

    request = {"prompts": prompts}
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if structured_output:
        request["guided_json"] = structured_output.model_json_schema()
    request.update(kwargs)

    result = handle.remote(request).result()
    results = result if isinstance(result, list) else [result]

    output = []
    for result_item in results:
        text = _extract_text(result_item)
        output.append(_parse_structured_output(text, structured_output) if structured_output else text)
    return output
```

Also removed:

```python
async def streaming_batch(...) -> AsyncGenerator[List[str], None]:
    """Stream batch inference results (async generator)."""
    raise NotImplementedError("Streaming not yet implemented")
```

---

## Orchestrator / actor noise of the same era

Deploy path filled with:

```python
print(f"    ✅ Deployed {gpu_deployment_name} on GPU {gpu_mapping[...]}")
print(f"  🧪 TEST MODE: Deploying 1 replica on GPU with most VRAM (...)")
```

Unslop: quieter deployment path; calculation results only when useful.
