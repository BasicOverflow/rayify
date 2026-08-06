# Wait helpers, silent excepts, and fallback deploys

## Slop tells

- Multi-path “status poller” that never needed to exist
- Nested `hasattr` / string-status checks
- Emoji progress + WARNING on timeout still returns `False` (soft failure soup)
- Bare `except:` that sleep-and-hopes
- When GPUs missing: invent a broken single-GPU deploy instead of failing

---

## AI-slop: `_wait_for_deployment_ready` (~90 lines)

```python
def _wait_for_deployment_ready(self, deployment_name: str, expected_replicas: int,
                               gpu_key: str = None, timeout: int = 300) -> bool:
    """Wait for deployment to have expected number of replicas fully initialized (RUNNING).

    Also waits for pending VRAM reservations to clear on the GPU to ensure replicas are fully loaded.
    Returns True if deployment is ready, False if timeout.
    """
    start_time = time.time()
    check_interval = 5
    max_checks = timeout // check_interval

    allocator = ray.get_actor("vram_allocator", namespace="system")

    for check_num in range(max_checks):
        try:
            status = serve.status()

            for app_name, app_info in status.applications.items():
                if hasattr(app_info, 'deployments') and deployment_name in app_info.deployments:
                    dep = app_info.deployments[deployment_name]

                    if hasattr(dep, 'status'):
                        dep_status_str = str(dep.status).upper()
                        if 'ERROR' in dep_status_str or 'FAILED' in dep_status_str:
                            print(f"    Deployment {deployment_name} is in error state")
                            return False

                    running = 0
                    error_count = 0
                    if hasattr(dep, 'replicas') and dep.replicas:
                        for replica in dep.replicas:
                            if hasattr(replica, 'state'):
                                state = str(replica.state).upper()
                                if state == 'RUNNING':
                                    running += 1
                                elif 'ERROR' in state or 'FAILED' in state:
                                    error_count += 1

                    if error_count > 0:
                        print(f"    ⚠️  Deployment {deployment_name} has {error_count} replicas in error state")
                        return False

                    pending_gb = 0
                    if gpu_key:
                        try:
                            gpu_info = ray.get(allocator.get_gpu_vram.remote(gpu_key))
                            if gpu_info:
                                pending_gb = gpu_info.get("pending", 0)
                        except:
                            pass

                    if check_num % 3 == 0:
                        print(f"    Deployment {deployment_name}: {running}/{expected_replicas} RUNNING, {pending_gb:.2f}GB pending on {gpu_key}")

                    if running >= expected_replicas:
                        print(f"    ✅ All {expected_replicas} replicas RUNNING on {gpu_key}")
                        return True

            time.sleep(check_interval)
        except Exception as e:
            time.sleep(check_interval)

    # Final check (same soup again) ...
    try:
        ...
    except:
        pass

    print(f"    WARNING: Timeout waiting for {deployment_name} to be fully ready")
    return False
```

## Unslopified

**Deleted entirely.** Ray Serve readiness / actor init failures surface as real errors. No fake readiness theater.

---

## AI-slop: no GPUs? “fall back” to a wrong deployment

```python
if not gpu_info_map:
    print(f"  WARNING: No GPU info available, falling back to single deployment")
    # Fallback to single deployment
    serve.run(
        VLLMModel.options(
            name=model_id,
            ray_actor_options={
                "num_gpus": 0.01,
                "memory": 2 * 1024 * 1024 * 1024,
            },
            autoscaling_config={
                "min_replicas": config["replicas"],
                "max_replicas": config["replicas"]
            }
        ).bind(
            model_id=model_id,
            model_name=config["name"],
            required_vram_gb=config["vram_gb"]
        ),
        name=model_id,
        route_prefix=f"/{model_id}"
    )
    continue
```

## Unslopified

```python
if not gpu_info_map:
    raise RuntimeError("No GPU info available from VRAM allocator")
```

---

## Same pass: single-GPU still builds a “consistency” router

```python
# Create router deployment that provides unified endpoint
# Router uses Ray Serve's built-in load balancing via deployment handles
if len(gpu_deployment_names) > 1:
    print(f"  Creating router deployment for unified endpoint: /{model_id}")
    self._create_router_deployment(model_id, gpu_deployment_names, gpu_app_names)
elif len(gpu_deployment_names) == 1:
    # Single deployment - no router needed, but create alias for consistency
    print(f"  Single deployment - accessible at /{gpu_deployment_names[0]} or /{model_id}")
    # Optionally create router anyway for consistent API
    self._create_router_deployment(model_id, gpu_deployment_names, gpu_app_names)
```

## Unslopified

```python
# Create router deployment that provides unified endpoint (only if multiple GPUs)
if len(gpu_deployment_names) > 1:
    print(f"  Creating router deployment for unified endpoint: /{model_id}")
    self._create_router_deployment(model_id, gpu_deployment_names, gpu_app_names)
```
