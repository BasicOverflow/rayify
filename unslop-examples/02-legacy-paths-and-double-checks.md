# Legacy branches, alias renames, and double-checks

## Slop tells

- “Legacy: detect GPU if target is None” after the call path always sets target
- `actual_required_gb = required_vram_gb` (rename for vibes)
- Re-validate what `reserve()` already decided
- Comments narrating the obvious (“use actual VRAM… no conservative multipliers”)

---

## AI-slop: optional GPU target + alias variable

```python
# Use VRAM requirement directly (already accounts for model + KV cache + overhead)
actual_required_gb = required_vram_gb

# Determine GPU key based on target_gpu_id
if target_gpu_id is not None:
    # We set CUDA_VISIBLE_DEVICES, so we know exactly which GPU we're using
    gpu_key = f"{k8s_node_name}:gpu{target_gpu_id}"
else:
    # Legacy: detect actual GPU being used (CUDA is already initialized)
    actual_gpu_id = torch.cuda.current_device()
    gpu_key = f"{k8s_node_name}:gpu{actual_gpu_id}"
```

## Unslopified

```python
# Determine GPU key - target_gpu_id is always provided now
if target_gpu_id is None:
    raise RuntimeError("target_gpu_id must be provided")
gpu_key = f"{k8s_node_name}:gpu{target_gpu_id}"
```

Then use `required_vram_gb` directly — no `actual_required_gb` clone.

---

## AI-slop: second VRAM free-check after reserve already passed

```python
# Memory check using daemonset's nvidia-smi data (consistent with reservation system)
# The reservation already validated availability, but we double-check using the same source
free_gb = gpu_info.get("free", 0)  # nvidia-smi free from daemonset
requested_by_vllm_gb = gpu_memory_utilization * total_memory_gb

if free_gb < requested_by_vllm_gb:
    available_gb = gpu_info.get("available", 0)
    raise RuntimeError(
        f"Insufficient GPU memory on {gpu_key}: need {requested_by_vllm_gb:.2f}GB, "
        f"have {free_gb:.2f}GB free (available: {available_gb:.2f}GB) from nvidia-smi"
    )
```

## Unslopified

Deleted. Reserve + load is enough; the “double-check” only adds race noise and branch fog.

---

## AI-slop: tautological math comment

```python
# Use actual VRAM requirement (no conservative caps)
total_gb = gpu_info.get("total", 16.0)
gpu_fraction = (required_per_replica_gb * replicas_for_gpu) / total_gb / replicas_for_gpu
```

(Multiply by `n` then divide by `n` — the replicas cancel. Pure AI rearrangement.)

## Unslopified

```python
gpu_fraction = required_per_replica_gb / total_gb
```

---

## Related: invent “available_mb” then never use it

```python
required_per_replica_mb = int(required_per_replica_gb * 1024)  # Convert to MB for resource request
available_mb = int(available_gb * 1024)
```

Deleted when nothing consumed those locals.
