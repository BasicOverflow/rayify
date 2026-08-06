# Example scripts: banners, emoji, comment manuals

## Slop tells

- `====` section banners for every micro-step
- Emoji success/fail in library + scripts
- Inline manuals explaining every kwarg like a README pasted into code
- Timing print for things that don’t matter
- Soft “if results look wrong warn else print” when you can just print

---

## 1. Deploy example: parameter encyclopedia

### AI-slop

```python
"""Deploy models using various strategies."""
# ...

# Deploy with custom vLLM kwargs
# Replicas: one per GPU (capped at number of available GPUs)
# test_mode=True: deploy only on GPU with most VRAM (useful for testing)
# max_num_seqs calculated automatically if not provided
# Model architecture params can be auto-detected from HuggingFace config
# scheduler.deploy_model(
#     model_id="qwen-custom",
#     replicas=1,  # Number of replicas (ignored when test_mode=True)
#     test_mode=True,  # Set to True to deploy only on GPU with most VRAM
#     model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
#     vram_weights_gb=0.763,  # Model weights only (KV cache calculated separately)
#     max_num_seqs=None,  # Max concurrent sequences per instance (optional, calculated if not provided)
#     max_model_len=2048,  # max prompt length
#     # Architecture params (auto-detected if not provided):
#     # hidden_dim=768,  # Model hidden dimension
#     # num_layers=12,   # Number of transformer layers
#     # dtype="int8",    # Model dtype (int8, fp8, fp16, bf16, fp32)
#     enforce_eager=True,  # pre-allocate necessary buffers to avoid lazy page faults
#     disable_custom_all_reduce=True,  # useful if running single-GPU, disables distributed reduce
#     kv_cache_dtype="fp8",  # reduce KV memory footprint
# )

scheduler.deploy_model(
    model_id="qwen-custom",
    replicas=6,  # Number of replicas (ignored when test_mode=True)
    test_mode=False,  # Set to True to deploy only on GPU with most VRAM
    model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
    vram_weights_gb=0.763,  # Model weights only (KV cache calculated separately)
    max_num_seqs=None,  # ...
    max_model_len=2048,
    enforce_eager=True,  # pre-allocate necessary buffers...
    # ...
)
```

### Unslopified

Configs live as data dicts without a novel’s worth of inline docs. Examples stay sparse: deploy → run → print one number.

---

## 2. Inference example: demo parade vs one path

### AI-slop

```python
# Warmup
print("Warming up...")
_ = inference(prompt, model_id=MODEL_ID, max_tokens=10, temperature=0.0)

# Synchronous inference
print("\n=== Synchronous Inference ===")
start = time.time()
result = inference(prompt, model_id=MODEL_ID, max_tokens=100, temperature=0.7)
elapsed = time.time() - start
print(f"Time: {elapsed:.3f}s")
print(f"Sample: {result}")

# Async inference
print("\n=== Async Inference ===")
async def test_async():
    ...
asyncio.run(test_async())

# Batch inference
print("\n=== Batch Inference ===")
...

# Async batch inference
print("\n=== Async Batch Inference ===")
...

# Structured output
print("\n=== Structured Output ===")
...
```

### Unslopified

```python
"""Test inference features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from ray_hive.inference import inference, inference_batch

MODEL_ID = "qwen-max-concurrency"
prompt = "Write a short poem about beer"
amount = 10_000

_ = inference(prompt, model_id=MODEL_ID, max_tokens=10, temperature=0.0)

prompts = [f"{prompt} {i}" for i in range(amount)]
start = time.time()
results = inference_batch(prompts, model_id=MODEL_ID, max_tokens=100, temperature=0.0)
elapsed = time.time() - start
print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")
```

---

## 3. Shutdown / client: emoji ops center

### AI-slop

```python
print("✅ Killed VRAM allocator actor (will be recreated fresh)")
print(f"Shutting down {len(apps)} application(s): {list(apps.keys())}")
print("✅ All applications shut down")
print(f"✅ Cleared {cleared} VRAM reservations")
print(f"⚠️  Model '{model_id}' not found in deployments")
print(f"✅ Shut down {model_id}")
```

### Unslopified

```python
serve.shutdown()
ray.get(allocator.clear_all_reservations.remote())
# ...
for app_name in apps_to_delete:
    try:
        serve.delete(name=app_name)
    except Exception:
        pass
ray.get(allocator.clear_reservations_by_prefix.remote(f"{model_id}-"))
```

Also: dropped unused `self._deployed_models` bookkeeping on the client that only existed so shutdown could `.clear()` it.

### AI-slop: `display_vram_state` multi-line report

```python
print("\nVRAM State (Per-GPU):")
print("-" * 80)
for gpu_key, info in sorted(state.items()):
    ...
    print(f"GPU {gpu_key}:")
    print(f"  Total: {total:.2f}GB, Free: {free:.2f}GB, Available: {available:.2f}GB")
    print(f"  Pending: {pending:.2f}GB, Active: {active:.2f}GB ({active_count} instances)")
print()
```

### Unslopified

```python
for gpu_key, info in sorted(state.items()):
    if len(gpu_key) > 50 or gpu_key.startswith('c'):
        continue
    print(f"GPU {gpu_key}: {info.get('available', 0):.2f}GB available / {info.get('total', 0):.2f}GB total")
```

---

## 4. Early test script banners

### AI-slop

```python
print(f"\n{'='*80}")
print(f"Testing {MODEL_ID} with {NUM_REQUESTS} concurrent inference requests")
print(f"{'='*80}\n")
...
print(f"\n{'='*80}")
print(f"Results Summary")
print(f"{'='*80}")
print(f"Total requests: {NUM_REQUESTS}")
print(f"Total time: {elapsed:.2f}s")
print(f"Average time per request: {elapsed/NUM_REQUESTS:.2f}s")
print(f"Throughput: {NUM_REQUESTS/elapsed:.2f} requests/second")
...
print(f"\n✅ All results written to: {output_file}")
print(f"❌ Error: ...")
except Exception as e:
    print(f"❌ Failed to test {MODEL_ID}: {e}")
    import traceback
    traceback.print_exc()
```

### Unslopified

```python
print(f"Testing {MODEL_ID}: {NUM_REQUESTS} requests...")
...
print(f"Results: {successes}/{NUM_REQUESTS} success, {elapsed:.2f}s total, {NUM_REQUESTS/elapsed:.2f} req/s")
print(f"Results saved to: {output_file}")
```

---

## 5. Script headers as product docs

### AI-slop

```python
"""
Deploy multiple models using the VRAM scheduler.

This is the main deployment script. It:
1. Initializes the VRAM allocator
2. Deploys all models from the MODELS configuration
3. Shows final VRAM state

Usage:
    python scripts/vram-scheduler/1_deploy_models.py
"""
# Add vram-scheduler directory to path (folder has hyphen, can't be imported as package)
# Import modules directly (since folder name has hyphen)
print("Initializing VRAM allocator...")
print("Deploying models...")
print("\nChecking VRAM state...")
```

### Unslopified

```python
"""Deploy multiple models using the VRAM scheduler."""
# ...
get_vram_allocator()
orchestrator = ModelOrchestrator.remote()
ray.get(orchestrator.apply.remote(MODELS))
```
