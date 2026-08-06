# KV / attention math: Ray actor blob → AttentionSpecs + VramReqs

## Slop tells

- `AutoConfig` probing and KV algebra living inside a Serve deployment’s `__init__`
- Manual `dtype_sizes` tables, print-debug of every intermediate GiB
- Weights / activations / KV mixed in the same method as CUDA pin + `LLM(...)`
- No way to unit-test or override attention without spinning Ray
- “Planning” happens after the replica process already started

---

## AI-slop: math baked into the model actor

```python
# vLLM Model Actor - one instance per GPU
# Each deployment targets a specific GPU via CUDA_VISIBLE_DEVICES
# Calculates optimal VRAM usage based on vLLM's PagedAttention memory model
# Supports CPU offloading for higher concurrency

@serve.deployment(...)
class VLLMModel:
    def __init__(
        self,
        model_id: str,
        model_name: str,
        required_vram_weights_gb: float,
        max_input_prompt_length: int,
        max_output_prompt_length: int,
        target_gpu_id: str = None,
        max_num_seqs: int = None,
        max_num_batched_tokens: int = None,
        gpu_utilization_target: float = 0.96,
        swap_space: float = 0.0,
        **vllm_kwargs
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_id
        # ... path hacks, allocator reserve ...

        from vllm import LLM
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden_dim = getattr(config, 'hidden_size', None) or getattr(config, 'd_model', None)
        num_layers = (
            getattr(config, 'num_hidden_layers', None)
            or getattr(config, 'num_layers', None)
            or getattr(config, 'n_layer', None)
        )
        num_heads = (
            getattr(config, 'num_attention_heads', None)
            or getattr(config, 'num_heads', None)
            or getattr(config, 'n_head', None)
        )

        kv_dtype = vllm_kwargs.get("kv_cache_dtype", "fp16")
        dtype = "int8" if kv_dtype in ["int8", "uint8"] else "fp8" if kv_dtype == "fp8" else "fp16"

        dtype_sizes = {
            "int8": 1, "uint8": 1, "fp8": 1,
            "fp16": 2, "bf16": 2, "fp32": 4,
        }
        dtype_size_bytes = dtype_sizes.get(dtype.lower(), 2)

        # Step 2a: Calculate KV cache per sequence
        kv_per_token_per_layer_bytes = 2 * hidden_dim * dtype_size_bytes
        kv_per_seq_bytes = kv_per_token_per_layer_bytes * num_layers * self.max_sequence_length
        kv_per_seq_gb = kv_per_seq_bytes / (1024**3)

        activation_per_token_bytes = hidden_dim * dtype_size_bytes * num_layers
        activation_buffer_bytes = activation_per_token_bytes * max_num_batched_tokens
        activation_buffer_gb = activation_buffer_bytes / (1024**3)

        kv_cache_total_gb = kv_per_seq_gb * max_num_seqs
        total_needed_gb = required_vram_weights_gb + kv_cache_total_gb + activation_buffer_gb

        vllm_init_kwargs = {
            "model": model_name,
            "max_model_len": self.max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            ...
        }
        self.llm = LLM(**vllm_init_kwargs)
        # re-reserve with total_needed_gb after load...
```

Earlier variants also did `print` thrash mid-formula:

```python
print(f"[{model_id}] KV cache calculation: hidden_dim={hidden_dim}, num_layers={num_layers}, ...", flush=True)
print(f"[{model_id}] KV cache per sequence: {kv_cache_per_seq_gb:.6f}GB ...", flush=True)
print(f"[{model_id}] ✅ Calculated max_num_seqs={max_num_seqs} for GPU {gpu_key} ...", flush=True)
print(f"[{model_id}] VRAM breakdown: weights=..., KV cache=..., activation=..., total=...", flush=True)
```

Problems:

1. **Wrong layer** — pure arithmetic next to CUDA + Serve lifecycle  
2. **Not extensible** — GQA/MQA/hybrid layers mean more `if` spaghetti in the actor  
3. **Late** — you discover OOM math after the worker is already up  
4. **Weights hardcoded** — caller passes `required_vram_weights_gb` instead of estimating from HF

---

## Unslopified: split responsibilities

### 1. AttentionSpecs — only how KV scales

```python
class BaseAttentionSpecs(ABC):
    def __init__(self, kv_bytes_per_element: float = 8, **hf_params: Any):
        self.hf_params = hf_params
        self.kv_bytes_per_element = kv_bytes_per_element

    @property
    def head_dim(self) -> int:
        if self.hf_params.get("head_dim") is not None:
            return self.hf_params["head_dim"]
        hidden_size = self._hf_any("hidden_size", "d_model")
        num_heads = self._hf_any("num_attention_heads", "num_heads", "n_head")
        return hidden_size // num_heads

    @property
    def kv_heads(self) -> int:
        return self._hf_any(
            "num_key_value_heads", "num_kv_heads",
            "num_attention_heads", "num_heads", "n_head",
        )

    @property
    def num_layers(self) -> int:
        return self._hf_any("num_hidden_layers", "num_layers", "n_layer", "n_layers")

    @property
    def kv_layers(self) -> int:
        return self.num_layers

    def kv_bytes_per_token(self) -> float:
        return 2 * self.kv_bytes_per_element * self.kv_layers * self.kv_heads * self.head_dim

    def kv_bytes_per_sequence(self, max_model_len: int) -> float:
        return self.kv_bytes_per_token() * max_model_len

    def calc_max_num_seqs_given_kv_cache(self, max_model_len: int, kv_cache_gib: float) -> int:
        kv_budget_bytes = kv_cache_gib * (1024 ** 3)
        bytes_per_seq = self.kv_bytes_per_sequence(max_model_len)
        return max(1, math.floor(kv_budget_bytes / bytes_per_seq))


# Non-standard attention = subclass, not if-blocks in Serve
class Qwen35AttentionSpecs(BaseAttentionSpecs):
    @property
    def kv_layers(self) -> int:
        return self.hf_params["num_attention_layers"]


class TensorParallelAttentionSpecs(BaseAttentionSpecs):
    def __init__(self, tp_size: int, **kwargs):
        super().__init__(**kwargs)
        self.tp_size = tp_size

    def kv_bytes_per_token(self) -> float:
        return super().kv_bytes_per_token() / self.tp_size
```

### 2. VramReqs — non-KV buckets + compose with attention

```python
class BaseVramReqs(ABC):
    attention_cls: type[BaseAttentionSpecs] = BaseAttentionSpecs

    def __init__(self, speculative_decoding_enabled: bool = False,
                 kv_cache_dtype_bytes: float = 8.0, **hf_params: Any):
        self.hf_params = hf_params
        self.attention = self.attention_cls(
            kv_bytes_per_element=kv_cache_dtype_bytes,
            **hf_params,
        )

    def calc_system_overhead_gb(self) -> float:
        return 0.5

    def calc_weights_gb(self) -> float:
        # HF-derived dense transformer estimate (override for MoE / published param counts)
        ...

    def calc_activation_gb(self, max_num_batched_tokens: int) -> float:
        ...

    def calc_kv_cache_gb(self, max_model_len: int, max_num_seqs: int) -> float:
        return (
            self.attention.kv_bytes_per_sequence(max_model_len) * max_num_seqs
        ) / (1024 ** 3)

    def calc_non_kv_vram_gb(self, max_num_batched_tokens: int) -> float:
        return (
            self.calc_system_overhead_gb()
            + self.calc_weights_gb()
            + self.calc_misc_vram_gb()
            + self.calc_activation_gb(max_num_batched_tokens)
        )
```

### 3. Planner — inverse problem *before* deploy

```python
def plan_deployment(
    vram_reqs: BaseVramReqs,
    used_vram_gb: float,
    live_free_vram_gb: float,
    live_total_vram_gb: float,
    max_model_len: int,
    input_len: int,
    output_len: int,
    max_num_batched_tokens_override: int | None = None,
) -> dict:
    available_vram_gb = live_free_vram_gb - used_vram_gb
    # estimate or take override for max_num_batched_tokens
    non_kv_vram_gb = vram_reqs.calc_non_kv_vram_gb(max_num_batched_tokens)
    kv_cache_gb = available_vram_gb - non_kv_vram_gb
    max_num_seqs = vram_reqs.attention.calc_max_num_seqs_given_kv_cache(
        max_model_len, kv_cache_gb
    )
    total_vram_gb = non_kv_vram_gb + kv_cache_gb
    return {
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": total_vram_gb / live_total_vram_gb,
        "kv_cache_gb": kv_cache_gb,
        "non_kv_vram_gb": non_kv_vram_gb,
        "total_vram_gb": total_vram_gb,
    }
```

Deploy path becomes: HF config → `build_vram_reqs` → `plan_deployment` per GPU → pass finished `engine_kwargs` down.

### 4. Actor — pin GPU, construct engine, stop thinking

```python
@serve.deployment(...)
class RayLLMActor(LLM):
    """Ray Serve replica — vLLM LLM engine pinned to one GPU."""

    def __init__(self, model_id: str, target_gpu_id: str, engine_kwargs: dict):
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu_id
        os.environ["VLLM_DISABLE_MARLIN"] = "1"
        self.model_id = model_id
        super().__init__(**engine_kwargs)
```

No HF probing. No KV formulas. No dtype tables. Deploy service already chose `max_num_seqs`, utilization, etc.

---

## Why this is the architectural unslop, not just deletions

| Before | After |
|--------|--------|
| Planning *is* the Serve constructor | Planning is pure Python on the controller |
| Attention hard-coded as `2 * hidden * layers * …` | `AttentionSpecs` with overridable `kv_heads` / `kv_layers` |
| Weights are a guess passed as float | `VramReqs.calc_weights_gb()` (or model-specific subclass) |
| Can’t unit-test without Ray | Specs + planner are plain objects |
| Hybrid / TP needs more actor branches | Subclass (or inject `attention_cls`) |

Same numbers, wrong house → right house.
