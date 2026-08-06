# Router overbuild: stats you’ll never read

## Slop tells

- Implement request-distribution counters “for observability”
- Then immediately in the test script: “we can’t easily call `get_stats`” + dead try/except note
- Five parallel arrays (`names`, `weights`, `counts`, `counters`, `connected_replicas`) for `random.choices`
- Comments narrating AWS-blog-post load balancing while selecting at random earlier

---

## AI-slop: router as mini metrics system

```python
class Router:
    def __init__(self):
        self.deployment_handles = []
        self.deployment_names = []
        self.weights = []
        self.request_counts = []
        self.round_robin_counters = []
        connected_replicas = []

        total_replicas = sum(replica_counts)

        for dep_name, app_name, replicas in zip(...):
            try:
                handle = serve.get_deployment_handle(dep_name, app_name=app_name)
                self.deployment_handles.append(handle)
                self.deployment_names.append(dep_name)
                connected_replicas.append(replicas)
                self.request_counts.append(0)
                self.round_robin_counters.append(0.0)
                print(f"    Router: Connected to {dep_name} ({replicas} replicas)")
            except Exception as e:
                print(f"    Router: Warning - Could not connect to {dep_name}: {e}")

        # Calculate weights based on connected deployments only
        connected_total = sum(connected_replicas)
        if connected_total > 0:
            self.weights = [r / connected_total for r in connected_replicas]
            for name, weight in zip(self.deployment_names, self.weights):
                print(f"    Router: {name} weight: {weight*100:.1f}%")
        else:
            # Fallback to equal weights if no replicas info
            self.weights = [1.0 / len(self.deployment_handles)] * len(self.deployment_handles)

    async def __call__(self, request):
        """Forward request using weighted round-robin to ensure all deployments get requests."""
        idx = min(range(len(self.deployment_handles)),
                  key=lambda i: self.round_robin_counters[i] / self.weights[i]
                  if self.weights[i] > 0 else float('inf'))
        handle = self.deployment_handles[idx]
        self.request_counts[idx] += 1
        self.round_robin_counters[idx] += 1
        ...

    def get_stats(self):
        """Get request distribution statistics."""
        total = sum(self.request_counts)
        if total == 0:
            return ["No requests processed yet"]
        stats = []
        for name, count, weight in zip(self.deployment_names, self.request_counts, self.weights):
            pct = (count / total * 100) if total > 0 else 0
            expected = weight * 100
            stats.append(
                f"{name}: {count} requests ({pct:.1f}% actual, {expected:.1f}% expected)"
            )
        return stats
```

### Dead consumer in the same test iteration

```python
# Try to get router stats if using router
if app_name == MODEL_ID:
    try:
        router_handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
        # Router has get_stats method, but we can't easily call it from here
        # The stats are tracked internally
        print("Note: Router tracks request distribution internally")
    except:
        pass
```

(Observability theater: build the API, then print that you can’t use it.)

---

## Unslopified

```python
class Router:
    def __init__(self):
        self.deployment_handles = []
        self.weights = []
        total_replicas = sum(replica_counts)

        for dep_name, app_name, replicas in zip(...):
            try:
                handle = serve.get_deployment_handle(dep_name, app_name=app_name)
                self.deployment_handles.append(handle)
                self.weights.append(replicas / total_replicas)
                print(f"    Router: Connected to {dep_name} ({replicas} replicas, {replicas/total_replicas*100:.1f}% weight)")
            except Exception as e:
                print(f"    Router: Warning - Could not connect to {dep_name}: {e}")

        if not self.deployment_handles:
            raise RuntimeError("Router: No deployment handles available")

    async def __call__(self, request):
        """Forward request to a deployment handle weighted by replica count."""
        handle = random.choices(self.deployment_handles, weights=self.weights)[0]
        result = await handle.remote(request)
        ...
```

`get_stats`, counters, round-robin bookkeeping, and the “Note: Router tracks…” block all deleted.

---

## Separate later spiral: library router as formula essay

Not pure “delete,” but the *tone* of AI-slop over-architecture:

```python
# Calculate performance factors (inverse of normalized response time)
# Faster GPUs get higher performance factor
min_response_time = min(self._response_times) if self._response_times else 0.1
max_response_time = max(self._response_times) if self._response_times else 1.0
response_range = max(max_response_time - min_response_time, 0.001)  # Avoid division by zero

performance_factors = []
for i, response_time in enumerate(self._response_times):
    # Normalize: faster = higher factor
    # Factor = (max - current) / range, normalized to [0.5, 2.0]
    normalized = (max_response_time - response_time) / response_range
    factor = 0.5 + (normalized * 1.5)  # Scale to [0.5, 2.0]
    performance_factors.append(factor)

# Dynamic weight = (base_weight * performance_factor) / load_factor
# This biases toward fast, less loaded GPUs
...
# Weighted Round-Robin with dynamic weights
```

Lots of formula narrative and edge clamps for a problem later code solved with direct capacity + measured speed, not five EMAs of invented constants.

---

## Moral of this pattern

If you are about to:

1. track percentages of traffic, and  
2. write a comment that nothing can read those percentages,

stop and delete the counters.
