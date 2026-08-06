# Placement Group APIs

APIs for creating and managing placement groups (collections of resources).

## Creating Placement Groups

### ray.util.placement_group()

Asynchronously create a placement group.

```python
import ray

# Create placement group
pg = ray.util.placement_group([
    {"CPU": 2, "GPU": 1},
    {"CPU": 4}
], strategy="PACK")

# Wait for ready
ray.get(pg.ready())
```

**Parameters:**
- `bundles`: List of resource requirement dictionaries
- `strategy`: Placement strategy
  - "PACK": Pack bundles into as few nodes as possible
  - "SPREAD": Place bundles across distinct nodes evenly
  - "STRICT_PACK": Pack into one node only
  - "STRICT_SPREAD": Place across distinct nodes (required)
- `name`: Name of the placement group
- `lifetime`: "detached" for independent lifetime, None for fate-sharing

**Returns:** PlacementGroup object

## PlacementGroup Methods

### ready()

Returns an ObjectRef to check ready status.

```python
ray.get(pg.ready())  # Blocks until ready
```

### wait()

Wait for placement group to be ready within specified time.

```python
pg.wait(timeout=60)
```

**Attributes:**
- `bundle_count`: Number of bundles
- `bundle_specs`: List of bundle specifications
- `is_empty`: Whether placement group is empty

## Placement Group Utilities

### ray.util.get_current_placement_group()

Get the current placement group for a task or actor.

```python
pg = ray.util.get_current_placement_group()
```

Returns None if no placement group is associated.

### ray.util.placement_group.get_placement_group()

Get a placement group by name.

```python
pg = ray.util.placement_group.get_placement_group("my_pg")
```

### ray.util.remove_placement_group()

Asynchronously remove a placement group.

```python
ray.util.remove_placement_group(pg)
```

### ray.util.placement_group_table()

Get state of placement group from GCS.

```python
state = ray.util.placement_group_table(pg)
```

## Using Placement Groups

```python
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Schedule task with placement group
@ray.remote
def task():
    return "done"

result = task.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0
    )
).remote()
```

