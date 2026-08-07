# Long-running jobs: Prefect + Ray

Agent crib for **long-running** conversions. Re-verify Prefect against [docs.prefect.io](https://docs.prefect.io/) (this file can go stale). Do **not** invent a repo-local checkpoint library — use Prefect + remote infra backends.

## When

Only after the dev **explicitly confirms** long-running / fault-tolerant mode (see [AGENTS.md](../AGENTS.md)). Short one-shot jobs never use this stack.

## Ask before coding

1. **Remote store** for workload progress/checkpoints (MinIO/S3, Postgres, Mongo, Neo4j from root `.env`) — reuse vs dedicated; stop if manual provisioning is needed.
2. **Training / experiment tracking** — ask if they want Weights & Biases (or similar). Only if yes, create **project-local** keys under `output/`.
3. Cross-check Prefect self-host + result persistence docs online.

## Architecture

```
Prefect server UI (compose :4200)
        │
   job container  (Prefect flow → Ray cluster)
        │
   remote store   (MinIO / DB — workload state)
        │
   Ray cluster
```

- Prefect = orchestrate, retry, logs, UI.
- Root `.env` = Ray + shared infra.
- Workload resume state = remote backend only (never host bind-mount for checkpoints/data).
- Prefect control-plane DB may use a compose volume or a host path placeholder the dev fills in.

## Env split

| Source | Contents |
|--------|----------|
| Root `.env` | `RAY_*`, MinIO/DB/Gitea toolbox; passed into `job` via compose `env_file` |
| `output/.env` | **Only** keys this project invents (W&B after approval, job-specific names). Create from scratch — do **not** add blank Prefect/W&B slots to root `.env.example` |
| `docker-compose.yml` | Service DNS for Prefect client→server only |

Do not put unused `PREFECT_*` keys in root or project templates “just in case.”

### Compose-only API wiring

Job service (when it talks to a sibling server) needs the server API over the compose network:

```yaml
environment:
  PREFECT_API_URL: http://prefect-server:4200/api
```

That is **compose glue**, not a root secret. UI is usually `http://localhost:4200` after port publish.

**Rare off-host UI:** if the browser is not on the Docker host, set on `prefect-server` what current docs call for UI API URL (e.g. `PREFECT_SERVER_UI_API_URL=http://<host>:4200/api`). Omit by default.

## Output files (long mode)

```
output/
├── flow.py
├── <ray_work>.py      # optional split
├── smoke_test.py
├── Dockerfile         # Alpine Python preferred
├── docker-compose.yml
├── .env               # only if project-only keys needed
└── requirements.txt
```

## Alpine Dockerfile skeleton

Prefer `python:3.12-alpine` (or current Alpine Python). Switch base only if a dep cannot build on Alpine.

```dockerfile
FROM python:3.12-alpine

WORKDIR /app
RUN apk add --no-cache build-base libffi-dev
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "flow.py"]
```

Install only what the job needs (`prefect`, `ray`, backend clients). Cross-check Alpine wheels for Ray; if Ray cannot install on Alpine, use the leanest alternative that works and note why in a one-line comment in the Dockerfile.

## docker-compose skeleton

```yaml
services:
  prefect-server:
    image: prefecthq/prefect:3-latest
    command: prefect server start --host 0.0.0.0
    environment:
      PREFECT_SERVER_API_HOST: "0.0.0.0"
    ports:
      - "4200:4200"
    volumes:
      # Pref control plane only — not workload checkpoints
      - ./prefect-data:/root/.prefect
      # or host placeholder: - /path/you-fill-in:/root/.prefect

  job:
    build: .
    env_file:
      - ../../.env
      # - .env   # only if project-local keys exist
    environment:
      PREFECT_API_URL: http://prefect-server:4200/api
    depends_on:
      - prefect-server
    restart: on-failure
```

- UI: open mapped `4200` on the host.
- No Prometheus/Grafana.
- No bind-mount for training data / checkpoint files.

Re-check official compose examples if Prefect splits server/services: [Self-hosted docker-compose](https://docs.prefect.io/v3/how-to-guides/self-hosted/docker-compose).

## Flow skeleton

Use Prefect tasks/flows (retries, logging). Ray does compute. Persist steps to the **chosen remote store**.

```python
import os
import ray
from prefect import flow, task, get_run_logger


@task(retries=2, retry_delay_seconds=30)
def connect_ray():
    ray.init(
        address=os.environ["RAY_ADDRESS"],
        namespace=os.environ["RAY_NAMESPACE"],
        ignore_reinit_error=True,
    )


@task
def load_progress():
    # read cursor / checkpoint from MinIO or DB (confirmed remote store)
    ...


@task
def run_step(state):
    # call Ray remotes / Train; write progress back to remote store
    ...
    return state


@flow(name="job")
def main():
    log = get_run_logger()
    connect_ray()
    state = load_progress()
    state = run_step(state)
    log.info("step done: %s", state)


if __name__ == "__main__":
    main()
```

- Log meaningful progress with Prefect so it shows in the UI.
- Ray Train/Tune: use their checkpoint APIs into the same remote store; still enter via the flow.
- Actors hold only temporary memory; reload from remote on restart.

Official: [tasks](https://docs.prefect.io/v3/concepts/tasks), [flows](https://docs.prefect.io/v3/concepts/flows), [retries](https://docs.prefect.io/v3/how-to-guides/workflows/retries), [results](https://docs.prefect.io/v3/concepts/results) — re-open online.

## Smoke

Same real Ray cluster and remote store as prod.

```bash
# host (API → server if server already up), or inside job service:
python smoke_test.py
```

- Tiny slice of the real path (one step / few items).
- Print proof from Ray **and** remote store (e.g. checkpoint key / row).
- Fail loud; no soft-fail soup.

## Anti-patterns

- Custom `LongRunningJob` / shared checkpoint base classes in this repo
- Host bind-mount as recovery for workload state
- Prefect/W&B blank keys in root `.env.example`
- Prefetching unused `PREFECT_UI_URL` / similar
