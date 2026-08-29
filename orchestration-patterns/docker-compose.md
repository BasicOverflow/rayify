# Job container: Docker + Ray

Agent crib for packaging rayified work. Re-verify Docker Compose against [docs.docker.com/compose](https://docs.docker.com/compose/) if needed.

## When

Every conversion ships this stack (see [AGENTS.md](../AGENTS.md)).

## Before coding

1. Confirm which remote store holds progress/checkpoints when the job needs durable state (MinIO/S3, Postgres, Mongo, Neo4j — reuse vs dedicated).
2. Pass root `.env` into the job service; add `output/.env` only for project-local keys (e.g. W&B after approval).

## Shape

```
docker compose up job
   └─ job container  (python <script>.py → Ray cluster)
         └─ remote store for progress / artifacts when needed
```

- Ray = compute on the existing cluster.
- Job container = deps + entrypoint; no orchestration UI.
- Remote backend = durable progress, checkpoints, outputs.

## Env layers

| Layer | Role |
|---|---|
| repo-root `.env` | `RAY_ADDRESS`, `RAY_NAMESPACE`, shared infra keys |
| `output/.env` | **Only** keys this project invents (W&B after approval, job-specific names). Create from scratch — do **not** add blank W&B slots to root `.env.example` |
| `docker-compose.yml` | Mount/pass env into the job service |

## Output tree

```
output/
├── <script>.py
├── <ray_work>.py         # optional
├── Dockerfile
├── docker-compose.yml
├── .env                  # optional project-local keys
└── requirements.txt
```

## Dockerfile

Prefer `python:3.x-alpine`. Install only what the job needs (`ray`, backend clients). Cross-check Alpine wheels for Ray; if Ray cannot install on Alpine, use the leanest alternative that works and note why in a one-line comment in the Dockerfile.

```dockerfile
FROM python:3.12-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "<script>.py"]
```

## docker-compose.yml

Single job service. Pass root `.env`; the job connects to `RAY_ADDRESS` on the LAN.

```yaml
services:
  job:
    build: .
    env_file:
      - ../.env
      - .env
    restart: "no"
```

Adjust `env_file` paths if compose is run from a different cwd. Add volume mounts only for read-only input data the container cannot reach over the network — not for checkpoint recovery (use the remote store).

## Entry script

- `ray.init(address=..., namespace=...)` from env.
- Persist/resume progress via the chosen remote backend when the workload needs it.
- Actors are disposable; reload state from remote on restart.
- Do **not** invent a repo-local checkpoint base-class package.

## Run

From `output/`:

```bash
docker compose up --build job
```

Or run `<script>.py` directly on the host with root `.env` loaded for local debugging — compose is still shipped as the standard run path.

## Avoid

- Prefect or other orchestration servers in compose
- Monitoring / dashboard UIs in compose (Prometheus, Grafana, Prefect UI, etc.)
- Host bind-mounts for checkpoint recovery
- Custom `LongRunningJob` / shared checkpoint base classes in this repo
- Blank W&B keys in root `.env.example`
