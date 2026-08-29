# rayify

A repository for converting Python scripts into Ray-optimized, cluster-ready code. This project provides resources and guidance for coding agents to transform standard Python scripts into distributed Ray applications that run on remote Ray clusters.

## Overview

This repository contains:
- **AGENTS.md**: Guide for coding agents to convert scripts to Ray
- **ray-resources/**: Ray documentation organized by topic
- **unslop-examples/**: Side-by-side AI-slop vs clean code (style bible for generated code)
- **orchestration-patterns/**: Docker Compose job-container packaging for Ray workloads
- **input/**: Drop your original Python scripts here
- **output/**: Rayified versions of your scripts will be placed here

## Project Structure

```
rayify/
├── input/                      # Place your original Python scripts here
├── output/                     # Rayified scripts will be generated here
├── ray-resources/              # Ray documentation and examples
├── unslop-examples/            # AI-slop vs clean pairs (code style)
├── orchestration-patterns/     # Docker Compose job-container crib
├── .env.example                # Shared Ray + infra backend key template
├── AGENTS.md                   # Main conversion guide for coding agents
└── README.md
```

## How to Use

1. **Place your script** in the `input/` folder
2. **Configure env** — copy the template and fill values:
   ```bash
   cp .env.example .env
   # set at least RAY_ADDRESS and a unique RAY_NAMESPACE per project
   ```
3. **Use AGENTS.md** as the conversion guide
4. **Generate** under `output/` (script + Dockerfile + docker-compose.yml)
5. **Run** — from `output/`:
   ```bash
   docker compose up --build job
   ```
   The job container connects to the Ray cluster and uses remote infra for durable state when needed.

## Key Features

- **Cluster-First Approach**: Connect to an existing Ray cluster via env (no implicit local Ray)
- **Per-project namespace**: Unique `RAY_NAMESPACE` per conversion/project
- **Docker packaging**: Every conversion ships `Dockerfile` + `docker-compose.yml` (job container only)
- **Workload backends via `.env`**: Ray, object store, DBs, search, registry/package indexes, etc. (keys in `.env.example`; policy in AGENTS.md)
- **Durable state on remote infra**: MinIO/S3, Postgres, Mongo, Neo4j, etc.; Alpine job image preferred; project-local env for extras (e.g. W&B) only when approved
- **Optional Ray Hive**: High-throughput LLM serving (vLLM on the cluster) when the user approves — see AGENTS.md
- **Unslop style + patterns**: Clean OOP, concurrency/object-store patterns, anti-slop examples

## Workflow

1. **Input**: Drop your original Python script into `input/`
2. **Env**: Filled root `.env` (from `.env.example`); unique project namespace
3. **Conversion**: Follow AGENTS.md — backends; Ray Hive / W&B when relevant
4. **Output**: Rayified script(s) + Docker/compose under `output/`
5. **Execution**: `docker compose up` in `output/`

See [AGENTS.md](AGENTS.md) for agent rules. See [`.env.example`](.env.example) for shared env key names. Job container packaging: [orchestration-patterns/docker-compose.md](orchestration-patterns/docker-compose.md).

## Resources

- **AGENTS.md**: Conversion guide (Ray + unslop + backends + Ray Hive)
- **.env.example**: Shared Ray/infra env key template (create project-local keys under `output/` when needed)
- **ray-resources/**: Ray API/docs and examples
- **unslop-examples/**: Anti-slop patterns agents must follow
- **orchestration-patterns/**: Docker Compose + remote-state patterns for packaged jobs
- **[ray-hive](https://github.com/BasicOverflow/ray-hive)**: Optional LLM serving SDK used on the Ray cluster

**Note**: This repository is designed to guide coding agents in performing conversions, not to automate the conversion process. The conversion is a manual process guided by the comprehensive resources and instructions provided.
