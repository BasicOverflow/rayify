# rayify

A repository for converting Python scripts into Ray-optimized, cluster-ready code. This project provides resources and guidance for coding agents to transform standard Python scripts into distributed Ray applications that run on remote Ray clusters.

## Overview

This repository contains:
- **AGENTS.md**: Guide for coding agents to convert scripts to Ray
- **ray-resources/**: Ray documentation organized by topic
- **unslop-examples/**: Side-by-side AI-slop vs clean code (style bible for generated code)
- **input/**: Drop your original Python scripts here
- **output/**: Rayified versions of your scripts will be placed here

## Project Structure

```
rayify/
├── input/              # Place your original Python scripts here
├── output/             # Rayified scripts will be generated here
├── ray-resources/      # Ray documentation and examples
├── unslop-examples/    # AI-slop vs clean pairs (code style)
├── .env.example        # Workload backend key template (Ray, S3, DBs, etc.)
├── AGENTS.md           # Main conversion guide for coding agents
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
4. **Generate the rayified version** in the `output/` folder (include a small real-cluster smoke script when applicable)
5. **Run** with the env loaded so `RAY_ADDRESS` / `RAY_NAMESPACE` (and any backends) are set

## Key Features

- **Cluster-First Approach**: Connect to an existing Ray cluster via env (no implicit local Ray)
- **Per-project namespace**: Unique `RAY_NAMESPACE` per conversion/project
- **Workload backends via `.env`**: Ray, object store, DBs, search, registry/package indexes, etc. (keys in `.env.example`; policy in AGENTS.md)
- **Optional Ray Hive**: High-throughput LLM serving (vLLM on the cluster) when the user approves — see AGENTS.md
- **Unslop style + patterns**: Clean OOP, concurrency/object-store patterns, anti-slop examples
- **Smoke tests**: Tiny runs on the real cluster/infra that print proof of success

## Workflow

1. **Input**: Drop your original Python script into `input/`
2. **Env**: Filled root `.env` (from `.env.example`); unique project namespace
3. **Conversion**: Follow AGENTS.md (backends, Ray Hive permission, implementation)
4. **Output**: Rayified script(s) + smoke test under `output/`
5. **Execution**: Run against the real cluster with env set

See [AGENTS.md](AGENTS.md) for agent rules. See [`.env.example`](.env.example) for env key names.

## Resources

- **AGENTS.md**: Conversion guide (Ray + unslop + backends + Ray Hive)
- **.env.example**: Env key template for cluster and workload services
- **ray-resources/**: Ray API/docs and examples
- **unslop-examples/**: Anti-slop patterns agents must follow
- **[ray-hive](https://github.com/BasicOverflow/ray-hive)**: Optional LLM serving SDK used on the Ray cluster

**Note**: This repository is designed to guide coding agents in performing conversions, not to automate the conversion process. The conversion is a manual process guided by the comprehensive resources and instructions provided.
