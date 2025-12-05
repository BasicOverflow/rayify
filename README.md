# rayify

A repository for converting Python scripts into Ray-optimized, cluster-ready code. This project provides comprehensive resources and guidance for coding agents to transform standard Python scripts into distributed Ray applications that run on remote Ray clusters.

## Overview

This repository contains:
- **AGENTS.md**: Comprehensive guide for coding agents to convert scripts to Ray
- **resources/**: Complete Ray documentation organized by topic
- **input/**: Drop your original Python scripts here
- **output/**: Rayified versions of your scripts will be placed here

## Project Structure

```
rayify/
├── input/              # Place your original Python scripts here
├── output/             # Rayified scripts will be generated here
├── resources/          # Ray documentation and examples
│   ├── ray-core/      # Core Ray concepts (tasks, actors, objects)
│   ├── data/          # Ray Data for batch processing
│   ├── train/         # Ray Train for distributed training
│   ├── tune/          # Ray Tune for hyperparameter tuning
│   ├── serve/         # Ray Serve for model serving
│   └── cluster/       # Cluster deployment guides
├── .env.example       # Example environment variables file
├── .env                # Your local environment variables (gitignored)
├── AGENTS.md          # Main conversion guide for coding agents
└── README.md          # This file
```

## How to Use

1. **Place your script** in the `input/` folder
2. **Use AGENTS.md** as a reference guide for conversion
3. **Generate the rayified version** in the `output/` folder
4. **Set environment variables** before running:
   ```bash
   export RAY_ADDRESS="ray://your-cluster:10001"
   ```

## Key Features

- **Cluster-First Approach**: All conversions assume connecting to an existing Ray cluster
- **Environment Variable Configuration**: Minimal setup using `RAY_ADDRESS`
- **Comprehensive Patterns**: Design patterns and anti-patterns for optimal performance
- **Use Case Examples**: Reference sections for data processing, training, tuning, serving, and more
- **Complete Documentation**: Full Ray API references and examples

## Workflow

1. **Input**: Drop your original Python script into `input/`
2. **Conversion**: Use AGENTS.md to guide the conversion process
3. **Output**: The rayified script goes into `output/`
4. **Execution**: Run the output script with `RAY_ADDRESS` set

## Environment Variables

The minimal required environment variable:
- `RAY_ADDRESS`: Cluster address (e.g., `ray://head-node:10001`)

Optional variables:
- `RAY_NAMESPACE`: Logical grouping namespace
- `RAY_RUNTIME_ENV`: JSON string for runtime environment
- `RAY_JOB_CONFIG`: JSON string for job configuration

### Setting Environment Variables

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** with your cluster details (this file is gitignored and won't be committed)

3. **Load the environment variables**:
   ```bash
   # On Linux/Mac
   export $(cat .env | xargs)
   
   # Or use a tool like python-dotenv
   pip install python-dotenv
   ```

See [AGENTS.md](AGENTS.md) for complete environment variable reference.

## Resources

- **AGENTS.md**: Main conversion guide with step-by-step instructions
- **resources/**: Comprehensive Ray documentation
  - Core concepts: tasks, actors, objects, scheduling
  - Design patterns and anti-patterns
  - API references
  - Examples for all Ray libraries

## Branch Strategy

**Important**: Different branches from `main` represent different instances of this project. Each branch may have:
- Different versions of the conversion guide
- Different resource documentation
- Different project configurations
- Different conversion strategies

Always check which branch you're working with and ensure you're using the correct version of the guide and resources for your specific use case.

## Prerequisites

- Python 3.7+
- Ray installed: `pip install "ray[default]"`
- Access to a Ray cluster (cluster address)
- `RAY_ADDRESS` environment variable set

## Quick Start

1. Clone this repository
2. Place your script in `input/`
3. Follow the conversion guide in `AGENTS.md`
4. Generate the rayified script in `output/`
5. Set `RAY_ADDRESS` and run the output script

## Design Patterns

This project references Ray's official design patterns:
- [Ray Core Patterns](https://docs.ray.io/en/latest/ray-core/patterns/index.html)
- All patterns are documented in `resources/ray-core/patterns/`

## Contributing

When working with this repository:
- Use `input/` for original scripts
- Use `output/` for converted scripts
- Follow the guidelines in `AGENTS.md`
- Reference resources in `resources/` directory

## Additional Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ray GitHub](https://github.com/ray-project/ray)
- [AGENTS.md](AGENTS.md) - Complete conversion guide

---

**Note**: This repository is designed to guide coding agents in performing conversions, not to automate the conversion process. The conversion is a manual process guided by the comprehensive resources and instructions provided.
