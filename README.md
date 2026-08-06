# rayify

A repository for converting Python scripts into Ray-optimized, cluster-ready code. This project provides comprehensive resources and guidance for coding agents to transform standard Python scripts into distributed Ray applications that run on remote Ray clusters.

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
├── ray-resources/     # Ray documentation and examples
├── unslop-examples/   # AI-slop vs clean pairs (code style)
├── .env.example
├── AGENTS.md          # Main conversion guide for coding agents
└── README.md
```

## How to Use

1. **Place your script** in the `input/` folder
2. **Use AGENTS.md** as a reference guide for conversion
3. **Generate the rayified version** in the `output/` folder
4. **Set environment variables** before running:
   ```bash
   export RAY_ADDRESS="ray://your-cluster:10001"
   export RAY_NAMESPACE="production"  # Required
   ```

## Key Features

- **Cluster-First Approach**: All conversions assume connecting to an existing Ray cluster
- **Environment Variable Configuration**: Minimal setup using `RAY_ADDRESS` and `RAY_NAMESPACE` (required)
- **Comprehensive Patterns**: Design patterns and anti-patterns for optimal performance
- **Use Case Examples**: Reference sections for data processing, training, tuning, serving, and more
- **Complete Documentation**: Full Ray API references and examples

## Workflow

1. **Input**: Drop your original Python script into `input/`
2. **Conversion**: Use AGENTS.md to guide the conversion process
3. **Output**: The rayified script goes into `output/`
4. **Execution**: Run the output script with `RAY_ADDRESS` and `RAY_NAMESPACE` set



See [AGENTS.md](AGENTS.md) for complete environment variable reference.

## Resources

- **AGENTS.md**: Conversion guide (Ray + unslop style)
- **ray-resources/**: Ray API/docs and examples
- **unslop-examples/**: Anti-slop patterns agents must follow



**Note**: This repository is designed to guide coding agents in performing conversions, not to automate the conversion process. The conversion is a manual process guided by the comprehensive resources and instructions provided.
