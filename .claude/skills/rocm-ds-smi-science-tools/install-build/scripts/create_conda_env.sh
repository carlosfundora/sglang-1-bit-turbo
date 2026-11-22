#!/bin/bash
# Simple scaffold for conda creation
env_name=${1:-rocm_ds_env}
echo "Creating conda environment: $env_name"
conda create -n "$env_name" python=3.10 -y || echo "Conda not available, fallback to uv"
