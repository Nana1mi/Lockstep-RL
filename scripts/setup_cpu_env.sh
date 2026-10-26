#!/usr/bin/env bash
# Isolated Python 3.11 CPU development environment. Does not install CUDA builds.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${1:-python3.11}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
"$python_bin" -m venv .envs/nano_rl
.envs/nano_rl/bin/python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cpu
.envs/nano_rl/bin/python -m pip install 'transformers==5.5.4' numpy xxhash safetensors tqdm matplotlib ruff
printf '%s\n' 'CPU setup complete. GPU stack remains GPU_PENDING; see GPU_RUNBOOK.md.'
