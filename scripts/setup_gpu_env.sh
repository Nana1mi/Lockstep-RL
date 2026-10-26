#!/usr/bin/env bash
# Manual candidate stack only. No driver/system changes; no GPU work is executed.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' 'GPU_PENDING candidate: Python 3.11; torch 2.11.0 cu130; triton 3.6.0; transformers 5.5.4; flash-attn 2.8.3.post2; CUDA compiler 13.0.88.'
  printf '%s\n' 'Manual install: bash scripts/setup_gpu_env.sh --allow-install /path/to/python3.11'
  exit 0
fi
if [[ "${1:-}" != "--allow-install" ]]; then
  printf '%s\n' 'Manual GPU dependency installation requires --allow-install. Use --dry-run to inspect.' >&2
  exit 2
fi
python_bin="${2:-python3.11}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
command -v c++ >/dev/null || { printf '%s\n' 'A C++ compiler is required; install build tools manually.' >&2; exit 1; }
"$python_bin" -m venv .envs/nano_rl
venv_python="$PWD/.envs/nano_rl/bin/python"
"$venv_python" -m pip install --upgrade 'torch==2.11.0+cu130' --index-url https://download.pytorch.org/whl/cu130
"$venv_python" -m pip install 'triton==3.6.0' 'transformers==5.5.4' packaging psutil ninja wheel setuptools numpy xxhash safetensors tqdm matplotlib datasets
"$venv_python" -m pip install 'nvidia-cuda-nvcc==13.0.88' 'nvidia-cuda-crt==13.0.88' 'nvidia-nvvm==13.0.88' 'nvidia-cuda-runtime==13.0.96'
export CUDA_HOME="$PWD/.envs/nano_rl/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
export FLASH_ATTN_CUDA_ARCHS=120
if [[ ! -e "$CUDA_HOME/lib/libcudart.so" && -e "$CUDA_HOME/lib/libcudart.so.13" ]]; then
  ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
fi
"$venv_python" -m pip install --no-build-isolation --no-deps 'flash-attn==2.8.3.post2'
"$venv_python" -m pip install --no-deps -e .
mkdir -p results
"$venv_python" -m pip freeze > results/gpu-environment-freeze.txt
printf '%s\n' 'Installation finished; all GPU execution is still GPU_PENDING. Run manual smoke checks next.'
