#!/usr/bin/env bash
# Manual, wheel-only rental candidate. Never builds FA2 or changes the driver.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' 'GPU_PENDING rental: L40S 48GB, Ada sm_89, Anhui region; Python 3.11; torch 2.8.0 cu128; Transformers 5.5.4; official FA2 2.8.3.post2 wheel. No source build.'
  printf '%s\n' 'bash scripts/setup_gpu_env.sh --allow-install --env /persistent/envs/nano_rl --wheel /persistent/wheels/flash_attn-...whl --python python3.11'
  exit 0
fi
if [[ "${1:-}" != "--allow-install" ]]; then
  printf '%s\n' 'Manual GPU dependency installation requires --allow-install. Use --dry-run to inspect.' >&2
  exit 2
fi
shift
python_bin=python3.11
env_dir=
wheel_file=
pip_index="${PIP_INDEX_URL:-https://pypi.org/simple}"
torch_index="${LOCKSTEP_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
wheelhouse=
while (($#)); do
  case "$1" in
    --env) env_dir="$2"; shift 2 ;;
    --wheel) wheel_file="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --pip-index-url) pip_index="$2"; shift 2 ;;
    --torch-index-url) torch_index="$2"; shift 2 ;;
    --wheelhouse) wheelhouse="$2"; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$env_dir" && -f "$wheel_file" ]] || { printf '%s\n' '--env and an existing pinned --wheel are required.' >&2; exit 2; }
[[ ! -e "$env_dir" ]] || { printf '%s\n' 'Use a new environment path; existing environments are never overwritten.' >&2; exit 2; }
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
"$python_bin" -m venv "$env_dir"
venv_python="$env_dir/bin/python"
if [[ -n "$wheelhouse" ]]; then
  export PIP_NO_INDEX=1
  export PIP_FIND_LINKS="$wheelhouse"
  "$venv_python" -m pip install 'torch==2.8.0+cu128'
else
  "$venv_python" -m pip install 'torch==2.8.0+cu128' --index-url "$torch_index"
fi
"$venv_python" -m pip install --index-url "$pip_index" 'transformers==5.5.4' packaging psutil wheel 'setuptools>=77' numpy xxhash safetensors tqdm matplotlib datasets ruff
"$venv_python" scripts/verify_gpu_wheel.py --wheel "$wheel_file"
"$venv_python" -m pip install --only-binary=:all: --no-deps "$wheel_file"
"$venv_python" -m pip install --no-deps --no-build-isolation -e .
"$venv_python" -m pip check
"$venv_python" -m pip freeze > "$env_dir/environment-freeze.txt"
printf '%s\n' 'Installed pinned binaries. GPU runtime validation is still GPU_PENDING; no GPU work was executed.'
