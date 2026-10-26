# WSL2 GPU runbook

All GPU experiments and large downloads are manual-only. Ordinary development
and CPU tests must not start CUDA work. Every executable GPU entry requires
`--allow-gpu`, and downloads require `--allow-download`.

The handoff is still being developed; consult IMPLEMENTATION_STATUS.md for entries
that are not yet prepared. Nothing below establishes that a memory budget works.

Run from Ubuntu WSL2, with the project at `/mnt/e/Lockstep-RL`.
The commands below are entry points, not claims that all experiments have passed.
See `PROGRESS.md` and result JSON for actual outcomes.

## Environment

```bash
cd /mnt/e/Lockstep-RL
bash scripts/setup_cpu_env.sh /path/to/python3.11
.envs/nano_rl/bin/python scripts/environment_probe.py --output results/stage0/environment.json
```

The independent nano_rl environment uses CPU-only torch for development. The old
.venv-linux inherited kg_train dependencies and is retained only as historical
groundwork. That old environment has a known CUDA 13 torch / CUDA 12 torchvision
import conflict. Do not modify either kg_train or Windows kg_project.

```bash
CUDA_VISIBLE_DEVICES= .envs/nano_rl/bin/python -m unittest discover -s tests -v
bash scripts/setup_gpu_env.sh --dry-run
# Explicitly install the unvalidated GPU candidate stack, without driver changes:
bash scripts/setup_gpu_env.sh --allow-install /path/to/python3.11
.envs/nano_rl/bin/python scripts/environment_probe.py --allow-gpu --output results/stage0/environment-gpu.json
.envs/nano_rl/bin/python scripts/attention_smoke.py --allow-gpu --output results/stage0/attention.json
HF_HUB_DISABLE_XET=1 .envs/nano_rl/bin/python scripts/download_model.py --allow-download
```

The downloader pins the resolved revision in `results/stage0/model.json`; reuse
that revision with `--revision` when reproducing. Weights live in ignored `.models/`.

## Training-memory probes

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True .envs/nano_rl/bin/python scripts/training_memory_probe.py --allow-gpu \
  --model .models/Qwen3-0.6B --batch-size 1 --sequence-length 256 \
  --chunk-size 32 --steps 2 --output results/stage0/training-alone-256.json

PYTORCH_ALLOC_CONF=expandable_segments:True .envs/nano_rl/bin/python scripts/training_memory_probe.py --allow-gpu \
  --model .models/Qwen3-0.6B --batch-size 1 --sequence-length 256 \
  --chunk-size 32 --steps 2 --gradient-checkpointing \
  --co-resident --inference-memory-fraction 0.3 \
  --output results/stage0/training-resident-256.json
```

These use synthetic fixed token IDs and a response-half mask to test numerical
and memory mechanics; they are not GRPO training or a learning-quality result.
Every step must have finite loss/gradients and a changed parameter. AdamW states
must actually materialize. The co-resident mode loads nano-vllm plus real KV and
generates before and after training; a dummy tensor reservation is insufficient.
Inference is eager here; graph feasibility is tested separately.

## Measurement boundaries

- Keep the laptop on AC power and record its performance mode before performance
  comparisons. Current power/performance mode has not been independently verified.
- Use TP=1, temperature=1, no top-k/top-p, and explicitly disable Qwen3 thinking in
  chat-template calls. Record exact prompt IDs and tokenizer/model revision.
- Synchronize GPU at timing boundaries. Report allocated and reserved process
  memory plus whole-device free memory; WSL/display allocations can vary.
- Preserve failures. Never overwrite a failed result to make a configuration look
  successful. Change output names when changing probe parameters.
- Do not infer that `enforce_eager=True` disables `torch.compile`.

## Documentation consulted (2026-10-26)

- [PyTorch checkpointing](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/checkpoint.md):
  non-reentrant checkpoint recomputes vocabulary-score chunks during backward.
- [Official FlashAttention](https://github.com/Dao-AILab/flash-attention): FA2 and
  FA4 provide distinct interfaces. A FA4 installation does not establish that the
  baseline's FA2 imports, paged cache calls, or SM120 execution work.
