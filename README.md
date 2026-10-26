# Lockstep-RL

RL rollout extension of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm),
fixed at `bb823b3e06983d71485a8e1f23715ebd87d98ef8`. MIT/upstream attribution is
preserved; [historical upstream results](docs/upstream-readme.md) are not our measurements.

The current revision implements the CPU-verifiable P0/P1/P2 changes before GPU rental.
All **69 tests pass on both torch2.11.0+cpu and torch2.8.0+cpu**, with27 focused
compatibility checks repeated after the final preflight/diagnostic adjustments.
Real CUDA accuracy, cache/graph behavior, learning effects and performance remain
**GPU_PENDING**. Current verification is recorded in [status](IMPLEMENTATION_STATUS.md),
[progress](PROGRESS.md) and `results/`. No real GPU result or training curve is supplied.

## CPU development

Use an independent Python 3.11 environment on Linux/WSL. Existing environments and
drivers are not modified. Base imports do not need FA2/Triton or initialize CUDA.

```bash
bash scripts/setup_cpu_env.sh /path/to/python3.11
CUDA_VISIBLE_DEVICES= .envs/nano_rl/bin/python -m unittest discover -s tests -v
.envs/nano_rl/bin/ruff check --select E9,F lockstep_rl nanovllm scripts tests example.py bench.py
.envs/nano_rl/bin/python scripts/validate_cpu_artifacts.py
.envs/nano_rl/bin/python scripts/run_rental.py --persistent-root /persistent/lockstep --dry-run
```

## Diagnostic conclusions that can be tested

- `enable_prefix_caching=False` disables lookup and hash registration. Separate
  `preemption_cached` and `preemption_recompute` experiments require actual
  preemption/re-prefill and positive/zero reused tokens respectively. The CPU
  6-block, two 512-token prompts, 300-token responses simulation verifies both.
- The default matrix samples **32 real local GSM8K questions × 128 response tokens**.
  Exact block-sized prompts use real solved few-shot text; only that context may
  be token-cropped, with the whole target question/chat wrapper retained. Provenance
  records source IDs/cropping. Eager is batch=1; batch companions are real trajectories.
  Every trajectory is the actual victim in its preemption replay.
- Independent eager/repeat processes define a noise floor. Reports retain pooled
  distributions, each trajectory's maximum error, threshold-exceeding trajectory
  count, raw scores/IDs/positions and trace. Pooled quantiles are token-weighted.
- `fp32_lm_head=True` converts live hidden/weight tensors every forward with
  autocast disabled. There is no stale FP32 weight shadow. HF scoring/training has
  the same option. A separate comparison and current-source throughput variant
  quantify its benefit/cost when run manually.

## Updates and training

`engine.update_weights(state_dict, measure_changes=True)` requires idle TP=1 and
returns `policy_version`, `timings` and `weight_changes`. Preflight validates the
whole mapping and finite dtype conversions, aggregates device flags for one host
read, and skips full equality work for identical tied source storage. Copy uses
existing packed loaders; logical KV/prefix state is invalidated, physical KV and
compatible graphs remain. Rejected preflight leaves the old version usable;
exceptions after copying starts make the engine `broken` and prevent generation.

Timers separate preflight, commit and invalidation; optional actor-change counting
has its own timer. The worker measures GPU-FP32 and CPU-BF16 sources separately.
Its **test-only** `--skip-invalidate` negative control must reuse stale KV and fail
the fresh-B score comparison. The production engine has no bypass flag.

GRPO uses frozen rollout/recomputed anchors, token clipping epsilon=0.2, population
group std and a global response-token denominator. **beta=0, no reference model**;
reference KL is not reported. Head projection gathers response hidden positions
before vocabulary computation and checkpoints chunks through backward.
The driver records actual actor-dtype weight changes and before/after logprobs on
one frozen probe trajectory after every successful sync; policy version alone is
not treated as evidence of a numerical change.

`train-smoke-*.json` retain the small integration settings. The two
`train-demo-simple*.json` presets use 16 prompts × G=8, 120 iterations,
512/128 disjoint arithmetic questions, evaluation every 10 iterations, and
learning rates 1e-6 / 5e-6. They are candidates, not evidence of learning.
`strict` reward parsing preserves old behavior; `lenient` also accepts currency,
trailing sentence periods, `{,}` and text units. Each sample has parse_status
(ok/no_answer/unparseable); JSONL/CSV record failure rates and optimization metrics.
Generation stops at any local generation_config EOS ID and records the full list.

## Rental handoff

All contrasts target **one rented NVIDIA A100-SXM4-80GB, Linux x86_64**. The old
RTX 5080/SM120 setup is historical and no longer applies. To avoid compiling FA2,
the candidate pins a published official torch-2.8 / CUDA-12 / cp311 / ABI-TRUE wheel;
the wheel filename, ABI and SHA256 are checked before installation. Availability
was checked against the [official release assets](https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3.post2);
runtime correctness is still pending. This also explains the supported torch floor
of 2.8; CPU regression is checked separately from CUDA.

Prepare model/data/wheel on persistent storage **before paying for GPU time** via
`prepare_rental_assets.py --allow-download`. No large download is performed by default.
Follow [GPU_RUNBOOK.md](GPU_RUNBOOK.md); `run_rental.py --allow-gpu` runs environment,
baseline, diagnostics, synchronization, GRPO and performance in order, stops on
failure, saves all raw output/configs/logs on persistent storage and records wall
time. The planning estimate is about **3–11 GPU hours including 30% contingency**,
not a measured runtime or cost quote. No card has been rented or started.

Limitations: no TP updates, async RL, LoRA sync, dynamic KV offload, automatic
rollback or guarantee of learning. FA2 ABI/kernels, numerical tolerances and
actual training budget must be verified on the rented card.
