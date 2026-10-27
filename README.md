# Lockstep-RL

RL rollout extension of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm),
fixed at `bb823b3e06983d71485a8e1f23715ebd87d98ef8`. MIT/upstream attribution is
preserved; [historical upstream results](docs/upstream-readme.md) are not our measurements.

The current revision implements the CPU-verifiable P0/P1/P2 changes before GPU rental.
CPU regression runs cover both torch2.11.0+cpu and torch2.8.0+cpu; exact current
counts and timings are in `results/cpu-tests*.txt` and the validation manifest.
The explicitly authorized 2026-10-27 L40S rental passed environment/FA2/Triton/graph
gates, all 24 short inference baseline workers, and a two-iteration GRPO smoke with
real optimizer updates and actor/probe-visible weight synchronization. Full numerical
diagnostics, long-context performance and learning effects remain **GPU_PENDING**.
Current evidence and limits are recorded in [status](IMPLEMENTATION_STATUS.md),
[progress](PROGRESS.md) and `results/`.

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
- The default matrix samples **32 real local GSM8K questions × 128 recorded tokens**.
  Exact block-sized prompts use real solved few-shot text; only that context may
  be token-cropped, with the whole target question/chat wrapper retained. Provenance
  records source IDs/cropping. Example order is seed-shuffled then rotated per trajectory.
  Eager is batch=1; batch companions are real trajectories.
  Every trajectory is the actual victim in its preemption replay.
- Primary accuracy includes tokens through the first EOS, inclusive. Fixed-workload
  post-EOS continuation is retained as a separate population/CSV, never mixed into
  primary quantiles or threshold counts. Every rollout records the first EOS index.
- Independent eager/repeat processes check repeatability. A zero floor is marked
  non-discriminative; HF BF16-vs-FP32 supplies a measured precision reference.
  Neither reference is an automatic bug verdict. Reports retain pooled
  distributions, each trajectory's maximum error, threshold-exceeding trajectory
  count, raw scores/IDs/positions and trace. Pooled quantiles are token-weighted.
- `fp32_lm_head=True` converts live hidden/weight tensors every forward with
  autocast disabled. `cache_fp32_lm_head=True` adds an inference-only nonpersistent
  FP32 buffer, initialized after full loading and refreshed after all copies in
  each successful commit, including tied-embedding-only updates. HF scoring/training
  retains live differentiable head scoring. Diagnostics/throughput compare both
  live-conversion and cached-compute variants, including persistent memory,
  initialization and synchronization maintenance costs.

## Updates and training

`engine.update_weights(state_dict, measure_changes=True)` requires idle TP=1 and
returns `policy_version`, `timings` and `weight_changes`. Preflight validates the
whole mapping and finite dtype conversions, aggregates device flags for one host
read, and skips full equality work for identical tied source storage. Copy uses
existing packed loaders; logical KV/prefix state is invalidated, physical KV and
compatible graphs remain. Rejected preflight leaves the old version usable;
exceptions after copying starts make the engine `broken` and prevent generation.

Timers separate preflight, commit and invalidation; their sum is `sync_seconds`.
`total_seconds` also includes optional actor-change counting, which has its own
timer and must not be quoted as synchronization time. The worker measures
GPU-FP32 and CPU-BF16 sources separately.
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
The L40S demo uses inference fraction0.4, microbatch16, accumulation8 and no backbone gradient checkpointing;
output-head chunk recomputation remains to bound vocabulary activations.
the 128-sequence objective is CPU-tested against a full batch. Speedup is unmeasured.
`strict` reward parsing preserves old behavior; `lenient` also accepts currency,
trailing sentence periods, `{,}` and text units. Each sample has parse_status
(ok/no_answer/unparseable); JSONL/CSV record failure rates and optimization metrics.
Generation stops at any local generation_config EOS ID and records the full list.

## Rental handoff

All contrasts target **one L40S 48GB (Ada sm_89), Anhui region, Linux x86_64**. The A100 and
RTX 5080/SM120 setups are historical and no longer applies. To avoid compiling FA2,
the candidate pins a published official torch-2.8 / CUDA-12 / cp311 / ABI-TRUE wheel;
the wheel filename, ABI and SHA256 are checked before installation. Availability
was checked against the [official release assets](https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3.post2);
runtime correctness is still pending. This also explains the supported torch floor
of 2.8; CPU regression is checked separately from CUDA.

Prepare a uv-managed Python3.11 on persistent storage with guarded
`setup_managed_python.py`, then create the GPU venv from its recorded absolute path.
Keep the same mount path/OS/architecture; moving the disk still requires rebuilding
venv entry points. See [uv configuration](https://docs.astral.sh/uv/reference/environment/#uv_python_install_dir).
Prepare model/data/wheel on persistent storage **before paying for GPU time** via
`prepare_rental_assets.py --allow-download`. No large download is performed by default.
Follow [GPU_RUNBOOK.md](GPU_RUNBOOK.md); `run_rental.py --allow-gpu` runs environment,
baseline, diagnostics, synchronization, GRPO and performance in order. Only the
environment/attention checks are hard gates; other job failures are saved and
independent jobs continue, with a nonzero final exit if anything failed.
`--stages diagnostics sync` retries a subset under a new run-id, always repeating
environment/attention. Hardware changes/user interruption still stop the run.
First manually run `--stages environment baseline` under `preflight-01`, review FA2
and graph evidence, then use a fresh run-id for the full suite. Roughly half an hour
for this first pass is an unmeasured planning estimate.
Every task has an explicit budget (expected wall seconds × 3 in
`configs/task-timeouts.json`). Workers inherit the parent deadline. Timeout124,
PID/PGID, port and cleanup records are saved; independent tasks/workers continue,
while failed sampling dependencies are skipped. Linux sessions/subreaper cleanup
covers orphan workers and escalates TERM to KILL; a surviving process is an
infrastructure stop. Current workers pick a fresh free NCCL port, or respect
`LOCKSTEP_NCCL_PORT`; initialization defaults to120s. Exact upstream code still
uses2333, which is checked before each baseline worker. Occupied ports fail without
killing their owner. This preflight cannot reserve a port across process startup.
The dependency-free regression command is
`python3 -S -m unittest tests.test_process_runner -v` on Linux/WSL2.
The runner saves all raw output/configs/logs on persistent storage and records wall
time. The planning estimate is about **5.14–18.98 GPU hours including 30% contingency**,
not a measured runtime or cost quote. No card has been rented or started.

Limitations: no TP updates, async RL, LoRA sync, dynamic KV offload, automatic
rollback or guarantee of learning. FA2 ABI/kernels, numerical tolerances and
actual training budget must be verified on the rented card.

The domestic rental has24vCPU/120GB RAM, a30GB system disk and50GB expandable data
disk; image Ubuntu22.04/CUDA12.8.0/Python3.10 with conda. Keep all interpreters,
venvs, caches, assets and results on the data disk. Exact GPU name, driver>=570,
mount point and network reachability are unresolved placeholders in rental.json.
Manual `rental_preflight.py` records the observed identity/data mount/network;
`run_rental.py` checks that record before any GPU subprocess. It does not infer a
name from the product label or accept a system-disk PERSIST.
Optional pip indexes/wheelhouse, HF-compatible mirror endpoints with pinned
revision and file SHA256, uploaded official wheel/Python archive, and persistent
conda Python3.11 registration are described in the runbook. No automatic source
switch or FA2 source build is used. Demo microbatch16 awaits the manual memory
probe; OOM fallback8 also changes accumulation to16 under a fresh run-id.
