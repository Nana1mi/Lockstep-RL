# Experiment definitions and evidence

All real GPU experiments are manual-only (`GPU_PENDING`). No fabricated results
are stored or plotted. CPU fixtures are verification tools, not measurements of
Qwen3 or nano-vllm GPU behavior. See PLAN.md for the full acceptance requirements.

| Experiment | Current entry point | Status |
| --- | --- | --- |
| WSL/dependency metadata | scripts/environment_probe.py | Implemented; no GPU by default |
| BF16 varlen/paged KV/Triton/graph primitives | scripts/attention_smoke.py --allow-gpu | GPU_PENDING |
| Original baseline | scripts/benchmark_suite.py exports the exact fixed Git archive | GPU_PENDING |
| Engine inference benchmark/memory | bench.py --allow-gpu | GPU_PENDING |
| Training full optimizer step | scripts/training_memory_probe.py --allow-gpu | GPU_PENDING |
| Co-resident inference + KV + training | same, --co-resident | GPU_PENDING |
| Token logprob overhead | benchmark_suite.py for full contract; sampler_benchmark.py for compute only | GPU_PENDING |
| Fixed trajectory HF/nano, three precisions | scripts/diagnose.py + scripts/score_hf.py | GPU_PENDING |
| Prefix/chunk/graph/batch/preemption paths | diagnose.py + nano_path_worker.py, actual trace requirements | GPU_PENDING |
| A-to-B / cache invalidation / graph reuse | weight_sync_experiment.py + weight_sync_worker.py | GPU_PENDING |
| Hot sync vs initialization | same, 3 independent fresh/hot runs for eager and graph | GPU_PENDING |
| GRPO rollout vs recomputed anchors | scripts/train_grpo.py --anchor ... --allow-gpu | GPU_PENDING |
| Independent validation/training curves | validation + metrics JSONL + scripts/plot_results.py | GPU_PENDING |

## Configuration and execution inventory

- `configs/diagnostics.json`: one sampled trajectory, seven replay cases, and two
  HF scoring processes. Errors are HF-minus-nano, FP32-minus-BF16, or
  variant-minus-eager for path comparisons; filenames identify the comparison.
- `configs/weight-sync.json`: construct a CPU BF16 checkpoint B, save tolerances
  before measurement, check 3 fresh repeats before 3 hot runs per backend.
  Parameter hashes and first/new-version cache traces are mandatory evidence.
- `configs/inference-benchmark.json`: 48 workers (2 sources, 2 backends, 4 shapes,
  3 process repeats), each with 3 nested steady repeats. Original source comes
  from commit bb823b3e06983d71485a8e1f23715ebd87d98ef8 without patching it.
- `configs/train-simple.json`, `configs/train-gsm8k.json`: group size, frozen
  anchor, update count, accumulation, optimizer and memory candidates. Independent
  local evaluation is available in `scripts/evaluate.py`; data downloads are
  explicitly manual through `prepare_gsm8k.py --allow-download`.
- Every GPU driver is opt-in; `scripts/validate_cpu_artifacts.py` prints/plans
  jobs and checks syntax only. Its saved plans are not measurements.
- `torch_compile` is controlled independently of the engine's `enforce_eager`.
  Force-eager compilation stance still uses the explicit CUDA/Triton attention
  backend; it does not select a new attention implementation.

Raw files are produced below each chosen output directory: manifest/report,
rollouts/trace, per-token JSON/CSV, timings/metrics CSV and plots. Full commands,
gate order, expected schema, success criteria and failure feedback are in
GPU_RUNBOOK.md. Do not infer GPU success from this implemented inventory.

## Raw records

- Preserve exact model/tokenizer revisions, upstream/project commits, local
  changes, seed, config, prompt and response IDs, and each per-token score.
- Save full scheduler traces for numerical path runs. A configuration flag alone
  is not evidence that a cache hit, chunk, graph replay or preemption happened.
- Compare exactly matched IDs/tokens/positions; delta = reference logp - rollout
  logp. exp(delta) is a same-trajectory diagnostic, distinct from training's
  current/anchor ratio. Record both out-of-range and actually active clip rates.
- Repeat performance groups at least three times; synchronize timing boundaries
  and separate cold initialization, compilation/capture, warmup and steady state.
- `bench.py` repeats warm the same prompts and retain prefix cache. Its JSON says
  so; do not interpret it as a cold/no-cache benchmark. Trace overhead is labeled.
- Establish output tolerances before A-to-B runs and preserve failed results.
- Every GPU result must record laptop power/performance mode and variability.
  Candidate memory settings are unmeasured until the manual probes pass.
