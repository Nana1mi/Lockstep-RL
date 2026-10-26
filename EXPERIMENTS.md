# Experiment definitions and evidence

All real GPU experiments are manual-only (`GPU_PENDING`). No fabricated results
are stored or plotted. CPU fixtures are verification tools, not measurements of
Qwen3 or nano-vllm GPU behavior. See PLAN.md for the full acceptance requirements.

| Experiment | Current entry point | Status |
| --- | --- | --- |
| WSL/dependency metadata | scripts/environment_probe.py | Implemented; no GPU by default |
| BF16 varlen/paged KV/Triton/graph primitives | scripts/attention_smoke.py --allow-gpu | GPU_PENDING |
| Original baseline | Fixed upstream checkout; manual runner still pending | Implementation pending |
| Engine inference benchmark/memory | bench.py --allow-gpu | GPU_PENDING |
| Training full optimizer step | scripts/training_memory_probe.py --allow-gpu | GPU_PENDING |
| Co-resident inference + KV + training | same, --co-resident | GPU_PENDING |
| Token logprob overhead | baseline vs extension runner | Implementation pending |
| Fixed trajectory HF/nano, three precisions | scripts/score_hf.py implemented; matrix driver pending | GPU_PENDING / orchestration pending |
| Prefix/chunk/graph/batch/preemption paths | scheduler trace implemented; runner pending | Implementation pending |
| A-to-B / cache invalidation / graph reuse | CPU update core tested; manual runner pending | Implementation pending |
| Hot sync vs initialization | manual runner | Implementation pending |
| GRPO rollout vs recomputed anchors | scripts/train_grpo.py --anchor ... --allow-gpu | GPU_PENDING |
| Independent validation/training curves | validation + metrics JSONL + scripts/plot_results.py | GPU_PENDING |

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
