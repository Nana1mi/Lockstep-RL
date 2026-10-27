# Experiment definitions and evidence

All real experiments remain **GPU_PENDING**. CPU fixtures are tests, never
measurements of Qwen3-0.6B or nano-vllm CUDA behavior. All comparisons use one
rented NVIDIA A100-SXM4-80GB and one pinned stack/UUID. See GPU_RUNBOOK.md.

| Experiment | Matrix and raw evidence |
|---|---|
| Environment | metadata, actual GPU/driver/ABI, BF16, FA2 varlen/paged KV, Triton, graph smoke |
| Baseline | exact fixed upstream git archive vs current; no patch to original source |
| Diagnostics | sample + eager/repeat/prefix/chunked/graph/padding/batch/cached-preemption/recompute-preemption/live/cached-fp32-head + HF BF16/FP32: 14 processes, 32 canonical trajectories each |
| Noise floor | independent same-config eager and repeat scores; retain pooled/per-trajectory maxima and exceedance counts |
| FP32 head | nano live FP32 head vs HF FP32, alongside BF16-head results; per-token scores and throughput variant |
| Weight sync | sample + per backend 3 fresh references, 3 GPU-FP32 hot, 3 CPU-BF16 hot, 3 GPU-FP32 negative controls: 25 workers |
| GRPO smoke | original short configs, both anchors and two-update smoke; integration only |
| GRPO demo | 16 prompts×G8, 120 iterations, 512/128 disjoint tasks, evaluation every10, lr1e-6/5e-6 separately |
| Performance | upstream/current/current-FP32-live/current-FP32-cached × 2 backends × 4 shapes × 3 independent processes =96 workers, each3 nested steady repeats |
| Sampler compute | original sampler math vs token-logprob compute; synthetic logits, labeled microbenchmark |

## Statistical and path definitions

Canonical IDs are traj-k. Eager/sample/repeat/FP32-head run batch1; real trajectories
group for batch and graph-padding. Every canonical preemption output is collected
only from its victim replay, with another real trajectory as companion. Repeated
companions are retained in all-rollouts but excluded from pooled statistics.
Trace replay_group identifies any repeated companion and joins to raw output.

Prompts are real local GSM8K questions with solved training examples as context.
Only the leading context may be token-cropped for exact block size; target and
chat wrappers remain whole. All source IDs, selected indices, solutions, template,
revision and cropping rules are preserved. Diagnostic/fewshot/train/validation
subsets are disjoint. No repeated-character or constant-token companions are used.

delta_logp = right/reference score - left/rollout score, and ratio=exp(delta_logp).
Pooled quantiles weight tokens; per_trajectory lists each request's maximum error;
trajectories_above_threshold counts requests with any token crossing threshold.
The noise-floor distribution is attached to every path comparison; differences
are not blindly subtracted. A zero repeat floor is marked non-discriminative;
measured HF-BF16/FP32 errors are a precision reference, not an automatic verdict
or tolerance. HF FP32 promotes identical BF16 effective values.
CSV retains raw logprobs, token, position and request ID; traces/config/rollouts
provide execution batches and paths. Finite maxima are not global error bounds.

Cached preemption requires actual preemption/re-prefill and prefix_reused_tokens>0.
Recompute preemption disables lookup/hash registration and requires zero reuse.
Do not describe the cached case as full recomputation. Configuration alone is not
proof that a path occurred; missing trigger aborts the run.

## Synchronization and training measurements

Lock atol/rtol before measurements and verify fresh-repeat stability first.
Normal hot/fresh parameter hashes must match and scores meet that tolerance.
GPU-FP32 vs CPU-BF16 source timings are separate groups. preflight/commit/invalidate
timers exclude the separately timed actor-change measurement; source preparation,
ready time, first generation and total recovery remain available. No whole
embedding is copied back to CPU just for tied equality checks.

The test-only --skip-invalidate mode must demonstrate stale prefix reuse AND fail
the fresh-B score comparison. If its scores are indistinguishable at predeclared
tolerance, the assay fails rather than claiming successful error detection.
No production API bypass is exposed. The CPU negative test uses functional cached
hidden values and catches a score difference, not just a stale-cache metadata flag.

Training JSONL retains each optimizer step's loss, ratio mean/min/max, clipping,
sample rewards/parse_status, response lengths and phase times; CSV adds common
summary fields. The driver saves rewarded rollout JSON/text, initial and repeated
independent validation, actual actor-dtype changed-value fraction/per-parameter
counts, and a frozen probe trajectory's raw before/after logprobs. Counted actor
values deduplicate tied storage; BF16-rounding-to-zero is a valid measured result.
Zero signal skips optimizer/sync; version increments are not learning evidence.
beta=0 with no reference: reference KL is not defined or reported.

## Persistent records and runtime estimate

The ordered rental runner writes configs, stdout/stderr, failure/manifest files,
trajectories, per-token and per-step data below persistent-root/results/run-id.
Every child command/exit/time and fixed GPU inventory are recorded. Never erase
an unfavorable repeat; use a new run-id after failure. stage-level planning
ranges are in configs/rental.json, about3–11 GPU hours with30% contingency; these
are assumptions until measured wall time is available. Asset download and wheel
preparation occur before paid GPU use. Optional old memory probes remain source
utilities, outside the revised mandatory sequence.

## Pre-rental review corrections

Primary statistics and A/B score tolerances include only the first-EOS-inclusive
response. Every raw record stores first_eos_index and primary/tail counts; no-EOS
responses use their whole length. Post-EOS data stay in JSON.post_eos and separate
*-post-eos.csv files, with zero-count/null distributions when empty. Raw complete
tokens remain available for fixed-length path triggering. Pooled quantiles and
trajectory thresholds cannot be influenced by that tail.

FP32 head has live-conversion and cached-inference variants. The nonpersistent
cache is built after full loading and refreshed only after every copy in a hot
commit, including tied-only sources; failure closes the engine. Both precision
and throughput comparisons are separate. Record cache bytes, KV blocks, init,
commit refresh and steady cost; live-minus-cached time includes memory effects,
so it is not a pure GEMM latency. Training keeps differentiable live weights.

Report synchronization as sync_seconds=preflight+commit+invalidate. total_seconds
and the training sync_wall_including_measurement also include actor-change
measurement, and cannot be quoted as pure synchronization time.

Only environment and attention are hard gates. Other failures remain in logs and
manifest and independent jobs continue, including within GRPO; any failure gives
a nonzero final exit. --stages selects retries under a fresh run-id and always
includes environment/attention. GPU identity changes and interruptions remain
infrastructure stops. Only required local asset subsets are validated per job.
Demo microbatch16/accumulation8 with no checkpointing is a speed candidate with
CPU-tested global-token objective equivalence, not a measured speedup. Few-shot
contexts use seed-shuffle plus per-trajectory rotation. Managed Python and venv
must share a fixed persistent mount path; arbitrary relocation is unsupported.
