# Core implementation decisions

## Delivery boundary

The target is GPU-validation-ready source, CPU verification and manual experiment
handoff. Dates in PLAN.md are advisory. Missing GPU measurements are explicitly
GPU_PENDING and do not prevent implementation of independent modules. Ordinary
test discovery never executes CUDA kernels or downloads model/data assets.

## Response and execution state

Sequence owns original prompt IDs, generated IDs, FP32-derived scalar logprobs,
raw loss masks, termination reason, request/group IDs and the admitted policy
version. Only Scheduler.postprocess appends response data. Temporary chunk
samples do not alter response state. Forced replay indexes the existing response
length, which survives preemption and re-prefill. Forced records and HF-scored
records are diagnostics and cannot enter GRPO as behavior-policy samples.

The engine serializes public admission, stepping, generation and updates with an
RLock. generate validates its complete batch before queueing work and requires an
idle scheduler; users mixing add_request/step must drain them first. This is a
synchronous engine, with one engine per process under upstream NCCL assumptions.
Per-request budgets must fit both context length and the single-request KV pool;
otherwise fail immediately instead of entering an impossible scheduling cycle.

## Online weights

Preflight resolves every HF key into the fixed target parameter/shard layout.
QKV and gate/up mapping uses upstream module metadata and loaders. Tied aliases
are identified by exact storage, offset, shape and stride; one alias may be
omitted, but supplied aliases must agree after target-dtype conversion. Missing,
unexpected, malformed, nonfloating, nonfinite or overflowing inputs are rejected
before writing. The initial implementation deliberately restricts updates to TP=1.

The copy plan holds source tensor references. Synchronous training keeps those
weights unchanged until update returns, so no full source snapshot is needed.
Copy uses existing loaders and parameter storage. GPU synchronization and storage
signature checks surround commit. Logical KV state is invalidated only after
successful copy/sync. Physical KV and graph buffers remain at their addresses.

State transition is ready -> updating -> ready with version increment, or ready
-> updating -> broken with no increment. Preflight failure leaves ready untouched.
A broken engine must be closed and constructed anew from a known complete
checkpoint; no automatic rollback is attempted.

## Training and scoring

Teacher forcing explicitly shifts hidden positions against next-token IDs. A
response starts at hidden index prompt_length-1. Right padding is excluded from
the response mask, and absolute positions derive from the attention mask.

Vocabulary scoring projects hidden chunks, gathers selected logits and subtracts
FP32 logsumexp. Non-reentrant checkpointing recomputes projection/score chunks in
backward, preventing autograd from retaining every vocabulary chunk. CPU tests
check scores, gradients, and saved activation shapes against full log_softmax.

The revised scorer first gathers response-mask hidden positions before any head
projection, then scatters scores into the original dense/masked API. Prompt and
padding positions never enter the vocabulary GEMM. Optional FP32 head disables
autocast locally and converts live hidden/weights on each call; no FP32 shadow
survives a hot update. The decoder hidden-state computation is graph-captured;
the head executes after graph replay, so live conversion also occurs in graph
mode. CPU tests compare full FP32 projections and hot weights; actual graph
behavior remains GPU_PENDING.

GRPO advantages use explicit group IDs and population standard deviation. Each
microbatch contributes a loss numerator divided by the full batch's valid-token
count. Frozen anchor/advantage tensors are built once and reused across update
steps. Current logprobs are recomputed after each optimizer step. Ratio outside
the clipping interval and sign-dependent actual clip activation are reported
separately. No KL reference model is loaded (beta=0).

Training mode stays enabled for HF activation checkpointing; dropout modules are
disabled individually and functional attention dropout must be zero. Parameter
updates are checked at a nonzero-gradient parameter region. Zero signal/masks
skip optimizer and sync; successful training batches synchronize once after all
configured updates, then admit only the next committed actor version.

## Numerical evidence

Experiment configurations separately control manual CUDA graph replay and
`torch_compile`. Compiler directives can be disabled with the public
[torch.compiler.set_stance("force_eager") API](https://docs.pytorch.org/docs/2.11/generated/torch.compiler.set_stance.html),
verified against versioned official documentation through Context7 and a CPU
backend-counter test. The stance does not remove the explicit Triton KV kernel.

HF FP32 reference first loads BF16 effective weights and promotes their values to
FP32. It does not recover nonexistent FP32 source precision. FP32-master trainer
comparisons are labeled as potentially including weight-cast differences.

Opt-in trace events record actual prefix reuse, scheduled token intervals,
preemption and decode graph selection. Path requirements reject missing triggers.
Statistics compare identical prompt/response IDs and token positions, retaining
raw per-token deltas and length-group summaries. Maximum error refers only to the
observed trajectories. Plotting uses saved data only; injected known-error tests
are labeled separately from natural observations.

The current matrix uses32 real local GSM8K trajectories, each128 response tokens;
solved few-shot context fills exact block lengths and its token cropping is
recorded. Independent eager/repeat runs measure noise, and every comparison lists
pooled errors, per-trajectory maxima and trajectory exceedance counts. Cached and
cache-disabled preemption are separate experiments, each canonical trajectory
being the actual victim in its retained replay. Disabling prefix caching disables
both lookup and hash registration, including self-cache after preemption.

Preflight now reduces each unique source tensor once after dtype conversion,
collects all finite/alias flags into one host read, and short-circuits identical
source storage. Nonidentical aliases compare on the source device. Three update
phases and optional actor-change measurement are timed independently. The manual
suite separates GPU-FP32/CPU-BF16 sources and includes a test-only invalidation
bypass; production update has no such switch. BF16 changed-value fractions and
frozen-probe before/after scores verify numerical visibility of a policy commit.

## Validation boundary

The manual HF/nano path matrix, three-way orchestration, A/B sync/recovery
benchmarks, original-source and sampler-compute comparisons, provenance and the
per-experiment handoff are implemented. Real CUDA primitives, graphs, memory
budgets and Qwen3 learning behavior remain GPU_PENDING. GPU findings may require
kernel compatibility, tolerances, memory settings or training-budget changes.

Raw training trajectories are persisted before reward/optimization in the manual
driver so failed updates retain their input data. The rollout timer explicitly
identifies that persistence overhead. Gradients are released after the update
batch. Initialization failure only cleans up a process group owned by that
constructor, preserving an unrelated existing group if initialization is rejected.
