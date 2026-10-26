# Implementation walkthrough and interview notes

These explanations describe implemented state and CPU evidence, not unexecuted
GPU performance or universal numerical guarantees. Exact code links should use
the project's current Git checkpoint when presenting the work.

1. **Prefill and decode.** Scheduler prioritizes waiting requests and budgets
   actual input tokens. Only the first selected prefill sequence may be partial.
   Decode schedules one new context position per sequence. Temporary prefill
   samples are discarded; a response token and its logprob are appended together
   only after final prefill or decode.
2. **KV and prefix lifetime.** Physical KV pages belong to ModelRunner; logical
   ownership, hashes and reference counts belong to BlockManager. Prefix hashes
   chain token blocks and verify token equality on lookup. Free blocks can still
   carry reusable cached content. New weights invalidate the logical mapping;
   zeroing GPU memory is unnecessary when no old page can be referenced.
3. **Preemption.** Internal KV pressure releases a running sequence's blocks and
   queues its entire existing context for re-prefill. Its response IDs/logprobs
   stay intact. Replay uses response length as its next forced-token index, so
   recomputation cannot accidentally advance or overwrite old trajectory scores.
4. **Probability definition.** V1 uses T=1 and no distribution processors. Scores
   are FP32 chosen-logit minus logsumexp from unmodified logits. The upstream
   exponential-race sampler overwrites probabilities, which cannot subsequently
   be interpreted as the original distribution.
5. **Fixed trajectory vs seed.** A seed cannot guarantee identical token choices
   after batching or numerical-path changes. Fixed replay supplies identical
   next tokens while retaining real scheduling, attention and KV paths. It is
   diagnostic scoring rather than a fresh sampled rollout, and training rejects it.
6. **Weight-update transaction.** Idle admission and an RLock give a synchronous
   boundary. Validate the full key/shard/shape/dtype/tie mapping before copying.
   Copy through upstream loaders without replacing storage, synchronize, reset
   logical cache, then commit the new version. A post-start copy failure is
   broken state, not partial success. Rebuild from a known checkpoint to recover.
7. **Graph compatibility.** Fixed structure, dtype, strides, parameter/KV addresses
   and graph buffers permit replay after value-only updates. Unsupported layout
   changes are rejected; unexpected storage replacement fails closed. Manual
   A/B scripts check both eager and actual graph replay plus storage preservation.
8. **Clipping anchors.** Rollout anchors include behavior/training backend
   differences at the first update. Recomputed anchors freeze a no-grad training
   score before the first optimizer step. Current scores must be recomputed for
   every step; re-detaching current scores would erase the intended ratio history.
9. **Mask and normalization.** Prompt/padding are excluded. Real EOS is included;
   a length cap does not fabricate EOS. Every accumulated microbatch numerator
   divides by the full batch's valid-token count, so long/short responses have
   the intended token weighting. Zero advantages or empty masks skip sync/version.
10. **Clipping diagnostics.** Outside the interval is not the same as active
    clipping. Positive advantage clips only above the upper bound; negative
    advantage clips only below the lower bound. Entirely clipped batches can
    have zero gradient; their ratio/clip diagnostics are retained without an
    optimizer step or version increment.
11. **Memory.** Chunking the output head alone does not bound autograd storage:
    checkpointing must recompute vocabulary chunks during backward. Full-parameter
    AdamW includes real gradients and moment state. Co-resident feasibility must
    be measured with an actual actor/KV allocation and optimizer step, not a
    dummy reservation or a successful forward-only pass.
12. **Numerical interpretation.** HF FP32 promotes the same BF16 weight values.
    It cannot recover original precision. Same-version delta/exp(delta), weight
    cast errors, and training current/anchor ratios are separately named. A
    sampled-token error is not full-vocabulary KL or a global error bound.
13. **Contribution boundary.** The scheduler, cache, TP and graph machinery are
    upstream. New work is the response contract, scoring/replay, guarded weight
    lifecycle, diagnostics/evidence, training integration and reproducible manual
    experiment system. TP update and async RL are not claimed as validated.

## Presentation wording supported today

Implemented token-aligned rollout records, fixed-trajectory diagnostics and an
idle fail-closed weight-update transaction on nano-vllm; integrated a token-level
GRPO variant with two frozen clipping anchors and global token normalization.
Verified core math, gradients, state transitions and tiny-Qwen3 parameter mapping
on CPU. Prepared real GPU validation; performance and learning-quality claims
await those measurements.

Do not use upstream throughput numbers as personal results, or replace missing
GPU evidence with phrases such as “fully correct,” “faster,” or “16GB guaranteed.”
