# Fixed baseline execution flow

Baseline: `bb823b3e06983d71485a8e1f23715ebd87d98ef8`; TP=1 is the first-version scope.

1. `LLMEngine.__init__` creates `Config`, `ModelRunner`, tokenizer and `Scheduler`.
   Even TP=1 initializes an NCCL process group on localhost port 2333. ModelRunner
   loads packed Qwen3 weights, warms up, allocates KV, and optionally captures
   decode graphs. `enforce_eager` skips these graphs, but compiled layer and
   sampler functions remain compiled.
2. `add_request` encodes a string (or accepts token IDs), creates a `Sequence`, and
   enqueues it. Chat-template selection belongs to the caller; the original
   example does not explicitly disable Qwen3 thinking.
3. `Scheduler.schedule` prioritizes waiting requests. `BlockManager.can_allocate`
   discovers reusable full prefix blocks and checks capacity. Only the first
   selected prefill sequence can consume a partial chunk. `num_scheduled_tokens`
   describes the actual chunk, not the full prompt.
4. `ModelRunner.prepare_prefill` builds packed input, absolute positions, cumulative
   lengths, physical KV slots, and optional prefix block tables. `prepare_decode`
   supplies each sequence's last token and full context length. Attention writes
   KV using Triton, then invokes varlen or paged FlashAttention. Decode can replay
   a padded CUDA graph; logits projection and sampling occur outside the graph.
5. `Sampler` casts logits to FP32, divides by temperature, computes softmax, and
   samples with an exponential race. It modifies `probs` in place. A future
   logprob implementation must retain the original distribution and gather the
   selected token score before this destructive operation.
6. `Scheduler.postprocess` hashes completed KV blocks and advances cached-token
   accounting. Incomplete prefill chunks discard their temporary sample. Only
   final prefill or decode appends a response token. EOS/length termination frees
   block references; logical prefix entries can survive for reuse.
7. Decode KV pressure preempts running requests from the tail, releases blocks,
   and queues them for prefill again. Their existing generated tokens stay in
   the sequence. Forced-token replay must advance only on a real new append.
8. `step` returns finished `(seq_id, completion_ids)` pairs; `generate` collects
   them, sorts by sequence ID, and decodes. Upstream has no request/group metadata,
   logprobs, masks, finish reasons, weight-update transaction or policy versions.
9. `loader.load_model` streams safetensors and resolves QKV/gate-up packed shards
   using the model mapping and parameter `weight_loader`. A future hot update
   must preflight all mappings before any copy, preserve storage/tied weights,
   invalidate logical caches, and fail closed if a copy fails.

## Stage-0 implications

- KV budget is derived from total GPU memory, existing use, and measured warmup
  overhead; 0.9 is not an independent KV-only fraction. Co-resident training must
  use a much smaller measured inference budget.
- Qwen3-0.6B has a 151936-token vocabulary. Merely looping over output chunks
  does not bound backward memory: autograd otherwise retains each chunk's
  `logsumexp` inputs. Checkpoint the projection-and-score chunks and verify both
  their values and gradients against full FP32 `log_softmax` on a small model.
- Original example/benchmark are evidence only when actually executed. Dependency
  imports or a successful BF16 GEMM do not establish prefill/decode/graph support.
