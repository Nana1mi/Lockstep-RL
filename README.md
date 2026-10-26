# Lockstep-RL

RL rollout infrastructure built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), fixed at `bb823b3e06983d71485a8e1f23715ebd87d98ef8`.

The first-version source is **GPU-validation-ready**: 55 CPU tests passed, manual experiment entry points and configurations are prepared, and real GPU correctness, memory feasibility and performance are `GPU_PENDING`. See [implementation status](IMPLEMENTATION_STATUS.md), [acceptance audit](docs/acceptance-audit.md), [progress](PROGRESS.md), [experiments](EXPERIMENTS.md), and the [manual GPU runbook](GPU_RUNBOOK.md).

## Scope and attribution

Prefix caching, chunked prefill, internal preemption, tensor parallelism, torch.compile and CUDA graphs come from upstream. First-version validation covers TP=1, dense Qwen3-0.6B, temperature=1, no top-k/top-p, and thinking disabled. The MIT license and upstream authorship are preserved. The [original README](docs/upstream-readme.md) contains historical upstream results; those numbers are not Lockstep-RL measurements.

## CPU development

Use WSL2 and an independent Python 3.11 environment. Prefer a Linux filesystem checkout for installations and GPU work. Existing Linux and Windows environments are untouched.

```bash
bash scripts/setup_cpu_env.sh /path/to/python3.11
CUDA_VISIBLE_DEVICES= .envs/nano_rl/bin/python -m unittest discover -s tests -v
.envs/nano_rl/bin/ruff check --select E9,F lockstep_rl nanovllm scripts tests example.py bench.py
.envs/nano_rl/bin/python scripts/train_grpo.py --dry-run
.envs/nano_rl/bin/python scripts/validate_cpu_artifacts.py
```

Base installation does not compile Triton/FlashAttention. CUDA-only imports are deferred. GPU entry points require `--allow-gpu`; model/dataset downloads require `--allow-download`. The runbook defines manual setup and experimental boundaries.

## Rollout contract

Every response has `request_id`, `group_id`, `prompt_ids`, `response_ids`, `response_logprobs` (computed in FP32), `loss_mask`, `finish_reason`, `policy_version`, and `sampling_config`. Actual EOS is retained; length truncation never fabricates EOS. Raw masks contain one for each actual response token; training padding is masked separately.

`generate()` retains `text` and `token_ids` aliases. `step()` retains upstream tuples by default; `step(return_rollouts=True)` returns complete records inside those tuples. Request/group identity is independent of completion order. Batch admission validates all requests before queueing any.

`forced_response_ids` scores fixed diagnostic trajectories through normal scheduling, chunking and preemption. Diagnostic records are rejected by the trainer. `record_trace=True` records actual execution evidence.

## Safe weight synchronization

`engine.update_weights(hf_model.state_dict())` requires an idle TP=1 engine. Complete preflight checks keys, packed QKV/gate-up mappings, shapes, dtype conversions and tied weights. Upstream loaders copy into existing parameter storage. Logical KV/prefix state is invalidated, while physical KV allocation and compatible graphs are retained.

Preflight rejection leaves the old version usable. Exceptions after commit starts mark the engine `broken`, prevent generation, and leave the version unchanged. Recovery requires closing it and creating an engine from a known complete checkpoint. Source parameters must remain unchanged throughout synchronization. Only successful full commits advance the policy version.

## GRPO definition

The implemented variant uses token-level clipped surrogate, epsilon=0.2, default group size 4, population-standard-deviation advantages and one global valid-token denominator across microbatches. **KL beta=0**; no reference model is loaded. `anchor=rollout` and `anchor=recomputed` freeze their logprob denominators across all updates on a batch. Advantages are frozen too. Empty masks and zero-advantage batches skip optimization and sync.

Checkpointed output-head chunks bound retained vocabulary activations. Gradient checkpointing, microbatch size and inference KV budget are configurable candidates; no 16GB memory setting has been established by measurement.

The simple numeric task and GSM8K share a fixed answer parser. Missing/unparseable answers score zero; truncated output uses the same rule. Training and validation IDs/prompts must be disjoint, and validation accuracy is reported separately.

## Diagnostics and results

Teacher forcing compares identical prompt/response IDs, next-token positions, padding and masks. FP32 reference scoring promotes the same effective BF16 weights. Error statistics include mean/P95/P99/max, ratios and grouped lengths. Path traces record actual prefix reuse, chunk ranges, graph batches and preemption.

Preserve raw trajectories, per-token scores, traces, configurations, code/model revisions and environment settings. `scripts/plot_results.py` plots supplied observations only. No Lockstep-RL benchmark or learning-quality result has been claimed.

GPU attention, graphs, A-to-B equivalence, co-resident memory, three-way numerics and Qwen3 training remain `GPU_PENDING`. Experiment orchestration and the staged manual handoff are implemented. Memory settings and tolerances may need revision after the first real measurements.

Multi-GPU weight updates, asynchronous RL, LoRA synchronization, production serving, automatic rollback and full batch invariance are outside v1 scope.
