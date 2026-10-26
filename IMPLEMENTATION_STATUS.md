# Implementation and verification status

Delivery target: GPU-validation-ready, as requested in the updated goal on
2026-10-26. All real GPU execution and large model/data downloads are manual.
`GPU_PENDING` does not block independent implementation. No full GPU gate passed.

| Module | Implementation | CPU evidence | Manual validation |
| --- | --- | --- | --- |
| A: rollout contract/logprob | Integrated, including replay and preemption | 6 rollout + 5 scoring tests passed | GPU_PENDING |
| B: consistency/replay/paths | Complete path matrix, HF BF16/FP32, compile controls, trace/statistics/plots | Path and experiment-plan checks, real tiny-HF shift/effective-weight tests passed | GPU_PENDING |
| C: transactional weight sync | TP=1 packed/tied updates, lock/broken/version/cache, A/B fresh/hot drivers | Mapping, rejection, partial aliases, failure, concurrency, initialization cleanup tests passed | GPU_PENDING |
| D: GRPO/rewards/evaluation | Complete core loop, two anchors, AdamW/SGD, checkpoints, standalone held-out evaluation, raw/failure metrics | Ragged gradients, frozen anchors, multi-step tiny Qwen, clipping/skip, casts, persistence and checkpoint tests passed | GPU_PENDING |
| E: experiments/benchmarks | Exact upstream export, full-contract/sampler comparisons, memory, sync, training and plots | Archive export, all guarded entries, six Dry Runs, docs, static/package checks passed | GPU_PENDING |

Current CPU result: **55 tests passed in 112.181s** in isolated Python 3.11 / torch 2.11.0+cpu,
including real micro-Qwen3 tests with random weights and no model downloads.
See results/cpu-tests.txt and results/cpu-environment-freeze.txt. Ruff E9/F,
compileall, Bash syntax, pip check, 16 documentation blocks, six JSON Dry Runs,
installer guards, wheel build and isolated wheel imports passed. All 34 library
Python sources are packaged and checked byte-for-byte against the working tree.
Historical environment import/build failures are recorded in PROGRESS.md; no
real GPU claim follows from CPU tests. Full requirements map: docs/acceptance-audit.md.

## Remaining manual validation and risks

1. FA2 build/ABI and sm_120 varlen/paged/graph kernels are not validated.
2. Real same-trajectory tolerances and cache/graph A-to-B equivalence are pending.
3. Actual FP32 optimizer + BF16 actor/KV memory may require configuration changes;
   no 16GB setting or offload capability is promised.
4. Reward variance and Qwen3 learning behavior require actual task runs. Zero
   signal is reported, not fabricated. Single-seed results remain single-seed.
5. GPU findings may require kernel/configuration/numerical fixes. The full ordered
   procedure and failure-feedback requirements are in GPU_RUNBOOK.md.

## Implementation order

1. Enforce manual-only execution guards; record changed goal and current state.
2. Integrate rollout records, sampler scores, sequence metadata and fixed replay;
   test scheduling/chunking/termination entirely on CPU.
3. Implement packed/tied parameter preflight, fail-closed update transaction,
   cache invalidation and version lifecycle; test with tiny CPU modules/mocks.
4. Implement GRPO batch construction, frozen anchors, accumulated-token objective,
   rewards, evaluation and full-loop integration using tiny CPU models in tests.
5. Build fixed-trajectory HF/nano scoring, three-way/path experiments, evidence
   validation, statistics and plotting; prepare complete manual GPU configurations.
6. Review integration and failure modes, run all allowed checks, update README and
   all runbooks, and save meaningful local Git checkpoints.

The detailed checklist in PLAN.md remains the technical acceptance reference;
its dates and measured GPU gates do not constrain this coding-only delivery.
