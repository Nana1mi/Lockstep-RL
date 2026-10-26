# Implementation and verification status

Delivery target: GPU-validation-ready, as requested in the updated goal on
2026-10-26. All real GPU execution and large model/data downloads are manual.
`GPU_PENDING` does not block independent implementation. No full GPU gate passed.

| Module | Implementation | CPU evidence | Manual validation |
| --- | --- | --- | --- |
| A: rollout contract/logprob | Integrated, including replay and preemption | 6 rollout + 5 scoring tests passed | GPU_PENDING |
| B: consistency/replay/paths | Trace, replay, statistics, HF scoring and plot entry implemented; matrix pending | Path evidence and real tiny-HF shift/effective-weight tests passed | GPU_PENDING |
| C: transactional weight sync | Integrated for TP=1, packed/tied weights, idle lock and broken state | 6 update tests passed | GPU_PENDING |
| D: GRPO/rewards/evaluation | Core trainer and synchronous loop integrated; manual simple/GSM8K entry points added | GRPO, reward, data split and real tiny-Qwen checkpointed updates passed | GPU_PENDING |
| E: experiments/benchmarks | Guarded env, attention, memory, benchmark, scoring and plot scripts; full suite pending | Guards/imports/Dry Run passed; static checks passed | GPU_PENDING |

Current CPU result: **36 tests passed** in isolated Python 3.11 / torch 2.11.0+cpu,
including real micro-Qwen3 tests with random weights and no model downloads.
See results/cpu-tests.txt and results/cpu-environment-freeze.txt. Ruff E9/F,
compileall and Bash syntax checks passed. Historical environment import/build
failures are recorded in PROGRESS.md; no real GPU claim follows from CPU tests.

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
