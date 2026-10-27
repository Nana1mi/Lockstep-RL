# Implementation and verification status

2026-10-27 pre-rental review: finish CPU changes before renting a GPU. No CUDA kernel,
model/data download or rental was started for this revision. Historical environment
attempts in PROGRESS are not current validation evidence.

| Priority | Change | CPU acceptance | GPU boundary |
|---|---|---|---|
| P0.1 | Prefix switch; cached/recompute preemption split | 6 blocks, 2×512 prompts, 2×300 responses; positive/zero reuse; full score alignment | Both real paths GPU_PENDING |
| P0.2 | Real trajectories, seeded context rotation; EOS-inclusive primary + separate tail; repeatability/precision references | K×14 dry coverage, EOS0/multiple/no-EOS, tail exclusion and rotation tests | Real numerical populations GPU_PENDING |
| P0.3 | Live and commit-refreshed cached FP32 head | Initial load/tied-only commit refresh, cache pointer/dtype/state exclusion, failure-closed behavior and live equivalence | Error benefit, cache memory and throughput cost GPU_PENDING |
| P1.4 | One reduction per source tensor, batched host read, alias fast path, phased timers, two source types, negative control | Packed/tied/overflow/transaction tests; functional stale-KV CPU simulation detects failure | GPU source timing and negative-control sensitivity GPU_PENDING |
| P1.6 | Strict/lenient rewards and sample failure status | Currency, period, nested thousands/units; original strict tests | Actual parse failure/reward rates GPU_PENDING |
| P1.5 | Smoke/demo presets, expanded arithmetic pool, actor-change/probe metrics | 512/128 disjoint pool; BF16 rounding counts; full CPU actor loop + fixed-probe changes | Both 120-iteration learning rates GPU_PENDING |
| P2.7 | Local generation config EOS lists | Either EOS terminates; fallback/malformed list and metadata tests | Actual Qwen generation GPU_PENDING |
| P2.8 | Response-only vocabulary projection | Dense value/gradient equivalence and actual projected-token count | Real resource/performance impact GPU_PENDING |
| Rental | A100-SXM4-80GB; persistent managed Python; soft failure continuation and --stages | Mocked hard/soft failures and startup errors, selected plans, interpreter path/host-fallback checks | Rental execution GPU_PENDING |

Full regression: **77 passed in136.612s on torch2.11.0+cpu**, **77 passed
in133.314s on torch2.8.0+cpu**. Earlier69-test runs are historical.
Raw logs and exact normalized source fingerprints are in `results/cpu-tests*.txt`
and `results/validation-manifest.json`. CPU2.11 and the separate CPU2.8 candidate
environment remain distinct; neither is evidence of CUDA compatibility.

Both A100 demos now use microbatch16/accumulation8 without checkpointing;
128-response objective equivalence is tested against a full batch. Speedup is
unmeasured. Public sync_seconds sums preflight/commit/invalidate; overall
total_seconds and measurement-inclusive wall time remain separate.

Remaining concerns are empirical: actual A100 FA2 import/ABI/kernels and graphs;
noise floor vs numerical errors; stability of the deliberately invalidated-cache
assay under prelocked tolerances; BF16 update visibility and genuine reward
variance/learning; actual runtime vs estimated hours. The fixed candidate memory
budget is configurable, without a new offload implementation or memory milestone.

The matrix deliberately fails when a requested path did not trigger. A negative
control is successful only when stale reuse is observed and its output comparison
fails. Code/tests do not substitute for these real GPU observations.
