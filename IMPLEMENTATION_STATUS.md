# Implementation and verification status

2026-10-27 revision: finish CPU changes before renting a GPU. No CUDA kernel,
model/data download or rental was started for this revision. Historical environment
attempts in PROGRESS are not current validation evidence.

| Priority | Change | CPU acceptance | GPU boundary |
|---|---|---|---|
| P0.1 | Prefix switch; cached/recompute preemption split | 6 blocks, 2×512 prompts, 2×300 responses; positive/zero reuse; full score alignment | Both real paths GPU_PENDING |
| P0.2 | 32 real-GSM8K trajectories, real companions, repeat noise floor, per-trajectory statistics | Dry plan covers K×13 jobs; fixture prompt preservation, grouping and weighted statistics tests | Local data/tokenizer + sampling GPU_PENDING |
| P0.3 | Live FP32 lm_head in nano/HF/trainer; separate diagnostics and benchmark | FP32 linear equivalence, autocast exclusion, gradients and hot-weight update | Error benefit and throughput cost GPU_PENDING |
| P1.4 | One reduction per source tensor, batched host read, alias fast path, phased timers, two source types, negative control | Packed/tied/overflow/transaction tests; functional stale-KV CPU simulation detects failure | GPU source timing and negative-control sensitivity GPU_PENDING |
| P1.6 | Strict/lenient rewards and sample failure status | Currency, period, nested thousands/units; original strict tests | Actual parse failure/reward rates GPU_PENDING |
| P1.5 | Smoke/demo presets, expanded arithmetic pool, actor-change/probe metrics | 512/128 disjoint pool; BF16 rounding counts; full CPU actor loop + fixed-probe changes | Both 120-iteration learning rates GPU_PENDING |
| P2.7 | Local generation config EOS lists | Either EOS terminates; fallback/malformed list and metadata tests | Actual Qwen generation GPU_PENDING |
| P2.8 | Response-only vocabulary projection | Dense value/gradient equivalence and actual projected-token count | Real resource/performance impact GPU_PENDING |
| Rental | A100-SXM4-80GB; official wheel-only installer; persistent prefetch and ordered runner | Guards, plan/order/config/ABI/hash tests; syntax/dry-runs | Rental execution GPU_PENDING |

Full suite: **69 passed in133.725s on torch2.11.0+cpu**, **69 passed in130.253s
on torch2.8.0+cpu**, plus27 focused target-version checks after final adjustments.
Raw logs and exact normalized source fingerprints are in `results/cpu-tests*.txt`
and `results/validation-manifest.json`. CPU2.11 and the separate CPU2.8 candidate
environment remain distinct; neither is evidence of CUDA compatibility.

Remaining concerns are empirical: actual A100 FA2 import/ABI/kernels and graphs;
noise floor vs numerical errors; stability of the deliberately invalidated-cache
assay under prelocked tolerances; BF16 update visibility and genuine reward
variance/learning; actual runtime vs estimated hours. The fixed candidate memory
budget is configurable, without a new offload implementation or memory milestone.

The matrix deliberately fails when a requested path did not trigger. A negative
control is successful only when stale reuse is observed and its output comparison
fails. Code/tests do not substitute for these real GPU observations.
