# GPU-validation-ready delivery audit — 2026-10-27 revision

This audit follows the updated user objective: implement all mandatory v1 code,
complete CPU-verifiable checks, and prepare real GPU experiments for the user.
It does not substitute CPU evidence for GPU measurements. Original PLAN.md dates
are advisory; its real-measurement checkboxes remain separate manual gates.

| Requirement | Current implementation/evidence | GPU boundary |
| --- | --- | --- |
| Fixed upstream and attribution | Git ancestry, exact archive export test, README attribution, original README retained | Performance baseline awaits manual execution |
| Token FP32 logprob | sampler.py, selected-logit minus logsumexp; CPU value/dtype/input-mutation tests | Compiled CUDA numerical behavior pending |
| Full response contract | rollout.py, Sequence, engine API; ID/group/shape/EOS/length/version tests | Actual Qwen3 generation pending |
| Chunked append and fixed replay | scheduler postprocess append gate, response-length replay index; chunk/preemption tests | Real kernels and paths pending |
| Template/thinking/distribution | T=1 validation, no processors, actual prompt IDs, hashed chat template; local-only tokenizer | Real tokenized model runs pending |
| HF next-token alignment | batch.py/scoring.py, full-logits tiny-Qwen3 equivalence, mask/shift tests | BF16/FP32 comparisons pending |
| Three equal-effective-weight references | BF16 load then FP32 promotion, real tiny-HF serialization/cast test | Numerical statistics pending |
| All required path experiments | diagnose.py/nano_path_worker.py and configs; actual-trigger validators | Prefix/chunk/graph/padding/batch/target-preemption pending |
| Statistics/raw data/plots | diagnostics.py, grouped mean/std/P95/P99/max, ratio, JSON/CSV, plot_results.py | No fictitious observation or curve supplied |
| Separate compile/graph controls | versioned Context7 docs, set_stance CPU backend-counter test, torch_compile config | Both CUDA execution choices pending |
| Complete update preflight | weight_update.py, real packed loaders, keys/shapes/dtypes/finite/overflow/alias tests | Full-size GPU copy pending |
| Version/cache/broken transaction | engine lock, idle check, post-copy version commit, invalidate; injected failure and concurrency tests | Real KV and graph preservation pending |
| Initialization failure safety | restored defaults, only owned process-group cleanup; CPU mock tests | CUDA initialization failure behavior pending |
| GRPO mathematics | group population std, signed token clipping, frozen anchors, total-token normalization; gradient equivalence tests | Real learning behavior pending |
| Empty/zero/clipped cases | skip without optimizer/sync; clip diagnostics retained; CPU tests | Data-dependent signal pending |
| Multiple optimizer steps | fresh current scores with frozen old/advantages; tiny CPU Qwen3 and full 2-iteration actor loop | Qwen3-0.6B loop pending |
| Logit/backward memory | checkpointed output-head chunks; score/gradient and saved-activation-shape tests | Real memory peaks pending |
| Cast/ratio/timing distinctions | separate weight-cast errors, pre-update comparison, training ratios, true clip and phase timers | Measured model statistics pending |
| Rewards and independent data | numeric parser, explicit malformed/missing rules, simple task split, GSM8K pinned preparation, leakage checks | Dataset download and GPU evaluation manual |
| Recoverable training artifacts | trajectories before reward/update, failure record, JSONL/CSV, summary, actor-effective checkpoint | Full-size checkpoint/eval pending |
| Experiment/benchmark system | exact original archive, full contract, sampler compute, memory, sync, anchors, validation and plots | All performance results pending |
| Isolated environment and resources | nano_rl CPU env; host-info, freeze, metadata; existing environments unchanged | GPU installer is an unvalidated candidate |
| No accidental GPU/download work | opt-in entry tests, imports with CUDA-init trap, dry-run and installer guard checks | Human must opt in explicitly |
| Installation/documentation | README, AGENTS, PROGRESS, IMPLEMENTATION_STATUS, EXPERIMENTS, GPU_RUNBOOK, design/flow/interview notes | Manual sequence documented |
| Package integration | wheel build, all 35 library sources including diagnostic prompt builder, isolated CPU imports without FA2/Triton; byte-for-byte source verification | GPU dependencies intentionally separate |

## Verification records

Complete regression counts/times for CPU2.11 and CPU2.8 are stored in their raw
logs and `results/validation-manifest.json`, with artifact/static/package checks.
The manifest binds the evidence to normalized source/config files; after the final commit its
`--verify-git` check must confirm the delivered HEAD matches that text.

- `results/cpu-tests.txt`: complete discovered CPU suite; includes tiny random
  models, no downloaded weights or real CUDA/distributed operations.
- `results/cpu-tests-torch28.txt` and `cpu-environment-torch28*`: independent target
  torch2.8 CPU compatibility, separate from actual CUDA validation.
- `results/cpu-artifact-checks.json` and `.txt`: documentation fences, nine GPU dry
  plans, installer opt-in guard and dependency consistency.
- `results/static-checks.txt`: Ruff E9/F; compileall and Bash syntax are also run.
- `results/package-build.txt` and `package-check.json`: local wheel and isolated
  import verification against the current source.
- `results/cpu-environment*.txt/json`, `host-info.txt`: reproducible CPU stack and
  actual resource inventory. Historical pre-update environment observations are
  explicitly identified in PROGRESS.md.

## Review conclusions and remaining risks

No unsupported GPU, TP-update, async, LoRA, batch-invariance, throughput or learning
claim is made. Real GPU validation on the A100-SXM4-80GB may require fixes to FA2 ABI,
paged/graph kernels, numerical tolerances, memory configuration or reward budget.
The code provides the required controls, trace/failure evidence and manual gates.
No sleep/offload direction has been selected without a memory measurement.

The first local checkpoint is db946ee; the final audited commit is recorded in
Git and the delivery message. The user-provided 初始任务.md is preserved unchanged
and remains untracked. No remote push, driver update or existing-environment
replacement is part of this delivery.

## Revised priority acceptance

P0 is covered by the exact6-block512/300 scheduler simulation with cache on/off,
32×13 diagnostic dry coverage, real-question/template/companion fixture tests,
token-vs-trajectory weighting, FP32-head/autocast/gradient/hot-weight checks.

P1 includes one reduction per unique source tensor and one batched flag read,
identical tied-storage fast path, phased timers, GPU-FP32/CPU-BF16 job separation,
and a functional stale-KV CPU assay that rejects the bypassed-invalidation output.
Reward tests preserve strict behavior and accept all three requested lenient
examples. Demo arithmetic sets are512/128 unique/disjoint at comparable difficulty;
both120-iteration configs dry-run. BF16-rounding change fractions and real actor
loop probe deltas are CPU-tested.

P2 includes local generation-config EOS lists and either-token stopping, and
response-only head value/gradient/token-count equivalence. Rental plan tests check
ordered stages, persistent paths, both learning rates, opt-in guards, pinned wheel
ABI/hash, and pre-GPU rejection of incomplete/leaked inputs. The official wheel
candidate is a torch2.8 binary; the old5080/SM120 workflow is no longer applicable.
No new memory/offload task is inserted into the revised mandatory sequence.
