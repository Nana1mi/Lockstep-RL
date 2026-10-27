# Experiment definitions and evidence

All real experiments remain **GPU_PENDING**. CPU fixtures are tests, never
measurements of Qwen3-0.6B or nano-vllm CUDA behavior. All comparisons use one
rented L40S 48GB in Anhui and one pinned driver/stack/UUID. A100 references below are historical. See GPU_RUNBOOK.md.

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
ranges are in configs/rental.json, about5.14–18.98 GPU hours with30% contingency; these
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

## 2026-10-27 任务超时与端口 CPU 实验

`CUDA_VISIBLE_DEVICES= .envs/nano_rl/bin/python -S -m unittest tests.test_process_runner -v`
实跑标准库子进程：挂起、TERM忽略、独立session孙进程、父进程立即退出、嵌套编排中断、端口占用、
清理后重新bind；也实跑总脚本的软超时继续与环境超时硬关口。未加载torch/FA2/Triton或初始化CUDA。
原始输出 `results/cpu-process-tests.txt`；集成/全套回归分别见 cpu-timeout-integration 与 cpu-tests 日志。

期限配置为预计墙钟秒×3，worker继承外层绝对期限；returncode124/timed_out以及进程组清理写入结果。
独立worker失败后继续，采样失败跳过依赖；不为不完整suite制造成功汇总。图/归档及元数据命令也有限时。
当前NCCL端口可覆盖/自动选空闲，原版基线2333不改；bind预检与实际启动间的竞态仍由有限任务时限兜底。
双层超时、首次约半小时与全量小时数均为未实测预算。真实A100行为、驱动卡死后的清理、FA2与graph
仍需人工上卡验证（GPU_PENDING）；若进程无法KILL，停止编排并保留现场，不自动关停租卡实例。

修复前证据：cpu-timewait-before-fix.txt记录一个真实连接进入TIME_WAIT后，普通bind预检
报EADDRINUSE。修复使用SO_REUSEADDR并实际listen检查；cpu-process-tests.txt的对应回归
与活跃IPv4/IPv6占用测试通过。没有把真实占用当成空闲，也没有修改/终止占用者。

最终结果：完整CPU2.11为96项/161.707秒，CPU2.8为96项/160.333秒；定向CPU2.8为43项/7.640秒，
无torch标准库为18项/4.182秒，全部OK。当前源码通过ruff和37库文件wheel字节一致性检查。
文档命令、11个JSON dry-run及源码指纹另存results；CPU日志不是GPU耗时/精度/学习数据。

## L40S 48GB 安徽区目标变更（当前）

按量计费，24vCPU/120GB RAM、30GB系统盘/50GB数据盘可扩容，镜像Ubuntu22.04/CUDA12.8.0/
Python3.10带conda。真实卡名、驱动版本、挂载点与四域网络尚未知，rental.json保留null。
驱动>=570、明确L40S卡名、独立数据盘与安徽区域确认后才可进入GPU编排；首次仅environment/baseline。
上一版A100配置、96项CPU日志及其时间仅为历史，当前测试证据以本轮results和指纹为准。

CPU新增验证覆盖卡名/驱动/区域/mount门槛、根分区bind拒绝、镜像endpoint+解析revision+文件SHA256、
篡改检测、官方wheel离线上传不访问GitHub、conda与已校验上传解释器注册。模型和数据均为几十字节
夹具，下载API/安装调用mock；没有连接镜像或下载真实资产，没有初始化CUDA或运行显存探针。
演示fraction0.4，microbatch16/acc8待真卡显存确认；OOM手动回退8/16并用新run-id。
当前小时数5.14–18.98（含30%）和3倍任务caps为未实测规划，preflight-01后按原始耗时校准。

本轮结果：CPU2.11 108项/138.349秒，CPU2.8 108项/135.199秒，L40S定向24项/1.871秒，
无site-packages进程模拟18项/4.223秒，全部OK。包内38份库源码通过字节校验；真实网络、磁盘、
显存、驱动/ABI与L40S吞吐仍GPU_PENDING。初次CRLF安装保护失败及修复前TIME_WAIT日志是历史失败证据，
当前通过状态见最新full CPU/artifact/package日志和源码指纹，不能把它们当作真卡验证。
