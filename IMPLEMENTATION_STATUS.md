# Implementation and verification status

**Current: full L40S campaign execution completed on clean ce9e354,13/13 jobs,
no failed jobs.** See `results/l40s-20261027/README.md` and the raw-result Release.
This publication does not turn execution success into unreviewed numerical or
learning claims. Earlier pre-rental/readiness statuses below are historical.


**2026-10-27 rental update:** Explicitly authorized L40S execution passed environment,
six attention/graph tests, 24 short inference baseline workers, a real two-iteration
GRPO update/sync smoke and the 16×256 co-resident memory probe. Full diagnostics,
long-context performance and learning claims remain GPU_PENDING. The older
pre-rental entries below are historical; details and raw data are in
`results/rental-20261027/README.md`, PROGRESS and EXPERIMENTS.


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
| Rental | L40S 48GB; persistent Python; per-task budgets, whole-tree cleanup, free-port checks, --stages | Real CPU hangs/nested sessions/orphans/SIGTERM/SIGKILL; port owner preserved; dependency skips and hard/soft continuation | Rental execution/runtime estimates GPU_PENDING |

Full regression: **108 passed in138.349s on torch2.11.0+cpu**, **108 passed
in135.199s on torch2.8.0+cpu**. Earlier78/94/96-test runs are historical.
Raw logs and exact normalized source fingerprints are in `results/cpu-tests*.txt`
and `results/validation-manifest.json`. CPU2.11 and the separate CPU2.8 candidate
environment remain distinct; neither is evidence of CUDA compatibility.

Both L40S demos now use microbatch16/accumulation8 without checkpointing;
128-response objective equivalence is tested against a full batch. Speedup is
unmeasured. Public sync_seconds sums preflight/commit/invalidate; overall
total_seconds and measurement-inclusive wall time remain separate.

Remaining concerns are empirical: actual L40S FA2 import/ABI/kernels and graphs;
noise floor vs numerical errors; stability of the deliberately invalidated-cache
assay under prelocked tolerances; BF16 update visibility and genuine reward
variance/learning; actual runtime vs estimated hours. The fixed candidate memory
budget is configurable, without a new offload implementation or memory milestone.

The matrix deliberately fails when a requested path did not trigger. A negative
control is successful only when stale reuse is observed and its output comparison
fails. Code/tests do not substitute for these real GPU observations.

Timeout revision: standard-library-only tests run with `python -S` in WSL, so no
PyTorch installation is needed to independently reproduce process/port checks.
See `results/cpu-process-tests.txt`. No CUDA kernels or paid instance were started.
Task caps are planning estimates, not performance measurements. Original upstream
port2333 and source remain unchanged; current workers support an override and
finite NCCL initialization. First paid pass is environment+baseline only.

Historical pre-L40S targeted torch2.8 CPU regression:43 passed in7.640s. Dependency-free
process/port acceptance:18 passed in4.182s. Those earlier docs/dry-run and package counts (37 library sources) are historical;
current L40S counts are recorded below and in results.

Current rental revision: L40S48GB/Ada sm_89, Anhui pay-as-you-go24vCPU/120GB RAM,
30GB system disk +50GB expandable data disk, Ubuntu22.04/CUDA12.8/Python3.10 conda image.
Observed name/driver/mount/network remain null until manual read-only preflight;
run_rental requires its matching record before CUDA. Driver>=570, same model,
region and driver are gates. Demo fraction0.4/mb16/acc8 needs manual memory evidence;
OOM fallback8/16 is documented. Explicit HF mirror source/revision/file hashes,
uploaded official wheel, pip/torch indexes and offline wheelhouse, persistent
conda/uploaded Python registration are CPU-tested using tiny mocked inputs.
Earlier96-test timings are historical; current evidence is in results/validation-manifest.json.

Final L40S evidence:108/138.349s (CPU2.11),108/135.199s (CPU2.8),24 targeted
L40S/rental tests/1.871s,18 dependency-free process tests/4.223s. Ruff, runbook
syntax/guarded dry runs and all38 packaged library sources pass. No rental-specific
GPU name, driver, mount or network value has been observed or filled in.


## 2026-10-27 ce9e354 正式 L40S 实验完成与公开归档

实验源码严格固定ce9e3542b3f31475ffa595e3f09b80e3f9190ed0，使用干净独立目录。
新的preflight于该提交执行，24/24 worker通过；27项实测耗时均低于对应上限的一半。
顶层manifest记录source_commit及源码干净状态，外部启动器显式将performance父阶段上限
从9000放宽到12000秒，源码configs/task-timeouts.json未改，覆盖记录进入manifest。
用户明确授权正式运行并取消十小时总截止；每任务超时、进程树清理和环境/attention硬关口保留。

2026-10-27 20:35:34（香港/北京时间）结束，13/13编排任务返回0，无任务超时/失败。
baseline24、diagnostics14、weight-sync25、performance96子任务完成；两档学习率各120轮，
另有两次更新检查、两种anchor冒烟、GSM8K冒烟和sampler。预检1173.01秒，
正式9683.30秒，合计10856.33秒（约3小时）。此前5–19小时是未实测保守范围，不能当作实测。

证据在results/l40s-20261027/：逐token主统计/post-EOS CSV、紧凑误差分布与逐轨迹统计、
同步分段耗时、逐轮训练CSV、配置、环境和manifest。详细轨迹/日志及服务器工具放Release附件，
通过publication.json记录原始服务器ZIP哈希、公用ZIP哈希、端点脱敏和独立模型/任务资产剔除项。
实验数值文件保持原始字节；不发布密码/令牌/私钥、权重、环境、缓存或checkpoint。

运行结束不等于科学结论全通过。诊断阈值越界、噪声参照、单seed学习曲线和GSM8K解析/截断
仍需分析；GSM8K末批解析失败50%，不能把该批统计当作全数据集解析率。
服务器GPU已空闲，定时巡检在确认完成和归档成功后删除；平台实例不会自动停止计费。
