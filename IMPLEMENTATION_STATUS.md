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

## 2026-10-27 公开数据的 CPU 复算

新增 analysis/analyze_l40s.py、完整报告、两张图与可复算 JSON（记录输入 SHA256）。
32 条轨迹的 3,361 个主统计 token 复算：nano BF16 vs HF FP32 平均误差0.009935，
FP32 head为0.007849，降低21.0%。735个post-EOS token保留为独立统计。
计时按边界报告：GPU FP32源三阶段同步47.9/47.5ms；包含验证输出的匹配恢复加速
eager88.0倍、graph290.1倍；不能混用48ms与重新初始化计时宣称端到端1000倍。
批大小/graph匹配只是已测两种分组的观察，不宣称已定位具体GEMM或任意陪跑请求独立性。
吞吐变化−2.96%至+0.74%，部分graph差异大于观测进程噪声，不能全部归为噪声。
两次120轮演示的clipping未激活；two-updates小型正对照第二次更新有效裁剪2.22%，
rollout冒烟ratio越界2.22%但有效裁剪为0。单seed验证提升不代表多seed稳定结论。
README删除过时待租卡说法，区分历史过程记录与当前结果。原始数值文件保持不变。
后续2个seed复现配置及6个匹配anchor/update_steps2配置仅生成并做CPU检查，GPU_PENDING。
未连接服务器、重新租卡、下载资产或启动GPU。新分析测试独立于此前两环境各109项核心回归。
本次新增分析测试在CPU2.11与CPU2.8各8项通过；8份后续配置dry-run、17组原统计与45项输入哈希核对、16个文档代码块语法检查均通过。检查清单见analysis/l40s-20261027/checks.json。

## 2026-10-27 后续多 seed / anchor 实验已授权启动

操作者重启原L40S实例后明确要求开跑。持久盘、固定ce9e354源码及环境保留；
驱动580.126.09不变，GPU UUID变更，因此本轮重新执行seed0/1/2，不将旧卡seed0并入新seed统计。
模型权重/config/generation config哈希与原清单相同，torch2.8.0+cu128、
Transformers5.5.4、FA2 2.8.3.post2、Triton3.4.0已核对。新环境/attention检查均通过。
本轮9个120轮配置：3个seed的recomputed/update_steps1复现，及3seed×2anchor的update_steps2对照。
lr1e-6、16prompt×G8、microbatch16/acc8、推理fraction0.4、同512/128乘法任务保持一致；
所有run从原始权重开始。真实启动器在源码目录外，manifest记录源码及配置准备提交d333ff5。
源目录保持干净，每任务验证UUID/驱动/挂载/源码/配置哈希，1200/1800秒超时，失败记录后继续；
环境或基础设施异常停止，保留进程树清理。没有总租卡截止，不自动续费或关机。
启动状态：running，首个seed0已写入训练、实际BF16变化及固定探针指标；后续实验尚无完成结论。
结果目录为持久盘results/followup-ce9e354-01，tmux会话lockstep-followup。
本机30分钟巡检已建立（id lockstep-rl），只依赖当前认证的SSH内存会话。
持久authorized_keys修改被自动审批拒绝且未执行；未保存密码、私钥或修改SSH访问权限。

## 2026-10-27 后续九项实验完成并备份

固定ce9e354源码干净、同一新L40S UUID/驱动/区域完成9/9，均120轮、exit0、无超时。
22:20:03至23:08:40（香港/北京时间），2916.87秒。单次更新复现seed0/1/2最终验证
95.31%/94.53%/95.31%；两次更新rollout为94.53%/95.31%/92.97%，
recomputed为92.97%/96.88%/93.75%。每个seed的两个anchor采用相同计划设置。
全部6个两次更新实验均激活clipping：按所有逐更新记录计，rollout318/446次、
recomputed219/476次存在有效裁剪；这是更新次数，不是token裁剪率。
验证采用默认temperature1.0采样，跨seed标准差同时包含训练与验证采样随机性。
n=3且任务简单，不据0.26百分点均值差宣称anchor优劣。小型完成记录见
results/followup-l40s-20261027/README.md与summary.json。
结果ZIP21,221,376字节已备份本机，SHA256为
8971ca9cb1ba66358dcc94228bd4290380f0f013abde693214b90f7c22e0b6de；
ZIP CRC及内部1201项大小/哈希核对通过。模型、checkpoint、环境、独立任务数据文件不入包。
初始SFTP下载helper120秒等待超时，底层传输随后成功；只读SSH流回退也完成并验证同一哈希，
失败/重试记录保留，不修改实验。两条临时认证连接均在备份后关闭，无持久SSH权限变更。
GPU空闲；巡检lockstep-rl已删除；没有替操作者关闭平台或停止计费。

## 2026-10-27 后续实验公开归档

后续九项结果发布至 https://github.com/Nana1mi/Lockstep-RL，
独立Release为l40s-followup-20261027，首次Release保持不变。
仓库保留64份原始字节不变的配置/CSV/报告/图/运行脚本，另提供9份逐更新steps.csv，
共1166次更新记录及三seed均值/样本标准差曲线。CPU两环境复算一致，
45项分析输入哈希与原始逐更新汇总交叉核对通过；图形已检查。
原始ZIP21,221,376字节直接作为Release附件，SHA256保持
8971ca9cb1ba66358dcc94228bd4290380f0f013abde693214b90f7c22e0b6de，
内部1201项哈希及ZIP CRC通过，私钥/令牌/活动SSH端点扫描无发现。
不发布模型权重、checkpoint、环境或独立任务数据文件。源码仍为ce9e354，
此次只发布证据和CPU分析/文档，不启动GPU或下载大模型。
