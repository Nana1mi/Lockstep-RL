# Lockstep-RL progress

## Latest pre-rental review (current; supersedes commit4b3a47c)

- Primary diagnostic populations stop at first EOS inclusive; raw fixed-workload
  tails remain separately available in JSON/CSV and cannot affect primary
  quantiles, trajectory counts or A/B tolerances. No-EOS and EOS-at-zero tested.
- Added cached FP32 inference head alongside live conversion. Complete load and
  successful commit refresh a nonpersistent buffer after all copies, including
  tied-embedding-only input. Forward never refreshes, cache storage is reused,
  and refresh failure breaks the engine. Diagnostics now14 processes, benchmark
  variants96 workers; extra cache bytes/KV blocks and conversion/compute variant
  differences are labeled separately from pure GEMM cost.
- Only environment/attention hard-gate the suite. Other child failures/startup
  errors are recorded while independent jobs continue; final failure is nonzero.
  --stages selects retries under a fresh run-id, always retaining environment
  checks. Required data subsets validate per job, so an unrelated missing split
  cannot block independent work. Hardware identity changes/interruption still stop.
- Demo microbatch16/accumulation8 with no checkpointing is tested against the
  complete128-response objective. No speedup is claimed before measurement.
  Added guarded uv-managed Python setup on persistent disk, refusing host fallback;
  same mount/OS/architecture is still required or venv must be recreated.
- Public sync_seconds is preflight+commit+invalidate; total and measurement-inclusive
  wall times stay separate. Per-trajectory example rotation is seed-reproducible.
  Repeat0 is non-discriminative; measured HF-BF16/FP32 is a precision reference,
  never an automatic bug verdict or a tolerance chosen after seeing results.
- Final full CPU suites:78 tests passed in142.601s on torch2.11 and138.100s on
  torch2.8. Ten GPU dry plans plus asset/managed-Python dry runs, nine document
  blocks, static/dependency checks and all35 wheel source-byte checks passed.
- Model-only stages also ignore the informational global asset-ready marker;
  missing diagnostic data is checked only for jobs that consume it. This boundary
  has a dedicated regression test in the final78-test suites.
- Full CPU logs and normalized source fingerprints are refreshed for this review.
  No CUDA work, rental, model/data or
  managed interpreter download has been performed by the agent.

The older revision below is retained as history, including its69-test counts and
previous live-only/stop-on-failure implementation details.

## 2026-10-27 priority revision (current delivery)

- Implemented P0.1–3, P1.4/6/5, P2.7–8 entirely with CPU development and tests.
  Exact acceptance map is IMPLEMENTATION_STATUS.md and docs/acceptance-audit.md.
- Prefix caching is configurable. The6-block, two512-prompt/two300-response CPU
  simulation shows positive cached reuse vs zero disabled reuse with score/token
  alignment. Cached and recompute preemption now remain separate measurements.
- Diagnostics use32 local real GSM8K trajectories,128-token responses, full real
  questions/templates plus cropped real solved context. Every preemption output
  is retained from its victim replay; eager/repeat define the noise floor. Raw
  scores and per-trajectory maxima/exceedance counts accompany pooled statistics.
- Live FP32 head is configurable in nano/training/scoring, tested against FP32
  linear, autocast and hot updates, and has independent diagnostic/performance
  variants. Training projects only response hidden positions.
- Weight preflight performs one reduction per unique source tensor and aggregates
  scalar flags, with identical alias storage bypassing equality work. Updates
  return phase timers; training logs actor-dtype change fractions and fixed-probe
  score deltas. The functional CPU stale-KV negative control fails as intended;
  actual GPU sensitivity still must be measured under locked tolerances.
- Smoke configs are renamed train-smoke-*; two120-iteration demos use512/128
  arithmetic tasks,16prompt×G8, lr1e-6/5e-6 and evaluation every10. Strict rewards
  remain compatible; lenient mode/status/failure rates and multiple EOS are added.
- Rental target is one A100-SXM4-80GB. The official FA2 release matrix has the
  pinned torch2.8/CUDA12/cp311/ABI-TRUE wheel; the old torch2.11/cu130 SM120 source
  build is not reused. setup_gpu_env is now wheel-only with ABI/hash checks, and
  does not overwrite an existing environment. Target2.8 CPU compatibility is
  checked in a separate project venv, preserving the original2.11 CPU environment.
- prepare_rental_assets and run_rental provide guarded CPU preparation and ordered
  persistent-disk execution, all commands/logs/raw output, GPU identity checks and
  an explicitly unmeasured3–11.1h planning range. No new mandatory memory task.
- Final full regression:69 tests passed on torch2.11.0+cpu in133.725s and on
  torch2.8.0+cpu in130.253s;27 focused target-version tests also pass. Artifact
  validation covers9 GPU dry plans plus the download-only asset dry plan,
  documentation syntax, installer guards and dependency consistency. The wheel
  contains all35 library sources with exact byte comparison and isolated imports.
- Final review removed even scalar device allocation from identical-source tied
  equality: it returns Python True directly, with a type assertion in acceptance
  tests. Both full CPU suites were rerun after that final adjustment.
- Final raw CPU counts/times are refreshed in results/cpu-tests*.txt and the
  normalized source manifest after the complete checks. All GPU gates remain
  GPU_PENDING; no card rented, no CUDA execution, no model/data downloaded here.

The following sections describe the prior delivery and intermediate history;
their test counts, hardware/build attempts and configuration names are historical.

## Current objective: GPU-validation-ready code delivery

The user updated the goal on 2026-10-26: implement all required modules, run CPU
checks, and prepare every GPU experiment for manual execution. Do not run CUDA
work or download large models/data autonomously. GPU validation does not block
independent coding. The technical scope remains in `PLAN.md`; dates are advisory.

### Final CPU-verified delivery

- All mandatory A–E source modules and their manual experiment entry points are
  implemented. The detailed requirement/evidence map is docs/acceptance-audit.md.
- Final full suite: **55 CPU tests passed in 112.181s**. Ruff E9/F, compileall,
  Bash syntax, dependency consistency, 16 documentation blocks and six dry plans
  passed. Wheel build and isolated CPU imports passed; 34 library sources match
  the packaged bytes. See results/ for raw records.
- Completed the subprocess-isolated three-way/path matrix, exact original Git
  source benchmark, sampler compute benchmark, prelocked-tolerance A/B recovery
  suite, and independent local evaluation. None was run on a real GPU.
- Added explicit torch.compile control separate from CUDA graph mode, checked
  against versioned official Context7 docs and a CPU backend-counter test.
- Added actual BF16 weight-cast diagnostics, all-clipped zero-gradient records,
  gradient release, early trajectory persistence/failure records and an optional
  actor-effective checkpoint with exact CPU round-trip tests.
- Review fixed partial parameter aliases, incomplete initial checkpoints, global
  dtype/device restoration and preserving unrelated existing process groups on
  rejected initialization. Concurrency tests verify new admission gets the
  successfully committed version.
- First checkpoint: db946ee. Final code/documentation checkpoint is delivered as
  HEAD and identified in the final response. No remote push was performed.
- All real GPU gates remain GPU_PENDING. Continue manually from GPU_RUNBOOK.md;
  FA2/SM120, numerical tolerances, actual memory and learning signal may need fixes
  after those measurements. No throughput or Qwen3 learning claim is made.

The sections below preserve the earlier development history; later delivery
status above supersedes their intermediate remaining-work notes.

### Intermediate implementation history

- A: Integrated FP32 selected-token scores, aligned response masks, EOS/length,
  request/group IDs, policy versions and sampling metadata. Added fixed-token
  replay that does not advance during temporary chunked-prefill samples or lose
  position after preemption. Legacy `step` token tuples remain available;
  `step(return_rollouts=True)` and `generate` expose complete records.
- C: Implemented full TP=1 key/shape/dtype/packed-shard/tied-weight preflight,
  upstream-loader in-place copies, idle-only version commit, logical cache reset,
  and fail-closed broken state. Admission/step/generate/update are serialized by
  an engine RLock. Small CPU tests use actual upstream packed loaders.
- D: Implemented token-normalized clipped GRPO, both frozen anchors, population
  group standard deviation, variable microbatch accumulation and skip behavior.
  A tiny CPU model passed two complete rollout/update/sync/next-rollout cycles.
  Added manual training, numeric rewards, independent validation and guarded
  GSM8K subset preparation. Real Qwen3 training remains GPU_PENDING.
- B/E: Added actual scheduler prefix/chunk/preemption trace events, graph batch
  replay metadata, fixed-trajectory error/ratio statistics and evidence checks.
  HF teacher forcing, same-effective-weight BF16/FP32 loading and saved-result
  plotting are implemented. Path orchestration and sync benchmarks remain.
  Existing attention/memory/benchmark scripts are manual-only.
- CPU evidence: all 36 tests passed in the independent CPU-only nano_rl
  environment (99.620s), including real randomly initialized one-layer Qwen3,
  actual activation-checkpoint invocation, next-token teacher-forcing equivalence,
  exact packed/tied hot-copy vs original safetensors loader and BF16-to-FP32
  effective-weight promotion. `results/cpu-tests.txt` contains the output.
- Ruff E9/F checks, Python compileall, Bash syntax checks and manual GPU installer
  Dry Run passed. No GPU installation or GPU execution occurred in these checks.
- Found and fixed an important integration issue during review: HF gradient
  checkpointing requires training mode. The trainer now keeps training mode and
  disables dropout modules individually, rather than using model.eval().
- All real GPU validations are GPU_PENDING. No Qwen3 inference/training,
  attention smoke, graph capture, or GPU benchmark has been run.

### Environment boundary update

- Stopped the previously started Qwen3 weight download after the user's update;
  its process reported `Terminated`. Partial data remains in ignored `.models/`.
- CUDA 13 compiler/runtime/CRT/NVVM installation completed only inside the old
  project `.venv-linux`. FA2 still has not been built successfully. No further
  GPU dependency work or GPU runs are being started automatically.
- Created an independent `.envs/nano_rl` Python 3.11 environment with CPU-only
  torch 2.11.0+cpu and development dependencies. Exact versions and metadata are
  saved in results/cpu-environment-freeze.txt and results/cpu-environment.json.
  Existing Linux kg_train and Windows
  kg_project environments have not been modified.
- Added opt-in guards to GPU and download scripts and CPU tests checking those
  guards. Environment probe defaults to metadata only; kernel work requires
  manual `--allow-gpu`.
- The old inherited environment failed to import HF Qwen3 because torch CUDA 13
  and torchvision CUDA 12.8 conflict. The independent CPU environment resolves
  this without changing the old environment; its complete suite passed.

### Intermediate next-work notes (completed in final delivery)

1. Finish the manual fixed-trajectory/path matrix and three-way comparison driver,
   with guaranteed trigger checks and raw provenance.
2. Finish A/B hot-update vs fresh-loader/initialization experiments and original
   upstream/logprob-overhead benchmark orchestration.
3. Complete experiment configurations, per-experiment runbook, failure capture,
   metrics/CSV/plot integration, then perform a full requirements audit.

## Historical environment groundwork (before updated GPU boundary)

### 2026-10-26 — environment and training-memory groundwork

- Cloned nano-vllm into the project root, branch `lockstep-rl-dev`, remote
  `upstream`, baseline `bb823b3e06983d71485a8e1f23715ebd87d98ef8`.
- Inspected all six required engine modules and wrote `docs/engine-flow.md`.
- Started Ubuntu WSL2; verified the GPU is visible: RTX 5080 Laptop, 16303 MiB,
  driver 596.49, compute capability 12.0.
- Created `.venv-linux` with read-only access to installed packages from
  `/home/rog/miniconda3/envs/kg_train`. New packages go into the project environment;
  the original environment is not modified. This is not yet a portable lockfile.
- `results/stage0/environment-initial.json`: Python 3.11.15, torch 2.11.0+cu130,
  Triton 3.6.0, Transformers 5.5.4; BF16 GPU GEMM passed. Installed FlashAttention 4
  exposes neither upstream FA2 entrypoint, so the inference backend is unverified.
- First FA2 install attempt failed during metadata generation: no `CUDA_HOME`.
  Existing `nvcc` is 12.8 whereas torch uses CUDA 13. Installed the official NVIDIA
  13.0.88 compiler into the project environment and started a bounded retry.
- Added checkpointed output-head chunk scoring. Five CPU tests passed, including
  full FP32 log-softmax value/gradient equivalence and absence of saved vocabulary
  activations. These tests do not establish real-model GPU memory feasibility.
- Added an executable full-parameter AdamW probe (FP32 master parameters, BF16
  autocast, FP32 scores), including optional inference/KV residency. Not yet run.
- Qwen3-0.6B download is in progress; the downloader resolves and records an exact
  Hub revision before fetching files.

### Remaining manual stage-0 evidence (GPU_PENDING)

- Usable FlashAttention varlen, paged KV decode, Triton writes, and CUDA graphs.
- Actual `example.py` / `bench.py` runs and short/long, single/multiple request paths.
- Full forward/backward/optimizer steps, alone and with real inference/KV residency.
- Measured memory configuration and decision on checkpointing/offload.
- At least three repeats for any reported performance comparison, with warmup,
  initialization, and steady execution recorded separately.


## 2026-10-27 最后租卡前修订：任务时限和进程/端口生命周期

- 新增标准库process_runner：Popen独立session、继承外层deadline、Linux subreaper接管孤儿，
  TERM宽限后KILL，正常/异常/中断都清理。PID/PGID/port/timeout/cleanup写入manifest及日志。
- rental、diagnostic、sync、benchmark和plot/archive统一有预算；只保留environment/attention硬关口。
  独立worker失败后继续，sample失败跳过依赖；清理失败/硬件变化仍作为基础设施停止。
- configs/task-timeouts.json集中记录预计秒数与3倍上限；dry-run可审阅，实际时间尚未测量。
- NCCL当前源码支持环境变量端口和默认120秒init等待；每worker预检空闲端口，上游2333单独检查。
  不改固定上游归档、不杀其他端口占用者、不承诺预检能消除bind竞态。
- CPU标准库模拟已执行，覆盖嵌套session/孤儿/忽略TERM/超时后继续/硬关口/端口所有者。
  精确当前测试数量、时长、完整输出见results，本轮全量验证后更新源码指纹和审阅ZIP。
- 文档给出首次--stages environment baseline与全新run-id全量安排。没有租卡、下载模型/数据，
  没有启动CUDA/真实训练/benchmark；真实运行时间与GPU验收仍GPU_PENDING。

本轮CPU查出并修复一项预检误报：最初仅bind会把TCP TIME_WAIT当作活跃占用。
先用真实loopback连接稳定复现（results/cpu-timewait-before-fix.txt），再改SO_REUSEADDR+listen；
修复后18项标准库测试通过，含TIME_WAIT、真正端口占用与IPv6 localhost检查。此失败日志
是有意保留的修复前证据，不是当前验收状态。

最终实跑：torch2.11 CPU全套96项/161.707秒，torch2.8 CPU全套96项/160.333秒，
目标版本定向43项/7.640秒，禁用site-packages的标准库模拟18项/4.182秒，全部通过。
文档/配置/干跑、ruff与37个库源码wheel内容校验通过；精确当前文件指纹见validation-manifest。
命令均限定CUDA_VISIBLE_DEVICES为空；没有执行真实GPU任务。审阅包增量基准为c279ed4。


## L40S 48GB 安徽区迁移（相对88a52c2）

按用户新目标更新当前配置/手册：L40S Ada sm_89、按量24vCPU/120GB、30GB系统盘与50GB数据盘，
Ubuntu22.04/CUDA12.8.0/Python3.10带conda镜像。保留原官方FA2 wheel/hash和Python3.11/torch2.8/cu128。
演示fraction0.4，microbatch16与关闭checkpointing保持候选；OOM8/acc16必须新run-id。
加入只读rental_preflight：未知实际值不猜填，读取卡名/驱动>=570、拒绝根盘及同设备bind的PERSIST，
记录同安徽区/GPU UUID/驱动/数据盘；总脚本在CUDA子任务之前验证。四域HEAD结果仅作为网络证据。
镜像源/两类revision显式传入模型和数据准备，manifest写入来源和逐文件SHA256；用前一次校验，
模型不每任务反复hash。支持官方wheel上传校验、pip index/torch cu128索引与离线wheelhouse。
持久Python支持原uv、已有conda prefix和上传官方archive校验后的解释器，拒绝宿主机base_prefix。
新CPU验证使用小型夹具与mock，不下载大模型/数据，不租卡、不执行GPU或真实显存/性能实验。

初次文档/CLI检查发现setup_gpu_env.sh被Windows编辑器写成CRLF，导致bash在保护开关之前报错。
已保留results/cpu-l40s-initial-checks.txt并恢复LF；随后重跑Linux语法/未授权安装保护与全套CPU。

本轮最终实跑：CPU torch2.11全套108项/138.349秒，CPU torch2.8全套108项/135.199秒，
定向L40S/rental24项/1.871秒，无site-packages进程测试18项/4.223秒，全部通过。
Linux findmnt JSON字段格式做了本机只读检查，不作为租卡挂载点实测；实际租卡预检值仍全部未知。
ruff通过，wheel包含38份库源码且字节一致、CPU隔离导入通过；文档/干跑及最终源码指纹另存results。
所有CPU测试命令均设置CUDA_VISIBLE_DEVICES为空；未执行GPU任务、未租卡、未下载真实模型/数据。
交付提供相对88a52c2的独立增量补丁，另保留相对固定上游的全量补丁与轻量ZIP。


## 2026-10-27 授权 L40S 远程验证（当前证据）

用户在本会话提供租卡 SSH、授权必需资产下载和远程 GPU 测试，并续费两小时用于完成实验环境。
本次实测：NVIDIA L40S，驱动580.126.09，安徽区；数据盘/root/gpufree-data，
PERSIST=/root/gpufree-data/lockstep；Python3.11、torch2.8.0+cu128、Transformers5.5.4、
官方FA2 2.8.3.post2固定wheel/hash。环境与缓存、模型、数据、结果均在持久盘；没有更换注意力后端。
直连网络受限，通过本机Clash的SSH loopback反向隧道获取官方资产；没有公开代理端口。

- 环境CUDA与六项注意力/分页KV/Triton/CUDA graph检查通过。
- preflight-01短基线24/24通过：原版/current/live-FP32/cached-FP32 × eager/graph ×3独立进程。
  仅short-single，32输入/32输出；不是完整96-worker、多长度/多请求性能结论。编排墙钟1178.52秒。
- readiness-grpo-02通过：seed0，temperature1，max_tokens128，4 prompts×G8，2轮，lr5e-6，
  microbatch4/accumulation8，checkpointing开启，enforce_eager=true，推理fraction0.3。
  真实奖励产生2次optimizer step、1次policy commit；另1轮因奖励全相同跳过。
  bf16 actor变化比例0.0658077364，固定探针logprob最大变化0.0465011597；
  三阶段同步和0.032931759秒（单次观测，不能概括同步性能）。这是闭环冒烟，不是学习效果证据。
- 演示显存探针通过：同卡fraction0.4，microbatch16，sequence256，无checkpointing，2步AdamW。
  峰值allocated36920615424 B、reserved38321258496 B；最低观察free8790540288 B。
  这是给定形状的容量证据，不保证2048-token长上下文或所有真实rollout形状。
- GSM8K四个互斥子集与Qwen3-0.6B官方资产下载、revision及文件hash清单完成。
- 发现并修复两个真实安装/入口问题：FA2 --no-deps安装缺einops；Transformers5.5.4
  apply_chat_template默认BatchEncoding，显式return_dict=False恢复token列表契约。
  最初训练失败与下载超时原始日志保留，不以重试覆盖。
- 修复后CPU2.11完整109项/135.595秒、CPU2.8完整109项/133.487秒，均通过；
  新回归覆盖Transformers5聊天token返回契约。CPU日志不是GPU精度证据。

完整32轨迹诊断、抢占/缓存对照、专门权重同步阴性对照、长上下文吞吐、两档学习率120轮学习实验
仍为GPU_PENDING。没有运行完整演示、没有保存训练checkpoint、没有自动续费或停止租卡实例。
原始证据目录见results/rental-20261027/README.md及轻量ZIP；真实模型、数据内容、环境、wheel均不入包。


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
