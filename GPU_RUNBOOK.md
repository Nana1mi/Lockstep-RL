# Lockstep-RL 租卡人工执行手册

目标为 **单张 NVIDIA L40S 48GB（Ada sm_89）、Linux x86_64**。
平台：安徽区按量计费，24vCPU、120GB RAM；系统盘30GB，数据盘50GB（可扩容，平台给定0.01元/GB/天）。
镜像为开发/CUDA12.8.0/Python3.10/Ubuntu22.04（cuda_only，带conda），运行栈仍用持久Python3.11。
所有 baseline、诊断、同步、训练、性能对照使用同L40S型号、同安徽区、同驱动/依赖和同一 GPU UUID。
A100-SXM4-80GB的旧设置仅为历史，不能继续用于当前预检或耗时引用。
RTX 5080 / SM120 / cu130 本地构建步骤已不再适用，旧记录仅保留为历史。
2026-10-27 操作者已租卡，并在本会话明确授权远程配置、下载必需资产及最小 GPU 验证。
本次已在干净ce9e354完成全部13个正式编排任务；完整诊断和性能原始数据已归档，
CPU复算及结论边界见analysis/l40s-20261027/REPORT.md；多seed和匹配anchor比较仍待验证。原始证据见results/l40s-20261027/README.md。本次授权不改变以后会话的人工GPU/下载确认边界。

## 0. 付费之前：源码、持久盘、依赖文件

目的：付费开机前备齐输入。需要 Linux/WSL、Python3.11、独立 CPU 环境及可挂载的
持久盘；下面 `/persistent/lockstep` 由操作者换成真实挂载位置。数据与结果不能只存临时系统盘。
源码须保留 Git 历史和固定上游对象，benchmark 会用 git archive 导出原版。

```bash
bash scripts/setup_cpu_env.sh /path/to/python3.11
CUDA_VISIBLE_DEVICES= .envs/nano_rl/bin/python -m unittest discover -s tests -v
.envs/nano_rl/bin/python scripts/validate_cpu_artifacts.py
PERSIST=/USER_MUST_SET_ACTUAL_DATA_DISK/lockstep
.envs/nano_rl/bin/python -m pip install datasets
.envs/nano_rl/bin/python scripts/prepare_rental_assets.py --persistent-root "$PERSIST" --dry-run
# 只有下面这条由人工决定执行；下载在没有付费 GPU 的 CPU 实例/挂载盘上完成。
.envs/nano_rl/bin/python scripts/prepare_rental_assets.py --persistent-root "$PERSIST" --allow-download
```

还要把解释器本身放在持久盘，不能依赖租卡镜像的 `/usr/bin/python3.11`。
在付费 GPU 开机前的 Linux CPU 节点上，用系统任意受支持的 python3 做一次性 bootstrap：

```bash
PERSIST=/USER_MUST_SET_ACTUAL_DATA_DISK/lockstep
python3 -m venv "$PERSIST/bootstrap"
"$PERSIST/bootstrap/bin/python" -m pip install uv
"$PERSIST/bootstrap/bin/python" scripts/setup_managed_python.py --persistent-root "$PERSIST" --dry-run
"$PERSIST/bootstrap/bin/python" scripts/setup_managed_python.py --persistent-root "$PERSIST" \
  --uv "$PERSIST/bootstrap/bin/uv" --allow-install
```

输出 python/、python-executable.txt、python-manifest.json。脚本要求 uv-managed 3.11，
验证真实 executable/base_prefix 均落在持久盘，拒绝回退宿主解释器或静默升级已有记录。
依据 [uv 持久安装目录配置](https://docs.astral.sh/uv/reference/environment/#uv_python_install_dir)。
bootstrap 仅用于准备，不作为 GPU 环境的 base Python。后续机器保持相同绝对挂载路径、Linux
架构和兼容系统库；若改变路径/平台，必须用持久解释器重建 venv，不能保证任意移动即可运行。

输入：公开 Qwen/Qwen3-0.6B、openai/gsm8k。输出：models/Qwen3-0.6B、data/gsm8k 的
train64/validation32/diagnostic32/fewshot32、官方 FA2 wheel、assets-ready.json。
manifest 保存解析后的精确 Hub revision、seed、选取索引；fewshot 保留原始 worked solution，
四个子集 ID/prompt 互不重叠。不要把这些文件重新打进代码 ZIP。
判据：模型 manifest complete、数据清单齐备、wheel SHA256 匹配；数据/网络失败保留原始错误，
在开机前处理，不能用重复字符替代真实题目或换成未记录的数据。

候选 GPU 栈固定 Python3.11、torch2.8.0+cu128、Transformers5.5.4、FA2 2.8.3.post2。
torch2.8 来自[官方安装矩阵](https://pytorch.org/get-started/previous-versions/)，FA2 使用
[官方 release 资产](https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3.post2)
中的 cu12/torch2.8/ABI-TRUE/cp311 wheel，URL/hash 固定在 configs/rental.json。
这是避免现场编译的组合，不代表已经验证 CUDA kernel。CPU2.11 与 CPU2.8 记录单独保留。

## 1. 开机后按顺序预检、安装与环境关口

目的：先获取未知项，再安装二进制依赖；不猜卡名、驱动或挂载点。
按顺序执行，前四项为只读或安装准备；只有第五项显式allow-gpu才执行CUDA任务。

1. `nvidia-smi`记录完整输出，并用下面查询取得单卡name/UUID/driver；驱动主版本必须>=570，
   不满足即停止。不要自动升级驱动。卡名预计类似NVIDIA L40S，但只有实际输出可填expected。
2. `df -h`与`findmnt`确认数据盘，设置PERSIST；必须是数据盘而非30GB系统盘。
3. 四个域名执行`curl -I`，保留成功/失败/HTTP状态。失败时按下一节显式选镜像或离线，不盲目重试。
4. 安装持久Python3.11与GPU环境，准备模型/数据/wheel并校验hash；解释器、缓存也必须在数据盘。
5. 第2节只运行`--stages environment baseline --run-id preflight-01`。
6. 人工审阅后，用新run-id执行全量；此前对照必须同型号、同区域、同驱动。

```bash
nvidia-smi
nvidia-smi --id=0 --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
df -h
findmnt
# 以下值只能在看到实际输出后填写；不能直接保留占位执行。
PERSIST=/USER_MUST_SET_ACTUAL_DATA_DISK/lockstep
export XDG_CACHE_HOME="$PERSIST/cache"
export HF_HOME="$PERSIST/cache/huggingface"
export PIP_CACHE_DIR="$PERSIST/cache/pip"
export UV_CACHE_DIR="$PERSIST/uv-cache"
EXPECTED_GPU_NAME='USER_MUST_COPY_EXACT_NVIDIA_SMI_NAME'
mkdir -p "$PERSIST"
python3 scripts/rental_preflight.py --persistent-root "$PERSIST" \
  --expected-gpu-name "$EXPECTED_GPU_NAME" --region 安徽区 --gpu-id 0 --check-network
```

预检不import torch、不执行CUDA、不下载资产。脚本用findmnt的设备号拒绝根文件系统及其bind目录，
把实际name/UUID/driver/挂载信息/区域/四域HEAD输出写到`$PERSIST/rental-preflight.json`，不覆写旧记录。
所有未知值在`configs/rental.json.preflight_pending`为null；这个记录才是已确认值，原模板不自动猜填。
总脚本启动前重新核对该记录与实际GPU、驱动、数据盘。平台区域由操作者在控制台确认，无法从
nvidia-smi推断；HTTP HEAD连通也不代表后续大文件一定可下载。可单独检查同四域：

```bash
for domain in pypi.org download.pytorch.org github.com huggingface.co; do
  curl -I --connect-timeout 5 --max-time 20 "https://$domain"
done
```

驱动门槛按本项目>=570执行；[CUDA12.8官方发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
列出工具包配套驱动，仍需真卡验证ABI/内核。L40S的48GB/864GB/s见[NVIDIA规格](https://www.nvidia.com/en-us/data-center/l40s/)。

```bash
# PERSIST沿用上面实际数据盘值，不要替换成系统盘上的/persistent目录。
# wheel须先按第0节预取，或从可信联网机器下载后上传到数据盘；不需要先下载模型。
WHEEL="$PERSIST/wheels/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
bash scripts/setup_gpu_env.sh --dry-run
MANAGED_PY=$(cat "$PERSIST/python-executable.txt")
bash scripts/setup_gpu_env.sh --allow-install --env "$PERSIST/envs/nano_rl" --wheel "$WHEEL" --python "$MANAGED_PY"
GPU_PY="$PERSIST/envs/nano_rl/bin/python"
"$GPU_PY" scripts/run_rental.py --persistent-root "$PERSIST" --run-id preflight-01 --stages environment baseline --dry-run
```

安装器只接受新环境路径、指定 wheel 并校验文件名/torch/CUDA/ABI/SHA256，FA2 不回退源码编译；
torch 自带 CUDA runtime，不安装编译器或升级系统驱动。若 wheel 不匹配，停止并在 CPU 准备阶段
修正镜像/版本，不在付费 GPU 上临时编译。输入是持久盘 wheel；输出为独立环境和 freeze。
环境测试成功必须同时有 BF16 finite 与 FA2 入口；attention_smoke 还须 status=passed。
反馈完整安装日志、pip freeze、nvidia-smi、wheel hash 与第一处失败，不能只给最后一行。

## 2. 按顺序执行总脚本

目的：执行 environment → baseline → diagnostics → weight sync → GRPO → performance，
统一原始输出与计时。前置：CPU 验证、资产准备和环境安装完成。无需再跑旧的显存阶段；
现有 memory probe 仅保留为遇到 OOM 时的可选排障工具。

```bash
PERSIST=/USER_MUST_SET_ACTUAL_DATA_DISK/lockstep
GPU_PY="$PERSIST/envs/nano_rl/bin/python"
# 第一次开机只跑环境、attention、短基线；约半小时是规划值，不是实测承诺。
"$GPU_PY" scripts/run_rental.py --persistent-root "$PERSIST" --run-id preflight-01 \
  --stages environment baseline --python "$GPU_PY" \
  --power-mode "记录真实租卡功耗/性能模式" --gpu-id 0 --allow-gpu
```

先停止在这里，人工审阅preflight-01的环境、attention和baseline原始结果，确认FA2、eager/graph
与对照可用后再执行下面的全量命令；不要把两段一次性串联。使用全新的run-id：

```bash
PERSIST=/USER_MUST_SET_ACTUAL_DATA_DISK/lockstep
GPU_PY="$PERSIST/envs/nano_rl/bin/python"
"$GPU_PY" scripts/run_rental.py --persistent-root "$PERSIST" --run-id review-01 \
  --python "$GPU_PY" --power-mode "记录真实租卡功耗/性能模式" --gpu-id 0 --allow-gpu
```

总脚本从模板生成绝对持久盘路径配置，全部输出位于 `$PERSIST/results/review-01`；
每个子进程的 console、命令、returncode、wall_seconds、GPU inventory 和总 GPU wall hours
都会保存。仅 environment 和 attention 失败会停止后续工作；普通阶段失败或子进程启动错误记录
后继续下一独立任务，最终有失败则以非零退出码结束。GPU 身份变化/用户中断等基础设施问题仍停止。
每步重新核对同一 GPU UUID/驱动。用新 run-id 重试，
不覆盖旧失败或不利结果。结束状态 executed_review_raw_results 仅表示命令执行完，仍需审阅数据。

任务预算固定在 `configs/task-timeouts.json`：每项 expected_seconds × 3，包含初始化/编译，
dry-run 的每个 job 都显示预算。当前总任务上限：environment/attention 各12分钟、baseline50分钟、
diagnostics150分钟、weight-sync108分钟、两档demo各480分钟、performance150分钟；其余见配置。
诊断/同步 worker 各10.5分钟、benchmark worker7.5分钟、绘图/归档各90秒。这些是L40S未实测预算，不是A100的实测换算，preflight-01后按各job实际时长重新校准。
应根据原始日志调整配置后用新 run-id 重跑。外层预算约束整项工作（含CPU准备），嵌套 worker
继承绝对截止时间，不能逐层叠加预算。检查 GPU inventory 的 nvidia-smi 与元数据命令另有30秒上限。

每个任务 Popen(start_new_session=True)，超时记 returncode124/timed_out=true；先对整棵子孙
进程树发TERM，最多等5秒，再KILL并最多等5秒。Linux subreaper 接管父进程退出后遗留的孤儿，
包括另开session的worker；正常退出和用户中断也清理，恢复原信号处理器。
参照 [Python subprocess](https://docs.python.org/3.11/library/subprocess.html#subprocess.Popen)
与 [Linux subreaper 语义](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html)。
仅支持Linux/WSL2主线程中的串行编排。超时与失败都保存PID/PGID、实际时限、清理结果和原始日志，
之后继续下一独立任务；采样失败会把依赖worker标记 skipped_dependency，其余独立worker继续。
有失败的suite不生成“完整通过”汇总，已有原始结果保留。若SIGKILL后仍有存活进程，作为基础设施
故障停止，不能把下一GPU任务叠在未清理的进程上；应由操作者检查系统/驱动，Agent不自动关机。

当前源码的NCCL默认直启端口仍为2333；用 `LOCKSTEP_NCCL_PORT=29501` 可明确覆盖，
`LOCKSTEP_NCCL_INIT_TIMEOUT_SECONDS` 默认120秒。总脚本/各suite会在每个任务启动前bind检查：
默认给当前源码的每个worker重新挑空闲loopback端口，同一worker的各rank共享它；显式设置端口时
严格检查该端口。检查不占用/保留端口，存在竞态时由有限NCCL初始化时限和任务时限兜底。
固定上游的归档代码保持原样，baseline worker始终检查其写死的2333，不宣称环境变量能改原版。
端口被其他进程占用时只记录失败，绝不杀占用者；预检使用SO_REUSEADDR+listen，允许可复用
连接的TIME_WAIT而拒绝活跃监听，基线还检查localhost的IPv6。其他不可复用占用仍保留错误，
以新run-id重试。不要同时在同一张卡上跑多套编排。

无需torch即可在Linux/WSL2复核本轮CPU进程模拟（`-S`禁用site-packages）：

```bash
CUDA_VISIBLE_DEVICES= python3 -S -m unittest tests.test_process_runner -v
```

例如只重跑诊断与同步；总脚本仍自动先跑环境和 attention，其他阶段不执行：

```bash
"$GPU_PY" scripts/run_rental.py --persistent-root "$PERSIST" --run-id retry-diag-sync-02 \
  --python "$GPU_PY" --stages diagnostics sync --power-mode "记录真实租卡功耗/性能模式" --allow-gpu
```

可选阶段为 environment、baseline、diagnostics、weight_sync（别名 sync）、grpo、performance。
普通失败的汇总状态为 completed_with_failures，硬关口失败为 blocked_by_environment；manifest
列出 failed_jobs/skipped_jobs/各退出码。一个失败组不会自动阻塞其他组或被当作通过。

未实测的L40S规划估算：环境0.1–0.3h、基线0.15–0.5h、诊断0.7–2.5h、同步0.5–1.8h、
GRPO 冒烟+两档演示1.8–7h、性能0.7–2.5h；合计3.95–14.6h，加30%预留约 **5.14–18.98 GPU小时**。
这不是吞吐实测或价格承诺；能预先上传的资产应在付费前准备，开机后安装/补齐下载和CPU开销
也计入租用墙钟时间。50GB数据盘还要容纳环境/wheel/cache/B模型/checkpoint，逐阶段用df/du检查
实际余量，必要时由操作者扩容；不能把空闲系统盘当PERSIST替代。
首轮真实耗时出来后按 manifest 调整预算。

以下各节给出总脚本已执行的阶段、输出和人工验收口径；需要单独重试时使用对应原始命令，
只更换输出目录。总脚本生成的所有配置在 run-id/configs/，不要省略实际配置。

## 3. 基线

输入：同一 Qwen3 权重/seed/T1、原版固定 commit 与当前源码；先 short-single 的 eager/graph。
`benchmark_suite.py` 导出原版源码，不打补丁；同一输入输出工作量。
输出 baseline/manifest.json、每进程 JSON/log、measurements.csv 和导入源码路径。
判据：正常退出、token 数量一致、原版对象存在；FP32-head 是当前版单独变体，不算上游功能。
失败反馈模型 revision、实际导入路径、commit/status、全日志；不依据一次差异宣称提速。

## 4. 多轨迹、噪声底线与 fp32 head

输入：32 条真实 GSM8K 题目、每条128 response tokens、prompt1024、block256。
真实已解答例题填充 few-shot context，允许仅该前缀按 token 裁剪，完整题目/template 保留；
prompt-provenance 保存例题 ID/裁剪信息；例题池先按 seed 打乱，再随 traj-k 轮换顺序。
题目超预算就增大配置重新规划，不能截掉题干。

矩阵为 sample +11 个 nano replay +HF BF16/FP32，共14进程，每个覆盖K条轨迹。
eager/repeat 同配置、独立进程、逐条 batch1，检查重复性；若误差为0，标为无区分度，
不把任意非零误差自动解释成异常。HF BF16-vs-FP32 是实测精度参照，并不是硬容差或全局误差界。
batch/padding 的陪跑都是真实轨迹；每条 canonical 轨迹在两种抢占实验里都当一次 victim。
preemption_cached 允许自复用且 reused_tokens>0；preemption_recompute 禁止 cache 且必须=0。
两者都须真实抢占/re-prefill，不能统称“抢占后全量重算”。

保留每步转换权重的 FP32 live head，并新增 cache_fp32_lm_head 变体：加载后建立非持久化
FP32 buffer，每次 commit 完成所有拷贝后刷新，forward 不更新缓存。刷新失败仍 broken，不能输出。
两种都与 HF FP32 比较，也互相比较；二者有效权重/数学相同。缓存约额外622MB（以实测字节为准），
其成本移到初始化/同步，不能只报告 steady 吞吐。训练端仍用可求导 live 权重，不使用 detached cache。
HF FP32 只是同一 BF16 有效权重的精度提升。torch_compile 与 enforce_eager 独立控制。

输出：逐路径 rollouts/all-rollouts/trace、逐 token 原始两端 logprob/ID/位置/差值 JSON/CSV，
每条记录 first_eos_index/primary_response_tokens/post_eos_tokens；主统计仅包含首 EOS 前（含 EOS）。
为路径触发保留到128的继续生成，EOS后尾段位于 JSON.post_eos 与单独的 -post-eos.csv，
不混入主分位数或超阈值轨迹数。没有尾段时记录 count0/null 分布，不捏造0误差样本。
noise-floor.json、汇总分布、每轨迹最大误差、超阈值轨迹数和图。统计须覆盖全部K条且请求匹配；
同组 companion 的重复不计入分位数。判据：所有 path 真实触发、token/mask/score 对齐；
误差结论同时看重复性、HF BF16/FP32 精度尺度与路径覆盖，不凭一个 P99 或零噪声底线判错。
失败反馈完整 token/trace/config/template/revision 和首次异常位置，保留整个失败组。

## 5. 同步与阴性对照

输入：固定 A、确定性扰动 B、锁定 atol/rtol=0.001；先 fresh 重复性，再 hot。
每 backend 各3fresh、3GPU-FP32 hot、3CPU-BF16 hot、3GPU-FP32 negative，加采样共25workers。
GPU-FP32 与训练来源一致；两个来源不能混成一个时延分布。

输出：参数 hash、parameter/KV/graph 存储证据、A缓存命中/首B无旧命中/新B命中、原始 scores，
preflight/commit/invalidate 三段计时，以及另外的变化率测量时间和源准备时间。
对外同步耗时用 sync_seconds=三段之和；total_seconds 含变化统计，禁止冒充同步耗时。
正常 hot/fresh 参数必须一致，分数在预锁容差内；失败不允许继续用混合版本。

阴性对照只在 weight_sync_worker 的 `--skip-invalidate` 测试参数中存在：必须真的复用旧 KV，
并且与 fresh-B 的分数比较不通过。若误差在既定容差内，就判定该实验未能检测到错误，
保留结果，重新设计更敏感但预先声明的扰动/轨迹；不能声称阴性对照已通过。
反馈完整 hash/trace/分段计时、source dtype/device、预声明容差及首个不同 token。

## 6. GRPO：冒烟、演示与解析

输入：train-smoke-simple/gsm8k 保留小规模；train-two-updates 专门验证两次 update 的冻结锚点。
两档演示分别用 lr1e-6/5e-6，16prompt×G8、120轮、train512/validation128、每10轮评测。
microbatch16、accumulation8、关闭 backbone gradient checkpointing；输出头分块重算仍保留，
smoke 小配置保留作排障。
当前演示fraction0.4约为总显存19.2GB的推理侧预算，剩余给FP32 actor、梯度、Adam状态、activation
与碎片；这不是KV-only份额，也不是已证明48GB能放下。microbatch16必须经人工显存探针确认。
如OOM，保留失败原始输出，用新run-id配置microbatch8且accumulation16；不要只改一个值。
CPU 验证128条数据与完整batch全局token归一目标一致，L40S 的实际显存和提速尚未实测。
默认 beta0 无 reference；不要报告不存在的 reference KL。演示 anchor=recomputed，
短冒烟另外比较 rollout/recomputed。推理预算为 L40S 候选0.4，非 KV-only 比例，仍按实测调整。

strict 保持原格式规则；lenient 接受货币符号、句末句点、{,}和文本单位。
每样本记录 ok/no_answer/unparseable、reward 和 text；解析失败率与答错率分开看。
训练仅对 response hidden 过 lm_head，prompt/padding 不做词表投影；EOS 保留进 mask。
generation_config 的任一 EOS 都停止，实际 EOS 列表写入 sampling_config/checkpoint。

输出：每轮原始 reward/解析状态、response长度、每 update 的 loss/ratio均值与极值/clip、
参数/梯度变化、真正 actor BF16 变化比例与各参数计数、固定探针轨迹前后 logprob，
独立验证与 checkpoint。CSV 是方便检索的摘要，完整 steps 在 metrics.jsonl。
判据：真实参数/actor/probe 变化、下一轮版本对应，独立验证不泄漏；学习效果必须看实际曲线。
zero_advantages/全裁剪跳过如实记录，不能拿版本递增替代学习证据或将零信号运行称为闭环成功。
OOM、解析大量失败、reward 全0/1、BF16变化为0时保留失败批/计数/长度/配置后再改；
不临时调假 reward。无需为这一版先实现动态显存卸载。

## 7. 性能

输入：同卡/同驱动/同供电、相同prompt/response工作量；上游/current/current-FP32-live/current-FP32-cached
各 eager/graph、4种长短/单多形状、3独立进程，每个进程3稳定重复，共96workers。
输出：全部逐次 JSON/CSV、初始化/预热/steady时间、吞吐、峰值、FP32-head吞吐变化百分比。
live/cached 分别对比 BF16；live减cached的总时间差单列，用来观察每步转换成本。额外 cache bytes
及实际 KV blocks 也记录，两者有内存驻留差异，因此该差不是孤立 GEMM 的纯计算时延。
Sampler 合成 logits microbenchmark 单独标注，不能冒充端到端结果。
判据：重复完整、输出数量一致、无失败，明确预热 prefix cache 的状态；嵌套重复不当成独立样本。
反馈所有正负差异和环境，不删不利重复。torch.compile/手工 CUDA graph 的成本分别描述。

## 8. 完成后带回的材料

- 完整命令、配置、两套 commit、model/dataset revision、环境 freeze/wheel hash 和固定 GPU inventory。
- 全部 per-token JSON/CSV、rollouts/trace、per-step JSONL/CSV、带 reward/parse_status 的好坏样例。
- 同步三段计时、每来源重复、吞吐/峰值、阴性对照、noise floor 与 FP32-head收益/代价。
- 首次失败阶段和原始错误、未通过项、墙钟 GPU 小时。大日志可保留开头和最后几百步，但不删原始数值表。

在真实材料回来之前，所有上述 GPU 验收均保持未完成。

## 国内网络与离线上传回退（均由人工选择）

先保留四域HEAD失败输出，再选择回退；脚本不自动从官方切到第三方。模型和数据支持HF兼容的
hf-mirror endpoint（不把ModelScope URL当HF API）；manifest同时记录实际endpoint、请求revision、
解析出的commit revision，以及模型文件/四份子集的大小和SHA256。首次使用资产时总脚本核验hash，
同一轮不反复hash整份权重。旧资产manifest若缺files，需要人工重新准备并建立完整清单。
镜像信任边界不同，哈希记录供复核，不声称第三方内容经过官方签名认证。所有下载仍需--allow-download。

```bash
# 可选：仅包索引走国内镜像；torch仍锁cu128，也可指向已备齐依赖的离线wheelhouse。
PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
bash scripts/setup_gpu_env.sh --allow-install --env "$PERSIST/envs/nano_rl" \
  --wheel "$WHEEL" --python "$MANAGED_PY" --pip-index-url "$PIP_MIRROR"
# 若download.pytorch.org不可达，先在可联网Linux x86_64/Python3.11机器上备齐所有wheel后上传：
# --wheelhouse "$PERSIST/wheelhouse" 禁止索引网络；torch仍要求2.8.0+cu128，ABI验证不变。
# 任选官方源或明确的HF镜像，实际选哪个就记录哪个，不自动fallback。
"$GPU_PY" scripts/prepare_rental_assets.py --persistent-root "$PERSIST" \
  --hf-endpoint https://hf-mirror.com --wheel "$WHEEL" --allow-download
```

`--model-revision`/`--dataset-revision`可指定已确认commit；默认main也先解析为具体commit再下载。
HF接口依据[HfApi endpoint](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
与[snapshot_download](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)。
模型/数据准备须用已装datasets/huggingface_hub的CPU环境或GPU venv（GPU_PY），不要用裸MANAGED_PY。
如果未在付费前准备模型/数据，在GPU环境安装好后、preflight-01之前用上面的命令补齐。
FA2 wheel若GitHub不通，在联网机器按rental.json固定URL下载后上传到数据盘wheels，传--wheel。
文件名与原官方SHA256仍严格校验；不换wheel、不回退源码编译。uv工具本身也可预下载wheel上传安装。

解释器首选第0节uv-managed路径。GitHub/Python下载不通时可在联网机器下载官方
python-build-standalone的Linux x86_64 Python3.11 install_only包及可信官方SHA256，上传数据盘。
固定包名/哈希由实际选定官方release获取，不猜值。人工核验后解压到数据盘，再注册：

```bash
PY_ARCHIVE="$PERSIST/uploads/USER_MUST_SET_OFFICIAL_PY311_ARCHIVE.tar.gz"
PY_SHA256=USER_MUST_SET_OFFICIAL_ARCHIVE_SHA256
printf '%s  %s\n' "$PY_SHA256" "$PY_ARCHIVE" | sha256sum -c -
mkdir -p "$PERSIST/python-upload"
tar -xf "$PY_ARCHIVE" -C "$PERSIST/python-upload"
# 以下解释器路径按实际archive内部结构确认；脚本只注册，不解压/安装其他内容。
python3 scripts/setup_managed_python.py --persistent-root "$PERSIST" --provider uploaded \
  --python "$PERSIST/python-upload/python/bin/python3.11" --source-archive "$PY_ARCHIVE" \
  --source-sha256 "$PY_SHA256" --allow-install
```

uv原生支持配置[Python安装镜像](https://docs.astral.sh/uv/reference/environment/#uv_python_install_mirror)，
并按其下载元数据校验SHA256；上面的上传路径直接注册已有解释器，manifest记录archive hash/来源。
如改用镜像自带conda，创建**新的持久prefix**，不修改base或既有环境：

```bash
conda create -y -p "$PERSIST/py311" python=3.11
python3 scripts/setup_managed_python.py --persistent-root "$PERSIST" --provider conda \
  --python "$PERSIST/py311/bin/python" --allow-install
MANAGED_PY=$(cat "$PERSIST/python-executable.txt")
# setup_gpu_env.sh --python "$MANAGED_PY" 仍另建独立GPU venv。
```

依据[conda prefix环境](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#specifying-a-location-for-an-environment)。
注册时要求Python3.11且executable/base_prefix都在PERSIST，拒绝宿主机回退；记录provider为conda或uploaded。
解释器已注册则拒绝覆写；换绝对挂载位置/平台仍须重新建venv，不能把指向旧路径的venv直接搬用。
所有pip/HF/uv缓存也设置在数据盘（HF_HOME、XDG_CACHE_HOME、PIP_CACHE_DIR、UV_CACHE_DIR）。

人工显存探针（不纳入自动开机流程，预检通过后再执行；Agent不运行）：

```bash
"$GPU_PY" scripts/training_memory_probe.py --allow-gpu --model "$PERSIST/models/Qwen3-0.6B" \
  --batch-size 16 --sequence-length 256 --steps 2 --optimizer adamw --co-resident \
  --inference-memory-fraction 0.4 --output "$PERSIST/results/memory-l40s-mb16.json"
```

探针是候选共驻形状，仍须结合真实GRPO峰值判断；不是所有长度/编译状态的显存保证。


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

本次可复用路径：`PERSIST=/root/gpufree-data/lockstep`，解释器`$PERSIST/envs/nano_rl/bin/python`。
已完成的run-id不可覆盖；正式全量须使用新run-id。重新开机或换机先重新只读预检，并检查数据盘挂载，
不能把本次卡名/驱动/UUID证据照搬到另一台机器。代理隧道只在本机SSH保持连接时可用；
已准备的模型/数据用于离线实验，后续新增下载需要重新建立隧道或验证直连。


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
