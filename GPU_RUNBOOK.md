# Lockstep-RL 租卡人工执行手册

目标为 **单张 NVIDIA A100-SXM4-80GB（Ampere sm80）、Linux x86_64**。
所有 baseline、诊断、同步、训练、性能对照使用同型号、同驱动/依赖和同一 GPU UUID。
RTX 5080 / SM120 / cu130 本地构建步骤已不再适用，旧记录仅保留为历史。
所有真实实验仍为 **GPU_PENDING**。先完成 CPU 测试，再由操作者租卡并运行本手册；
Agent 不启动 GPU、不租卡、不自动下载大型模型或数据。

## 0. 租卡之前：源码、持久盘、依赖文件

目的：付费开机前备齐输入。需要 Linux/WSL、Python3.11、独立 CPU 环境及可挂载的
持久盘；下面 `/persistent/lockstep` 由操作者换成真实挂载位置。数据与结果不能只存临时系统盘。
源码须保留 Git 历史和固定上游对象，benchmark 会用 git archive 导出原版。

```bash
bash scripts/setup_cpu_env.sh /path/to/python3.11
CUDA_VISIBLE_DEVICES= .envs/nano_rl/bin/python -m unittest discover -s tests -v
.envs/nano_rl/bin/python scripts/validate_cpu_artifacts.py
PERSIST=/persistent/lockstep
.envs/nano_rl/bin/python -m pip install datasets
.envs/nano_rl/bin/python scripts/prepare_rental_assets.py --persistent-root "$PERSIST" --dry-run
# 只有下面这条由人工决定执行；下载在没有付费 GPU 的 CPU 实例/挂载盘上完成。
.envs/nano_rl/bin/python scripts/prepare_rental_assets.py --persistent-root "$PERSIST" --allow-download
```

还要把解释器本身放在持久盘，不能依赖租卡镜像的 `/usr/bin/python3.11`。
在付费 GPU 开机前的 Linux CPU 节点上，用系统任意受支持的 python3 做一次性 bootstrap：

```bash
PERSIST=/persistent/lockstep
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

## 1. 人工安装与环境关口

目的：确认二进制 ABI、BF16、FA2 varlen/paged KV、Triton 和 graph primitives。
前置：A100-SXM4-80GB 已由操作者选定，持久盘挂载且资产校验通过；宿主驱动不由脚本修改。

```bash
PERSIST=/persistent/lockstep
WHEEL="$PERSIST/wheels/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
bash scripts/setup_gpu_env.sh --dry-run
MANAGED_PY=$(cat "$PERSIST/python-executable.txt")
bash scripts/setup_gpu_env.sh --allow-install --env "$PERSIST/envs/nano_rl" --wheel "$WHEEL" --python "$MANAGED_PY"
GPU_PY="$PERSIST/envs/nano_rl/bin/python"
"$GPU_PY" scripts/run_rental.py --persistent-root "$PERSIST" --run-id review-01 --dry-run
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
PERSIST=/persistent/lockstep
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

例如只重跑诊断与同步；总脚本仍自动先跑环境和 attention，其他阶段不执行：

```bash
"$GPU_PY" scripts/run_rental.py --persistent-root "$PERSIST" --run-id retry-diag-sync-02 \
  --python "$GPU_PY" --stages diagnostics sync --power-mode "记录真实租卡功耗/性能模式" --allow-gpu
```

可选阶段为 environment、baseline、diagnostics、weight_sync（别名 sync）、grpo、performance。
普通失败的汇总状态为 completed_with_failures，硬关口失败为 blocked_by_environment；manifest
列出 failed_jobs/skipped_jobs/各退出码。一个失败组不会自动阻塞其他组或被当作通过。

未实测的规划估算：环境0.1–0.25h、基线0.1–0.3h、诊断0.4–1.5h、同步0.3–1h、
GRPO 冒烟+两档演示1–4h、性能0.4–1.5h；合计2.3–8.55h，加30%预留约 **3–11.1 GPU小时**。
这不是吞吐实测或价格承诺；安装/下载应在开机前处理，CPU 开销仍计入租用墙钟时间。
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
CPU 验证128条数据与完整batch全局token归一目标一致，A100 的实际显存和提速尚未实测。
默认 beta0 无 reference；不要报告不存在的 reference KL。演示 anchor=recomputed，
短冒烟另外比较 rollout/recomputed。推理预算为 A100 候选0.6，非 KV-only 比例，仍按实测调整。

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
