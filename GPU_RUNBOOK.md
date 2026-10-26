# Lockstep-RL 人工 GPU 验证手册

所有真实 GPU 项目均为 **GPU_PENDING**。GPU 命令和大型模型/数据下载由操作者亲自执行，Agent 只进行 CPU 开发。CPU 测试通过不代表 CUDA 数值、显存或性能通过。技术定义见 PLAN.md，实现状态见 IMPLEMENTATION_STATUS.md。

## 0. 准备、环境与反馈约定

目标硬件为 RTX 5080 Laptop、sm_120、约 15.92GiB 显存。已读取 WSL 资源：24 核 Core Ultra 9 275HX、约 15GiB RAM、4GiB swap；Linux 文件系统约有 906GiB 空闲、E 盘约有 194GiB 空闲。运行前重新检查，不要混淆主存、磁盘和显存。

优先使用 Linux 文件系统中的已提交代码，避免 /mnt/e 小文件 I/O：

```bash
mkdir -p ~/projects
git clone --branch lockstep-rl-dev /mnt/e/Lockstep-RL ~/projects/Lockstep-RL
cd ~/projects/Lockstep-RL
mkdir -p results/manual configs/local
set -o pipefail
```

也可继续使用 /mnt/e/Lockstep-RL。以下命令都从仓库根目录运行。不得修改 Windows kg_project、现有 Linux kg_train、系统驱动或系统 CUDA。现有 kg_train 的 Python 仅作为基础解释器，新包进入项目的独立 nano_rl：

```bash
bash scripts/setup_cpu_env.sh /home/rog/miniconda3/envs/kg_train/bin/python
CUDA_VISIBLE_DEVICES= .envs/nano_rl/bin/python -m unittest discover -s tests -v
.envs/nano_rl/bin/ruff check --select E9,F lockstep_rl nanovllm scripts tests example.py bench.py
```

每次实验使用新的目录，例如 diag-01、diag-02，保留失败记录。console 日志放在输出目录旁边，不能先在待创建的输出目录中放 tee 文件，否则非空目录保护会拒绝运行。失败后停止依赖它的后续阶段。

先记录真实供电/性能模式并生成本地配置：

```bash
read -r -p '实际供电与性能模式: ' LOCKSTEP_POWER_MODE
export LOCKSTEP_POWER_MODE
.envs/nano_rl/bin/python - <<'PY'
import json, os
from pathlib import Path
mode = os.environ.get('LOCKSTEP_POWER_MODE', '').strip()
if not mode:
    raise SystemExit('必须填写实际模式')
for name in ['diagnostics', 'weight-sync', 'inference-benchmark', 'train-simple', 'train-gsm8k']:
    data = json.loads(Path(f'configs/{name}.json').read_text())
    if 'power_mode' in data:
        data['power_mode'] = mode
    if name.startswith('train-'):
        data['save_final_checkpoint'] = True
    Path(f'configs/local/{name}.json').write_text(json.dumps(data, indent=2) + '\n')
data = json.loads(Path('configs/local/inference-benchmark.json').read_text())
data['cases'] = {'short-single': data['cases']['short-single']}
Path('configs/local/baseline-smoke.json').write_text(json.dumps(data, indent=2) + '\n')
data = json.loads(Path('configs/local/train-simple.json').read_text())
data['grpo']['update_steps'] = 2
data['iterations'] = 2
Path('configs/local/train-two-updates.json').write_text(json.dumps(data, indent=2) + '\n')
PY
```

所有实验共同的反馈包：完整命令、commit/status、实际配置、console 日志、原始 JSON/JSONL/CSV、trace、pip freeze、驱动/CUDA/attention backend、实际供电模式、首次失败阶段。保留非零退出码，不只提供最后一行报错。下面每节给出目的、前置、输入、完整命令、输出/日志、判据、错误处理及额外反馈内容。

## 1. WSL2、CUDA 与依赖冒烟

目的：确认 Linux BF16、Triton、FA2 varlen/paged KV 和 graph primitives。前置：WSL2、Python 3.11、CPU 测试通过。候选栈为 torch 2.11.0+cu130、Triton 3.6.0、Transformers 5.5.4、FA2 2.8.3.post2、CUDA compiler 13.0.88；GPU 兼容性尚未验证。

```bash
bash scripts/setup_gpu_env.sh --dry-run
bash scripts/setup_gpu_env.sh --allow-install /home/rog/miniconda3/envs/kg_train/bin/python \
  2>&1 | tee results/manual/install-gpu-01.log
.envs/nano_rl/bin/python scripts/environment_probe.py --allow-gpu \
  --output results/manual/environment-gpu-01.json 2>&1 | tee results/manual/environment-gpu-01.log
.envs/nano_rl/bin/python scripts/attention_smoke.py --allow-gpu \
  --output results/manual/attention-01.json 2>&1 | tee results/manual/attention-01.log
.envs/nano_rl/bin/python -m pip freeze > results/manual/gpu-freeze-01.txt
```

输入是小型随机 BF16 张量，不加载模型。环境 JSON 只证明 torch CUDA 和 imports，attention JSON 须 status=passed，包含 varlen、分页 prefix、Triton 写入、decode、改变输入后的 graph replay；预设 atol/rtol=0.02。

常见错误：无 nvcc/头文件、CUDA major 不同、sm_120 无 kernel、FA4 缺少 FA2 入口。反馈完整编译日志、编译器和包版本；不能修改驱动、混入旧环境或悄悄更换 attention。旧继承环境曾有 torch cu130/torchvision cu128 冲突，独立环境不安装 torchvision。该关口不通过就先停止真实实验。

## 2. 模型、示例、固定原版与基本路径

目的：准备精确版本并跑通 example.py、bench.py 及原版源码对照。前置：阶段 1 通过。

```bash
HF_HUB_DISABLE_XET=1 .envs/nano_rl/bin/python scripts/download_model.py --allow-download \
  --destination .models/Qwen3-0.6B 2>&1 | tee results/manual/model-download-01.log
.envs/nano_rl/bin/python example.py --allow-gpu --model .models/Qwen3-0.6B \
  --max-tokens 32 --memory-fraction 0.5 --token-budget 512 --max-num-seqs 8 \
  2>&1 | tee results/manual/example-01.log
.envs/nano_rl/bin/python bench.py --allow-gpu --model .models/Qwen3-0.6B \
  --num-seqs 4 --input-len 32 --output-len 32 --fixed-length --repeats 3 \
  --token-budget 512 --max-num-seqs 8 --max-model-len 2048 --memory-fraction 0.5 \
  --output results/manual/current-smoke-01.json 2>&1 | tee results/manual/current-smoke-01.log
.envs/nano_rl/bin/python scripts/benchmark_suite.py --allow-gpu \
  --config configs/local/baseline-smoke.json --output-dir results/manual/baseline-smoke-01 \
  2>&1 | tee results/manual/baseline-smoke-01.console.log
```

输入：固定解析后的 Hub revision、短合成 token 和关闭 thinking 的 chat。输出：_lockstep_model.json、精确版本、示例文本、重复吞吐/显存、实际导入源码路径。原版由固定 commit git archive 提取，无源码补丁；外部 runner 统一预算。

判据：完整 manifest=complete；示例有效；固定输出数量与预算一致；原版/当前版 eager 和 graph 正常退出。这里不能据一次速度差异声称改进。缺少权重必须报错，不能使用未初始化参数。首次编译久时检查进程/日志，不要观察超时就重复启动。反馈原版 commit、实际源码路径、模型 revision 和失败入口。

## 3. 完整 optimizer 与训推合计显存

目的：测 forward/backward/optimizer.step，包含真实 AdamW moment state。前置：阶段 2 通过。0.3 是推理端总预算比例，不是独立 KV-only 比例；所有值均为候选。

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True .envs/nano_rl/bin/python scripts/training_memory_probe.py \
  --allow-gpu --model .models/Qwen3-0.6B --batch-size 1 --sequence-length 256 \
  --chunk-size 32 --steps 2 --optimizer adamw --output results/manual/memory-alone-256-01.json \
  2>&1 | tee results/manual/memory-alone-256-01.log
PYTORCH_ALLOC_CONF=expandable_segments:True .envs/nano_rl/bin/python scripts/training_memory_probe.py \
  --allow-gpu --model .models/Qwen3-0.6B --batch-size 1 --sequence-length 256 \
  --chunk-size 32 --steps 2 --optimizer adamw --gradient-checkpointing \
  --co-resident --inference-memory-fraction 0.3 --output results/manual/memory-resident-256-01.json \
  2>&1 | tee results/manual/memory-resident-256-01.log
```

输入：固定合成 token，FP32 master、BF16 autocast、FP32 分块 scores。JSON 含每步 loss、grad norm、参数 delta、forward/backward/step 的 allocated/reserved/peak 和设备 free。resident 模式必须有真实 actor、KV 和训练前后生成。

判据：两步均有限且参数变化；status=passed；KV bytes/blocks>0。仅 forward 成功不算。随后用新文件名测试实际更长长度/microbatch。OOM 保留 failure_memory，检查其他占用、长度、microbatch、checkpointing、KV；可单独用 --optimizer sgd 测候选。小探针不能证明全部 GSM8K 长度安全。若候选都失败，反馈最低失败配置/各阶段峰值，再决定受限 offload；当前不宣称实现 offload。

## 4. Token logprob、三方数值与全部执行路径

目的：相同 prompt/response IDs 的 teacher forcing 及真实路径对照。前置：1–3 通过。矩阵固定 256-token block、512-token 完整 chat、32 response tokens。

```bash
.envs/nano_rl/bin/python scripts/diagnose.py --dry-run --config configs/local/diagnostics.json
.envs/nano_rl/bin/python scripts/diagnose.py --allow-gpu --config configs/local/diagnostics.json \
  --output-dir results/manual/diag-01 2>&1 | tee results/manual/diag-01.console.log
```

共 10 个独立进程：采样、7 条 replay 路径、HF BF16、HF FP32。FP32 提升同一份 BF16 有效权重；禁用 autocast/TF32/BF16 reduced precision reduction，记录 backend。模型不同时驻留。

配置的 torch_compile 独立控制编译指令：true 使用 default stance，false 使用 force_eager。它和 enforce_eager 的手工 CUDA graph 开关是两个不同因素；显式 Triton KV kernel 仍然存在。需要对照编译影响时另存配置，不能把改变两个因素的结果当作单因素图对照。

输出：每路径 rollouts/trace/report、逐 token delta JSON/CSV、mean/P95/P99/max、exp(delta)、长度分组、直方图。判据：token/score/mask 对齐；prefix 确实复用；chunk 有范围；graph/padding 确实 replay；目标请求确实被抢占并 re-prefill。配置但没触发会失败。统计 completed 不等于所有数值已证明正确。

单独评分命令：

```bash
.envs/nano_rl/bin/python scripts/score_hf.py --allow-gpu --model .models/Qwen3-0.6B \
  --trajectory results/manual/diag-01/sample/rollouts.json --dtype float32 \
  --attention-backend eager --output results/manual/hf-fp32-eager-01.json
```

未触发抢占时查 block、prompt 边界、同轮准入和 KV 上限，不能删掉触发检查。误差大时反馈原始 token/shift/数值设置/trace/分组结果；不要事后统称 BF16 噪声。已知错误注入仅在标明 synthetic/known-error 的 CPU 测试中，不能算自然发现。有限最大误差不是全局界。

## 5. A→B 权重、缓存、graph 与恢复时间

目的：hot B 与直接加载 B 的参数/输出相等，比较恢复时间。前置：阶段 4 对应路径通过，有存放 B checkpoint 的磁盘空间。

```bash
.envs/nano_rl/bin/python scripts/weight_sync_experiment.py --dry-run --config configs/local/weight-sync.json
.envs/nano_rl/bin/python scripts/weight_sync_experiment.py --allow-gpu \
  --config configs/local/weight-sync.json --output-dir results/manual/sync-01 \
  2>&1 | tee results/manual/sync-01.console.log
```

输入：A 及一个 q_norm 参数确定性 BF16 扰动得到的 B。每 backend 先 3 次 fresh-B 再 3 次 hot-B，含采样共 13 进程。预设 atol/rtol=0.001，在运行前保存并先检查 fresh 重复性，不能看热更新失败后放宽。

输出：A 的 cache hit、首条 B 旧缓存零命中、B 新缓存命中、参数 hash、parameter/KV/graph 地址检查、逐 token 对照、timings.csv、恢复图。source preparation、ready、首条验证生成分别计时，总恢复包含生成预热。

判据：hot/fresh 参数 hash 完全一致；输出符合预先容差；hot version=1；物理 KV 与兼容 graph 保留；缓存正确。preflight 拒绝和 broken 复制异常由 CPU 注入测试验证，不另建 GPU 故障矩阵。参数不一致先查 packed/tied/dtype；参数一致输出不符先查缓存、graph、配置和 fresh 重复性。反馈 B manifest、全部 trace/hash、预设容差和首个失败 token，不能继续使用混合权重。

## 6. GRPO、两种锚点、GSM8K 与独立评测

目的：rollout→reward→advantage→optimizer→sync→下一版本。前置：阶段 3 的实际预算和阶段 5 更新通过；把实测长度、KV、optimizer/microbatch 写入 local 配置，accumulation 必须覆盖全部 G=4。

```bash
.envs/nano_rl/bin/python scripts/train_grpo.py --dry-run --config configs/local/train-simple.json
.envs/nano_rl/bin/python scripts/train_grpo.py --allow-gpu --config configs/local/train-two-updates.json \
  --anchor recomputed --output-dir results/manual/simple-two-01 2>&1 | tee results/manual/simple-two-01.console.log
.envs/nano_rl/bin/python scripts/train_grpo.py --allow-gpu --config configs/local/train-simple.json \
  --anchor rollout --output-dir results/manual/simple-rollout-01 2>&1 | tee results/manual/simple-rollout-01.console.log
.envs/nano_rl/bin/python scripts/train_grpo.py --allow-gpu --config configs/local/train-simple.json \
  --anchor recomputed --output-dir results/manual/simple-recomputed-01 2>&1 | tee results/manual/simple-recomputed-01.console.log
```

KL beta=0，无 reference。默认 FP32 master/BF16 autocast/分块输出头。简单算术可选 add/multiply。奖励认最后完整 answer/boxed/final-#### 标记内合法数字，缺失/解析失败为 0，截断用同规则；train/validation ID 与 prompt 必须不重合，EOS 计入 loss，prompt/padding 排除。

输出：配置/来源/完整任务、rollouts、JSONL/CSV、summary、验证与训练图；启用 save_final_checkpoint 写 BF16 actor-effective checkpoint，不改 FP32 master。判据：至少两个真实 policy commits 与下一轮对应版本；更新时梯度/参数变化；第二次 current 改变但 anchor/advantage 冻结。零信号不 optimizer/sync/递增；全裁剪仍有诊断。completed_no_learning_signal 不算真实学习闭环已验证，应调整难度/预算。

训练轨迹在 reward/update 前保存，失败会保留 failure.json；该驱动的 rollout 时间明确包含轨迹持久化。权重 cast 诊断单独计时，不能混入后端数值差异或 optimizer 的 current/anchor ratio。

GSM8K 只手动准备，保存 dataset revision、seed、固定索引：

```bash
.envs/nano_rl/bin/python scripts/prepare_gsm8k.py --allow-download --train-count 64 --validation-count 32 \
  --output-dir data/gsm8k 2>&1 | tee results/manual/gsm8k-data-01.log
.envs/nano_rl/bin/python scripts/train_grpo.py --allow-gpu --config configs/local/train-gsm8k.json \
  --train-file data/gsm8k/train.jsonl --validation-file data/gsm8k/validation.jsonl \
  --output-dir results/manual/gsm8k-01 2>&1 | tee results/manual/gsm8k-01.console.log
.envs/nano_rl/bin/python scripts/evaluate.py --allow-gpu --model results/manual/gsm8k-01/checkpoint-final \
  --config configs/local/train-gsm8k.json --train-file data/gsm8k/train.jsonl \
  --validation-file data/gsm8k/validation.jsonl --output-dir results/manual/gsm8k-eval-01 \
  2>&1 | tee results/manual/gsm8k-eval-01.console.log
```

评测含泄漏检查、原始轨迹和 T=1 accuracy。一个 seed 只能报单次结果。记录 reward、独立 accuracy、零方差/跳过/截断、ratio 越界/实际 clip、weight cast error，以及 rollout/reward/training/sync/诊断耗时。OOM 回阶段 3；全 0/1 奖励检查解析、模板、thinking、长度/难度，不能改造假 reward。参数无变化检查 clipping/梯度/精度/optimizer，不能硬设 ratio=1。反馈失败批完整轨迹、reward、冻结锚点、实际 steps、版本和验证集。

## 7. 完整性能与采样计算开销

前置：相关正确性/显存关口通过，固定供电模式及其他占用。完整矩阵为 2 源码×2 backend×4 长短/单多请求×3 独立进程，进程内再 3 次稳定重复；嵌套重复不能当成 9 个独立进程。

```bash
.envs/nano_rl/bin/python scripts/benchmark_suite.py --allow-gpu --config configs/local/inference-benchmark.json \
  --output-dir results/manual/performance-01 2>&1 | tee results/manual/performance-01.console.log
.envs/nano_rl/bin/python scripts/sampler_benchmark.py --allow-gpu --power-mode "$LOCKSTEP_POWER_MODE" \
  --output-dir results/manual/sampler-01 2>&1 | tee results/manual/sampler-01.console.log
```

输出：逐进程原始输入/输出 ID、初始化/编译预热/稳定耗时、峰值、CSV、均值/标准差和图。主对照测完整 rollout 数据契约成本；microbenchmark 用合成 BF16 logits 隔离 sampling+logprob 计算，不能冒充端到端改善。原版已有加速不算个人贡献。稳定阶段保留预热 prefix cache；新进程不等于冷磁盘/编译缓存，报告已注明。

判据：固定输出工作量一致、重复完整、无错误、数据可追溯。更快或更慢只由实际数据决定。分析功耗/频率波动、不同 KV 分配，不删除不利重复；反馈全部逐次结果，不只给均值图。

## 8. 人工验证完成清单

- [ ] CUDA/FA2/SM120/graph primitives 通过
- [ ] 固定原版和当前基本路径通过
- [ ] 完整 optimizer 与同驻留预算通过
- [ ] token 对齐、三方数值和全部路径证据齐备
- [ ] A→B 参数/缓存/版本/graph/容差通过
- [ ] 简单任务单/多 update、两种 anchor 通过
- [ ] GSM8K 与独立验证完成，注明 seed/预算
- [ ] 性能重复、原始数据、图和环境资料齐备

以上默认未通过，仅在人工实测证据齐备后勾选。GPU-validation-ready 是代码交付状态，不代表真实 GPU 关口已经通过。
