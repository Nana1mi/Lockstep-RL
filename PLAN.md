# nano-vllm RL Rollout 引擎项目计划书

Oct 26, 2026 · @··

## 1. 项目概述

本项目把 nano-vllm 扩展为可服务 RL 训练的 rollout 引擎。第一版在 2027-02-14 前完成四件事：正确的 token logprob、可复现的训推一致性诊断、安全的在线权重同步、最小 GRPO 训练闭环；其中核心闭环在 2027-01-17 前可演示。

**一句话定位**：基于 nano-vllm 的 RL rollout 引擎，支持 token logprob 与安全权重同步，并提供可复现的训推一致性验证。

**必须完成（第一版）**

1. **Rollout 数据契约**：每个生成 token 对应一个正确的 logprob 和基础 loss mask；每条 response 带终止原因、请求标识与 policy version。
2. **训推一致性诊断**：用同一条固定 token 轨迹，测量 nano-vllm 与 HF 前向的 logprob 差异，并完成规定路径的对照。发现有意义的异常就归因和修复；差异很小时如实报告测得范围、覆盖条件和局限。
3. **安全权重同步**：无活跃请求时免重启更新权重，旧 KV 与 prefix cache 不会被误用；复制失败后引擎不再输出 rollout。
4. **最小训练闭环**：rollout → reward → 组内优势 → GRPO 更新 → 权重同步 → 下一轮 rollout，在单卡上跑通，并验证数据与参数的对应关系。clipping 锚点可在 rollout logprob 与训练端重算的 old logprob 之间切换。

**可选深入（最多选一项）**：可控 staleness 实验、partial rollout、sleep/wake 显存管理、局部 batch invariance。若显存探针证明最小闭环必须释放推理显存，可提前实现受限的 sleep/offload，但不因此同时开展其他扩展。

**与个人方向的关系**：研究生阶段关注 AI infra，当前重点学习 RL infra 与异步 RL。stale rollout、重要性权重截断和动量相关研究需要区分后端数值差异与策略版本滞后；本项目可作为相关研究的工程验证工具，是否进入论文实验由后续研究问题和证据决定。

**范围边界**：第一版以单卡、小型 dense 模型、同步训练闭环为主。TP 验证、多卡性能、大 MoE、完整 batch-invariant 推理、生产级服务和自动故障回滚都不在必做范围内。新增能力属于对 nano-vllm 的扩展，不预先主张行业首创或论文创新。

## 2. 现状与边界

固定上游基线为 [commit bb823b3](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8)（2026-10-26 读取）。实验同时记录上游基线和项目自身的 commit；以后若升级上游，重新建立对应基线。

| 能力                                          | 固定基线的现状                                            | 在本项目中的角色                                  |
| --------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------- |
| Prefix caching、TP、torch.compile、CUDA graph | 已有                                                      | 作为基础，不计入新增贡献；第一版只验证 TP=1       |
| Chunked prefill                               | 已有；一轮 prefill 调度中只有首个被选中的序列允许部分执行 | logprob 和重放位置遵循实际追加 token 的逻辑       |
| `add_request()` / `step()`                    | 已有；`step()` 只返回已结束序列                           | 扩展返回数据与请求元数据，step 接口本身不算新增   |
| 显存不足时抢占、重新 prefill                  | 已有内部机制                                              | 一致性实验的路径之一；不同于对外 pause/resume API |
| Token logprob                                 | 输出接口未提供                                            | 新增                                              |
| 在线权重更新、policy version                  | 无完整公开支持                                            | 新增                                              |
| Sleep/wake                                    | 无                                                        | 按实际需要选做                                    |

现状依据：[LLMEngine](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/llm_engine.py)、[Scheduler](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py)。

**硬件与环境**

- GPU：租用单张 NVIDIA A100-SXM4-80GB（Ampere / sm80），Linux x86_64；所有对照固定同一型号、同一驱动与依赖栈。原 RTX 5080 / SM120 构建步骤不再适用。
- 模型：Qwen3-0.6B 完成全部主线；Qwen3-1.7B 只作可选的规模补充验证，不阻塞第一版。
- 依赖：租卡候选为 Python 3.11、PyTorch 2.8.0+cu128、Transformers 5.5.4 与官方 FA2 2.8.3.post2 预编译 wheel（见 configs/rental.json 的 ABI、URL、SHA256）；禁止现场回退源码编译。CPU 开发另保留 2.11.0+cpu，租卡执行前验证目标 2.8 CPU 兼容性。
- 推理显存：`gpu_memory_utilization=0.9` 是推理端的总预算比例，扣除已有占用和运行开销后余下的分给 KV。训推同卡时按两者合计峰值设预算，不能只按权重大小估算。见 [ModelRunner](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py)。
- 训练显存：阶段 0 用 0.6B、小 microbatch、短序列做全参训练探针，按实测决定是否需要梯度检查点或 offload。词表约 15.2 万，训练端 logits 是主要显存风险之一（见 3.4）。若改用 LoRA，需另行定义 adapter 或合并权重的同步语义，不默认宣称支持。
- 三方一致性实验的对照模型可以顺序加载和评分，不要求三份模型同时驻留 GPU。

## 3. 技术方案

方案包含四个必做模块和一个可选模块。每个模块先定义数据、执行边界和正确性标准，再报告性能。

### 3.1 Token logprob 与 rollout 数据契约

第一版固定 `temperature=1`，不启用 top-k、top-p 或其他改变分布的 logits processor。设模型 logits 为 z：

```latex
p_{\mathrm{model}}=\operatorname{softmax}(z),\qquad q_{\mathrm{rollout}}=\operatorname{softmax}(z/T)
```

T=1 且没有额外处理时，两者的概率定义相同；执行后端之间仍可能有数值差异。

**计算与追加规则**

- 保留未被采样操作破坏的 logits，用 FP32 计算 `log_softmax` 后取被选 token 的值；也可以用数学等价的“被选 logit 减 `logsumexp`”，但先验证等价性和性能。
- 上游 [Sampler](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/sampler.py) 的 `probs.div_(...)` 会原地修改概率张量，不能在采样后对它取 log 当作生成概率。
- 只有 `postprocess` 真正追加 response token 时才追加对应 logprob。未完成的 chunked prefill 产生的临时采样结果不进入 response，也不推进固定 token 的重放位置。
- logprob 按标量序列保存，默认不保存全词表概率；全词表对照只用于少量诊断前缀。

**第一版数据契约**

| 字段                     | 粒度 / 形状   | 含义与约束                                                   |
| ------------------------ | ------------- | ------------------------------------------------------------ |
| `request_id`、`group_id` | 每条 response | 标识请求及所属 prompt 组；不靠完成顺序推断组别               |
| `prompt_ids`             | 长度 Lp       | 实际送入模型的 prompt token，包含约定的 chat template 和特殊 token |
| `response_ids`           | 长度 Lr       | 实际生成的 token；生成了 EOS 就保留                          |
| `response_logprobs`      | 长度 Lr，FP32 | 每个 response token 在生成时所用分布下的 logprob；训练中作为冻结数据 |
| `loss_mask`              | 长度 Lr       | 有效 response token（含实际 EOS）为 1，批处理补齐位置为 0    |
| `finish_reason`          | 每条一个      | 至少区分 `eos` 与 `length`；达到长度上限时不补造 EOS         |
| `policy_version`         | 每条一个      | 成功提交到推理端的权重版本；第一版一条 response 不跨版本     |
| `sampling_config`        | 每批或每条    | temperature、长度限制、EOS 规则、模型/tokenizer 标识、chat template 版本，以及 Qwen3 思考模式开关（第一版固定关闭 thinking） |

对齐断言针对 `response_ids`、`response_logprobs`、`loss_mask` 三者。训练端若追加过滤规则，另建训练 mask，不覆盖原始轨迹。HF 对照评分必须使用与 rollout 完全相同的 chat template 和 thinking 设置，否则 prompt token 不同，比较失去意义。

以后支持 T≠1 时，记录实际行为分布 q 的 logprob，并写清目标策略的温度及 ratio 分子、分母的口径。这一扩展不进入第一版。

### 3.2 训推一致性诊断

#### 3.2.1 同轨迹 HF 对照

先由 nano-vllm 生成轨迹，再把完全相同的 `prompt_ids + response_ids` 交给 HF 做 teacher forcing，只比较相同前缀下相同 response token 的概率。对齐 next-token shift、position IDs、attention mask、chat template、EOS 和 padding；测量时关闭 dropout，记录模型与 tokenizer 的版本。

#### 3.2.2 引擎内部固定 token 重放

增加一个仅供诊断的内部入口，例如 `forced_response_ids`，不对外提供新的 scoring 服务。

1. 保存一份固定的 `prompt_ids + response_ids`。
2. 每个真正生成新 token 的步骤，计算下一个预定 token 的 logprob，并强制把它追加到序列。
3. 只有追加成功才推进重放位置；chunked prefill 的中间步骤不推进。
4. 改变执行路径时重复使用同一条轨迹，保留正常的 KV 分配、调度、prefill 和 decode 路径。随机种子只是辅助复现信息，不能代替固定轨迹。

阶段 1A 的重放覆盖正常 decode 与 chunked prefill；抢占后重新 prefill 的重放放到阶段 1B 实现。诊断模式得到的是指定 token 的评分，不代表一次真实随机采样，与正常 rollout 分开标识。

#### 3.2.3 三方数值对照

比较 nano-vllm BF16、HF BF16 和 HF FP32，三方使用同一组有效权重数值：把同一份 BF16 权重提升到 FP32 只是用更高精度做计算对照，并不能恢复原本不存在的权重精度。

HF FP32 是更高精度的参照，但不能单凭“谁离它更远”就判定谁有 bug。记录并控制 autocast、TF32、attention backend 等设置，必要时做逐层或局部算子对照。参考 [PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)。

#### 3.2.4 路径对照与触发证据

| 路径因素            | 实验设置                                           | 必须记录的触发证据                                 | 阶段 |
| ------------------- | -------------------------------------------------- | -------------------------------------------------- | ---- |
| Prefix cache        | 同版本下预热共享前缀，再与清空逻辑缓存后的重放比较 | 实际复用的 block / token 数                        | 1A   |
| Chunked prefill     | 降低 token 调度预算，使目标 prompt 分块执行        | chunk 次数和每次的 token 范围                      | 1A   |
| CUDA graph 与 eager | 同一轨迹分别运行，另覆盖 batch padding             | 实际 decode batch、选中的 graph batch、是否 replay | 1B   |
| Batch 组合          | 目标请求与不同数量、长度的其他请求一起执行         | 目标请求每步实际所在的 batch                       | 1B   |
| 抢占与重新 prefill  | 用足以触发 KV 压力的请求组合，同时保证单请求能完成 | 目标请求确实被抢占并重新 prefill                   | 1B   |

先做单因素对照，发现需要解释的交互时再加组合实验。在该基线中，`enforce_eager=True` 只绕过手工 CUDA graph 路径，不会关闭所有 `torch.compile`，两者分别记录和控制。

#### 3.2.5 验收

- 完成固定轨迹对照和路径覆盖记录，给出误差分布和数值设置。
- 对超出参照范围的异常做归因；能修复就验证修复，并报告性能代价。
- 误差很小或没有显著路径效应时，如实报告测得范围、测试条件和局限。不把“必须找到 bug”设为交付条件，也不把有限测试的最大值称为全局误差上界。
- 用少量已知错误用例验证工具的检测能力，例如故意把评分位置错开一位，或使用未失效的旧缓存。这些只是测试，不算自然发现的问题。

### 3.3 安全权重同步

第一版只支持同模型结构、同目标 dtype、固定参数布局下的更新，并且只在无活跃请求时执行。

**成功流程**

1. 停止接收新 rollout，等已接收的请求完成，确认调度器空闲、在途 GPU 工作结束。
2. 写入任何参数之前，完整校验键集合、packed shard 映射、形状、dtype 转换和 tied weights。
3. 复用上游 [loader](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/utils/loader.py) 的映射和 `weight_loader`，从内存张量原位复制，保持参数存储地址和共享关系不变。
4. 同步并检查错误后，重置 block manager 的逻辑状态、引用计数和 prefix cache 索引，保证旧 KV 不会再被命中。
5. 全部成功后提交新的 `policy_version`，再恢复生成。版本号对应一次成功的完整权重提交，不在每轮 rollout 后机械递增。

**失败处理（简化）**：预检失败时拒绝更新，原版本继续服务。复制开始后出现任何异常，就把引擎标记为 `broken`：不再返回 rollout，也不提交新版本，必须完整重新加载一个已知 checkpoint 才能恢复。单进程交替执行时 trainer 在同步期间不会修改参数，因此不另做源权重快照，也不实现自动回滚。用一个注入异常的单元测试确认 `broken` 标志生效即可，不为此建测试矩阵。

**缓存与 CUDA graph 边界**

- 热更新只做逻辑缓存失效，保留已分配的 GPU KV 张量，不清零也不重新分配；物理显存释放属于 sleep/wake。
- 只有模型结构、shape、dtype、参数与 KV 的存储地址、捕获路径都保持兼容时，才复用已有 graph；地址或配置变化就重新捕获。
- 在 eager 和 graph replay 两种路径下分别验证“加载 A → 更新为 B”。图的地址约束见 [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)。

**验收**

1. 参数：热更新到 B 与直接加载 B，在相同 dtype、布局和映射下参数值一致，包括 packed 参数和共享权重。
2. 输出：固定轨迹和执行配置，对比热更新 B 与新加载 B，使用事先记录的容差，并以新加载 B 的重复性作参照；不能事后用“BF16 噪声”解释差异。
3. 缓存：用确实产生过命中的 prompt 预热 A，更新到 B 后旧缓存命中数为零，之后可以建立 B 的新缓存。
4. 异常：预检拒绝生效；注入异常后 `broken` 标志生效、不再生成。
5. 性能：热更新与重新初始化对比，计入各自的同步、缓存处理、预热和图捕获，报告恢复到可生成状态的总耗时和峰值显存。

### 3.4 最小训练闭环

#### 3.4.1 算法口径

PyTorch 训练端与 nano-vllm 推理端在单卡上交替执行。第一版是一个明确定义的 GRPO 变体：token 级 clipped surrogate，按有效 token 总数归一化，clipping 锚点可配置。不宣称复现所有 GRPO 变体。

| 项目          | 起步设置                                                     |
| ------------- | ------------------------------------------------------------ |
| 模型和采样    | Qwen3-0.6B；T=1；无 top-k/top-p；关闭 thinking               |
| 组大小        | 每个 prompt 生成 G=4 条，可配置且 G≥2                        |
| 优势          | 组内减均值、除以总体标准差加小常数；零方差组优势为 0         |
| Clip          | token 级 clipped surrogate；初始 ε=0.2                       |
| Clipping 锚点 | `anchor=rollout`（默认）或 `anchor=recomputed`               |
| KL            | β=0，不加载 reference model，并在 README 写明                |
| 每批更新次数  | 默认 1 次；另做 2 次更新的机制测试                           |
| 训练内存配置  | microbatch、累积步数、序列预算、优化器与精度由阶段 0 探针确定后写入配置 |

对同一 prompt 的 G 个奖励 R\_i，用总体标准差（`correction=0`），ε\_A 起步取 1e-6：

```latex
\bar R=\frac{1}{G}\sum_{i=1}^{G}R_i,\quad \sigma_R=\sqrt{\frac{1}{G}\sum_{i=1}^{G}(R_i-\bar R)^2},\quad A_i=\begin{cases}(R_i-\bar R)/(\sigma_R+\varepsilon_A), & \sigma_R>0\\ 0, & \sigma_R=0\end{cases}
```

同一条 response 的有效 token 共用其序列优势。奖励、优势和锚点 logprob 都不带梯度，在该批数据的所有优化步骤中保持冻结。训练端用 teacher forcing 计算带梯度的当前 logprob：

```latex
r_{i,t}(\theta)=\exp\left(\ell^{\theta}_{i,t}-\ell^{\mathrm{anchor}}_{i,t}\right)
```

```latex
\mathcal{L}(\theta)=-\frac{\sum_{i,t} m_{i,t}\,\min\left[r_{i,t}(\theta)A_i,\ \operatorname{clip}\left(r_{i,t}(\theta),1-\epsilon,1+\epsilon\right)A_i\right]}{\sum_{i,t} m_{i,t}}
```

**Clipping 锚点（可配置）**

- `anchor=rollout`：分母是冻结的 rollout logprob。第一次更新时参数还没变，ratio 就已经包含后端数值差异，clip 会作用在这部分差异上。
- `anchor=recomputed`：在本批第一次更新前，用训练端 no-grad 前向重算 old logprob 并冻结，额外开销是一次前向。训练端自洽，但目标不再包含行为分布与训练端之间的差异。
- 两种设置都记录 ratio 分布和 clip 实际生效比例，用同一套代码直接对比。这一对比与重要性权重截断的研究问题直接相关。参考 [veRL Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) 与 [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)。

**mask、截断与归一化**

- 实际生成的 response token（含 EOS）为有效位置，prompt 与 padding 不计入。长度截断时保留已生成 token、记录 `finish_reason=length`、不补造 EOS，奖励按固定任务规则计算。
- 梯度累积时，各 microbatch 的 loss 分子都按整个累计批次的有效 token 总数归一化，避免变长 microbatch 各自求均值后改变权重。
- 不能把 ratio 硬编码为 1，也不能用当前 logprob 的 detach 值代替冻结的锚点。

**训练端 logprob 的显存**：Qwen3-0.6B 的词表约 15.2 万。batch 4、长度 1024 时，完整 FP32 logits 约 2.5GB，并会一直保留到 backward。训练端按序列分块计算“被选 logit 减 `logsumexp`”，不物化全词表 `log_softmax`；阶段 0 的显存探针按这种实现测量。

#### 3.4.2 流程与验证

先用奖励容易验证的短任务调通数据、梯度和权重同步，再上固定的 GSM8K 子集。训练集与独立验证集分开，明确答案提取、缺失答案和截断输出的评分规则。

- 整批优势全为零或没有有效训练 token 时，跳过 optimizer update 并记录原因；这一轮不同步、也不递增版本。
- 对有学习信号且实际更新的批次，检查梯度、参数变化和更新后的推理输出；没有变化就定位是 clipping、梯度路径还是数值问题。
- 2 次更新的机制测试确认：第二次前向用的是更新后的参数，而锚点 logprob 和优势仍然冻结。
- 分别记录 ratio 越界比例与 clip 实际生效比例：正优势且 ratio 超过上界，或负优势且 ratio 低于下界，才算生效。
- 更新前，在同版本、同有效权重下比较 HF 与 rollout logprob，用于诊断后端差异；优化过程中的 current/rollout ratio 还包含正常的参数变化，单独命名记录。训练端用 FP32 主权重、推理端用 BF16 时，权重下转换的差异也单独记录。

该目标只用于第一版同步闭环，不宣称能对任意异步轨迹做精确无偏校正。

### 3.5 可选深入（最多选一项）

| 方向                  | 内容                                                         | 前置与边界                                                   |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 可控 staleness        | rollout 使用落后 k 个训练更新步的权重；测 ratio、越界比例、clip 生效比例和训练行为，并对比两种锚点 | 完成一致性基线和闭环；单卡分时实验不宣称多 GPU 并行加速      |
| Partial rollout       | 外部暂停、取消、恢复；记录 token 区间级版本；跨版本时用新权重对完整上下文重新 prefill | 完成安全更新；旧 token 保留原始 logprob，不追溯改写版本      |
| Sleep/wake            | 首版只支持空闲、关闭 CUDA graph 时释放 KV 与权重并恢复       | 由峰值显存需求决定；报告释放量、恢复正确性和唤醒耗时         |
| 局部 batch invariance | 只针对已定位的一个数值来源做确定性修复                       | 只承诺验证过的模型、硬件、dtype 和 batch 范围；不等于跨后端逐位一致 |

## 4. 分阶段计划与里程碑

第一版完成线定在 2027-02-14，按每周 10–15 小时估算约 16 周、160–240 小时；2027-01-17 阶段 3 结束时，核心闭环已经可以演示。比上一稿推迟两周，换来的是 1A 不再被压缩、1B 不再与主线并行，排期在阶段 0 结束后按实际环境和显存结果再校准。

&#91;embedded content: 项目路线图 · 6 个阶段、5 个验收关口\]

每个菱形是进入下一阶段前必须通过的验收标准，没通过就不往下走。1.7B 模型、额外的组合消融和局部确定性优化都不是完成线的前提。若以 NeurIPS 2027 为研究目标，把 2027 年 5 月仅当作内部排期假设，正式时间以会议公告为准；本工程项目不以论文录用为验收条件。

### 任务清单

**阶段 0 · 租卡后原版冒烟（原排期仅作历史参考）**

- [ ] 在 Linux / WSL2 中建立并记录可运行的依赖版本组合
- [ ] 跑通 `example.py`、`bench.py` 和短/长、单/多请求的基本路径，记录吞吐与显存
- [ ] 可选排障：若真实训练 OOM，再用现有完整 optimizer / 同驻留探针定位；这一版不新增显存任务或 offload 实现
- [ ] 读完 llm\_engine、scheduler、block\_manager、model\_runner、sampler、loader，写一页调用流程笔记

**阶段 1A · 数据与一致性工具（11/09–11/29）**

- [ ] Sampler 返回被选 token 的 FP32 logprob，避开被原地修改的概率张量
- [ ] postprocess 按真实的 token 追加逻辑维护 logprob、mask、终止原因和版本
- [ ] 固定 chat template 与 thinking 开关，写入 `sampling_config`
- [ ] 固定 token 重放覆盖正常 decode 与 chunked prefill 中间步骤
- [ ] HF teacher forcing 对齐 shift、position、mask、模板和 EOS
- [ ] 完成 prefix cache 与 chunked prefill 两个路径的单因素对照
- [ ] 用少量已知错误用例验证测试工具；建立重复性对照和初始容差

**阶段 2 · 安全权重同步（11/30–12/13）**

- [ ] `update_weights()` 预检键集合、packed 映射、形状、dtype 和 tied weights
- [ ] 原位复制，完成后同步并检查错误；异常即进入 `broken`
- [ ] 重置逻辑 KV 与 prefix cache 状态，保留 GPU KV 分配，成功后提交版本
- [ ] 在 eager 和 graph replay 下完成 A→B 对照与缓存测试
- [ ] 测热更新与重新初始化到可生成状态的耗时和峰值显存

**阶段 3 · 最小训练闭环（12/14–01/17）**

- [ ] 把组大小、优势、clip、锚点、归一化、更新次数、KL 和训练内存设置写入配置
- [ ] 实现 GRPO 变体与两种锚点，验证多次更新时锚点和优势保持冻结
- [ ] 检查梯度累积的归一化，以及零优势或空 mask 批次的跳过逻辑
- [ ] 先过简单可验证任务，再跑 GSM8K 子集并评估独立验证集
- [ ] 分开记录更新前的一致性和优化中的 ratio、clip 生效比例；记录版本、截断与零方差组比例
- [ ] 记录生成、奖励、训练、权重同步各自的耗时

**阶段 1B · 深入诊断与交付（01/18–02/14）**

- [ ] 固定 token 重放补上抢占后重新 prefill 的路径
- [ ] 顺序运行 nano BF16、HF BF16、HF FP32 三方对照，记录数值设置
- [ ] 完成 CUDA graph、batch 组合、抢占三个路径的单因素实验并保存触发证据
- [ ] 对异常归因并验证修复；没有显著异常就如实报告测量范围
- [ ] 补齐复现命令、配置、原始指标、环境信息、已知局限和失败案例
- [ ] 完成 README 和只含实测事实的简历条目

## 5. 实验与评测设计

每个数字都要回答一个明确的问题。新增推理功能的开销以原版 nano-vllm 为主基线，热更新对比重新初始化。HF `generate` 只作补充基线，且需采用合理 batching、相同生成预算和可比的输出功能，不能把上游已有的加速算作本项目的改进。

| 要回答的问题             | 对照设置                                              | 报告指标                                                     |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ |
| 返回 logprob 的成本      | 固定基线 vs 只加 logprob 的版本，生成工作量可比       | 输出 tok/s、总耗时、峰值显存，冷启动与预热后分开报告         |
| 训推差异有多大           | 同有效权重、同轨迹：nano BF16 / HF BF16 / HF FP32     | \|Δlogp\| 的均值、P95、P99、最大值；exp(Δlogp) 分布；指定阈值下的越界比例 |
| 执行路径的影响           | 固定 token 重放，单因素切换                           | 同上，按路径和长度分组，附实际触发计数                       |
| 权重同步是否正确         | 热更新 B vs 新加载 B，另做新加载 B 的重复性对照       | 参数一致；固定路径输出满足事先规定的容差；缓存和版本状态正确 |
| 失败是否被阻断           | 预检拒绝；注入复制异常                                | 预检后原版本照常服务；`broken` 后不再生成                    |
| 权重同步省了多少时间     | 热更新 vs 重新初始化到可生成状态                      | 总耗时与分项耗时（ms）、峰值显存；写明是否含图捕获           |
| 锚点选择的影响           | `anchor=rollout` vs `anchor=recomputed`，其余配置相同 | 首次更新前 ratio 分布、clip 生效比例、训练曲线               |
| RL 循环效率              | 相同任务与训练预算下的每轮耗时分解                    | 生成、奖励、训练、同步各自耗时，每轮端到端耗时与有效样本量   |
| 训练是否产生可解释的变化 | 简单任务机制验证、训练曲线、独立验证集                | reward、验证准确率、零方差组与跳过更新比例、截断比例；注明训练预算 |

**误差与 ratio 的命名**：同版本对照中定义 Δlogp = HF logprob − rollout logprob，exp(Δlogp) 只用于诊断同有效权重下的概率比。优化过程中的 current/anchor ratio 单独命名和记录。ratio 越界比例不等于 loss 中 clip 实际生效的比例，也不等于 TIS 的截断比例。

**测量规范**

- 固定模型与 tokenizer 版本、chat template 与 thinking 设置、上游与本项目的 commit、prompt 集、轨迹、长度预算和采样配置；记录驱动、CUDA、PyTorch、Triton、flash-attn 与 attention backend。
- 正确性对照使用固定 token 重放；性能实验使用相同的请求长度和输出预算。为控制变量而强制固定输出长度的测试，与真实任务结果分开报告。
- 区分初始化、编译与图捕获、预热和稳定运行；计时边界等待 GPU 工作完成。笔记本固定供电与性能模式，并记录重复运行的波动。
- 性能每组至少重复 3 次，报告均值与波动。训练效果只有一个随机种子时，明确标为单次结果。
- 输出容差在比较热更新结果之前确定，不在失败后放宽。
- 有限样本上的最大误差只是该测试集的最大值，不是理论上界。
- 单个被采样 token 的 logprob 差不等于完整词表 KL；需要 KL 时，在少量固定前缀上计算全词表分布并写明方向。
- 数学示例（不是实测）：Δlogp=0.05 时 exp(Δlogp)≈1.051；是否足以影响训练要靠实验回答。

**原始材料**：保存配置、环境清单、轨迹 ID、逐次与分组指标和作图脚本，保证每张图都能追溯到原始记录。

## 6. 风险与应对

最大的管理风险是时间被研究和实习挤占。第一版完成线是核心闭环、规定诊断与复现材料齐备；可选扩展随时可以放弃。

| 风险                     | 早期信号                                            | 应对                                                         |
| ------------------------ | --------------------------------------------------- | ------------------------------------------------------------ |
| 时间被其他工作挤占       | 连续两周没有可提交的进展                            | 优先保 1A、2、3；砍掉 1.7B 和可选扩展，不放宽核心正确性要求  |
| Blackwell 依赖组合不可用 | 原版 prefill、decode 或 graph 测试失败              | 记录报错，排查时间累计不超过阶段 0 的一半；仍失败就重新评估环境和排期 |
| SDPA 兜底扩大范围        | 需要另行适配 varlen、paged KV 或 KV 写入            | 当作独立的 eager 正确性后端：显式收集 KV、处理序列边界和因果位置，单独验证和计时。上游 [Attention](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py) 同时用 varlen/paged FlashAttention 与 Triton KV 写入 kernel，替换不是一行修复 |
| 训练端 logits 撑爆显存   | 小 batch 下 backward 即 OOM                         | 按序列分块计算被选 token 的 logprob；再依次调 microbatch、长度和梯度检查点 |
| 训推合计显存不足         | 真实 optimizer step OOM，或 KV 预算不够完成请求     | 阶段 0 提前发现；调 microbatch、长度和 KV 预算；必要时提前实现受限 offload/sleep |
| 复制中途失败导致混合权重 | 注入异常测试未拦住生成                              | 预检在写入前完成；复制异常即 `broken`，完整重载已知 checkpoint 后才恢复 |
| Qwen3 思考模式使长度失控 | response 普遍很长、大量 `length` 截断、答案解析失败 | 固定关闭 thinking 并写入 `sampling_config`；HF 评分使用同一模板 |
| 没有发现显著数值问题     | 各误差分位数都很小，路径切换影响有限                | 报告测得范围与覆盖条件，保留检测工具和正确性结果，不强行制造优化主张 |
| GRPO 能跑但缺少学习信号  | 大量零方差组、频繁跳过更新、验证指标不变            | 先用更简单的任务确认梯度路径，再调组大小和任务难度；第一版保持 T=1 |
| reward 上升但解释不清    | 训练/验证划分或答案提取不稳定                       | 固定数据划分、评分规则和预算；先排查数据泄漏、解析错误、长度变化 |
| AI 辅助的代码讲不清      | 说不出本阶段新增的状态和不变量                      | 每阶段写一页设计说明：改了什么、为什么、怎么验证；逐项能解释自己提交的核心代码 |

## 7. 成果呈现

交付物包括可公开复现的代码仓库、README、实验配置和结果，一条基于实测的简历条目，以及面试准备材料。

### README 结构

1. 动机：RL 训练为什么需要 rollout 概率、权重版本和同步边界
2. 固定上游基线与新增差异：逐项链接到自己的 commit，不认领上游已有机制
3. 数据契约：token、概率、mask、终止原因、版本、chat template 与 thinking 设置
4. 固定 token 重放与一致性诊断：数值设置、路径触发证据、结果与局限
5. 权重同步：成功流程、失败处理、逻辑缓存失效、CUDA graph 约束与验证
6. 最小 GRPO 变体：冻结锚点及其两种选择、优势、mask、归一化、KL、跳过更新与数据预算
7. Benchmark：硬件、版本、计时边界、复现命令、原始结果与图表
8. 局限：已验证范围、单卡分时执行、未验证的能力和可选扩展状态

适合上游的 logprob 或通用修复可以单独整理成 PR。重点展示改动价值、测试和技术讨论，不把是否合并当作效果保证。

### 简历条目模板

以下均待实测；没完成或没有证据的部分直接删掉，不保留占位符，不预设改善方向。

**通用版（不需要先发现或修复显著数值问题）**

> 基于 nano-vllm 实现面向 RL 训练的 rollout 引擎，新增 token logprob 与请求级 policy version；实现免重启权重同步、旧 KV 逻辑失效与复制失败隔离；建立固定轨迹的训推一致性验证，在【模型/精度/路径范围】下报告概率差异；在单卡 GRPO 闭环中验证权重与样本的对应关系，并对比两种 clipping 锚点下的 ratio 与 clip 行为。logprob 吞吐开销【实测值】，权重同步耗时【实测值】，较重新初始化【实测变化】。

**优化版（只在定位并验证改善后使用）**

> 在上述引擎基础上，定位【具体来源】导致的 logprob 差异，通过【具体改动】使【规定条件】下 P99 绝对 logprob 差异从【A】降至【B】，吞吐变化【C】；提供路径消融、固定轨迹对照与复现脚本。

简历只写已完成的行为、证据和贡献边界：不写未验证的 TP 或异步并行收益，不用“深入理解”代替交付，不把基线本身相对 HF 的加速算作个人改进。

### 面试准备清单

- [ ] prefill/decode 如何调度，当前 chunked prefill 有什么限制
- [ ] block 如何分配回收，prefix cache 如何哈希，权重更新后为什么要失效
- [ ] 抢占后的重算路径为什么可能影响数值，怎样用固定轨迹检验
- [ ] logprob 属于哪个分布，采样器的原地操作如何影响实现
- [ ] 固定 token 重放为什么比固定随机种子更适合路径对照
- [ ] 序列级 policy version 的前提是什么，跨版本 partial rollout 还要补什么
- [ ] 权重更新成功和失败时分别允许哪些动作，`broken` 状态如何恢复
- [ ] 逻辑缓存失效和物理显存释放有什么区别，CUDA graph 复用有哪些条件
- [ ] 两种 clipping 锚点有什么区别，各自把哪部分差异放进了目标
- [ ] 零优势、梯度累积、多次更新分别怎么处理
- [ ] 训练端 logprob 为什么要分块计算
- [ ] 同版本数值差异、正常策略更新和 staleness 如何分别测量
- [ ] batch invariance 与跨后端一致性有什么区别
- [ ] 每个数字的基线、范围、计时边界和复现命令是什么

### 参考资料

- [nano-vllm 固定基线](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)
- [PyTorch Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
- [veRL Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [vLLM Async Reinforcement Learning](https://docs.vllm.ai/en/latest/training/async_rl/)
- [vLLM Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)
- [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)

资料只用于定义与设计参考，不替代本项目的实测；实施时记录实际采用的依赖版本和文档日期。

### 修订记录

- 第二版（2026-10-26）：吸收 GPT 修订稿的固定 token 重放、路径触发证据、训练显存探针、明确的 GRPO 定义、1A/1B 拆分，以及“不必发现数值问题”的验收口径。在此基础上把 1A 恢复为 3 周、抢占路径重放移到 1B、1B 改为串行，完成线调整为 2027-02-14；失败处理简化为预检加 `broken` 标志；clipping 锚点改为可配置；补充训练端 logits 分块和 Qwen3 思考模式开关。

## 2026-10-27 优先级修订（覆盖上文旧排期的执行顺序）

先在 CPU 上完成并验收 1 → 2 → 3 → 4 → 6 → 5 → 7/8，再人工租卡。
本轮不新增显存/offload 任务，GPU 运行顺序为环境、基线、诊断、同步、GRPO、性能。

| 顺序 | 实现与 CPU 验收 | 租卡后才可成立的结论 |
|---|---|---|
| P0.1 | enable_prefix_caching 开关；6 blocks、两个512 prompt、各300 response：cached有复用、recompute无复用，scores逐位对齐 | 两种真实抢占路径分别报告，不混称全量重算 |
| P0.2 | 32条真实GSM8K轨迹、128 response；fewshot真实解答填充，traj-k；真实组批、逐条eager、repeat噪声底线；轨迹级统计与dry-run全覆盖 | 数值分布、轨迹最大值与超阈值轨迹数 |
| P0.3 | live FP32 head、不缓存影子权重；训练评分对应选项；CPU数值/梯度/热更新正确 | FP32 head vs HF FP32误差与吞吐代价 |
| P1.4 | 单张量一次归约、标量统一读回、同源别名快路、三段计时、GPU-FP32来源、测试专用跳过失效阴性对照 | 各来源分组时延；真实阴性对照必须复用且不符合fresh-B容差 |
| P1.6 | strict保持旧行为、lenient兼容货币/句点/{,}/text单位；每样本parse_status与失败率 | 格式失败率、答错率与奖励分别观察 |
| P1.5 | smoke/demo分开；512/128题、16prompt×G8、120轮、lr1e-6/5e-6、每10轮评测；真实actor变化率与固定探针变化 | BF16有效更新可见性与独立学习曲线，版本号不能替代证据 |
| P2.7/8 | generation_config多EOS；response-only head，值/梯度与全量计算一致 | 真实模型EOS行为及性能 |
| 租卡准备 | 固定A100-SXM4-80GB；官方预编译wheel/ABI/hash；持久盘预取与总脚本；CPU检查顺序/配置/保护开关 | GPU_PENDING；约3–11.1小时为未实测规划值，原始输出全存持久盘 |

源码和 CPU 验收通过状态以 IMPLEMENTATION_STATUS.md、results/cpu-tests.txt 及指纹清单为准。
旧实测类复选框继续保持未勾选；没有 GPU 数据时不声称完成数值、性能或学习效果验收。

## 同日开机前复核补充（优先于上一版实现细节）

1. 记录每条固定工作量轨迹的首 EOS；主统计仅用 EOS 前（含 EOS），尾段单独 JSON/CSV。
   不删除尾段路径证据，也不让它影响主分位数、轨迹阈值或 A/B 容差判断。
2. FP32 head 同时保留 live 转换与 cached 推理变体。缓存作为非持久 buffer，在初始完整加载和
   成功 commit 后刷新；tied embedding-only 更新也必须覆盖，刷新失败保持 broken。诊断14进程、
   性能96workers，两种成本/额外显存/初始化与同步维护分开报告，禁止把 live 转换成本冒充纯 FP32 计算成本。
3. 总脚本仅 environment/attention 是硬关口，普通子任务失败保存后继续，汇总返回非零；
   --stages 允许新 run-id 选阶段重试并自动复跑环境关口。硬件身份变化/用户中断仍停止。
4. A100 演示改 microbatch16、accumulation8、关闭 checkpointing，目标与 full batch 的
   全局 token 归一做 CPU 对照；不宣称尚未实测的提速。用 uv-managed Python 放入持久盘，
   固定挂载路径/平台，移动时重建 venv。同步对外用三阶段和 sync_seconds，统计耗时另列。
5. eager/repeat 若为0，只说明可重复且无区分度；HF BF16/FP32 是精度参照而非错误判据。
   few-shot 例题池按 seed 打乱并逐轨迹轮换，保留次序 provenance。

本轮仍只做 CPU 验证，所有真实 GPU 数值、性能、学习与运行时间保持 GPU_PENDING。

## 同日任务超时补充（租卡前最后一轮）

- 每个任务按未实测预计墙钟时间的3倍限时，配置与dry-run公开预算；外层截止时间约束嵌套worker。
- 所有长期子任务独立session，超时/正常退出/中断均清理整棵子孙进程树；TERM后KILL，Linux
  subreaper接管孤儿。超时失败保存后继续独立任务，依赖失败明确跳过；清理失败属于基础设施停止。
- NCCL当前源码支持LOCKSTEP_NCCL_PORT和有限初始化等待；每个任务启动前检查端口，默认每worker
  选空闲端口。固定上游仍用2333并单独检查，不修改对照源码，不杀无关占用者。
- 首次人工开机仅--stages environment baseline，约半小时为规划值；审阅后以新run-id跑全量。
  该执行安排不等于已租卡；所有真实GPU验收仍GPU_PENDING。
