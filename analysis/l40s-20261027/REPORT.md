# L40S 数据复算与结论边界

2026-10-27；仅 CPU 分析，未重新启动 GPU。实验源码
`ce9e3542b3f31475ffa595e3f09b80e3f9190ed0`，上游
`bb823b3e06983d71485a8e1f23715ebd87d98ef8`。一张 L40S、一个 Qwen3-0.6B、固定依赖。
本报告是事后描述性分析，不是预注册的假设检验或跨模型结论。

## 数据和复算方法

输入为仓库 `results/l40s-20261027/full/` 的原始 CSV、训练初始汇总，以及公开
[Release ZIP](https://github.com/Nana1mi/Lockstep-RL/releases/tag/l40s-20261027)
里的逐题验证结果、trace、阴性对照和逐更新指标。公开 ZIP SHA256：
`8a45aae2863881e622c409ca63c1c3da74e3b25ec97641846b009f0587a82fd1`。
分析脚本不修改这些输入。输入哈希、17 个比较的逐轨迹最大值和全部复算结果见
[recomputed.json](recomputed.json)。运行命令见仓库 README；`--no-plots` 只需 Python 标准库。

主统计严格截至第一个 EOS（含）；32 条轨迹原始定长 4,096 token 中，3,361 个进入主统计，
735 个进入独立 post-EOS 文件。汇总按 token 加权；P99 为排序后的线性插值，
不是将每条轨迹的 P99 再平均。轨迹数、超阈值轨迹数和各轨迹最大值同时保留。
同一条轨迹内 token 有相关性，不能把 3,361 当作独立样本数计算置信区间。

## 训推一致性

| 比较 | 平均绝对 logprob 误差 | P99 | 误差 > 0.1 | 诊断 ratio 越出 [0.8,1.2] |
| --- | ---: | ---: | ---: | ---: |
| nano BF16 vs HF FP32 | 0.009935 | 0.121591 | 1.52% | 0.208% |
| nano FP32 head vs HF FP32 | 0.007849 | 0.097167 | 0.89% | 0.149% |
| HF BF16 vs HF FP32 | 0.011561 | 0.135622 | 2.23% | 0.208% |
| nano BF16 vs HF BF16 | 0.013880 | 0.168388 | 3.54% | 0.565% |

FP32 head 将平均误差降低 21.0%，P99 降低 20.1%。在这批轨迹上，nano BF16 的平均误差
小于 HF BF16 对 HF FP32 的误差；这不说明 nano 在任意任务上都更准确。
HF FP32 是本实验的数值参照，不是数学精确值。

![诊断及同步](diagnostics-sync.png)

eager 的两个独立进程重放、batch-1 graph 对 eager 的主统计误差均为零。
零 repeat 底线没有误差尺度上的区分度；HF BF16-vs-FP32 是更有用的精度参照。
prefix cache、chunked prefill、cached/recompute 两种抢占和组批路径均应分别解释。

batch 用例和 graph_padding 用例的 3,361 个候选分数逐位相同。trace 显示前者实际 batch 4、
后者实际 batch 3、backbone graph bucket 4。但 `model_runner.py` 在 graph replay 后
对输出切片 `[:bs]` 再计算 head：后者的 head 实际是 3 行，不是所有矩阵都补到 4 行。
因此能报告“这两个组批配置结果相同，且不同于 batch 1”，还不能归因为某个具体 GEMM，
也不能排除所有其他请求内容的影响。需要额外的形状/伴随请求受控实验才能建立更强因果结论。

按 **nano rollout_logprob** 分箱：小于 −1 的 355 个 token 平均误差 0.05178；
大于等于 −0.1 的 2,305 个 token 平均误差 0.000652，相差 79.5 倍。
分箱边界靠哪一端的分数会影响数字，故明确记录分箱依据。这里只说明低概率段误差更大。
上表的 ratio 越界率仅描述后端分数差异，不含优势符号、真实更新或梯度，不能当作有效 clipping 比例。

## 权重同步：必须匹配计时边界

下表为三个独立重复的均值，源为 GPU FP32，目标为普通 BF16 actor。
此同步矩阵没有开启 cached FP32 head，不能用它代表缓存 FP32 权重刷新耗时。

| 秒 | eager | graph |
| --- | ---: | ---: |
| 预检 | 0.038155 | 0.037786 |
| commit | 0.009628 | 0.009628 |
| invalidate | 0.000118 | 0.000124 |
| 三阶段同步合计 | 0.047901 | 0.047537 |
| 热更新至 ready | 0.067067 | 0.066547 |
| 重新加载至 ready | 44.905351 | 48.345878 |
| 匹配 ready 边界的加速比 | 669.6× | 726.5× |
| 热更新至完成验证输出 | 0.529035 | 0.169194 |
| 重新加载至完成验证输出 | 46.558654 | 49.082481 |
| 匹配验证输出边界的加速比 | 88.0× | 290.1× |

CPU BF16 源三阶段同步为 0.895 秒 / 0.875 秒。GPU FP32 的路径更接近真实训练。
不能将 48 ms 的 API 三阶段与 45–48 秒初始化边界相除，称为端到端约 1,000 倍。
权重变化统计单独计时，不计入 `sync_seconds`。

热更新与直接加载 B 的参数哈希一致。六个故意跳过 cache 失效的阴性对照均在
`vs-fresh.json` 显式记录检测成功，最大 logprob 差异 0.33117；不是仅从 exit 0 推断。
这些结果验证测试能发现当前构造的 stale-KV 错误，不覆盖所有潜在缓存错误。

## 推理吞吐

96 个 worker：四实现 × 两 backend × 四 workload × 三独立进程；每个进程内三次 steady
重复。独立重复数是 3，不能把嵌套的九个 steady 样本当作九个独立进程。
四 workload 为输入/输出长度 32/32、1024/64，分别单请求和四请求。

普通 BF16 扩展相对固定上游变化 −2.96% 至 +0.74%。不能统一称为测量噪声：
graph long-multi 下降 1.65%，上游/当前的进程均值变异系数仅 0.053% / 0.046%。
不过实现按块运行、只有三进程，并未随机交错或做等效性检验；不作“零开销”的声明。

FP32 live head 相对当前 BF16：eager 吞吐下降 11.4–12.9%，graph 下降 34.1–37.6%；
cached head 分别下降 3.1–9.6%、12.2–14.9%。graph 下缓存消除了约 60–67% 的吞吐损失。
这量化的是完整实现的吞吐代价：两者还可能改变持久显存/KV 容量，不能严格解释为孤立的
格式转换算子和 GEMM 算子耗时分解。同期同步矩阵未测 cached-head 刷新。
当前实现的 graph/eager 加速范围 3.90–4.53×，short-single 为 4.41×。

![训练及性能](training-performance.png)

性能图按同 workload 的上游均值归一化；误差线是三个进程均值的样本标准差，
不是置信区间，也未传播上游分母的不确定性。训练图只有 seed 0，不能画 seed 误差带。

## GRPO：学习变化与 clipping 分开解释

每轮 16 prompt × G=8，120 轮，验证 128 题，训练 512 题，lr 1e-6/5e-6。
beta=0，无 reference model；不能将 old-policy 诊断量表述为 reference KL。
两次演示均为 recomputed anchor、每批一次更新：ratio=1，有效 clipping=0。
它们仍可产生策略梯度；归一化优势导致标量 loss 接近零不意味着梯度为零。

| seed 0 | lr=1e-6 | lr=5e-6 |
| --- | ---: | ---: |
| 初始准确率 | 109/128 (85.16%) | 109/128 (85.16%) |
| 最终准确率 | 122/128 (95.31%) | 111/128 (86.72%) |
| 初错末对 / 初对末错 | 18 / 5 | 15 / 13 |
| 有效更新轮 / 跳过轮 | 81 / 39 | 90 / 30 |
| 最终准确率差 | +10.16 个百分点 | +1.56 个百分点 |

lr=5e-6 最低验证准确率 46.875% 在 iteration 19，即第 20 轮。lr=1e-6 的最终结果按
相同题目 ID 配对，准确率差的配对标准误约 3.65 个百分点，差/标准误为 2.78；
精确双侧 McNemar p=0.01062（18/5 discordant pairs）。计算采用条件二项分布，
与 [statsmodels 的 exact McNemar 定义](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
一致，不需导入 statsmodels。任务是结构化合成乘法且训练仅一个 seed；题目间独立性并不充分，
这只是给定验证样本上的描述，不是跨 seed 的学习稳定性证据。高学习率对应 p=0.85055。
初始准确率较高也限制了任务难度与可推广性。

需要区分两个字段：`ratio_outside_fraction` 与按优势方向判定的 `clip_active_fraction`。

| 冒烟用例 / 更新 | ratio 越界 | 有效 clipping |
| --- | ---: | ---: |
| rollout anchor，第一次更新，ratio_max=1.42764 | 2.22% | 0 |
| recomputed anchor，第一次更新 | 0 | 0 |
| two-updates，第二次更新 | 2.22% | 2.22% |

因此现有数据已含一个真正激活 clipping 的小型正对照，但只有一批 1 prompt × G=4 的
有效样本，不是有统计力度的 anchor 对照。`update_steps=2` 也不保证后续实验必然越界/激活。
不能在简历里称已完成系统性的 clipping anchor 效果比较。

## 尚缺什么

多 seed 稳定性、匹配 anchor 的学习/裁剪对照、较难任务，以及 cached-head 同步维护成本
仍未验证。GSM8K 只做闭环冒烟，末批解析失败 50% 不能推广为整体解析率或学习效果。
后续设计见 [FOLLOWUP.md](FOLLOWUP.md)；已准备配置和 CPU dry-run，不曾重新租卡或开跑。
