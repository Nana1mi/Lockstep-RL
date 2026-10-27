# 后续实验计划与执行记录

以下为执行前的配置与验收计划。操作者随后明确授权上卡，九项实验已完成；
因分配到新 GPU UUID，实际增加一次 seed 0 复现，三 seed 全在同一张新卡上运行。
见 [完成记录](../../results/followup-l40s-20261027/README.md)。原计划的 GPU_PENDING 描述仅为历史状态。
配置生成器为 `analysis/prepare_followups.py`，以原 lr=1e-6 演示配置为基底。

## 第一组：单次更新的 seed 复现

新增 seed 1、2，各 120 轮、recomputed anchor、update_steps=1。
现有 seed 0 只有在源码、模型 revision/文件哈希、数据生成、GPU 型号/区域、驱动和依赖均匹配时
才能复用。若换驱动或训练实现，补跑 seed 0；旧结果单独展示，不能悄悄合成同批复现。
固定 512/128 train/validation 题目，训练随机性由 seed 控制，确保不是换题造成的变化。
报告三条完整曲线、各 seed 起终点、均值和 seed 标准差；n=3 不作稳定性或泛化的强结论。

## 第二组：匹配的 anchor 比较

seed 0、1、2 × rollout/recomputed，共六个配置。每组仅 anchor 不同，均 update_steps=2；
lr=1e-6、120 轮、16 prompt × G=8、microbatch16/accumulation8 与现有演示保持一致。
每个 run 重新从相同初始权重开始，不继承第一组训练后的 checkpoint。相同 seed 固定题目与采样设置，
但策略分叉后 rollout 不必相同；按 seed 配对比较最终指标和学习曲线。

先审阅首个短段的逐更新 ratio、优势、有效 clipping 与零方差 group 比例。
`update_steps=2` 给 recomputed anchor 产生非一 ratio 的机会，**不保证裁剪激活**。
若全程 `clip_active_fraction=0`，如实标为未激活，不能把结果当作 clipping 效果实验。
需要更大步长或更多更新来构造正对照时，另建明确标为 stress 的配置，不能事后调参后混入
原比较；同一对照的两种 anchor 必须采用相同设置。

每个 step 保留 reward、parse_status/失败率、ratio min/mean/max、越界率、有效裁剪率、
old-policy KL 诊断量、loss、response 长度、梯度与实际更新计数、同步探针和 BF16 改变比例。
beta 仍为 0，无 reference KL。保存所有验证题目 ID/结果、实际配置、commit 和来源清单。
跳过轮与 optimizer 更新轮分别计数，避免把“120 轮”误称为“120 次有效梯度更新”。

## 配置与无 GPU 检查

Linux/WSL，在仓库根目录执行。`--model` 此处可为尚不存在的未来持久盘路径：dry-run 不加载模型。
输出目录必须是新的，防止覆盖审阅过的计划。

```bash
CUDA_VISIBLE_DEVICES= python analysis/prepare_followups.py --model /persistent/lockstep/models/Qwen3-0.6B --output .build/followups
for config in .build/followups/*.json; do
  CUDA_VISIBLE_DEVICES= python scripts/train_grpo.py --config "$config" --dry-run
done
```

共八个配置；CPU unit tests 覆盖配对参数一致、seed、总 batch 和 update 次数。
GPU 启动仍需人工授权、同栈 preflight 和新的 run-id；保留逐任务超时及进程组清理。
不要据“每轮约两秒”推断租卡总费用：加载、编译、评估、同步和 preflight 都计入墙钟时间，
两次更新改变成本，旧吞吐也不保证新实验耗时。按首段实测重新估时。

## 难度扩展

更难的三位数乘两位数值得另做，但需要单独的数据生成器、train/validation 去重和 CPU 验收。
本次未修改任务生成器，也未宣称已有更难任务结果。不要在同一 anchor 比较中同时换任务难度，
以免无法解释差异。
