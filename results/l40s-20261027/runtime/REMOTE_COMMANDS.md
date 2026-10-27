# L40S 正式实验：连接与查看

实验已于2026-10-27 17:34（香港/北京时间）在服务器tmux启动。
源码固定ce9e3542b3f31475ffa595e3f09b80e3f9190ed0，干净目录repo-ce9e354。
新preflight通过后自动执行全套。没有十小时总截止；每任务超时仍保留。

## 1. 在本机PowerShell连接

```powershell
ssh -p 30705 root@YOUR_RENTAL_HOST
```

输入平台密码。下面的bash命令都在SSH登录后的服务器终端执行。

## 2. 最方便的综合进度面板

```bash
P=/root/gpufree-data/lockstep
python3 "$P/ops/monitor_status.py" --watch 10
```

显示GPU利用率/显存/温度/功耗、容器CPU配额与采样使用率、内存占用（含缓存）、
数据盘空间、当前阶段、worker完成数、失败和训练最新iteration/reward/loss。
第一次没有CPU采样百分比，第二次刷新起显示。Ctrl+C只退出这个查看面板。
CPU和内存按cgroup实际配额24核/120GiB读取，避免free/top显示宿主机约1TB造成误解。

## 3. 看tmux运行窗口（只读）

```bash
tmux ls
tmux attach -r -t lockstep-ce9e354
```

离开：先Ctrl+B，松开，再按小写d。SSH断开不会中断服务器tmux。
窗口每30秒显示阶段和worker进度。只读接入不接受中断实验的按键。

## 4. 看总日志与训练指标

```bash
P=/root/gpufree-data/lockstep
tail -n 40 -F "$P/results/campaign-ce9e354-01/console.log"
```

另开SSH窗口看演示训练的逐轮CSV（阶段尚未开始时会等待文件出现）：

```bash
P=/root/gpufree-data/lockstep
tail -n 5 -F "$P/results/full-ce9e354-01/train-demo-simple/metrics.csv"
```

另一档学习率：

```bash
tail -n 5 -F "$P/results/full-ce9e354-01/train-demo-simple-lr5e-6/metrics.csv"
```

iteration从0开始。reward并非要求每轮单调上升；看完整曲线和验证集。
仅看日志时Ctrl+C结束tail，不会停止实验。

## 5. 单项服务器指标

```bash
nvidia-smi -l 2
```

GPU每2秒刷新；Ctrl+C退出后可分别运行：

```bash
top -c
```

```bash
df -h /root/gpufree-data
```

```bash
git -C /root/gpufree-data/lockstep/repo-ce9e354 rev-parse HEAD
git -C /root/gpufree-data/lockstep/repo-ce9e354 status --porcelain --untracked-files=no
```

最后一条应无输出。编译/切换worker期间GPU利用率短时为0正常。
容器内nvidia-smi进程名Not Found可能是PID命名空间差异，不能单独据此判定孤儿进程。

## 6. 输出目录与下载

- 总编排：`/root/gpufree-data/lockstep/results/campaign-ce9e354-01/`
- 新提交预检：`/root/gpufree-data/lockstep/results/preflight-ce9e354-01/`
- 全量：`/root/gpufree-data/lockstep/results/full-ce9e354-01/`
- 每个目录manifest保留状态、失败、耗时；总目录与全量manifest记录固定提交。
- 全部结束后自动打包原始结果/日志/配置，不含权重、数据集内容、环境和checkpoint。

完成后在你本机PowerShell下载：

```powershell
scp -P 30705 root@YOUR_RENTAL_HOST:/root/gpufree-data/lockstep/exports/campaign-ce9e354-01-results.zip .
```

## 7. Codex巡检

每30分钟自动只读检查；重要阶段变化、失败、疑似挂起、空间紧张或完成时通知。
可以结束本轮对话、切换页面；本机要保持开机、联网、Codex应用运行，巡检才能访问本地密钥。
本机睡眠或应用退出时，服务器实验仍继续，自动巡检无法按时执行。
不需要让本轮一直生成，也不需要保持SSH窗口打开。实验不依赖之前的Clash代理隧道。
任务完成不会自动停止平台实例，计费需你在平台管理。
