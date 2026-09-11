# QAM → online / DAWN：单 seed 实验

任务：`cube-double-play-singletask-task1-v0`。本目录保存实验实现、固定协议和结果整理代码；训练在已授权的远程机器上执行。

三组已于 2026-09-06 完成，数据与图表校验通过。最终成功率（100 episodes）：native 99%、warm 91%、random 66%。[完整结果](results/summary.md) · [学习曲线](results/learning_curves.png) · [评估表](results/evaluation_table.csv)。

新增两组 QAM replay 消融也已完成：warm + QAM replay 79%、random + QAM replay 83%。[五组完整报告](results_five/summary.md) · [五组学习曲线](results_five/learning_curves_five.png) · [补充实验设计](REPLAY_ABLATION.md)。两组均保持原 DAWN 的 20k warmup、UTD 0.25、batch 1024 及其余超参数。

Policy Decorator 继承 critic + QAM replay 的单 seed 试跑也已完成：最终随机残差成功率 **79%**，均值残差 **90%**（均为 100 episodes）。[六组结果与 PD 超参数报告](results_policy_decorator/summary.md) · [PD 学习曲线](results_policy_decorator/learning_curves_policy_decorator.png) · [六组 CSV](results_policy_decorator/comparison_six.csv) · [训练前固定协议](POLICY_DECORATOR_PROTOCOL.md)。实际完成 50k primitive steps / 10,500 次更新；60 份原始评估和全部训练源文件校验通过。

- 完整设计：[PROTOCOL.md](PROTOCOL.md)
- 数据来源及 SHA-256：[DATA_PROVENANCE.md](DATA_PROVENANCE.md)
- 实验入口：[run.py](run.py)
- DAWN residual SAC：[dawn_agent.py](dawn_agent.py)
- 三组调度及短流程检查：[launch.py](launch.py)
- 原始评估校验和图表：[summarize.py](summarize.py)

## 三组

1. `native`：官方 cube-double `edit_scale=0` 的 QAM-edit，实际为 QAM 原生 online continuation。
2. `warm`：冻结相同 QAM offline policy，DAWN residual online；继承 offline current/target critic。
3. `random`：同上，但随机初始化 critic，target 从新 critic 复制。

共享一个 seed=0、500k 次 offline 更新的 checkpoint。每组 50k 个原始环境步。DAWN 前 20k 步 base-only warmup 包含在总预算中。DAWN 两组用相同初始 residual、温度与新 optimizer；采用 QAM 的 10-member critic backbone，以支持直接迁移权重。

## 远程位置和使用

远程工作目录为 `/root/autodl-tmp/qam_dawn_o2o`。当前 Task2 机器为 `root@connect.bjb3.seetacloud.com:16628`；Task1 延长训练使用 `root@connect.bjb2.seetacloud.com:46219`（最早机器端口为 `12749`）。凭据不保存在本目录。

在远程工作目录中：

```bash
# 查看最近调度事件、当前阶段的进度与日志。
cat runs/suite_status.json
cat runs/offline/progress.json

# 调度器有进程锁；运行中的 suite 不应重复启动。
# 仅当原调度器已停止、且需要从现有 checkpoint 恢复时执行：
venv/bin/python launch.py

# 三组全部完成后，从原始 episode 记录校验并生成报告：
venv/bin/python summarize.py --runs runs --out results

# 五组结果合并，保留原始三组报告：
venv/bin/python summarize_five.py --runs runs --out results_five

# 新增 Policy Decorator 结果校验、报告及轻量证据包：
venv/bin/python export_policy_decorator.py
```

`runs/offline/` 为共享 offline 训练；`runs/native/`、`runs/warm/`、`runs/random/` 为正式 online 结果。`runs/smoke_*` 仅用于短流程验证，不计入正式结果。

新增两组保存在 `runs/warm_qam_replay/` 与 `runs/random_qam_replay/`。其调度入口为 `launch_qam_replay.py`，状态为 `runs/replay_suite_status.json`，源码快照为 `runs/replay_source_manifest.json`。共享的 frozen base proposal 缓存位于 `data/qam_base_cache_seed0/`，哈希记录包含在五组结果中。

Policy Decorator 正式结果和权重保存在 `runs/policy_decorator_qam_replay/`，调度入口为 `launch_policy_decorator.py`，状态为 `runs/policy_decorator_suite_status.json`。该组使用 8k learning starts、30k 渐进探索长度和初始自动熵系数 1；其余 QAM 适配细节见协议。这是完整配置比较，不能解释为只改变 progressive exploration 的消融。官方代码 commit 和默认值核查保存在 `POLICY_DECORATOR_SOURCE_AUDIT.json`。

每个阶段保存配置、指标、逐 episode 评估记录、断点及完成标记。`latest.pkl` 包含恢复所需训练状态；`final.pkl` 保留阶段结束的 agent 状态。`DONE.json` 为成功结束标记，`FAILED.json` 表示需要检查异常。`runs/source_manifest.json` 和 `runs/pip_freeze.txt` 分别记录启动时源码和 Python 依赖。

## 结果解释

学习曲线采用七个预先指定的 online 评估点，每点 50 个固定 reset seed 的 episode；最终另评估 100 episodes。所有组的起点评估进行精确一致性检查。两组 DAWN 的 warmup 动作、回报、discount 和 base proposals 要求逐位一致；观测的原始哈希检查发现约 4.1e-13 的差异，调查后采用绝对容差 1e-10、相对容差 0，并在结果中保留具体差异和原始哈希。

一个训练 seed 只能支持流程验证和趋势观察。`native` 与 DAWN 的差异包含 policy 更新方式、replay 和优化日程；warm 与 random 的对比用于隔离 critic 初始化的影响。不能将前者解释为只改变 critic 初始化的消融。

## DAWN 延长训练（2026-09-08）

已从原 `warm` 的 50k 环境步 / 7,500 次更新 checkpoint 继续到 260k 环境步 / 60,000 次更新，保持全部 DAWN 超参数、online-only replay、优化器和随机数状态。正式结果目录为 `runs/warm_extended/`。45,001 次更新时为 **91% / -141.66**，60,000 次更新时为 **92% / -141.78**（成功率 / 平均回报，均为 100 episodes），未追上 native 的 99% / -68.66。[完整对照表](results_warm_continuation/summary.md) · [环境步数与更新次数双轴曲线](results_warm_continuation/learning_curves.png) · [CSV](results_warm_continuation/comparison.csv) · [固定协议](WARM_CONTINUATION_PROTOCOL.md)。

当前机器为单张 RTX 4090。首次启动与短程恢复验证入口为 `launch_warm_continuation.py`。`runs/warm_continuation_suite_status.json` 和 `runs/warm_extended/progress.json` 记录进度。训练进程断开且旧进程已停止后，可直接运行 `venv/bin/python run_warm_continuation.py` 从完整 `latest.pkl` 恢复；已有完成标记时不会重复训练。`venv/bin/python export_warm_continuation.py` 校验并导出表格、曲线和原始评估证据。已取回并核验 47 个归档文件及 40 个报告源文件，图表已完成视觉检查。

这里对齐的是累计更新次数；DAWN batch 1024，native batch 256，而且继续收集环境数据。旧 checkpoint 缺少最后一个 chunk 的剩余四个动作，续训在保存的环境状态重新采样。其余状态完整继承，恢复后的原 50-episode 评估与历史结果一致。短程连续运行与保存后恢复的完整状态逐项哈希一致，包含尚未执行完的更新。

## Task2 两组原始预算实验（2026-09-08）

用户确认只运行 QAM native 与 DAWN / pretrained Q / online-only 两组。任务改为 `cube-double-play-singletask-task2-v0`，其余沿用最初设置：seed 0、重新从头预训练 QAM 500k、共享 task2 checkpoint、50k online primitive steps，不延长。实现通过 [run_task2.py](run_task2.py) 选择环境，原训练循环保持不变；固定设置见 [TASK2_PROTOCOL.md](TASK2_PROTOCOL.md)。

两组已全部完成，最终各评估 100 episodes：**QAM native 99% / -219.47，DAWN / pretrained Q / online-only 54% / -495.53**（成功率 / 平均回报）。共享 task2 offline checkpoint 的 50-episode 成功率为 56%；50k online 环境步对应 native 45,001 次、DAWN 7,500 次更新。[完整报告](results_task2/summary.md) · [学习曲线](results_task2/learning_curves.png) · [对照表 CSV](results_task2/comparison.csv)。曲线各点评估 50 episodes，最终主对照表使用另行评估的 100 episodes。

正式与 smoke 数据均隔离在 `runs/task2/`。任务/奖励审计、两组 smoke 与最终结果校验均通过；已取回并验证 103 个归档文件、40 个报告源文件及训练源码哈希，图表已完成视觉检查。单张 RTX 4090 复用已有环境与数据；任务审计确认相同轨迹中有 99,441 条 reward 和 972 条 mask 因任务改变而不同。本轮仅一个训练 seed，且保留两种方法各自的 replay、batch 与更新日程。

查看 `runs/task2/suite_status.json`、`runs/task2/offline/progress.json` 或对应 online 目录的 `progress.json` 获取状态。仅在旧调度器已停止时重新运行 `venv/bin/python launch_task2.py`，可跳过已完成阶段并从 checkpoint 恢复。输出报告目录为 `results_task2/`。

## Task2 residual-scale 消融（2026-09-08）

用户确认新增 scale=0.05、0.2、0.3，以已有 0.1 为对照。保持 seed 0、共享 task2 offline 500k checkpoint、继承 Q、online-only replay、50k primitive online steps 和全部其他训练设置。只通过新入口 [run_task2_scale.py](run_task2_scale.py) 改变 residual scale，原训练循环和历史结果保持不变；详见 [固定协议](TASK2_SCALE_PROTOCOL.md)。

三组已全部完成，最终各评估 100 episodes：**scale=0.05：70% / -440.30；0.1（原有）：54% / -495.53；0.2：61% / -470.94；0.3：70% / -422.14**（成功率 / 平均回报）。0.05 更早改善，0.3 前期下降后恢复；两者最终均比 0.1 高 16 个百分点，但仍低于 native 的 99%。仅一个训练 seed，不能确认通用最优 scale。[完整报告](results_task2_scale/summary.md) · [学习曲线](results_task2_scale/learning_curves.png) · [最终 scale 对照图](results_task2_scale/scale_sensitivity.png) · [CSV](results_task2_scale/comparison.csv)。

原入口与新入口 scale=0.1 的短程最终 checkpoint 完全一致；三组并发 smoke 及正式运行已验证相同初始化、4,020 条 warmup chunks 和训练预算。单张 RTX 4090 上三组并行运行，各进程显存预分配比例 .28。已下载并校验 198 个归档文件、80 个报告输入和 35 个训练源文件哈希；两张图均通过视觉检查。状态、日志、配置和权重位于 `runs/task2/scale_sweep/`，报告在 `results_task2_scale/`。只在旧调度器停止后才可重启 launcher；静态 scale 在恢复前由配置文件校验。

## Task1 / Task2 极端 residual scale 扩展（2026-09-08）

用户要求将 scale 范围扩展到 0.01 和 1.0，并在 task1、task2 都运行。仅新增这四组，各自复用对应任务的 offline 500k checkpoint，保持 seed 0、继承 Q、online-only replay、50k primitive steps / 7,500 updates。原训练循环和原先 scale 实验代码均保持不变；[新入口](run_scale_extremes.py) · [固定协议](SCALE_EXTREMES_PROTOCOL.md)。

四组已于 2026-09-08 14:09:22 UTC 完成训练、最终评估和报告导出。两个任务的 scale=.1 新旧入口短程 checkpoint 一致，四组并发 smoke 通过。每组完成 50k primitive steps / 7,500 updates，使用同一任务对应的 offline checkpoint。最终主指标为 100 个配对 episodes 的 sampled-residual 评估：

| Scale | Task1 成功率 / 回报 | Task2 成功率 / 回报 |
|---|---:|---:|
| .01（新增） | 85% / -172.92 | 61% / -476.17 |
| .05（原有） | 未运行 | 70% / -440.30 |
| .1（原有） | 91% / -161.17 | 54% / -495.53 |
| .2（原有） | 未运行 | 61% / -470.94 |
| .3（原有） | 未运行 | 70% / -422.14 |
| 1（新增） | 62% / -258.00 | 4% / -849.71 |

当前单 seed、50k online 预算下，扩大范围未超过先前最好结果：Task1 在已测三种 scale 中以 .1 最好；Task2 的 .05/.3 仍以 70% 并列最高。Scale=1 在两个任务均退步。原生 QAM 参考两任务均 99%，但有 45,001 次 online 更新，不能视为相同更新预算。

[完整报告](results_scale_extremes/summary.md) · [对照 CSV](results_scale_extremes/comparison.csv) · [学习曲线](results_scale_extremes/learning_curves.png) · [全部已测 scale](results_scale_extremes/all_scales.png)。曲线每点 50 episodes；均值 residual 的 50-episode 诊断单列保存。日志、权重仍在远端 `runs/scale_extremes/`，导出包含原始评估、审计与图表，不含模型权重。

完整 warmup 数组审计已通过：Task2 原始 replay 完全一致；Task1 的模拟器状态最多相差 6.56e-13，转为训练 float32 后最多相差 7.00e-17，动作/奖励/折扣完全一致。另保留原 Task1 scale=.1 在 20k 的历史 episode 33 回报差异（-530 对比 offline -533，均值差 .06），成功率与其他字段一致。训练实现未因这些报告审计而更改；详细证据见 `runs/scale_extremes/warmup_audit/` 与 `results_scale_extremes/validation.json`。

## Task1 / Task2 新增数据消融（2026-09-08）

已在新机器配对复跑两任务的 native QAM：一组照常均匀采样 offline+online replay，一组只允许采样原 offline 数据。共享各任务的 offline 500k 完整 agent、Adam 与 RNG 状态；seed 0、batch 256、50k primitive 采集步、5k learning start、45,001 次更新。仅 offline 组仍运行相同采集循环，但全部新 transitions 与训练隔离，以保留原 RNG 和评估调用日程。详见 [固定协议](NEWDATA_ABLATION_PROTOCOL.md)。

四组正式运行于 2026-09-08 14:59:38 UTC 启动，15:15:38 UTC 完成训练、审计与导出。最终 100 个配对 episodes 的结果：

| Task | 使用 offline+online 数据 | 仅使用 offline 数据 | 新数据带来的成功率差值 |
|---|---:|---:|---:|
| 1 | 99% / -68.66 | 81% / -208.41 | +18 pp |
| 2 | 99% / -219.47 | 71% / -418.04 | +28 pp |

表中为成功率 / 平均回报。以相同 50 episodes 比较起点与终点，Task1 为初始 76% → 仅旧数据 76% / 使用新数据 98%；Task2 为初始 56% → 72% / 98%。使用新数据组累计只有 2.661% 的抽样序列涉及新数据，仍明显优于相同更新次数的仅旧数据组；这是单 training seed 下的结果，不支持跨 seed 稳定性结论。[完整报告](results_newdata_ablation/summary.md) · [CSV](results_newdata_ablation/comparison.csv) · [采集步数曲线](results_newdata_ablation/learning_curves.png) · [更新次数曲线](results_newdata_ablation/update_curves.png)。

两任务短程验证均通过：all 组最终 checkpoint 与原 native 精确一致；offline 组与零训练环境交互的纯 offline 更新参考精确一致。正式 all 组最终 checkpoint 也与历史 native 精确一致。每个 batch 审计实际采样起点及 5-step 窗口，两个 offline 组各 11,520,256 次序列抽样中，新数据窗口数均为零。Task1 初始 episode 33 有回报差异（-533 / -500），两组初始成功率相同，其余 episode 字段一致；完整模型、优化器与 reset 状态哈希一致，保留原始记录并在报告中披露。Task2 起点评估完全一致。

源文件在启动后冻结，正式数据与状态位于 `runs/newdata_ablation/`；入口为 `launch_newdata_ablation.py`，导出入口为 `export_newdata_ablation.py`。新克隆镜像的系统 NVIDIA EGL vendor 文件为空，本轮使用独立配置 `runs/newdata_ablation/egl_vendor.json` 指向已安装驱动，保留原 EGL backend；恢复训练时需要相同 `__EGL_VENDOR_LIBRARY_FILENAMES` 环境变量。不要在旧调度器仍运行时重启。

已取回并核验 247 个归档文件、114 个报告输入和 42 个训练源文件哈希；两张图均通过视觉检查。完整轻量证据包保存在 `results_newdata_ablation_bundle.tar.gz`，模型权重仍保留在远端各正式 run 目录。

## Task1 / Task2 critic target entropy 消融（2026-09-09）

已完成 DAWN Soft TD 与普通 TD 的配对运行。唯一变化是 critic backup 删除 entropy 项，actor SAC entropy 和自动 alpha 保留；仍为继承各任务 offline500k Q / target 参数、冻结 base、scale=.1、online-only replay、20k warmup、50k primitive steps、7,500 updates、batch 1024、10 critics、minimum 聚合。详见 [固定协议](TD_ABLATION_PROTOCOL.md)。

两任务短程完整 checkpoint 均精确复现原 Soft TD；普通 TD 与其初始化/warmup 一致、base 不变。四组正式运行于 2026-09-09 08:23:08 UTC 启动。诊断通过原有评估轨迹记录折扣回报，并在固定 warmup batch 上计算两种 counterfactual target、entropy 项、Q 差与残差梯度，不添加训练数据或随机抽样调用。

最终各评估 100 个配对 episodes：

| Task | 本轮 Soft TD | 普通 TD（actor entropy 保留） | 成功率差值 |
|---|---:|---:|---:|
| 1 | 91% / -160.24 | 93% / -150.58 | +2 pp |
| 2 | 50% / -578.03 | 62% / -483.16 | +12 pp |

表中为成功率 / 平均回报。普通 TD 在本轮改善了结果，但没有消除与 native 的差距；只有一个 training seed，不能据此确认跨 seed 稳定收益，也不能区分继承 Q 时的目标切换影响与长期目标改变的影响。初始固定 batch 的 entropy target 项均值约为 Task1 -1.74、Task2 -1.78；相应 chunk reward 的平均绝对值约为 4.94 / 6.67。[完整报告](results_td_ablation/summary.md) · [结果 CSV](results_td_ablation/comparison.csv) · [学习曲线](results_td_ablation/learning_curves.png) · [critic 诊断](results_td_ablation/critic_diagnostics.png)。

Task1 Soft 最终 checkpoint 与历史完全一致，评估回报有少量差异；Task2 新 Soft 从历史 54% 变为本轮 50%。后者的初始化、warmup 与前 25k 训练指标一致，随后 replay 中出现微小观测差异并伴随轨迹/训练分叉，具体数值来源尚未定位。主对照使用本轮配对实测值；复现证据保存在 `runs/td_ablation/task2_soft_historical_reproduction.json`。初始 Task1 配对评估中一个 episode 的回报差异也保留在报告中。

训练与 checkpoint 位于 `runs/td_ablation/`，入口为 `launch_td_ablation.py`；导出入口 `export_td_ablation.py` 生成 `results_td_ablation/` 表格、曲线和原始证据。使用独立 `runs/td_ablation/egl_vendor.json` 适配新机驱动，原训练源码保持不变。仅在旧进程停止后重启 launcher。

最终报告于 2026-09-09 08:45:26 UTC 完成导出。已取回并核验 370 个归档文件、210 个报告输入和 46 个训练/诊断源文件哈希；学习曲线与 critic 诊断图通过视觉检查。轻量证据包保存在 `results_td_ablation_bundle.tar.gz`，模型权重保留在远端。
