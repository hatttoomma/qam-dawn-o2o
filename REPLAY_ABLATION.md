# DAWN replay 补充实验

用户要求：保持原先两种 DAWN 设置，将 data replay buffer 改为 QAM setting，并加入原三组结果。复用相同 seed 0、500k offline checkpoint，各新增 50k 原始环境步；原实验结果及训练源码保留。

新增 `warm_qam_replay` 与 `random_qam_replay`。除 replay 接入外，保留原来的 frozen QAM、residual 网络与初始化、critic 网络、current/target 初始化规则、SAC entropy TD、10 个 critic 的 minimum 聚合、batch 1024、UTD 0.25、lr 1e-4、tau 0.01、alpha 初始化与自动调整、20k base-only warmup、完整执行 5 步 action chunk、评估 seed 与评估日程。

## Replay 的实际含义

- 使用原 QAM `ReplayBuffer`，初始装入全部 1,000,000 条 offline primitive transitions；每个 online primitive transition 继续加入同一 buffer。
- 原 QAM `sample_sequence` 从所有可用序列起点均匀采样，序列长度 5。混合比例不固定为 50:50：online 起点占比从学习开始时约 1.96% 增至终点约 4.76%。记录实际抽样比例。
- 使用原 sampler 的累计 reward、mask 和 terminal/timeout 处理。序列跨 episode 时按 QAM 的 `valid[..., -1]` 屏蔽 critic MSE；仍按整个 batch 求均值。actor 与温度更新沿用原 DAWN。有效序列上的 DAWN 更新通过数值对照检查。
- 这是 primitive transition 均匀采样，包括滑动窗口；原 DAWN 组则采集并均匀采样实际执行的 decision chunks。因此该实验改变完整 replay 机制，不能解释为只改变固定 offline:online 比例。
- DAWN 需要 cached base/next-base proposals。为 offline 每条 transition 的 observation/next observation 从冻结 QAM 预计算一次，两个新组共享完全相同的缓存。缓存 key 由固定 seed、字段类型和行号决定，独立于训练与评估 RNG。
- 对 online decision 的起点与终点使用实际采集时的 base/next-base proposal。滑动窗口涉及的其他中间状态在首次被抽到时生成并缓存 proposal。缓存始终来自同一 frozen QAM。
- 前约 20k 个环境步仍只采集、不学习。offline replay 的存在不会提前触发训练。总更新次数仍为 7,500。

## 校验与输出

检查原 sampler 的输出及 RNG 推进一致、跨 episode mask、缓存行对齐、有效 batch 上与原 DAWN 更新一致、初始化与旧组一致、frozen QAM 不变、完整预算和逐 episode 评估。warmup 与旧 DAWN 比较时，动作/回报/discount/proposals 要求逐位相同，观测容许既有的 1e-10 绝对误差。

两个新训练目录单独保存，新增源码使用独立 source manifest；原始三组结果不覆盖。最终输出五组评估表、学习曲线，以及同一 critic 初始化下切换 replay 的差值。
