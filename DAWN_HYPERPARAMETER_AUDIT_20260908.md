# DAWN 参数与实现差异复核

2026-09-08。只读核对；没有修改训练实现或启动新实验。主对象为 Task1/Task2 的 QAM offline → DAWN pretrained Q / online-only replay。random Q 和 QAM replay 消融在下文另行区分。

此前“多数设置和 DAWN 相同”的描述不够精确：部分标量默认值一致，但 step 的计数单位、critic 的聚合/网络/梯度裁剪、奖励和评估动作存在实质差异。

## 来源与范围

- [DAWN 论文 v1，附录 B/E](https://arxiv.org/html/2602.10539v1)：论文没有 OGBench cube-double 的官方配置。
- 官方代码固定 commit `71122b7fa89568bc2d49831fed8cb1b01e5a91e8`，与首次实验的 SOURCE_AUDIT.json 一致。本地参考仓库 HEAD 和干净工作区已核验。
- [官方训练程序](https://github.com/Guozheng-Ma/DAWN/blob/71122b7fa89568bc2d49831fed8cb1b01e5a91e8/online_dawn/pi_dec_diffusion_maniskill2_dawn.py)、[官方网络](https://github.com/Guozheng-Ma/DAWN/blob/71122b7fa89568bc2d49831fed8cb1b01e5a91e8/nets/dawn.py)、[ManiSkill 启动配置](https://github.com/Guozheng-Ma/DAWN/blob/71122b7fa89568bc2d49831fed8cb1b01e5a91e8/online_dawn/run_dawn_maniskill.sh)。涉及 step/reward/clipping 的精确结论限于核对过的该公开实现，不外推所有论文实验的隐藏配置。
- 我们的实际来源：PROTOCOL.md、TASK2_PROTOCOL.md、run.py、dawn_agent.py、vendor/qam/agents/qam.py、vendor/qam/utils/networks.py、runs 下的配置/评估/训练日志。

## 设置比较

| 项目 | 原论文 / 对应公开实现 | 我们的实验 |
|---|---|---|
| Base policy | 示范 BC 的 Diffusion Policy / BeT | 500k 次 QAM offline RL 更新，选最后 checkpoint |
| Critic 初始化 | 新建 Q，target 复制 current | warm 继承 QAM current/target Q，online optimizer 新建；random 组新建 Q |
| Critic 数量与聚合 | twin critics，actor/target 都 min2 | 10 critics，actor/target 都 min10 |
| Critic 结构 | 公开实现：512 维嵌入、2 个 LN residual blocks，各 block 扩展到 2048，再接 value head | QAM 的普通 4×512 GELU MLP，每隐藏层激活后 LN，无 skip connection，便于直接继承 Q |
| Residual actor | 论文 B.3 写 2×256；公开代码是 3×256 ReLU | 跟公开代码：3×256 ReLU，state-only，无 actor LN |
| Residual scale | 任务相关，论文覆盖 0.05–0.3；没有 cube-double 值 | 两任务固定 0.1，尚未做 scale 消融 |
| Action chunk | Diffusion Policy 预测 16 步，执行 4 步 | 预测/执行 5 步；输出维度 5×5=25 |
| Base sampling | Diffusion 的公开默认 4 DDIM steps | QAM 10 Euler steps，单 proposal |
| Warmup | 公开代码约 20k replay/chunk transitions | 20k primitive steps；Task2 实际 20,002 步、4,020 chunks |
| UTD | 0.25 / chunk transition | 0.25 / primitive step，完整 5 步 chunk 对应约 1.25 updates |
| 收集/更新日程 | 默认 16 环境并行；收集 64 chunks 后更新 16 次 | 单环境；每个 chunk 后补足累计更新数，通常 1–2 次 |
| Online budget | ManiSkill launcher 1,000,000 chunk transitions，约 245k 次训练更新 | 50k primitive steps，7,500 次更新；Task2 replay 最终 10,038 chunks |
| Discount | 公开 ManiSkill 程序每 chunk bootstrap 用 gamma=0.97（Peg/Turn）或 0.9（Push） | 每 primitive step 0.99；完整 chunk bootstrap 为 0.99^5≈0.95099 |
| Reward | SeqActionWrapper 对 sparse rewards 求和，然后每 chunk 减 1 | 保持 OGBench 原始每步 -2/-1/0；chunk 内折扣累加 |
| Replay | online-only，缓存 current/next base action；ManiSkill launcher 容量 1m | 原始 DAWN 组 online-only、同样缓存 base；使用持续增长的列表，当前预算未涉及淘汰 |
| Critic clipping | 每个 critic 分别 clip global norm 到 50 | 全部 10 个 critic 的联合梯度 clip 到 50 |
| Action clipping | 环境 ClipAction；Q/actor loss 中直接用未裁剪的 base+residual | 环境、replay、actor/target Q 输入统一使用裁剪后的动作 |
| Evaluation | residual 使用 tanh(mean)，base 仍随机 | 主表 sampled residual；另外记录 tanh(mean) 辅助评估 |
| Repeats | 论文主图 8 training seeds | 1 training seed，50-episode 曲线及最终 100-episode 主评估 |

论文 B.3 的简化网络描述和公开网络不完全一致；不能把“原版 critic”简单写成普通 2×256 MLP，也不能把我们 3 层 residual actor 误认成偏离官方代码。

一致的主要标量和选择：batch 1024，actor/critic/alpha learning rate 1e-4，Adam，target tau=0.01，每次 critic 更新后更新 actor/target，alpha 初始 0.01 并自动调整，SAC entropy TD，log-std 范围 [-20,2]，输出头 orthogonal gain 0.01，冻结 base、state-only residual、critic 评估 summed action、base-only warmup 且 warmup 不训练、无 progressive exploration。熵定义同为未乘 residual scale 的 residual 坐标，target entropy 按整个 chunk 维度取负；我们的 -25 是维度变化的结果。框架默认隐藏层初始化和 LayerNorm 数值实现并非逐位相同。

QAM replay 消融有意额外改变了数据设计：保留 offline+online，使用 QAM sequence sampler；它不是原版 DAWN 的 online-only replay。当前 task2 只运行原始 online-only 版本。

## 对计数单位的推导

官方 SeqActionWrapper.step 一次执行至多 4 个 primitive actions，但 global_step 每个环境只加 1；ReplayBuffer 也只新增一个条目。因此公开代码的 20k warmup 是约 20k chunks，完整 chunk 时约 80k primitive steps。我们的 20k primitive steps 对应约 4k chunks。提前终止时换算不是严格整数倍。

我们实际 (50,000−20,000)×0.25=7,500 次更新。按新 chunk 数据量算，我们的更新密度约为官方数值的 5 倍；按总预算算，我们只有约 10k chunks，远少于官方 ManiSkill launcher 的 1m chunks。这两点同时成立。不能只说我们的 UTD 低于原版，也不能按原始环境步把 50k 与公开代码的 1m 直接作 20 倍比较。

gamma 也应在相同单位下解释。0.99 是 primitive discount，完整 chunk 的 0.95099 小于 0.97；不能因 0.99>0.97 就声称我们规划视野一定更长。OGBench 的奖励累加尺度也不同，不能从相同 alpha 初值推出相同的奖励/熵权衡。

## 已有证据与解释边界

| 证据 | 可以支持的结论 | 不能推出的结论 |
|---|---|---|
| Task1 100-episode DAWN：7,500 / 45,001 / 60,000 更新为 91% / 91% / 92%；native 45,001 更新为 99% | 延长训练改善回报，但此 seed 未追上 native；更新次数不是唯一因素 | Task2 延长训练也一定无效；DAWN 已收敛到全局最优 |
| Task2 相同 50-episode 曲线：offline 56%，DAWN 54%，native 98% | 这次短预算下 DAWN 基本没有改善 | 多 seed 统计结论 |
| Task2 tanh(mean) residual 辅助评估 58%（50 eps），sampled residual 54%（50 eps） | 仅改变评估动作不足以消除主要差距 | 残差采样在所有任务都不重要 |
| Task1 原始 online-only 91%，QAM replay 79%（两者最终 100 eps） | 当前均匀混合 offline 数据没有改善 | 所有 offline mixing 比例都无用 |
| Task2 最后一个训练 minibatch：scaled residual abs mean=0.059737，clip fraction=0.015898 | residual 已明显非零；不是绝大多数动作被裁剪 | residual 已饱和；局部半径一定足够 |
| Task2 alpha 最后为 0.005559；30 个已记录更新采样点未显示 alpha 爆炸 | 不支持把持续 alpha 爆炸当作已证实原因 | 完全排除最早未记录的瞬态或熵目标不匹配 |
| Task2 30 个有更新的日志点联合 critic gradient norm 均 >50，范围 62.63–249.60；最终 119.713 | ensemble 联合 clipping 确实在日志采样点生效，值得单独核查 | 全部 7,500 次更新都触发；Adam 参数步幅同比例缩小；裁剪已被证明导致失败 |

最后一步的联合 gradient clipping 系数约 50/119.713=0.418，仅指送入 Adam 前的梯度；不能把它解释为有效学习率直接变成 0.418 倍。

## 当前最值得检验的机制

1. **局部残差与 QAM online 的策略调整能力不同。** 当前策略为 clip(a_QAM(s,z)+0.1 tanh(u(s,xi)))。它冻结 QAM flow，residual 又不看本次 sampled base proposal。对于同一状态的不同动作模式，它不能直接按 proposal 定制修正，也难以重新分配距离较远的模式概率。如果失败需要改变行为顺序或选择另一种动作模式，局部修正可能不足。状态分布可随残差改变，故这不是全局不可改善的数学结论。state-only 也是 DAWN 官方默认，属于迁移适配假设，而非单纯实现偏差。
2. **critic 的局部动作梯度可能仍不可靠。** min10 比 min2 的聚合规则更保守；当前/target 从 QAM 转到 residual soft Bellman target 时，还同时改变了下一动作策略与原 mean−0.5std 聚合。继承权重不等于继承同一个 value-learning 问题。联合梯度裁剪和不同 LN 结构又增加了差异。需要测逐 critic 分歧、动作梯度一致性、Q(full)−Q(base) 与实测 rollout 改善的关系，而不只看平均 TD loss。
3. **原 DAWN 价值学习设计的收益在 QAM 上可能部分重复。** QAM 已有训练过的 critic 和 LayerNorm；我们没有比较 residual 去掉 LN/anchoring 的消融，不能把两项组件各自的作用从当前结果中分离。20k 不更新在 50k 总预算中占 40%，而原 ManiSkill launcher 中只有约 2%。
4. **短预算和 anchor 数量不足仍可能影响 task2。** 此轮只有 4,020 个初始 replay chunks、单环境和 7,500 更新。Task1 的延长结果限制了“只需更多更新”的解释，但不能代替 task2 的验证。

建议下一步先统一报告 primitive/chunk/update 三种计数及 paired sampled/tanh(mean) 评估；随后做单变量消融：min10 对固定两 critic minimum、联合 clipping 对逐 critic clipping、scale 0.1 对 0.2、state-only 对 state+base proposal。保留继承 QAM backbone 以避免网络重建和初始化同时变化。以上只是分析建议，本次没有启动这些实验。
