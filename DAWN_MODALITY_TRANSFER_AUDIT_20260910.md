# DAWN：输入模态与迁移差距核对

日期：2026-09-10。此次只审阅论文、源码和已有结果，未改训练配置或启动实验。

官方仓库远端 HEAD 与本地干净参考仓库一致：`71122b7fa89568bc2d49831fed8cb1b01e5a91e8`。当前 cube5 设置应以 CUBE5_PROTOCOL.md 和 run_cube5.py 为准，不能沿用早期 state-only / soft-TD / batch1024 的描述。

## 可以确认的事实

- DAWN 的 Figure 11 包含六个 state 配置和两个 visual 配置。其显著改善并不限于视觉输入；Table 10 明确把 PegInsertionSide、TurnFaucet、PushChair 列在 state 配置下。已视觉检查 PDF 第 7 页。
- 公开主程序 `make_env` 明确设置 `obs_mode='state'`、`reward_mode='sparse'`。不存在“官方主实验是视觉，而 cube 是 state”这一前提。
- DAWN 官方仓库当前只提供 ManiSkill state 训练入口。其上游 Policy Decorator 的 RGBD 程序把冻结 base 的视觉 embedding 与附带 state 特征交给 residual actor/critic；这不能当作 DAWN 未公开视觉实验的逐项复现证明。
- 官方代码使用 3×256 ReLU Gaussian actor；默认 actor 输入为当前 state，支持可选 state+base action。其 critic 是 512 维投影、两个含 LayerNorm 的 residual blocks（内部扩展至 2048）、value head。Twin critics，actor/target 均 min2。
- 官方在线主循环使用 soft TD；评估使用 tanh(mean) residual，base diffusion 仍从随机噪声采样。官方 base 观察两步历史，预测 16 步、执行 4 步；residual 一次输出对应四步修正。
- 官方 launcher 是 1M chunk transitions；`SeqActionWrapper` 一次执行最多四个 primitive actions，global_step 每个向量环境只增加一次。因此 20k warmup 是约 20k chunks，不是 20k primitive steps。1M launcher 对应约 245k 更新；提前终止会影响 primitive 换算。
- 当前 cube5：50k primitive steps，20k primitive warmup，约 4k warmup chunks，7,500 更新，batch256，hard TD（actor entropy/自动 alpha 仍在），state+base actor，继承 10 个 QAM critics，min10，四层 512 GELU/LN，无 skip blocks。评估主口径为 sampled residual。
- 官方梯度裁剪分别对每个 critic 做 norm50；我们对整个十 critic 参数树联合 norm50。相同阈值不等于同一种裁剪。
- 官方任务环境定义包含任务/目标相关 state 信息及新物体适配；base checkpoint 来自 Policy Decorator 的任务示范 BC。我们基于 play 数据作任务奖励学习，之前 pure-BC 消融仍模仿同一份 play 数据。这两种 BC 数据设定不可视为相同。

## 解释及边界

输入是 state 并不阻止 residual RL 生效：官方 state 结果和我们 Task1 的改善均为直接反例。特征表示质量、尺度和历史信息仍可能影响学习，但这是更具体的假设，不能笼统归因于 state/visual。

当前优先假设是预算/价值学习实现，以及 base 的剩余错误能否被局部修正。示范 BC 的局部偏差与 QAM 剩余失败可能不同；这是待检验的机制，不是现有论文或视频已经证明的因果结论。类似 offline success 不保证类似局部可改进空间。

原始 DAWN 的 base-only 数据与 critic LN 旨在帮助价值学习；它们不提供逐 episode 不退步的约束。Task5 完整 100 次固定配对评估中 offline 和 DAWN 都为 76%，有 14 次得益与 14 次退步。这证实策略有实际改变，不识别 critic、actor、随机探索或局部半径中哪一项是主因。

当前方法已经改变原版 critic、TD、batch、actor 输入、时间计数和评估；其结果不能单独作为“原版 DAWN 不适用于 cube”的证据。随机 Q 对照只控制初始化，不会消除网络、聚合、数据和目标的差异。

既有 Task1 延长训练到 45,001 / 60,000 更新后的成功率为 91% / 92%，native 为 99%。这不支持“只需对齐更新数就一定追平”，但没有验证 Task5 在论文级别预算下的表现。

## 建议的区分性实验（未启动）

1. 用官方 state-ManiSkill checkpoint/代码，明确记录 primitive、chunk、update 三个轴，并加入 50k primitive 截断点，验证短预算能保留多少论文收益。
2. 固定 cube base、数据日程、评估口径，比较官方价值网络配置与当前配置；网络结构对照应双方随机 Q，不能把结构变化与继承 Q 同时当作一个因素。先整体复现，再拆 min2、block、clipping 等因素。
3. 检验 BC/QAM 的区别时，先使用同一份任务一致数据训练二者，在独立 validation 上匹配 base success，再使用相同 online 配置；play BC 不能直接代替专家示范 BC。比较得益与退步，并补 online training seeds。

## 核对来源

- [论文 v1](https://arxiv.org/html/2602.10539v1)：Figure 11、Appendix B/E。
- [官方训练入口](https://github.com/Guozheng-Ma/DAWN/blob/71122b7fa89568bc2d49831fed8cb1b01e5a91e8/online_dawn/pi_dec_diffusion_maniskill2_dawn.py)：146–174（环境/动作块）、299–315（评估）、434–480（actor/critic）、510–633（计数/replay）、737–831（更新）。
- [官方网络](https://github.com/Guozheng-Ma/DAWN/blob/71122b7fa89568bc2d49831fed8cb1b01e5a91e8/nets/dawn.py)。
- [官方任务 launcher](https://github.com/Guozheng-Ma/DAWN/blob/71122b7fa89568bc2d49831fed8cb1b01e5a91e8/online_dawn/run_dawn_maniskill.sh)。
- [官方环境适配](https://github.com/Guozheng-Ma/DAWN/blob/71122b7fa89568bc2d49831fed8cb1b01e5a91e8/envs/maniskill_fixed.py)。
- [上游 Policy Decorator RGBD 入口](https://github.com/tongzhoumu/policy_decorator/blob/92ba9ba442587ae5989c286355ace2200a0537fb/online/pi_dec_diffusion_maniskill2_rgbd.py)：548–595、615–632；仅用于解释上游表征接口。
- 本地：CUBE5_PROTOCOL.md、actor_input_agent.py、run_cube5.py、BC_PROTOCOL.md、results_cube5/summary.md、results_warm_continuation/summary.md、output/task5_offline_vs_dawn_all100/manifest.json。
