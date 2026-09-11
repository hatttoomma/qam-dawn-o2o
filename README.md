# QAM → online / residual RL experiments

OGBench `cube-double-play-singletask-task1`–`task5` 上的 offline-to-online 实验代码。此仓库保存从实际训练服务器取回的源码，包括 QAM native、DAWN 派生 residual RL、Policy Decorator、BC 起点与相关消融。

## 当前主要设置

- 每个 task 独立进行 QAM offline 500k updates；训练 seed=0。
- QAM native：batch 256、UTD 1、5k steps 开始更新、offline + online 均匀 replay、50k online steps。
- 官方 cube-double 配置为 `inv_temp=1`、`edit_scale=0`，所以这里的 QAM-EDIT baseline 实际没有启用 editor。
- 最近的 DAWN 派生版本：冻结 QAM base、普通 TD、batch 256、UTD 0.25、online-only replay、20k base-only warmup、residual scale 0.1。
- Residual actor 输入 state + 实际 sampled base action chunk（37 + 25 = 62 维）；3×256 ReLU，tanh-Gaussian 输出。
- Critic ensemble=10，actor Q 和 target Q 均取 minimum；actor entropy 和自动 alpha 保留。
- 两种方法的 action chunk 都为 5。
- Task5 另外比较继承 critic / 随机 critic，完成 500k online，并从完整训练状态续训到 1M。

DAWN 派生实现包含我们为 QAM / OGBench 做的改动，并非对 DAWN 原论文全部默认设置的原样复现。较早消融有不同的 TD、batch、replay 和 actor input；各协议是对应实验的准确说明。

## 文件入口

| 内容 | 文件 |
|---|---|
| QAM 与初始 residual 训练、评估、checkpoint | [`run.py`](run.py) |
| 最新 state + base action residual | [`actor_input_agent.py`](actor_input_agent.py) |
| 五任务训练 | [`run_cube5.py`](run_cube5.py)、[`CUBE5_PROTOCOL.md`](CUBE5_PROTOCOL.md) |
| Task5 500k 长训 | [`run_task5_long_critic.py`](run_task5_long_critic.py)、[`TASK5_LONG_CRITIC_PROTOCOL.md`](TASK5_LONG_CRITIC_PROTOCOL.md) |
| Task5 500k → 1M 续训 | [`run_task5_continue_1m.py`](run_task5_continue_1m.py)、[`TASK5_CONTINUE_1M_PROTOCOL.md`](TASK5_CONTINUE_1M_PROTOCOL.md) |
| Policy Decorator | [`policy_decorator_agent.py`](policy_decorator_agent.py)、[`POLICY_DECORATOR_PROTOCOL.md`](POLICY_DECORATOR_PROTOCOL.md) |
| BC 起点 | [`bc_pretrain.py`](bc_pretrain.py)、[`BC_PROTOCOL.md`](BC_PROTOCOL.md) |
| 其余消融 | `run_*ablation.py`、`*_PROTOCOL.md` |
| 验证 / 报告 / 视频 | `audit_*.py`、`verify_*.py`、`summarize*.py`、`render_*.py` |
| 原始实验记录 | [`docs/REMOTE_README.md`](docs/REMOTE_README.md) |

## 环境与数据

训练环境为 Linux + NVIDIA GPU，依赖锁定于 [`requirements.txt`](requirements.txt)。在仓库根目录准备环境：

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
```

数据来自 OGBench，由训练入口通过 `ogbench.make_env_and_datasets` 读取/下载；默认位置是仓库下的 `data/`，也可使用 `--data` 指定。数据来源和哈希见 [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md)。实际运行还需要主机 CUDA / EGL 驱动；本次打包只验证源码完整性和 Python 语法，没有在本地重新训练。

## 如何使用这份代码

本仓库是**实际实验源码快照**。训练逻辑未为上传而改写。历史 launcher 和 continuation 脚本会验证原始 checkpoint、评估记录及 source manifest 的哈希，因此仅 clone 代码不能直接恢复历史实验。

- 继续已有实验：将原服务器对应的 `runs/`、`data/` 另外保留或恢复到相同目录结构，再使用相应 launcher；先确认没有重复运行中的训练进程。
- 从头运行最初 Task1 QAM baseline：可运行以下命令。它们会执行完整训练，产生计算开销。

```bash
python run.py --stage offline --out runs/offline
python run.py --stage native --out runs/native --offline-checkpoint runs/offline/final.pkl
```

- 最新五任务与 1M 续训：按对应协议和 `launch_*.py` 的前置检查准备历史依赖。`launch_cube5.py` 复用 Task1/2，Task5 continuation 依赖对应 500k 完整 checkpoint。
- 数据、checkpoint、replay、运行日志、视频和结果归档没有上传；`.gitignore` 已覆盖这些路径。
- 历史文档中的 `results*/` 链接及绝对服务器路径是原运行记录，clone 后只有另外取回相应产物才可使用。

## 评估口径

训练时 residual 为随机采样。最近 Task5 长训在 50k、100k、200k、500k、750k、1M 同时保存 sampled 和 mean-residual 评估；每种方式每点 100 episodes，0k 为关闭 residual 的 offline baseline。

Mean residual 指 `tanh(mean)`，不是多个 sampled actions 的平均；QAM base action 仍按原 flow policy 采样。原先五任务的主结果为 sampled residual，mean residual 是额外诊断。比较时应明确标注口径、episode 数和 online budget，不能逐 checkpoint 选择两种方式中较高的值。

## 来源与完整性

[`REMOTE_SOURCE_MANIFEST.json`](REMOTE_SOURCE_MANIFEST.json) 记录取回时的 148 个远程文件及 SHA-256。原始 `README.md` 和 `.gitignore` 分别完整保存在 `docs/REMOTE_README.md` 和 `docs/REMOTE_GITIGNORE.txt`，其余源文件保持原路径和原内容；仓库根目录 README 与 .gitignore 是上传时添加的整理版本。

QAM 上游固定 commit：`2726d767c9a0a7a46d49693f0391f73dc2cf58ac`；来源核对见 [`SOURCE_AUDIT.json`](SOURCE_AUDIT.json)。Vendored QAM 代码保留其 [`MIT License`](vendor/qam/LICENSE)。此快照未为项目自有代码额外选择开源许可证。
