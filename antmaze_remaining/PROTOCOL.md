# AntMaze-large tasks 2–5, seed 0

User-approved continuation of the task1 experiment. Each task has its own freshly trained QAM offline checkpoint. Do not reuse another task's checkpoint: OGBench relabels rewards for each task.

## Training

- QAM: official vanilla QAM, not QAM-EDIT. Source commit `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`; official source and historical residual implementation remain unchanged and are checked by SHA256.
- QAM budgets: 500,000 offline updates, 50,000 online environment steps, 45,001 online updates (start at step 5,000 inclusive).
- QAM parameters: inv_temp=10, fql_alpha=0, edit_scale=0, horizon=1, batch=256, UTD=1, 10 critics, target mean−0.5std, lr=3e-4, discount=.99, tau=.005. Original uniform offline+online replay, remaining official defaults unchanged.
- DAWN starts from its task's QAM **offline500k** checkpoint, after native completion passes checks. It inherits Q and target Q with fresh Adam state and freezes the base policy.
- DAWN budget: **80,000 base-only warmup +20,000 additional training environment steps =100,000 total online steps**. No parameter updates during warmup; 5,000 updates total.
- DAWN unchanged settings: naive TD (no entropy in the critic target), actor entropy and auto-alpha retained, batch256, UTD.25, state+base-action input (29+8), residual scale.1, horizon1, ensemble10 with minimum actor-Q and target-Q, online-only replay including warmup. Actor/critic/alpha lr1e-4, tau.01, discount.99, grad clip50, initial alpha.01, target entropy−8.
- One training seed0 per task; task ID is the only task-specific setting.

## Evaluation and preservation

QAM's official 50-episode evaluation every50k is unchanged. Supplemental comparison uses100 fixed episodes, identical within each task across checkpoints and methods. The evaluator fixes Gymnasium, NumPy and action-space reset RNGs and restores training NumPy RNG.

DAWN evaluates at0, **90k and100k** for both mean and sampled residual. Mean residual is primary; the QAM base remains stochastic. Checkpoints are retained at80k,90k,100k. At80k the base is unchanged and its0k result is reused without an extra rollout. The90k evaluation leaves training RNG unchanged, verified by an audit marker; update counts are2,500 at90k and5,000 at100k.

The earlier task1 run remains preserved, including its100k evaluation. This continuation does not relaunch task1 or manufacture a90k task1 result.

## Execution

Remote root: `/root/autodl-tmp/qam_antmaze_task2_5`; same host port45557 and existing `/root/autodl-tmp/antmaze_qam_venv` runtime as task1. Read-only symlinks share the verified official/legacy sources; data files are reused with checksum validation. Credentials are not stored.

Queue: verify all four task registrations and data/source hashes; run a short task2 native+DAWN smoke (scaled80/90/100 checkpoints); then task2 native→DAWN, task3 native→DAWN, task4 native→DAWN, task5 native→DAWN serially on one GPU. Any failed stage stops the queue.

Results: `results/task{2,3,4,5}/{native,dawn}`; suite status `results/suite_status.json`. Each stage records wall-clock start/finish times. Completion requires native DONE and DAWN CHECKS_PASSED, exact initial paired records, frozen base and checkpoint hashes, requested update counts, and both90k/100k evaluation files.

QAM50k vs DAWN100k is not an equal-environment-budget comparison; report env steps and update counts separately. These one-seed runs do not estimate across-training-seed uncertainty.
