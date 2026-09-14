# Antmaze-large task1: official vanilla QAM then DAWN

Authorized on 2026-09-13. Environment: `antmaze-large-navigate-singletask-task1-v0`.
One training seed, 0, continuing the prior experimental convention. Only task1 is authorized for the initial rollout; tasks2–5 are deferred.

## Native QAM

Upstream: https://github.com/ColinQiyangLi/qam, commit `2726d767c9a0a7a46d49693f0391f73dc2cf58ac` (verified current HEAD).
Settings come from `experiments/reproduce.py`, QAM/antmaze-large-navigate, plus `main.py` and `agents/qam.py` defaults. The official main, agent, environment loader, sampler, optimizer, and evaluation implementations are byte-identical, checked against `source_manifest.json` before each stage.

| Parameter | Value |
|---|---|
| Offline updates | 500,000 (user override of official 1,000,000) |
| Online environment steps | 50,000 (user override of official 500,000) |
| Method | Vanilla QAM, no editor or FQL actor |
| inv_temp / fql_alpha / edit_scale | 10 / 0 / 0 |
| Horizon / action_chunking flag | 1 / True (one primitive action) |
| Batch / UTD | 256 / 1 |
| Online training starts | Step 5,000 inclusive; 45,001 online updates |
| Replay | Uniform over all offline data and collected online data; balanced_sampling=False |
| Critic target | Ensemble mean − 0.5 × standard deviation |
| Ensemble | 10 |
| Discount / target tau | 0.99 / 0.005 |
| Adam learning rate / global gradient clip | 0.0003 / 1 |
| Actor / critic hidden dimensions | Four layers of 512 |
| Actor / critic LayerNorm | False / True |
| Flow steps / best-of-n | 10 / 1 |
| Other agent switches | target_actor=True, residual=False, clip_adj=True, use_target_grad=True |
| Data fraction / reward transformation | 1.0 / official OGBench relabeling, sparse=False |
| Official evaluation | 50 episodes every 50k steps, unchanged |

Operational additions: local logging instead of W&B upload; preserve checkpoints (`auto_cleanup=False`); save complete offline/final policy state; supplementary independent fixed 100-episode evaluation at offline500k and online50k. Supplementary evaluation uses a separate environment and preserves training NumPy/Python RNG state. The official seed schedule uses 12 seeds; this pilot uses one seed0, not that multi-seed aggregate. Training environment reset behavior is exactly the upstream main, including its lack of an explicit reset seed.

Antmaze's `add_noise()` uses global NumPy randomness in addition to Gymnasium's per-environment RNG. The supplemental paired evaluator seeds global NumPy and the action space for each reset, then restores global NumPy state. This is required for matching initial observations; `env.reset(seed)` alone failed the initial smoke audit. Official QAM evaluation remains unchanged. DAWN's per-episode training reset uses the same isolation with a separate training seed range so checkpoint restore reproduces the collected state without consuming the replay-sampling RNG.

## DAWN residual

Begins only after native QAM finishes and passes configuration, source, finite-metric, checkpoint, and update-count checks. Its initialization is the **offline500k** state, never the native-online final state. No gate selects a checkpoint on evaluation success.

| Parameter | Value |
|---|---|
| Total online env steps / base-only warmup | 150,000 / 80,000, user confirmed |
| Batch / UTD / updates | 256 / 0.25 per primitive env step / 17,500 |
| TD | Plain TD, no critic-target entropy bonus |
| Actor input | State concatenated with sampled base action |
| Residual | 3×256 ReLU, tanh Gaussian, scale 0.1 |
| Critic | Inherit offline current and target Q; fresh Adam states |
| Ensemble / actor-Q and target-Q aggregation | 10 / minimum |
| Replay | Online-only uniform growing replay, including warmup transitions |
| Learning rates / clip / target tau | 1e-4 for actor, critic, alpha / 50 / 0.01 |
| Actor entropy / automatic alpha | Enabled / enabled; initial alpha0.01 |
| Action / horizon / target entropy | 8 / 1 / −8 |
| Evaluation | 100 paired episodes; mean residual primary, sampled residual secondary |

The immutable historical `run.py` residual collection/update loop and `actor_input_agent.py` are reused. Only environment, task-specific QAM inverse temperature, horizon and automatically derived observation/action dimensions change from cube. The hard-TD hook, state+base-action switch, budgets, evaluation hooks and audits match the prior design. Base flow weights must stay unchanged throughout DAWN. Initial fixed evaluation records must exactly match the native offline evaluation.

## Execution and artifacts

New host: `connect.bjb1.seetacloud.com:45557`; root `/root/autodl-tmp/qam_antmaze`; venv `/root/autodl-tmp/antmaze_qam_venv`.
Credentials are not stored in this project.

Dataset transport can use the `yonghoon96/ogbench-mirror` mirror (or its hf-mirror.com frontend). Both file SHA256 values were first computed from complete downloads from the official Berkeley endpoint, independently checked against the mirror LFS metadata, and are hardcoded in `download_data.py`. Installation accepts files only after exact size, SHA256, and ZIP CRC verification. Changing the transport does not change dataset contents.

`suite.py` runs a small isolated native smoke (10 offline +12 online), residual smoke (80 warmup/200 total), production native, then production DAWN serially on the one GPU. Production always starts fresh after smoke. A failed phase stops the suite.

Results live in `results/{smoke_native,smoke_dawn,native,dawn}`. `results/suite_status.json` identifies the active child. `native/DONE.json` and `dawn/CHECKS_PASSED.json` are completion gates. Original native CSVs, supplementary paired evaluations, configs, source hashes and checks are retained.

Do not describe 50k-native versus 150k-DAWN endpoints as an equal-data-budget comparison. Count online gradient updates separately from the shared offline pretraining.
