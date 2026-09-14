# AntMaze-large tasks 1–5: 40k warmup +10k training, balanced replay

User-authorized seed0 DAWN comparison. Reuse each task's own verified vanilla QAM offline500k checkpoint. No offline retraining and no native QAM rerun.

- Warmup40,000 primitive env steps: frozen base-only stochastic rollout, no actor/critic/target/alpha updates.
- Then10,000 primitive env steps of frozen base + sampled residual rollout;50,000 total, UTD.25,2,500 updates.
- Two separate buffers. Offline buffer is the1M-transition original training dataset with that task's reward/mask relabeling, actions clipped exactly as prior QAM training. The validation dataset is excluded. Online buffer grows uniformly and retains all warmup transitions.
- Each batch256 contains exactly128 uniform offline draws and128 uniform online draws, with replacement. The ratio applies to critic, actor and alpha updates.
- Offline critic regression uses the recorded dataset behavior action, not a reconstructed residual action. Frozen QAM supplies stochastic base actions for the sampled current and next states on demand. These are distinct from the dataset behavior actions. An independent JAX stream keyed by seed and update index preserves rollout randomness and checkpoint resumability. Online samples retain their collection-time base and next-base actions as before.
- Offline discounts are gamma times task-relabeled masks; horizon1 means there is no multi-step sequence conversion. Time-limit/terminal treatment otherwise remains that of the original dataset and collection loop.
- Unchanged DAWN: naive TD (no entropy in critic backup), actor entropy and auto-alpha retained; actor input state29+base action8; scale.1; Q and target Q inherited with fresh Adam; actor3x256ReLU/tanhGaussian, critic10x(4x512GELU+LN), actor-Q/target-Q minimum; lr1e-4, clip50, tau.01, discount.99, alpha.01, target entropy−8. QAM base inv_temp10, edit_scale0, horizon1, always frozen.
- Evaluate0k,45k,50k on the same100 fixed episodes used previously. Mean residual primary, sampled residual secondary; base remains stochastic. At40k save a checkpoint and reuse the unchanged0k base evaluation. Midpoint45k has1,250 updates,50k has2,500. Evaluator preserves training NumPy RNG.
- Compare with saved task-specific offline, native QAM50k and prior DAWN80k+20k (task1 old100k checkpoint). The new arm changes both budget and replay, so differences cannot be attributed to replay alone. These are single-seed comparisons.

Remote root `/root/autodl-tmp/qam_antmaze_balanced40k`; host port45557, existing venv `/root/autodl-tmp/antmaze_qam_venv`. Official source and actor/critic implementation stay byte-identical. Only the isolated copy of the old collection loop keeps the offline dataset and delegates batch assembly to `balanced_replay.py`; original results and code stay intact.

Queue validates sources/checkpoints/tasks and sampler semantics, then a48-step task1 smoke using the real offline500k checkpoint (40 warmup,2 updates,2 episodes). Production tasks1–5 run serially on one GPU. Failures stop the queue. Each task verifies inherited hashes, exact initial100 episode records, unchanged settings, no warmup updates, batch128/128, all2,500 update counts, frozen base and final checkpoint SHA256, evaluation grids and RNG preservation.

Results `results/taskN/dawn`; suite status `results/suite_status.json`. Runtime files include evaluations. Local artifacts `output/antmaze_large_balanced_20260914`. Credentials must not be written to files.
