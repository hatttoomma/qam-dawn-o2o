# AntMaze tasks1–5: resume balanced50k and train10k with online-only replay

User-authorized continuation of all five seed0 DAWN runs from `qam_antmaze_balanced40k`. Do not restart from offline, retrain QAM or reset optimizers.

- Start at online env step50,000 / update2,500. Stop at60,000 / cumulative update5,000: exactly10,000 additional env steps and2,500 additional updates. No new warmup. Historical warmup40k remains in the cumulative step/update schedule.
- Recover the original50k `latest.pkl`, which includes the same complete agent state as `final.pkl` plus the50k online replay, NumPy/rollout JAX keys, current episode and action trace. Check agent/state hashes against the published50k final checkpoint. Restore the environment by replaying the saved episode trace and verify the observation matches. All actor/Q/target-Q/alpha parameters and optimizer states are carried over.
- The entire retained50k online buffer, including40k base-only warmup transitions, remains available and grows to60k. Each new update uses256 online samples, uniform with replacement; zero offline samples. Do not clear the buffer or restrict it to the new10k transitions.
- Frozen QAM base still comes from the task's original offline500k policy. Rollout remains sampled base+sampled residual. Naive TD, batch256, UTD.25, state29+base8, scale.1, horizon1, ensemble10/minimum Q/target aggregation, actor entropy+auto-alpha, lr1e-4, tau.01, discount.99, clip50 all remain unchanged. No optimizer or entropy reset.
- Preserve old50k mean and sampled100-episode evaluations. Add55k/60k evaluations for both modes using the same100 fixed episodes. Mean residual is the primary summary; base policy remains stochastic. Evaluation preserves training NumPy RNG.55k has3,750 cumulative updates;60k has5,000.
- The original unmodified online-only UTD.25 collection/update loop is reused with source SHA checks. Thin wrappers handle exact state restoration, configuration auditing and evaluations; no update equation changes.
- Compare resumed55k/60k with the parent balanced50k checkpoint, old DAWN online-only100k and QAM native50k. The continuation adds both interaction and updates while changing replay, so it does not isolate replay alone.

Remote root `/root/autodl-tmp/qam_antmaze_online60`, same host root@connect.bjb1.seetacloud.com:45557 and `/root/autodl-tmp/antmaze_qam_venv/bin/python`. Prefer hostname (direct IP is unreliable). Credentials are never stored in files.

One GPU queue validates sources and all5 full50k resume checkpoints, then a task1 smoke resumes the real50k state, reproduces the first2 original50k mean/sampled episodes and continues8 env steps /2 updates. Production task1–5 resume serially. Results are separate from old runs. Checkpoint copies are independent; old checkpoints are unchanged. A failed stage stops the queue; do not automatically change parameters or restart an incomplete run.

Completion requires exact resume state and replay hashes, environment restoration, no new warmup,256 online/0 offline samples per update,10k new env steps/2,500 new updates, unchanged frozen base,55k/60k evaluations with paired initial states, finite metrics and final checkpoint SHA256. Runtime measures only the continuation including its evaluations, not the prior50k runtime.

Local artifacts: `output/antmaze_large_online60_20260914`; remote results: `results/taskN/dawn`. Reuse the existing completion monitor and pause after reporting all five tasks.
