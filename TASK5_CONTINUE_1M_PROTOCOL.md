# Task5: continue the completed 500k runs to 1M online steps

Authorized: extend both inherited-Q and random-Q Task5 arms. Continue their corresponding completed states from `runs/task5_long_critic_shared_20260910/{warm,random}/latest.pkl`; preserve all agent parameters, target networks, optimizer and alpha states, replay records, episode trace, observation, rollout keys, NumPy RNG and update counters. Verify the agent state in latest equals the evaluated final checkpoint. Do not reset the agent, critic, buffer or warmup. Historical outputs and training sources remain unchanged.

- Task: `cube-double-play-singletask-task5-v0`, seed 0, frozen QAM 500k offline checkpoint.
- Target: 1,000,000 total primitive online steps, adding 500,000 to each arm.
- Original warmup threshold remains 20,000; no additional warmup. Final cumulative updates: 245,000, adding 125,000.
- Keep ordinary TD (no target entropy bonus), actor entropy and automatic alpha, batch256, UTD0.25 per primitive step, state37+base chunk25 input, scale0.1, online-only uniform growing chunk replay, horizon5, gamma0.99, min10 critic aggregation, and all architectures/optimizer settings unchanged.
- New evaluations: 750k and 1M, each with the same 100 fixed episodes, sampled and tanh(mean) residual separately. Base actions remain sampled. Nominal cumulative updates: 182,500 and 245,000. Retain slim milestone weights and full latest state. Complete-chunk intermediate milestones may overshoot by up to four primitive steps; final budget is exact.
- Retain the existing 500k evaluations as the continuation's starting reference. Report success/return and paired gains/losses versus both offline and the corresponding 500k policy.

Budget-boundary detail: the historical runner could truncate a chunk at exactly 500k, omitted artificial partial chunks from replay, and did not save the unexecuted action suffix. Restore the saved environment state and replan one chunk at 500k; do not invent missing replay or reset the episode. This is continuation of the actual saved run, not a claim of bitwise equivalence to an uninterrupted run originally configured for 1M. Diagnose and record whether the last recorded next observation matches the saved current observation. The optimizer, replay and counter states must otherwise remain exact.

Before formal launch, run each arm for 120 additional primitive steps and verify the extra update count is 30. Compare a short interrupted/resumed inherited-Q continuation against uninterrupted execution from the same 500k state; compare agent, replay, episode trace, observation and RNG state. Test pauses happen at complete-chunk checkpoints with an unchanged final smoke budget.

Outputs: `runs/task5_continue_1m_20260910/{warm,random}`. Preserve source checkpoint hashes and immutable source manifests. This remains one training seed; do not infer a stable performance advantage from small differences.
