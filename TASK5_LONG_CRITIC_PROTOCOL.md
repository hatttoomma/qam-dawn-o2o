# Task5: critic initialization × long online budget

Authorized 2026-09-10. Two fresh seed-0 runs start from the same frozen QAM 500k offline checkpoint. The only between-arm change is online critic initialization: inherited current and target Q, or random current Q with target equal to current Q. Both reset optimizer states. Both use identical actor/alpha initialization and RNG streams; verify identical base-only warmup replay before interpreting results.

Environment: `cube-double-play-singletask-task5-v0`. Offline checkpoint SHA256: `fc0ac82260d16262da8e0c1d7266fd19fc024dfc52c4b40b26e64b6463a1938a`.

| Setting | Both arms |
|---|---|
| Total online budget | 500,000 primitive environment steps, including warmup |
| Base-only warmup | 20,000 primitive steps, no gradient updates |
| Update/data ratio | 0.25 updates per primitive step after warmup |
| Final gradient updates | 120,000 |
| Replay | Uniform sampling from growing online-only chunk replay |
| Batch size | 256 chunk transitions |
| Critic TD target | Ordinary TD, no target entropy bonus |
| Actor objective | Maximize minimum Q plus SAC entropy; automatic alpha retained |
| Actor input | State (37) + the sampled base action chunk (25) |
| Actor | 3 × 256 ReLU, tanh Gaussian residual |
| Residual scale | 0.1, final action clipped to [-1, 1] |
| Critics | 10 × (4 × 512 GELU + LayerNorm) |
| Q aggregation | Minimum of all 10 for target and actor |
| Actor/critic/alpha LR | 1e-4 |
| Target Polyak tau | 0.01 |
| Initial alpha; target entropy | 0.01; -25 |
| Actor/critic gradient clipping | Global norm 50 per optimizer |
| Action chunk; primitive discount | 5; 0.99 |
| Base policy | Frozen QAM, edit_scale=0, inv_temp=1 |

Retain the original `run.py` residual collection/update loop and original conditioned agent implementation, with the same hard-TD substitution used in the existing Task5 experiment. Do not overwrite historical sources or results. The new wrapper only changes total budget, evaluation/checkpoint grid, and initialization arm.

Evaluate offline and at 50k, 100k, 200k, 500k using the same 100 reset seeds (500000–500099), base RNG seeds (800000–800099), and residual RNG seeds (900000–900099) as prior evaluation. Report sampled residual and tanh(mean) separately; the base policy is sampled in both. Record success rate, average return, episode lengths and paired success gains/losses versus offline. No checkpoint selection or hyperparameter selection on these evaluations. This is one training seed and does not establish robustness across seeds.

Milestone evaluations occur at the first complete chunk crossing each threshold, as in the historical loop; record the actual primitive step (at most four steps later). Final training stops at exactly 500k. Artificial partial chunks cut by the final budget are not inserted into replay. Terminal chunks use their executed length and zero bootstrap only for true termination; time limits retain bootstrap.

Retain slim milestone checkpoints and a latest checkpoint containing replay, optimizer states, RNGs, episode trace, and simulator reconstruction state. Before launch, run short inherited/random checks, verify matching non-Q initialization and warmup replay, and compare an interrupted/resumed inherited run with an uninterrupted run at the same final budget. Test interruptions occur only at complete-chunk checkpoints so they do not alter action execution.

Output: `runs/task5_long_critic_shared_20260910/{warm,random}`. At 50k, 100k, 200k and 500k the nominal cumulative updates are 7,500, 20,000, 45,000 and 120,000 respectively.

Launch amendment: independent-process warmup replay byte hashes did not consistently agree despite fixed seeds and matching chunk counts. The exact numerical source is unresolved. To guarantee the intended paired-data comparison, collect one base-only warmup with zero gradient updates using the unchanged collection loop; stop at its complete-chunk boundary (Task5: 20,003 steps). Branch the identical replay, episode trace, observation, rollout keys, and NumPy RNG state into both arms, attaching each arm's freshly initialized agent and its intended critic/target. Reconstruct and verify the simulator state through the existing restore procedure. The offline evaluation is shared because the policy is identical. Each arm then independently collects the remaining 479,997 primitive steps. Both still have a logical 500k online budget and exactly 120k gradient updates. This changes how the common prefix is shared, without changing either arm's training equations or hyperparameters. Preserve interrupted attempts separately; use a new output directory and recheck short-run branching and exact resume before formal launch.
