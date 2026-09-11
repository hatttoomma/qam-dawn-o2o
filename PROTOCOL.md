# QAM / QAM + DAWN, cube-double task1, seed 0

User-requested exploratory single-seed experiment, 2026-09-06.

## Fixed user choices

- Task: cube-double-play-singletask-task1-v0.
- Offline: 500,000 gradient updates; online: 50,000 primitive environment steps.
- Action horizon: 5, fully executed before replanning, for all arms.
- Seed: 0, explicitly chosen as a workflow/trend pilot.
- Official cube-double QAM_EDIT uses edit_scale=0, inv_temp=1. The user explicitly chose to preserve this degenerate setting. The baseline is therefore QAM with native online continuation, not a baseline with an active editor.
- All three arms consequently share the exact same offline checkpoint.
- DAWN: SAC entropy in the TD target; 10 critics; minimum across all 10 for both target backup and residual actor optimization.
- Replay: native QAM uses uniform offline+online replay; DAWN uses online-only replay, including base-policy warmup data.

## Implementation choices

QAM source: ColinQiyangLi/qam at 2726d767c9a0a7a46d49693f0391f73dc2cf58ac.
DAWN reference: Guozheng-Ma/DAWN at 71122b7fa89568bc2d49831fed8cb1b01e5a91e8.
The DAWN residual and losses are ported to JAX, using the existing QAM critic architecture so that warm versus random initialization changes only critic weights.

Common task semantics: unmodified OGBench task rewards, discount 0.99 per primitive step, gamma^5 chunk backup, timeouts bootstrap and true terminations do not. Critic architecture: the QAM 10-member, four-hidden-layer, 512-width GELU MLP with LayerNorm. Flow sampling: 10 Euler steps, one candidate, no Q search. All running flows and offline BC priors are frozen in DAWN arms.

Native baseline: unchanged QAM losses, batch 256, lr 3e-4, UTD 1 per primitive step, target tau .005, mean-minus-.5-std target, mean actor Q, starts updating at online step 5,000. Preserve complete offline agent including optimizer and target networks. Replay contains offline data plus all online transitions; use original QAM sequence sampler.

DAWN arms: fresh online residual network, state-only input, 3x256 ReLU backbone following the released DAWN code, orthogonal .01 mean/log-std heads, log-std mapped via tanh to [-20,2], tanh-Gaussian unscaled residual, action scale .1. Critic and actor lr 1e-4, batch 1024, UTD .25 per primitive step, target tau .01, norm clipping 50, automatic entropy coefficient initialized .01, target entropy -25 (whole unscaled chunk). New optimizer states in both arms. Warm arm copies current and target Q from the offline checkpoint. Random arm uses new Q weights and target=current Q. Residual and entropy initial states are identical across arms.

First 20,000 online primitive steps: frozen QAM only, no residual/critic/temperature updates. These steps ARE included in the total 50,000 steps. No progressive exploration or explicit critic-only warmup. Both DAWN arms use the same seeded base-only prefix. Replay stores actually executed chunks, discounted rewards, bootstrap discounts, base proposals and next-state base proposals, following DAWN's cached-base design. Base proposals are generated from the frozen stochastic QAM flow. Critic sees the clipped executed action; actor/target actions use the same final clipping. Entropy is conditional residual entropy before multiplication by .1; base-policy mixture entropy is not estimated.

## Evaluation and integrity

Evaluation grid: 0, 5k, 10k, 20k, 30k, 40k, 50k online steps. For residual runs, evaluate at the first completed chunk at/after each intermediate threshold, recording the actual step (at most four steps later); the final budget is exactly 50k. The base-only warmup similarly ends at the first chunk boundary at/after 20k. This preserves uninterrupted chunk execution despite early episode termination. 50 fixed-reset episodes per point; 100 episodes for the final additional evaluation. Training and evaluation RNGs are separate. Primary common evaluation uses stochastic flow samples and sampled residuals, with residual disabled during the warmup prefix. Also record deterministic-residual evaluation at the final checkpoint as a secondary DAWN-specific diagnostic. No evaluation transitions enter replay. Report success, return, time, AUC and individual episode records; a single seed cannot establish statistical superiority.

Offline evaluation at 0, 100k, 250k, 500k, with the last fixed checkpoint used online regardless of score. Source hashes, dataset hashes, package freeze, configs, offline checkpoint hash, frozen flow hashes, optimizer initialization and final hashes, and warmup replay hash are persisted. Numerical and integration checks must pass before full runs start. Intermediate resumable checkpoints are retained.
