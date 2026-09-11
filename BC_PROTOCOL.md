# BC pretraining replacement: four online arms

User approved four arms, all with randomly initialized critics. Task `cube-double-play-singletask-task1-v0`, seed 0, offline 500,000 gradient updates, online 50,000 primitive environment steps. All earlier QAM-pretrained results remain unchanged.

## Offline BC and transfer

Pure flow-matching behavior cloning on the same 1m offline transitions. Policy architecture is the exact QAM `ActorVectorField`: 4×512 GELU, no LayerNorm, observation dimension 37, action chunk dimension 25 (5 primitive actions × horizon 5). Match Gaussian noise to dataset actions using uniform interpolation time; mean squared velocity error multiplied by QAM's last-step valid-sequence mask. Batch 256, Adam 3e-4, betas (0.9,0.999), eps 1e-8, global norm clip 1; no critic, rewards, adjoint matching, or Q guidance in the objective. Initialize from the same seed-0 QAM slow-flow random weights; sample with 10 Euler steps and clip actions to [-1,1]. Select the last fixed 500k checkpoint regardless of evaluation.

At the offline→online transition copy the trained BC flow into QAM's slow, fast, target-slow, and target-fast slots. All slots therefore produce the BC policy initially. Use QAM best_of_n=1, inv_temp=1, residual=False, edit_scale=0, fql_alpha=0. With one candidate, critic values cannot affect action selection.

All four online arms use the **same fresh random critic draw** as the earlier random-critic DAWN implementation (`seed + 71000`), with target=current critic. BC never trains these weights. An untrained placeholder is stored in the transfer checkpoint for compatibility with existing QAM checkpoint loaders. No QAM-pretrained critic is loaded. All online optimizer states start fresh. In particular the native QAM arm has no offline QAM optimizer to inherit; its BC actor weights are transferred, and its combined online optimizer is newly initialized. This transition detail is explicit when comparing to the old native arm, which continued its full QAM offline optimizer.

## Online arms (all random Q)

| Arm | Base policy | Replay | Online settings |
|---|---|---|---|
| BC → native QAM | BC-initialized QAM, trainable | Original QAM uniform offline+online primitive sequences | Original QAM losses; lr 3e-4, batch 256, UTD 1, update starts 5k, tau .005, mean-minus-.5-std target/mean actor Q, no entropy TD; 45,001 updates |
| BC → DAWN, online-only | Frozen BC | Original DAWN executed decision chunks, online-only | Unchanged DAWN: 20k base-only warmup, lr 1e-4, batch 1024, UTD .25, tau .01, alpha0 .01 autotune, residual scale .1; 7,500 updates |
| BC → DAWN, QAM replay | Frozen BC | Original QAM offline+online uniform primitive sequence sampler + cached BC proposals | Same DAWN settings and critic valid mask as the previous QAM-replay experiment; 7,500 updates |
| BC → Policy Decorator, QAM replay | Frozen BC | Same QAM replay + cached BC proposals | Unchanged PD adaptation: learning starts 8k, progressive p(t)=min(t/30k,1) per chunk, pre-learning gated uniform residual, alpha0=1, lr 1e-4, batch 1024, UTD .25, tau .01, scale .1; 10,500 updates |

Shared residual backbone: state-only 3×256 ReLU, tanh Gaussian, log std in [-20,2], .01 orthogonal output heads, target entropy -25, global gradient norm clip 50. Target and actor Q both use minimum across all 10 QAM critics (4×512 GELU + LayerNorm). SAC entropy remains in residual TD targets. Gamma .99 per primitive step; same reward, termination, timeout, action clipping, sequence mask and action chunk semantics as before. A new BC-only offline proposal cache is built; never reuse QAM-pretrained policy proposals.

## Evaluation and checks

Offline evaluation: 0/100k/250k/500k, 50 fixed reset episodes. Online primary grid: 0/5k/10k/20k/30k/40k/50k, 50 episodes, final 100 episodes. Primary residual evaluation continues stochastic residual sampling; final mean-residual diagnostics follow each previous arm's evaluation protocol (DAWN: 50, PD: both 50 and 100, PD mean curves at every grid point). Base policy remains stochastic. Training and evaluation randomness stay separate.

Check pure-BC reward independence, valid masking, exported policy equality, unchanged random critics throughout pretraining, identical random Q initialization across all four online arms, new BC cache provenance, initial evaluations, final frozen-base hashes, finite metrics, update counts, replay counts, and source/evaluation hashes. Meaningful unit tests and small offline/online end-to-end runs precede formal runs. Report all ten arms with matched comparisons: DAWN random-Q old vs BC for each replay; native and PD comparisons have the documented initialization/optimizer differences. One seed supports exploratory trends, not statistical claims.
