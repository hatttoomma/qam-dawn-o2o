# UTD comparison figure

Question: at batch 256 and 50k primitive online steps, how does raising UTD from
0.25 to 1 change success and return on Task1 and Task2? Takeaway follows final
validated results, with one training seed and known rollout reproducibility limits.

Standalone scientific PNG/PDF using Matplotlib, consistent with earlier exported
experiment figures. Use two task rows and two metric columns (success and mean
return). Seven predeclared 50-episode checkpoints exist per arm; the original
runner has no denser evaluations. Preserve that schedule and use unconnected
ordered dots, not interpolated learning curves. Primary exact table uses final
100-episode assessments. Retain both old control and new measurements unchanged.

Single blue root: UTD 1 uses filled circles #2463A6; UTD 0.25 uses open squares
#82A9D0. Grey grid and a vertical dotted warmup boundary at 20k. Success axes
span 0--100%; return axes may differ across tasks with visible labels. Output
13x9 inches, clear title/subtitle, shared legend and footer. Footer states batch
256, 7,500 versus 30,000 updates, actor entropy retained and evaluation sample
sizes. No smoothing or across-seed uncertainty bands. Internal research artifact
without publication branding.

Inputs: 4 arms x 7 checkpoints, final 4x100 episodes and mean-residual 4x50
episodes; export enriched tables including task, UTD, batch, updates, step,
evaluation role, sample count, success and return. Output paths:
results_utd_ablation/evaluation_checkpoints.png and .pdf. Inspect the final PNG
for labels, marker distinctions, clipping and consistency with validated CSV.
