# Batch-size experiment figures

Question: at fixed ordinary-TD objective, interaction budget and update count,
how does batch 256 compare with batch 1024 on each cube-double task?
Takeaway is determined only after validation of the four completed runs.

Surface: standalone scientific PNG/PDF figures alongside the experiment report,
using Matplotlib, consistent with the existing exported experiment artifacts.
Primary table reports final 100-episode success and mean return. Figure reports
the seven predeclared 50-episode evaluations per arm, with task as row and metric
as column. The source runner was checked for denser evaluations: only the seven
fixed checkpoints exist. Preserve the experimental evaluation schedule and use
unconnected ordered dots, not an interpolated continuous learning curve.

Data: 4 arms x 7 checkpoints = 28 aggregate rows, backed by episode-level data;
4 x 100 final episodes, 4 x 50 mean-residual diagnostic episodes. Retain step,
task, batch, evaluation role, sample count, success, return and initialization
metadata in exported tables. No fabricated intermediate observations, smoothing,
best-checkpoint selection or across-seed uncertainty bands.

Palette policy: single blue root (#2463A6) with supporting shade #82A9D0 and
neutral axes/grid. Batch 256 uses filled circles; 1024 uses open squares. The
same mapping applies to both tasks and metrics. Success axes are 0--100%; return
axes may differ between tasks and are visibly labeled. A dotted vertical guide
marks the 20k warmup boundary. Overlapping warmup dots are intentional.

Footprint: 13 x 9 inches, two rows/two columns, title/subtitle and shared legend.
Caption states ordinary TD with actor entropy retained, seed 0, scale 0.1,
online-only replay, 50 episodes per point, and 7,500 updates per arm.
Files: results_td_batch_ablation/evaluation_checkpoints.png and .pdf.
QA: inspect the final PNG, ensure labels/legend/caption do not overlap, and
check all plotted values against validated CSV. This is an internal experiment
artifact without publication branding.
