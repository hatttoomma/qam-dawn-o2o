# Task2 comparison figures

Use standard Matplotlib PNG/PDF with two panels: success rate and mean return
against primitive online environment steps, 0 to 50k. Plot only the seven
measured 50-episode points for each of the two task2 arms. Grey circles for
native QAM, blue squares with a dashed line for pretrained-Q DAWN with online-only replay. No
smoothing, extrapolation or seed uncertainty bands. Mark DAWN's 20k warmup
boundary, and distinguish 50-episode curves from the final 100-episode table.

Title and caption must identify cube-double task2, training seed 0, the new
shared 500k offline checkpoint, chunk length 5 and 50k online environment steps.
Keep task1's historical 50k results in a separate comparison table, not mixed
into the task2 curves. Do not substitute the extended task1 DAWN run.

Verify raw episode aggregates, evaluation seeds, checkpoint/task manifests,
source hashes, actual delayed chunk boundaries and final100/first50 agreement
before plotting. Visually inspect the resulting PNG before sharing it.
