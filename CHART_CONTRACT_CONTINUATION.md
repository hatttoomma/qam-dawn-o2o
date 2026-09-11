# DAWN continuation figures

Use standard Matplotlib PNG/PDF with four panels: success and mean return
against primitive online environment steps, and against cumulative online
optimizer updates. Plot measured 50-episode checkpoints only, no smoothing.
Grey native QAM; blue DAWN with distinct continuation markers. Label the 50k
resume boundary and the 45,001-update matching point. Native's horizontal
reference, if drawn, must be explicitly identified as its final measured score.
Show exact 100-episode checkpoint results separately in a CSV/Markdown table.
No seed uncertainty bands: all data use training seed 0. Verify hashes and
episode-level aggregates, identical restored evaluation outcomes, exact
milestone updates and first-50 agreement with the 100-episode evaluations.
State that continuation increases environment data and uses batch 1024 versus
native's batch 256. Visually inspect the rendered PNG before sharing it.
