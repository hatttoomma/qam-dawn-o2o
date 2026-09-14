# 100k online comparison: all20 runs complete

Both benchmarks use tasks1–5, seed0 and100 paired evaluation episodes per task. DAWN mean residual is the primary comparison.

|Benchmark|Task|QAM native SR / return|DAWN mean SR / return|DAWN sampled SR / return|
|---|---:|---:|---:|---:|
|antmaze-large|1|97% / -407.22|96% / -584.57|97% / -588.14|
|antmaze-large|2|85% / -592.49|84% / -663.46|79% / -689.59|
|antmaze-large|3|97% / -297.00|93% / -361.08|90% / -378.36|
|antmaze-large|4|92% / -421.62|0% / -1000.00|0% / -1000.00|
|antmaze-large|5|94% / -416.03|90% / -488.80|92% / -496.31|
|cube-double|1|100% / -58.09|99% / -99.57|98% / -100.73|
|cube-double|2|99% / -190.43|89% / -297.63|91% / -301.07|
|cube-double|3|100% / -193.59|49% / -467.18|38% / -511.41|
|cube-double|4|93% / -426.80|5% / -925.98|7% / -872.39|
|cube-double|5|98% / -204.37|79% / -390.93|74% / -419.69|

Equal-weight macro averages across all5 tasks within each benchmark:

|Benchmark|QAM native SR / return|DAWN mean SR / return|DAWN sampled SR / return|
|---|---:|---:|---:|
|AntMaze-large|93.0% / -426.87|72.6% / -619.58|71.6% / -630.48|
|Cube-double|98.0% / -214.66|64.2% / -436.26|61.6% / -441.06|

Settings and interpretation:

- Same100k online environment budget; QAM native95001 online updates versus DAWN5625. Update counts are not equal compute.
- DAWN0–40k uses base-only rollout with no updates;40–50k uses128offline+128online atUTD0.25;50–100k uses256online atUTD0.0625.
- Naive TD, state+base action, inherited Q/target, frozen base, scale0.1,10critics/minimum, actor entropy and autoalpha remain fixed.
- Each task reuses its original offline500k checkpoint. Native Antmaze restarts online from that checkpoint; Cube native resumes the complete original50k state.
- Antmaze native export has99999 online transitions and partial environment state, so it is not an exact resume checkpoint. Its100k environment steps and95001 updates are verified.
- Return definitions differ: Antmaze counts unsuccessful steps; Cube sums per-object goal penalties. Do not combine raw returns across benchmarks.
- One training seed;100 evaluation episodes do not establish stability across training seeds.

Source: final_complete_snapshot.json, SHA256 70ee02f29c9dbc6883f3447e8a1f7809d5d62f66468db246ee888d0f9da68a87. Completed results passed source/checkpoint hashes, schedule, update, replay, aggregation and paired-state checks.
