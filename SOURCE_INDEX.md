# Source index: completed 100k online experiments

The repository keeps the original Cube experiment files at the root. New AntMaze stages and the final two-benchmark experiment live in the directories below. Training sources are byte-identical to the retrieved server files, including their pinned official / legacy dependencies. Duplicate dependencies intentionally preserve each stage's original source manifest and import resolution.

| Repository path | Original remote project | Role |
|---|---|---|
| repository root | `/root/autodl-tmp/qam_dawn_o2o` | Cube offline 500k, native 50k, historical residual ablations |
| `antmaze/` | `/root/autodl-tmp/qam_antmaze` | Task1 official QAM 500k + 50k; initial DAWN |
| `antmaze_remaining/` | `/root/autodl-tmp/qam_antmaze_task2_5` | Tasks2–5 official QAM and initial DAWN |
| `antmaze_balanced/` | `/root/autodl-tmp/qam_antmaze_balanced40k` | All5 tasks: 40k warmup +10k balanced replay |
| `antmaze_online60/` | `/root/autodl-tmp/qam_antmaze_online60` | 50→60k online-only continuation at UTD0.25 |
| `antmaze_online60_utd00625/` | `/root/autodl-tmp/qam_antmaze_online60_utd00625` | Tasks1/2/3/5: 50→60k at UTD0.0625 |
| `scale100k/antmaze/` | `/root/autodl-tmp/qam_antmaze_scale100k_dawn` | DAWN complete-state continuation to100k; task4 starts at50k, others at60k |
| `scale100k/antmaze_native/` | `/root/autodl-tmp/qam_antmaze_scale100k_native` | Native fresh online100k from the same offline500k checkpoints |
| `scale100k/cube/` | `/root/qam_dawn_outputs/scale100k_20260914` | Native50→100k plus fresh DAWN40k/mixed50k/low-UTD100k |

## Integrity and ownership

- [Remote-source inventory](provenance/20260914/remote_sources.json): 500 retrieved files with original path, SHA256, size and host label. Root remote README / gitignore are preserved under `docs/REMOTE_README.md` and `docs/REMOTE_GITIGNORE.txt`; the root README is the curated entry point.
- [Original import](REMOTE_SOURCE_MANIFEST.json): retained without alteration. All148 previously imported remote source files are still unchanged at their preserved paths.
- [Local additions](provenance/20260914/local_additions.json): original hashes of reporting tools and protocols. [Packaging changes](provenance/20260914/packaging_changes.json) separately records the three report-tool path adaptations. No optimizer, rollout, replay, loss, evaluation or launcher code is refactored for upload.
- [Final experiment snapshot](reports/100k_20260914/final_complete_snapshot.json): original frozen data, SHA256 `70ee02f29c9dbc6883f3447e8a1f7809d5d62f66468db246ee888d0f9da68a87`. It includes all final episode records, actual configurations, update/replay audits, source hashes and final checkpoint hashes; it contains no model weights or replay samples.
- `tools/verify_sources.py` verifies retrieved source hashes, packaging hashes, Python syntax, and all three final-run source manifests against repository files. `scale100k/validate_completed.py` independently checks all20 run endpoints and paired evaluation states.

Historical protocols describe the budgets used at that time. The current entry point is [scale100k/PROTOCOL.md](scale100k/PROTOCOL.md), supplemented by [RUNNING_100K.md](docs/RUNNING_100K.md). Historical `fetch_*.py` and plotting scripts outside `scale100k/` retain original machine/workspace paths; they are archived tools, not portable CLI interfaces. The three current report/monitor entry points use repository-relative output defaults.

The local curation branch preserves original training sources. Datasets, checkpoints, replay, virtual environments and videos remain outside Git. QAM upstream license files are retained in every pinned source tree; project-specific code retains its existing licensing status.
