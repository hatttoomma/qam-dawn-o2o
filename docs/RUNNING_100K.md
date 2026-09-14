# Running and auditing the current experiments

## Verify or regenerate the published results

From the repository root, no SSH, GPU, dataset or checkpoint is needed for the first two commands:

```bash
python3 tools/verify_sources.py
python3 scale100k/validate_completed.py
python3 scale100k/plot_final.py
```

Plotting requires NumPy and Matplotlib. The default input is the frozen `reports/100k_20260914/final_complete_snapshot.json`; outputs go to the same report directory. To generate into a separate directory, give both report commands the same `--output-dir` (and `--snapshot` when using another snapshot). The plot reads that directory's newly generated validation summary.

## Training environment

The original runs used Python3.10, Linux and RTX4090. Install the appropriate requirements in a separate virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install -r requirements-antmaze.txt  # includes requirements.txt
export MUJOCO_GL=egl
```

`requirements.txt` remains the original Cube dependency file. The AntMaze addition pins the extra official-QAM imports to observed server versions; actual selected versions for both hosts are recorded in `provenance/20260914/*_runtime_versions.json`. CUDA and NVIDIA EGL are host dependencies; `scale100k/cube/egl_vendor.json` records the actual original NVIDIA library path and must be present on a compatible host. The Cube suite preserves OMP/BLAS thread counts1 and disables the CUDA command buffer; AntMaze suites preserve thread counts4. These settings are part of the measured source snapshot.

## Historical artifacts required to reproduce the continuations

These launchers restore measured historical states and deliberately stop when inputs or hashes do not match. `inputs.json` is the authoritative per-task checkpoint / parent mapping. Do not bypass those checks or treat a model-only checkpoint as a full continuation state.

| Stage | Required external artifacts |
|---|---|
| Cube native100k | Original per-task native50k `latest.pkl`, `final.pkl`, `DONE.json`, paired eval records, offline500k checkpoint, relabeled dataset audit |
| Cube DAWN100k | Original per-task offline500k checkpoint, dataset and `runs/cube5/TASK_AUDIT_PASSED.json`; starts a fresh residual optimizer and collector |
| AntMaze native100k | Original per-task `offline_500k.pkl`, `offline_checkpoint.json`, dataset manifest, actual agent config and fixed offline evaluation |
| AntMaze DAWN100k | Complete50k balanced state for task4; complete60k low-UTD state for tasks1/2/3/5; parent final/latest checkpoint, completion and integrity records, config, dataset manifest and eval records |

A complete residual state includes model/optimizer/alpha, replay, random-number generators, action/episode state and simulator restoration information. Model weights alone are insufficient. The new AntMaze native100k `latest.pkl` exports only99999 online transitions plus partial simulator state; it is **not** a valid exact-continuation artifact. Its measured endpoint remains valid.

Training and historical artifact paths are intentionally unchanged. When recreating the original host layout, copy the corresponding source directory from [SOURCE_INDEX.md](../SOURCE_INDEX.md) into its listed remote project root and restore the external artifacts there. Keep existing result directories intact; the suites are designed for empty target outputs and reject an existing partial or completed run. No source copy or training launch is required merely to inspect these results.

Once the original layout, datasets, EGL and all input hashes have been restored, the measured suite entry points are:

```bash
# AntMaze DAWN: /root/autodl-tmp/qam_antmaze_scale100k_dawn
/root/autodl-tmp/antmaze_qam_venv/bin/python -u dawn_suite.py
# AntMaze native: /root/autodl-tmp/qam_antmaze_scale100k_native
/root/autodl-tmp/antmaze_qam_venv/bin/python -u suite.py
# Cube: /root/qam_dawn_outputs/scale100k_20260914
/root/autodl-tmp/qam_dawn_o2o/venv/bin/python -u suite.py
```

Run each command from its indicated directory. The suites first run isolated smoke checks, then formal tasks serially. The AntMaze suites share a GPU lock. These are full training commands and consume GPU time; this repository update did not run them again.

For a new experiment with new checkpoint locations, use a new run directory and record any path/configuration changes in a new source/adapter manifest rather than editing the archived measured run in place. The current repository snapshot is an audit-preserving integration, not a new generic training framework.

## Monitoring

`python3 scale100k/fetch_status.py --full` reads the three original remote queues using already authenticated SSH sessions and writes to ignored `output/scale100k_live/`. Host/port/socket/root defaults remain in `QUEUES`; credentials are not stored. Live output is separate from the frozen report snapshot. All20 runs in the bundled report were already complete when this code was uploaded.
