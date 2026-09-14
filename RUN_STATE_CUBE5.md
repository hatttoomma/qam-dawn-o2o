# Completed cube-double five-task experiment

All new offline/online training and final evaluations completed 2026-09-09 15:38:48 UTC. The original export finished 15:39:13 UTC (23:39 Hong Kong time). Last process check 16:23:10 UTC: no training/launcher processes, no failures, GPU 0% utilization and 0 MiB allocated. No unfinished experiments in this suite.

Task1/2 reuse verified native results and runs/actor_input_ablation/task{1,2}_base_action; Task3–5 each have new task-specific offline500k and native50k/DAWN50k runs. Seed0. QAM-EDIT uses official cube-double inv_temp=1 and edit_scale=0. DAWN plain TD, batch256, UTD.25, state37+base25, inherited current/target Q, frozen base, scale.1, online-only replay, 20k warmup within 50k and 7,500 updates. Gaussian, actor entropy and automatic alpha retained. QAM has 45,001 updates.

Final100 success / mean return:
- Task1: QAM 99% / -68.66; DAWN 94% / -129.72.
- Task2: QAM 99% / -219.47; DAWN 58% / -479.38.
- Task3: QAM 99% / -209.53; DAWN 36% / -564.90.
- Task4: QAM 61% / -576.39; DAWN 8% / -883.07.
- Task5: QAM 98% / -252.88; DAWN 76% / -399.70.
- Equal task mean: QAM 91.2% / -265.386; DAWN 54.4% / -491.354.

DAWN mean-residual secondary50 success: Task1 94%, Task2 70%, Task3 36%, Task4 8%, Task5 80%. This changes evaluation only, not training. Initial/end repeat differences: zero. Complete final/latest agent states match for all ten endpoints. Five full warmup snapshots checked. One training seed per task/method; no across-seed superiority claim.

Final report: results_cube5/summary.md; comparison.csv and aggregate.csv; final_performance.png/pdf and evaluation_checkpoints.png/pdf. All 611 final archive members, 319 report sources and 58 frozen training/protocol sources locally verified. Both figures visually checked; endpoint left-label clipping fixed with a report-only layout change. Original verified export retained separately as results_cube5_initial_export_bundle.tar.gz; original extracted report under results_cube5_initial_export/. No model or training changes during this correction.

Final archive: results_cube5_bundle.tar.gz, 16,615,839 bytes, SHA256 10c203bcb026c3b076e1d5bbe81abb82d75861c4d4d167daad82bb7cc5f5fd22. Local receipt: results_cube5/local_verification.json.

Remote weights remain under /root/autodl-tmp/qam_dawn_o2o on root@connect.bjb1.seetacloud.com:44915; control socket /tmp/icra_qam_44915.sock. Python venv/bin/python. Runtime RTX4090/580.76.05, three concurrent new jobs at .30 GPU memory fraction. Training sources unchanged and frozen; all launcher stages ended successfully. Do not restart unless explicitly requested for new work.
