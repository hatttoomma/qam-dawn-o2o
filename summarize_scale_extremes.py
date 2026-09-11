"""Validate both tasks and report every measured DAWN residual scale."""
import json
from pathlib import Path

import numpy as np
import summarize_task2_scale as evidence
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/scale_extremes'
OUT = ROOT / 'results_scale_extremes'
load, checked, digest, track = evidence.load, evidence.checked, evidence.digest, evidence.track
SOURCES = evidence.sources


def task_root(task): return ROOT / ('runs' if task == 1 else 'runs/task2')
def arms(task):
    result = [(.01, RUNS / f'task{task}_scale001', True),
              (.1, task_root(task) / 'warm', False),
              (1., RUNS / f'task{task}_scale100', True)]
    if task == 2:
        result += [(scale, ROOT / 'runs/task2/scale_sweep' / name, False)
                   for scale, name in [(.05, 'scale005'), (.2, 'scale020'), (.3, 'scale030')]]
    return sorted(result)


def main():
    OUT.mkdir(exist_ok=True)
    for n, h in load(RUNS / 'source_manifest.json').items(): assert digest(ROOT / n) == h, n
    assert load(RUNS / 'CHECKS_PASSED.json')['status'] == 'passed'
    audit = load(ROOT / 'runs/task2/TASK_AUDIT_PASSED.json');assert audit['status'] == 'passed'
    warmup_audit = load(RUNS / 'warmup_audit/audit.json');assert warmup_audit['status'] == 'passed'
    assert warmup_audit['extraction_source_sha256'] == digest(ROOT / 'audit_scale_warmup.py')
    track(ROOT / 'audit_scale_warmup.py')
    assert warmup_audit['raw_observation_atol'] == 1e-10 and warmup_audit['training_float32_atol'] == 1e-12
    for name, entry in warmup_audit['arms'].items():
        p = ROOT / entry['snapshot'];track(p);assert digest(p) == entry['snapshot_sha256']
        ref = warmup_audit['arms'][entry['comparison_with']]
        with np.load(p) as a, np.load(ROOT / ref['snapshot']) as b:
            assert set(a.files) == set(b.files)
            for k in a.files:
                x, y = a[k], b[k];assert x.shape == y.shape and x.dtype == y.dtype
                assert np.isfinite(x).all() and np.isfinite(y).all()
                xf, yf = x.astype(np.float32), y.astype(np.float32)
                if k in ('observations', 'next_observations'):
                    np.testing.assert_allclose(x, y, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(xf, yf, rtol=0, atol=1e-12)
                else: np.testing.assert_array_equal(x, y)
                assert entry['diagnostics'][k] == dict(raw_changed_elements=int(np.sum(x != y)),
                    raw_max_abs_diff=float(np.max(np.abs(x - y))),
                    float32_changed_elements=int(np.sum(xf != yf)),
                    float32_max_abs_diff=float(np.max(np.abs(xf - yf))))
    curves, natives, rows, evals, episodes, pairs = {}, {}, [], [], [], []
    historical_warmup_eval_differences = []
    for task in (1, 2):
        root = task_root(task);curves[task] = {}
        offline = load(root / 'offline/DONE.json');assert offline['step'] == 500000
        off_eval = checked(root / 'offline/eval_500000_050.json')
        cfg_ref = load(root / 'warm/config.json');init = load(root / 'warm/initial_hashes.json')
        warmup = load(root / 'warm/warmup.json');ref = checked(root / 'warm/eval_050000_100.json')
        assert init['critic'] == offline['q_hash'] and init['target'] == offline['target_q_hash']
        native = checked(root / 'native/eval_050000_100.json')
        native_pts = [checked(p) for p in sorted((root / 'native').glob('eval_*_050.json'))]
        assert len(native_pts) == 7 and native['records'][:50] == native_pts[-1]['records']
        assert load(root / 'native/DONE.json')['updates'] == 45001
        natives[task] = (native_pts, native)
        for scale, folder, new in arms(task):
            cfg = load(folder / 'config.json');done = load(folder / 'DONE.json')
            for k, v in cfg_ref.items():
                if k != 'out': assert cfg[k] == v, (task, scale, k)
            assert set(cfg) - set(cfg_ref) <= {'task', 'residual_scale', 'environment'}
            assert cfg['offline_sha256'] == offline['checkpoint_sha256']
            assert cfg['online_steps'] == done['steps'] == 50000 and done['updates'] == 7500
            assert load(folder / 'initial_hashes.json') == init
            current_warmup = load(folder / 'warmup.json')
            if new:
                assert {k: v for k, v in current_warmup.items() if k != 'replay_hash'} == {
                    k: v for k, v in warmup.items() if k != 'replay_hash'}
                assert warmup_audit['arms'][folder.name]['warmup_replay_hash'] == current_warmup['replay_hash']
            else: assert current_warmup == warmup
            assert done['final_flow_hash'] == init['flow'] == offline['flow_hash']
            assert done['final_critic_hash'] != init['critic'] and done['final_actor_hash'] != init['actor']
            if scale != .1:
                actual = load(folder / 'actual_agent_config.json');selected = load(folder / 'scale_config.json')
                assert actual['residual_scale'] == selected['residual_scale'] == cfg['residual_scale'] == scale
                assert actual['action_dim'] == 25 and actual['tau'] == .01
                assert actual['critic_hash'] == init['critic'] and actual['target_hash'] == init['target']
                assert actual['actor_hash'] == init['actor']
                manifest = load(folder / 'task_manifest.json')
                for k in ['environment', 'reward_task_id', 'target_cube_xyzs', 'rewards_hash', 'masks_hash']:
                    assert manifest[k] == audit['tasks'][task - 1][k], (task, scale, k)
                if new: assert actual['task'] == cfg['task'] == task
            if (folder / 'final.pkl').exists(): assert digest(folder / 'final.pkl') == done['checkpoint_sha256']
            p = folder / 'metrics.jsonl';track(p)
            metrics = [json.loads(s) for s in p.read_text().splitlines()]
            for m in metrics:
                for v in m.values():
                    if isinstance(v, (int, float)): assert np.isfinite(v)
            last = metrics[-1];assert last['step'] == 50000 and last['updates'] == 7500
            assert 0 <= last['residual_abs_mean'] <= scale + 1e-6
            pts = [checked(p) for p in sorted(folder.glob('eval_*_050.json'))];assert len(pts) == 7
            for nominal, e in zip([0, 5000, 10000, 20000, 30000, 40000, 50000], pts):
                assert nominal <= e['step'] <= min(nominal + 4, 50000)
                assert not e['deterministic_residual']
                assert [r['initial_hash'] for r in e['records']] == [r['initial_hash'] for r in off_eval['records']]
            for e in pts[:4]:
                if task == 1 and scale == .1 and e['step'] == 20004:
                    # Preserve this already-existing raw evaluation discrepancy.
                    # All other episode fields and all new base-only evals match.
                    differences = [dict(episode=a['episode'], field=k, recorded=a.get(k), offline=b.get(k))
                        for a, b in zip(e['records'], off_eval['records'])
                        for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
                    assert differences == [dict(episode=33, field='return_', recorded=-530., offline=-533.)]
                    historical_warmup_eval_differences.append(dict(task=task, scale=scale,
                        step=e['step'], differences=differences, mean_return_difference=.06))
                else: assert e['records'] == off_eval['records']
            final = checked(folder / 'eval_050000_100.json')
            mean = checked(folder / 'eval_050000_050_mean_residual.json')
            assert final['records'][:50] == pts[-1]['records']
            assert not final['deterministic_residual'] and final['residual_enabled']
            assert mean['deterministic_residual'] and mean['residual_enabled'] and mean['episodes'] == 50
            assert [r['initial_hash'] for r in final['records']] == [r['initial_hash'] for r in ref['records']]
            assert [r['initial_hash'] for r in mean['records']] == [r['initial_hash'] for r in off_eval['records']]
            curves[task][scale] = pts
            row = dict(task=task, residual_scale=scale, new_experiment=new, seed=0,
                offline_updates=500000, online_primitive_steps=50000, online_updates=7500,
                final_episodes=100, success_percent=100 * final['success'], return_mean=final['return_mean'],
                mean_episode_length=float(np.mean([r['length'] for r in final['records']])),
                curve_final_success_percent=100 * pts[-1]['success'], mean_residual_episodes=50,
                mean_residual_success_percent=100 * mean['success'], mean_residual_return_mean=mean['return_mean'],
                success_auc_percent=float(np.trapz([e['success'] for e in pts], [e['step'] for e in pts]) / 500),
                replay_chunks=last['replay_chunks'])
            for key in ['residual_abs_mean', 'action_clip_fraction', 'alpha', 'entropy', 'logstd', 'critic_grad_norm']:
                row['last_train_' + key] = last[key]
            rows.append(row)
            for role, data in [('curve50', pts), ('final100', [final]), ('mean_residual50', [mean])]:
                for e in data:
                    evals.append(dict(task=task, scale=scale, role=role, primitive_steps=e['step'],
                        episodes=e['episodes'], success_percent=100 * e['success'], return_mean=e['return_mean']))
                    for rec in e['records']: episodes.append(dict(task=task, scale=scale, role=role, primitive_steps=e['step'], **rec))
            a, b = final['records'], ref['records']
            pairs.append(dict(task=task, scale=scale, reference_scale=.1, episodes=100,
                both_success=sum(bool(x['success']) and bool(y['success']) for x, y in zip(a, b)),
                scale_only_success=sum(bool(x['success']) and not bool(y['success']) for x, y in zip(a, b)),
                reference_only_success=sum(not bool(x['success']) and bool(y['success']) for x, y in zip(a, b)),
                neither_success=sum(not bool(x['success']) and not bool(y['success']) for x, y in zip(a, b))))
    for filename, data in [('comparison.csv', rows), ('evaluations.csv', evals), ('episodes.csv', episodes), ('paired_vs_scale010.csv', pairs)]:
        evidence.write_csv(OUT / filename, data)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    plot(curves, natives, rows)
    lines = ['# Task1 / Task2: DAWN residual-scale 扩展', '',
        '每任务复用原有 offline 500k checkpoint；seed 0；继承 Q；50k primitive online steps / 7,500 updates。仅改变 scale。', '',
        '| Task | Scale | 新增 | 成功率（100 eps，sampled） | 平均回报 | 均值残差成功率（50 eps） |',
        '|---|---:|---|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['task']} | {r['residual_scale']:g} | {'是' if r['new_experiment'] else '原有'} | {r['success_percent']:.0f}% | {r['return_mean']:.2f} | {r['mean_residual_success_percent']:.0f}% |")
    lines += ['', 'Native 参考（100 eps、50k primitive steps、45,001 updates）：', '']
    for task in (1, 2):
        e = natives[task][1];lines.append(f"- Task{task}: {100 * e['success']:.0f}% / {e['return_mean']:.2f}")
    lines += ['', '保持原来的 frozen QAM、state-only actor、min10、联合 critic clip=50、SAC entropy TD、20k base-only warmup、online-only replay、batch=1024、UTD=.25/primitive step、lr=1e-4、tau=.01。',
        '曲线各点 50 episodes，最终主表 100 episodes。均值残差为独立的 50-episode 诊断，base 仍随机。',
        'Warmup 审计：初始化权重、步数、动作、奖励与折扣匹配；部分模拟器状态有浮点末位差异，原始 replay 哈希因此可能不同。保存了完整 warmup 数组快照，并验证原始状态绝对差 ≤1e-10、转为训练 float32 后差 ≤1e-12。详见 runs/scale_extremes/warmup_audit/audit.json。',
        '保留历史原始数据：Task1 原 scale=.1 在 20k 的第 33 号评估 episode 回报为 -530，offline 对应值为 -533，因此该点平均回报相差 .06；成功率、长度、初始状态一致。本轮四组所有 warmup 评估记录均与各自 offline 评估完全一致。',
        'Task1 没有运行 .05/.2/.3，图表不填补这些值。只有一个 training seed，不能确认通用最优 scale。', '',
        '![Extreme-scale learning curves](learning_curves.png)', '', '![All measured scales](all_scales.png)', '']
    (OUT / 'summary.md').write_text('\n'.join(lines))
    track(Path(evidence.__file__).resolve())
    (OUT / 'validation.json').write_text(json.dumps(dict(status='passed', sources=SOURCES,
        historical_warmup_eval_differences=historical_warmup_eval_differences,
        report_source_sha256=digest(Path(__file__)), checks=['historical training files unchanged',
        'scale010 wrapper parity on both tasks', 'correct task checkpoints and reward hashes',
        'only scale changes in training configuration', 'effective static scales persisted',
        'identical initial weights; audited warmup numerical equivalence per task', 'frozen flow unchanged', 'exact budgets',
        'finite metrics and verified episode means', 'paired initial states and seeds',
        'final100 first50 equals learning-curve endpoint', 'no fabricated missing task1 scales']), indent=2) + '\n')
    print(json.dumps(rows, indent=2), flush=True)


def plot(curves, natives, rows):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    styles = [(.01, '#82A9D0', 'o', ':'), (.1, '#2463A6', 's', '--'), (1., '#164A7B', '^', '-')]
    for task in (1, 2):
        for col, key, mul, label in [(0, 'success', 100, 'Success rate (%)'), (1, 'return_mean', 1, 'Mean episode return')]:
            ax = axes[task - 1, col]
            for scale, color, marker, ls in styles:
                pts = curves[task][scale]
                ax.plot([e['step'] / 1000 for e in pts], [e[key] * mul for e in pts],
                    label=f'Scale {scale:g}', color=color, marker=marker, markersize=4.5, linewidth=1.8,
                    linestyle=ls, markerfacecolor='white' if scale == .01 else color)
            pts = natives[task][0]
            ax.plot([e['step'] / 1000 for e in pts], [e[key] * mul for e in pts], label='QAM native',
                    color='#53575D', marker='x', linestyle='--', linewidth=1.5, markersize=4)
            ax.set_title(f'Task{task}', loc='left', fontsize=12)
            ax.set_xlabel('Primitive online steps (thousands)');ax.set_ylabel(label)
            ax.set_xlim(-.5, 50.5);ax.set_xticks([0, 10, 20, 30, 40, 50])
            if col == 0: ax.set_ylim(0, 103)
            ax.axvline(20, color='#ADB2B8', linestyle=':', linewidth=1)
            ax.grid(axis='y', color='#E8EAED')
    fig.suptitle('Cube-double: residual scales 0.01, 0.1 and 1', x=.08, ha='left', fontsize=15)
    fig.text(.08, .93, 'Seed 0 | Task-specific 500k offline checkpoints | Inherited Q | Online-only replay | 50 episodes per curve point', fontsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', bbox_to_anchor=(.5, .055), ncol=4, frameon=False)
    fig.text(.08, .033, '20k base-only warmup; 50k primitive steps. DAWN: 7,500 updates; native: 45,001. Sampled residual actions.', fontsize=9)
    fig.text(.08, .012, 'Final tables use 100 paired episodes. One training seed; no smoothing or seed uncertainty bands.', fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.875, bottom=.155, hspace=.40, wspace=.24)
    for ext in ('png', 'pdf'): fig.savefig(OUT / ('learning_curves.' + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for task in (1, 2):
        data = [r for r in rows if r['task'] == task]
        for col, key, native_key, mul, label in [(0, 'success_percent', 'success', 100, 'Success rate (%)'),
                                               (1, 'return_mean', 'return_mean', 1, 'Mean episode return')]:
            ax = axes[task - 1, col]
            for r in data:
                x, y = r['residual_scale'], r[key]
                ax.scatter([x], [y], color='#2463A6', s=50, zorder=3)
                ax.annotate(f'{y:.0f}%' if col == 0 else f'{y:.1f}', (x, y), xytext=(0, 9),
                            textcoords='offset points', ha='center', fontsize=9)
            ax.axhline(natives[task][1][native_key] * mul, color='#53575D', linestyle='--', linewidth=1)
            ax.set_xscale('log');ax.set_xlim(.008, 1.25)
            xs = [r['residual_scale'] for r in data]
            ax.set_xticks(xs, [f'{x:g}' for x in xs]);ax.minorticks_off()
            ax.set_xlabel('Residual scale (logarithmic axis)');ax.set_ylabel(label)
            ax.set_title(f'Task{task}', loc='left', fontsize=12);ax.grid(axis='y', color='#E8EAED')
            if col == 0: ax.set_ylim(0, 110)
            else: ax.margins(y=.23)
    fig.suptitle('Cube-double: all measured residual scales after 50k online steps', x=.08, ha='left', fontsize=15)
    fig.text(.08, .93, '100 paired evaluation episodes | Sampled residual | Seed 0 | 7,500 DAWN updates | Dashed: native reference', fontsize=10)
    fig.text(.08, .04, 'Task1 intermediate scales 0.05 / 0.2 / 0.3 were not run. Measured dots only; no interpolation.', fontsize=9)
    fig.text(.08, .017, 'Task-specific offline checkpoints. Native uses 45,001 updates. A single training seed does not establish an optimal scale.', fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.875, bottom=.14, hspace=.40, wspace=.24)
    for ext in ('png', 'pdf'): fig.savefig(OUT / ('all_scales.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__': main()
