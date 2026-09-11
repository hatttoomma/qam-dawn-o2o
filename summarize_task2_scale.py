"""Validate and report the four task2 DAWN scales, with native as context."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TASK = ROOT / 'runs/task2'
RUNS = TASK / 'scale_sweep'
OUT = ROOT / 'results_task2_scale'
ARMS = [('scale005', .05), ('warm', .1), ('scale020', .2), ('scale030', .3)]
sources = {}


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def track(p): sources[str(p.relative_to(ROOT))] = digest(p)
def load(p):
    track(p)
    return json.loads(p.read_text())


def checked(p):
    e = load(p)
    records = e['records']
    assert e['episodes'] == len(records)
    assert [r['episode'] for r in records] == list(range(len(records)))
    assert [r['reset_seed'] for r in records] == list(range(500000, 500000 + len(records)))
    assert all(r['length'] > 0 and r['length'] <= 500 for r in records)
    success = np.array([r['success'] for r in records])
    returns = np.array([r['return_'] for r in records])
    assert np.isin(success, [0, 1]).all() and np.isfinite(returns).all()
    np.testing.assert_allclose(success.mean(), e['success'], atol=1e-12)
    np.testing.assert_allclose(returns.mean(), e['return_mean'], atol=1e-10)
    return e


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    for n, h in load(RUNS / 'source_manifest.json').items(): assert digest(ROOT / n) == h, n
    assert load(RUNS / 'CHECKS_PASSED.json')['status'] == 'passed'
    assert load(TASK / 'TASK_AUDIT_PASSED.json')['status'] == 'passed'
    offline = load(TASK / 'offline/DONE.json')
    offline_eval = checked(TASK / 'offline/eval_500000_050.json')
    task_manifest = load(TASK / 'offline/task_manifest.json')
    reference_config = load(TASK / 'warm/config.json')
    initial = load(TASK / 'warm/initial_hashes.json')
    warmup = load(TASK / 'warm/warmup.json')
    assert offline['step'] == 500000 and task_manifest['reward_task_id'] == 2
    assert reference_config['environment'] == 'cube-double-play-singletask-task2-v0'
    assert initial['critic'] == offline['q_hash'] and initial['target'] == offline['target_q_hash']
    native = checked(TASK / 'native/eval_050000_100.json')
    native_curve = [checked(p) for p in sorted((TASK / 'native').glob('eval_*_050.json'))]
    assert len(native_curve) == 7 and native['records'][:50] == native_curve[-1]['records']
    rows, eval_rows, episode_rows, curves, finals = [], [], [], {}, {}
    pair_rows = []
    baseline_final = checked(TASK / 'warm/eval_050000_100.json')
    for name, scale in ARMS:
        folder = TASK / 'warm' if name == 'warm' else RUNS / name
        cfg = load(folder / 'config.json')
        done = load(folder / 'DONE.json')
        assert {k: v for k, v in cfg.items() if k not in ('out', 'residual_scale')} == {
            k: v for k, v in reference_config.items() if k != 'out'}, name
        assert cfg['offline_sha256'] == offline['checkpoint_sha256']
        assert cfg['online_steps'] == done['steps'] == 50000 and done['updates'] == 7500
        assert load(folder / 'task_manifest.json') == task_manifest
        assert load(folder / 'initial_hashes.json') == initial
        assert load(folder / 'warmup.json') == warmup
        assert done['final_flow_hash'] == initial['flow'] == offline['flow_hash']
        assert done['final_critic_hash'] != initial['critic'] and done['final_actor_hash'] != initial['actor']
        if name != 'warm':
            selected = load(folder / 'scale_config.json')
            actual = load(folder / 'actual_agent_config.json')
            assert cfg['residual_scale'] == selected['residual_scale'] == actual['residual_scale'] == scale
            assert selected['offline_sha256'] == offline['checkpoint_sha256']
            assert actual['critic_hash'] == initial['critic'] and actual['target_hash'] == initial['target']
            assert actual['actor_hash'] == initial['actor'] and actual['action_dim'] == 25
        if (folder / 'final.pkl').exists(): assert digest(folder / 'final.pkl') == done['checkpoint_sha256']
        metrics_path = folder / 'metrics.jsonl'
        track(metrics_path)
        metrics = [json.loads(s) for s in metrics_path.read_text().splitlines()]
        for m in metrics:
            for v in m.values():
                if isinstance(v, (int, float)): assert np.isfinite(v)
        last = metrics[-1]
        assert last['step'] == 50000 and last['updates'] == 7500
        assert 0 <= last['residual_abs_mean'] <= scale + 1e-6
        points = [checked(p) for p in sorted(folder.glob('eval_*_050.json'))]
        assert len(points) == 7
        for nominal, e in zip([0, 5000, 10000, 20000, 30000, 40000, 50000], points):
            assert nominal <= e['step'] <= min(nominal + 4, 50000)
            assert not e['deterministic_residual']
            assert [r['initial_hash'] for r in e['records']] == [r['initial_hash'] for r in offline_eval['records']]
        assert points[0]['records'] == offline_eval['records']
        # Every warmup evaluation remains the identical frozen base policy.
        assert all(e['records'] == offline_eval['records'] for e in points[:4])
        final = checked(folder / 'eval_050000_100.json')
        mean = checked(folder / 'eval_050000_050_mean_residual.json')
        assert final['records'][:50] == points[-1]['records']
        assert final['residual_enabled'] and not final['deterministic_residual']
        assert mean['episodes'] == 50 and mean['deterministic_residual'] and mean['residual_enabled']
        assert [r['initial_hash'] for r in final['records']] == [r['initial_hash'] for r in baseline_final['records']]
        assert [r['initial_hash'] for r in mean['records']] == [r['initial_hash'] for r in offline_eval['records']]
        rows.append(dict(task='task2', residual_scale=scale, seed=0, offline_updates=500000,
            online_primitive_steps=50000, online_updates=7500, batch_size=1024,
            final_episodes=100, success_percent=100 * final['success'], return_mean=final['return_mean'],
            mean_episode_length=float(np.mean([r['length'] for r in final['records']])),
            curve_final_success_percent=100 * points[-1]['success'],
            mean_residual_episodes=50, mean_residual_success_percent=100 * mean['success'],
            mean_residual_return_mean=mean['return_mean'],
            success_auc_percent=float(np.trapz([e['success'] for e in points], [e['step'] for e in points]) / 500),
            last_train_residual_abs_mean=last['residual_abs_mean'], last_train_clip_fraction=last['action_clip_fraction'],
            last_train_alpha=last['alpha'], replay_chunks=last['replay_chunks']))
        curves[name], finals[name] = points, final
        for role, es in [('curve', points), ('final100', [final]), ('mean_residual50', [mean])]:
            for e in es:
                eval_rows.append(dict(scale=scale, role=role, primitive_steps=e['step'], episodes=e['episodes'],
                    success_percent=100 * e['success'], return_mean=e['return_mean']))
                for rec in e['records']:
                    episode_rows.append(dict(scale=scale, role=role, primitive_steps=e['step'], **rec))
        a, b = final['records'], baseline_final['records']
        pair_rows.append(dict(scale=scale, reference_scale=.1, episodes=100,
            both_success=sum(bool(x['success']) and bool(y['success']) for x, y in zip(a, b)),
            scale_only_success=sum(bool(x['success']) and not bool(y['success']) for x, y in zip(a, b)),
            reference_only_success=sum(not bool(x['success']) and bool(y['success']) for x, y in zip(a, b)),
            neither_success=sum(not bool(x['success']) and not bool(y['success']) for x, y in zip(a, b))))
    for filename, data in [('comparison.csv', rows), ('evaluations.csv', eval_rows),
                           ('episodes.csv', episode_rows), ('paired_vs_scale010.csv', pair_rows)]:
        write_csv(OUT / filename, data)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    plot(curves, rows, native_curve, native)
    lines = ['# Task2: DAWN residual-scale 消融', '',
        '共享同一个 task2 QAM offline 500k checkpoint；seed 0；50k primitive online steps；7,500 次更新。仅 residual scale 改变。', '',
        '| Residual scale | 最终成功率（100 eps，sampled） | 平均回报 | 均值残差成功率（50 eps） |',
        '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['residual_scale']:g} | {r['success_percent']:.0f}% | {r['return_mean']:.2f} | {r['mean_residual_success_percent']:.0f}% |")
    lines += ['', f"原生 QAM 参考：{native['success'] * 100:.0f}% / {native['return_mean']:.2f}（100 eps，45,001 次更新）。",
        '0.1 与 native 为原有已完成结果，其他三组为新增实验。所有 DAWN 初始权重、optimizer、RNG、任务数据和 base-only warmup 一致。',
        '其余设置：继承 current/target Q，冻结 QAM，state-only residual，10 critics / min10，联合 clip=50，SAC entropy TD，online-only replay，20k warmup，UTD=.25/primitive step，batch=1024，lr=1e-4，tau=.01。', '',
        '曲线每点评估 50 episodes；最终 sampled 主表为 100 episodes；tanh(mean) 诊断只有 50 episodes，base 仍随机。只有一个训练 seed，不能据此确认通用最优 scale。', '',
        '![Learning curves](learning_curves.png)', '', '![Final scale sensitivity](scale_sensitivity.png)', '']
    (OUT / 'summary.md').write_text('\n'.join(lines))
    (OUT / 'validation.json').write_text(json.dumps(dict(status='passed', sources=sources,
        report_source_sha256=digest(Path(__file__)), checks=['unchanged training sources',
        'scale010 wrapper checkpoint parity', 'effective scales match configs', 'only scale differs in formal configs',
        'same pretrained task2 current and target Q', 'same actor/optimizer initial states',
        'identical base-only warmup replay and evaluations', 'frozen flows unchanged',
        'exact 50k primitive steps and 7500 updates', 'finite metrics', 'paired evaluation states',
        'episode aggregates recomputed', 'final100 first50 equals curve endpoint']), indent=2) + '\n')
    print(json.dumps(rows, indent=2), flush=True)


def plot(curves, rows, native_curve, native):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    styles = [('scale005', '0.05', '#82A9D0', 'o', ':'), ('warm', '0.1 (reference)', '#2463A6', 's', '--'),
              ('scale020', '0.2', '#164A7B', '^', '-'), ('scale030', '0.3', '#4D83BA', 'D', '-.')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for name, label, color, marker, ls in styles:
        pts = curves[name]
        for ax, key, mul in [(axes[0], 'success', 100), (axes[1], 'return_mean', 1)]:
            ax.plot([e['step'] / 1000 for e in pts], [e[key] * mul for e in pts],
                    label='Scale ' + label, color=color, marker=marker, markersize=5,
                    markerfacecolor='white' if name in ('scale005', 'scale030') else color,
                    linestyle=ls, linewidth=1.8)
    for ax, key, mul in [(axes[0], 'success', 100), (axes[1], 'return_mean', 1)]:
        ax.plot([e['step'] / 1000 for e in native_curve], [e[key] * mul for e in native_curve],
                label='QAM native', color='#53575D', linewidth=1.7, linestyle='--', marker='x', markersize=5)
        ax.axvline(20, color='#ADB2B8', linestyle=':', linewidth=1)
        ax.set_xlim(-.5, 50.5)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_xlabel('Primitive online steps (thousands)')
        ax.grid(axis='y', color='#E8EAED')
    axes[0].set_ylim(0, 103)
    axes[0].set_ylabel('Success rate (%)')
    axes[1].set_ylabel('Mean episode return')
    fig.suptitle('Cube-double task2: DAWN residual-scale learning curves', x=.075, ha='left', fontsize=15)
    fig.text(.075, .89, 'Seed 0 | Shared 500k offline checkpoint | Inherited Q | Online-only replay | 50 episodes per point', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .07), ncol=5, frameon=False, fontsize=9)
    fig.text(.075, .04, 'Dotted boundary: 20k base-only warmup. DAWN: 7,500 updates; native context: 45,001 updates.', fontsize=9)
    fig.text(.075, .015, 'Sampled residual actions; final table uses 100 paired episodes. One training seed; no smoothing or seed uncertainty bands.', fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.23, wspace=.24)
    for ext in ('png', 'pdf'): fig.savefig(OUT / ('learning_curves.' + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    xs = [r['residual_scale'] for r in rows]
    for ax, key, ylabel, ref in [(axes[0], 'success_percent', 'Success rate (%)', native['success'] * 100),
                                (axes[1], 'return_mean', 'Mean episode return', native['return_mean'])]:
        vals = [r[key] for r in rows]
        ax.scatter(xs, vals, s=60, color='#2463A6', zorder=3)
        for x, y in zip(xs, vals): ax.annotate(f'{y:.0f}%' if key == 'success_percent' else f'{y:.1f}',
            (x, y), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=10)
        ax.axhline(ref, color='#53575D', linestyle='--', linewidth=1, label='QAM native reference')
        ax.set_xticks(xs, ['0.05', '0.1', '0.2', '0.3'])
        ax.set_xlim(.02, .33)
        ax.set_xlabel('Residual scale')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#E8EAED')
        ax.legend(frameon=False, loc='lower left', fontsize=9)
        if key == 'success_percent': ax.set_ylim(0, 110)
        else: ax.margins(y=.25)
    fig.suptitle('Cube-double task2: performance after 50k online steps', x=.075, ha='left', fontsize=14)
    fig.text(.075, .87, '100 paired evaluation episodes | Sampled residual | Seed 0 | 7,500 DAWN updates', fontsize=10)
    fig.text(.075, .02, 'Measured scales only; no interpolation. Native is context, with 45,001 updates. One training seed.', fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.77, bottom=.18, wspace=.25)
    for ext in ('png', 'pdf'): fig.savefig(OUT / ('scale_sensitivity.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__': main()
