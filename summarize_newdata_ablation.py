"""Audit data eligibility and report the paired native QAM experiment."""
import json
import pickle
from pathlib import Path

import numpy as np
import summarize_task2_scale as evidence
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/newdata_ablation'
OUT = ROOT / 'results_newdata_ablation'
load, checked, digest, track = evidence.load, evidence.checked, evidence.digest, evidence.track


def update_count(step): return max(0, step - 5000 + 1)


def main():
    OUT.mkdir(exist_ok=True)
    for n, h in load(RUNS / 'source_manifest.json').items(): assert digest(ROOT / n) == h, n
    checks = load(RUNS / 'CHECKS_PASSED.json')
    assert checks['status'] == 'passed' and checks['original_native_checkpoint_parity']
    assert checks['zero_interaction_offline_checkpoint_parity'] and checks['offline_new_data_leakage'] == 0
    assert load(RUNS / 'runtime_setup.json')['probe_exit_code'] == 0
    rows, eval_rows, episode_rows, pairs, historical, prefixes, initial_differences = [], [], [], [], [], [], []
    curves, finals = {}, {}
    for task in (1, 2):
        root = ROOT / ('runs' if task == 1 else 'runs/task2')
        offline = load(root / 'offline/DONE.json');assert offline['step'] == 500000
        original_cfg = load(root / 'native/config.json')
        old = checked(root / 'native/eval_050000_100.json')
        old_base = checked(root / 'offline/eval_500000_050.json')
        old_done = load(root / 'native/DONE.json');assert old_done['updates'] == 45001
        historical.append(dict(task=task, role='historical_native_only', success_percent=100 * old['success'],
            return_mean=old['return_mean'], episodes=100))
        curves[task], finals[task] = {}, {}
        initial, config, manifest, first_arrays = None, None, None, None
        for kind in ('all', 'offline'):
            folder = RUNS / f'task{task}_{kind}'
            cfg = load(folder / 'config.json');selected = load(folder / 'requested_config.json')
            for k, v in original_cfg.items():
                if k != 'out': assert cfg[k] == v, (task, kind, k)
            assert set(cfg) - set(original_cfg) <= {'task', 'train_data', 'environment'}
            assert cfg['offline_sha256'] == selected['offline_sha256'] == offline['checkpoint_sha256']
            assert cfg['task'] == task and cfg['train_data'] == kind and cfg['seed'] == 0
            assert cfg['online_steps'] == 50000 and cfg['native_start'] == 5000
            ini = load(folder / 'initial_state.json');effective = load(folder / 'effective_qam_config.json')
            assert ini['network_step'] == 500001 and ini['flow'] == offline['flow_hash']
            assert effective['batch_size'] == 256 and effective['lr'] == 3e-4
            assert effective['edit_scale'] == effective['fql_alpha'] == 0 and effective['inv_temp'] == 1
            assert effective['tau'] == .005 and effective['num_qs'] == 10 and effective['rho'] == .5
            assert effective['action_chunking'] and effective['horizon_length'] == 5
            task_manifest = load(folder / 'task_manifest.json')
            assert task_manifest['reward_task_id'] == task
            if initial is None: initial, config, manifest = ini, effective, task_manifest
            else: assert initial == ini and config == effective and manifest == task_manifest
            done = load(folder / 'DONE.json');audit = load(folder / 'DATA_AUDIT_PASSED.json')
            assert done['steps'] == audit['collected_transitions'] == 50000
            assert done['updates'] == audit['update_batches'] == 45001
            assert audit['batch_samples'] == 45001 * 256
            assert audit['status'] == 'passed' and audit['offline_data_unchanged']
            assert audit['original_native_loop'] and audit['original_sequence_sampler']
            assert done['initial_flow_hash'] == ini['flow'] != done['final_flow_hash']
            assert audit['offline_size'] == 1000000
            if kind == 'offline':
                assert audit['online_start_samples'] == audit['sequences_touching_online'] == 0
                assert audit['largest_sampled_index'] < audit['offline_size'] == audit['training_pool_size']
            else:
                assert audit['training_pool_size'] == 1050000
                assert 0 < audit['online_start_samples'] <= audit['sequences_touching_online']
            p = folder / 'preupdate_collection.npz';track(p)
            assert digest(p) == audit['first_collection_prefix']['snapshot_sha256']
            assert audit['first_collection_prefix']['transitions'] == 5000
            with np.load(p) as z: arrays = {k: z[k] for k in z.files}
            assert all(len(a) == 5000 and np.isfinite(a).all() for a in arrays.values())
            if first_arrays is None: first_arrays = arrays
            else:
                differences = {}
                assert set(arrays) == set(first_arrays)
                for k, a in arrays.items():
                    b = first_arrays[k];assert a.shape == b.shape and a.dtype == b.dtype
                    if k in ('observations', 'next_observations'): np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
                    else: np.testing.assert_array_equal(a, b)
                    differences[k] = dict(changed_elements=int(np.sum(a != b)), max_abs_diff=float(np.max(np.abs(a - b))))
                prefixes.append(dict(task=task, transitions=5000, differences=differences))
            if (folder / 'final.pkl').exists():
                assert digest(folder / 'final.pkl') == done['checkpoint_sha256']
                with (folder / 'final.pkl').open('rb') as f: checkpoint = pickle.load(f)
                assert checkpoint['step'] == 50000 and checkpoint['updates'] == 45001
                assert int(checkpoint['agent']['network']['step']) == 545002
                del checkpoint
            p = folder / 'metrics.jsonl';track(p)
            metrics = [json.loads(s) for s in p.read_text().splitlines()]
            assert metrics[-1]['step'] == 50000 and metrics[-1]['updates'] == 45001
            for m in metrics:
                assert m['updates'] == update_count(m['step'])
                for v in m.values():
                    if isinstance(v, (int, float)): assert np.isfinite(v)
            pts = [checked(p) for p in sorted(folder.glob('eval_*_050.json'))]
            assert [e['step'] for e in pts] == [0, 5000, 10000, 20000, 30000, 40000, 50000]
            for e in pts:
                a = load(folder / f'data_audit_{e["step"]:06d}_050.json')
                assert a['update_batches'] == update_count(e['step'])
                assert a['collected_transitions'] == e['step'] and a['batch_samples'] == update_count(e['step']) * 256
                if kind == 'offline': assert a['online_start_samples'] == a['sequences_touching_online'] == 0
                assert not e['deterministic_residual']
            final = checked(folder / 'eval_050000_100.json')
            assert final['records'][:50] == pts[-1]['records']
            assert load(folder / 'data_audit_050000_100.json')['index_chain_sha256'] == audit['index_chain_sha256']
            assert not final['deterministic_residual']
            curves[task][kind], finals[task][kind] = pts, final
            rows.append(dict(task=task, train_data=kind, seed=0, offline_updates=500000,
                primitive_collection_steps=50000, new_transitions_eligible_for_training=50000 if kind == 'all' else 0,
                additional_updates=45001, batch_size=256, batch_samples=audit['batch_samples'],
                final_episodes=100, final_success_percent=100 * final['success'], final_return_mean=final['return_mean'],
                initial_50_success_percent=100 * pts[0]['success'], initial_50_return_mean=pts[0]['return_mean'],
                final_50_success_percent=100 * pts[-1]['success'], final_50_return_mean=pts[-1]['return_mean'],
                mean_episode_length=float(np.mean([r['length'] for r in final['records']])),
                sampled_online_starts=audit['online_start_samples'], windows_touching_online=audit['sequences_touching_online'],
                new_window_share_percent=100 * audit['sequences_touching_online'] / audit['batch_samples'],
                historical_native_success_percent=100 * old['success'],
                identical_checkpoint_to_historical_native=done['checkpoint_sha256'] == old_done['checkpoint_sha256'] if kind == 'all' else None,
                historical_offline50_success_percent=100 * old_base['success']))
            for role, es in [('curve50', pts), ('final100', [final])]:
                for e in es:
                    eval_rows.append(dict(task=task, train_data=kind, role=role, primitive_collection_steps=e['step'],
                        additional_updates=update_count(e['step']), episodes=e['episodes'],
                        success_percent=100 * e['success'], return_mean=e['return_mean']))
                    for rec in e['records']: episode_rows.append(dict(task=task, train_data=kind, role=role, step=e['step'], **rec))
        a, b = finals[task]['all']['records'], finals[task]['offline']['records']
        assert [r['initial_hash'] for r in a] == [r['initial_hash'] for r in b]
        # Preserve any initial reward differences rather than replacing observations.
        base_a, base_b = curves[task]['all'][0]['records'], curves[task]['offline'][0]['records']
        differences = [dict(episode=x['episode'], fields={k: dict(all=x[k], offline=y[k])
            for k in x if x[k] != y[k]}) for x, y in zip(base_a, base_b) if x != y]
        # Observed and inspected before training completed: identical full initial
        # agents, reset states and success; one task1 episode has a return mismatch.
        # This exact exception is disclosed, never used to replace measured data.
        expected = [dict(episode=33, fields={'return_': dict(all=-533.0, offline=-500.0)})] if task == 1 else []
        assert differences == expected, (task, differences)
        initial_differences.append(dict(task=task, exact_episode_records_match=not differences,
            identical_full_initial_agent=True, identical_success=True, differences=differences,
            all_return_mean=curves[task]['all'][0]['return_mean'],
            offline_return_mean=curves[task]['offline'][0]['return_mean']))
        for x, y in zip(curves[task]['all'], curves[task]['offline']):
            assert [r['initial_hash'] for r in x['records']] == [r['initial_hash'] for r in y['records']]
        pairs.append(dict(task=task, episodes=100, both_success=sum(bool(x['success']) and bool(y['success']) for x, y in zip(a, b)),
            all_only_success=sum(bool(x['success']) and not bool(y['success']) for x, y in zip(a, b)),
            offline_only_success=sum(not bool(x['success']) and bool(y['success']) for x, y in zip(a, b)),
            neither_success=sum(not bool(x['success']) and not bool(y['success']) for x, y in zip(a, b)),
            all_minus_offline_success_pp=100 * (finals[task]['all']['success'] - finals[task]['offline']['success']),
            all_minus_offline_return=finals[task]['all']['return_mean'] - finals[task]['offline']['return_mean']))
    for name, data in [('comparison.csv', rows), ('evaluations.csv', eval_rows), ('episodes.csv', episode_rows),
                       ('paired_outcomes.csv', pairs), ('historical_reference.csv', historical)]:
        evidence.write_csv(OUT / name, data)
    (OUT / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    (OUT / 'preupdate_prefix_comparison.json').write_text(json.dumps(prefixes, indent=2) + '\n')
    (OUT / 'initial_evaluation_comparison.json').write_text(json.dumps(initial_differences, indent=2) + '\n')
    plot(curves)
    lines = ['# QAM 新增 online 数据消融', '',
        '每任务两组在新机器配对复跑。共享对应任务的 offline 500k checkpoint 与完整训练状态；seed 0；均完成 50k primitive 采集步、45,001 次更新、batch 256。唯一处理变量是新增数据是否可被训练采样。', '',
        '| Task | 训练 replay | 最终成功率（100 eps） | 平均回报 | 被抽到的新数据起点次数 |',
        '|---|---|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['task']} | {'offline + online' if r['train_data'] == 'all' else '仅 offline'} | {r['final_success_percent']:.0f}% | {r['final_return_mean']:.2f} | {r['sampled_online_starts']:,} |")
    lines += ['', '新增数据的配对效果（all − offline）：', '']
    for p in pairs: lines.append(f"- Task{p['task']}: 成功率 {p['all_minus_offline_success_pp']:+.0f} pp；回报 {p['all_minus_offline_return']:+.2f}。")
    lines += ['', '与初始策略比较时，统一使用相同 50 episodes：', '',
        '| Task | 初始成功率 | 仅旧数据最终成功率 | 使用新数据最终成功率 |', '|---|---:|---:|---:|']
    for task in (1, 2):
        a, b = [next(r for r in rows if r['task'] == task and r['train_data'] == kind) for kind in ('all', 'offline')]
        lines.append(f"| {task} | {a['initial_50_success_percent']:.0f}% | {b['final_50_success_percent']:.0f}% | {a['final_50_success_percent']:.0f}% |")
    lines += ['', '使用新数据组实际抽到的序列中，涉及新 transitions 的累计比例：' + '；'.join(
        f"Task{r['task']} {r['new_window_share_percent']:.3f}%" for r in rows if r['train_data'] == 'all') + '。这是整个训练期间的实测采样比例，不是 50/50 replay，也不等于新数据对 loss 的数值贡献比例。']
    lines += ['', '公平性：两组均更新 BC prior、adjoint-matching actor 和 critic，延续 Adam 状态与 agent RNG；相同奖励、chunk=5、lr=3e-4、target tau=.005、10 critics、mean−.5std target、mean actor Q、global norm clip=1、edit_scale=0。',
        '仅 offline 组也照常采集，以保留原训练循环和 RNG/评估调用时序，但新收集的 transitions 完全隔离，任何 loss 都不能使用。不能把该组称为零环境交互运行；其学习部分等价于继续做纯 offline 更新。',
        '每个训练 batch 均记录原 sampler 的采样索引和 5-step 窗口界限：仅 offline 组必须为零新数据窗口；原 offline 数组前后哈希一致。首次更新前的 5k 采集前缀配对比较见 preupdate_prefix_comparison.json。',
        '短程完整 checkpoint 验证：all 复现原 native；offline 复现完全无训练环境交互的纯 offline 更新；两任务均通过。',
        '初始评估复现差异：Task1 两组成功率同为 76%，但 episode 33 回报为 -533 / -500，初始均值为 -218.28 / -217.62；其余 episode 字段完全一致。模型/优化器/RNG 与 reset 状态哈希相同；首次更新前动作、奖励、mask 完全一致，观测最大差 4.51e-17。数值差异可能被接触动力学放大，但该 episode 的具体原因未经轨迹级验证。保留原记录，未以旧结果替换。Task2 起点评估完全一致。',
        '曲线每点 50 episodes；最终主表 100 episodes。初始值与最终主表分母不同，增长趋势使用配对的 50-episode 曲线。只有一个 training seed，不能外推跨 seed 稳定性。',
        '历史 native 仅作为复现参考，主对照一律使用本机本轮 all 组。克隆镜像 NVIDIA EGL vendor 文件为空，已使用独立 GLVND 配置指向新机已安装的驱动；没有改训练算法。', '',
        '![By collection steps](learning_curves.png)', '', '![By additional updates](update_curves.png)', '']
    (OUT / 'summary.md').write_text('\n'.join(lines))
    track(Path(evidence.__file__).resolve())
    (OUT / 'validation.json').write_text(json.dumps(dict(status='passed', sources=evidence.sources,
        report_source_sha256=digest(Path(__file__)), checks=['fixed training source hashes', 'paired full initial states and configurations',
        'correct task-specific checkpoints', 'original native / pure offline smoke checkpoint equivalence',
        'zero new-data windows in offline condition', 'identical collection and update budgets',
        'preupdate collection prefix comparison', 'paired evaluation seeds and initial states',
        'explicit task1 initial episode33 return exception; all other initial fields exact',
        'episode means recomputed', 'final100 first50 equals endpoint50', 'historical reference not substituted']), indent=2) + '\n')
    print(json.dumps(rows, indent=2), flush=True)


def plot(curves):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    for by_updates in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for task in (1, 2):
            for col, key, factor, label in [(0, 'success', 100, 'Success rate (%)'), (1, 'return_mean', 1, 'Mean episode return')]:
                ax = axes[task - 1, col]
                for kind, color, marker, ls, title in [('all', '#2463A6', 'o', '-', 'Offline + online training data'),
                        ('offline', '#82A9D0', 's', '--', 'Offline training data only')]:
                    pts = curves[task][kind]
                    x = [(update_count(e['step']) if by_updates else e['step']) / 1000 for e in pts]
                    ax.plot(x, [e[key] * factor for e in pts], color=color, marker=marker, linestyle=ls,
                        linewidth=2, markersize=4.5, markerfacecolor=color if kind == 'all' else 'white', label=title)
                ax.axhline(curves[task]['all'][0][key] * factor, color='#8B9096', linewidth=1, linestyle=':', label='Initial offline policy')
                if not by_updates: ax.axvline(5, color='#ADB2B8', linewidth=1, linestyle=':')
                ax.set_title(f'Task{task}', loc='left', fontsize=12);ax.set_ylabel(label)
                ax.set_xlabel('Additional gradient updates (thousands)' if by_updates else 'Primitive collection steps (thousands)')
                ax.set_xlim(-.5, 45.5 if by_updates else 50.5)
                ax.set_xticks([0, 10, 20, 30, 40, 45] if by_updates else [0, 10, 20, 30, 40, 50])
                if col == 0: ax.set_ylim(0, 103)
                ax.grid(axis='y', color='#E8EAED')
        fig.suptitle('Cube-double: availability of newly collected training data', x=.08, ha='left', fontsize=15)
        fig.text(.08, .93, 'Seed 0 | Task-specific offline 500k checkpoints | Matched 45,001 updates / batch 256 | 50 episodes per point', fontsize=10)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', bbox_to_anchor=(.5, .06), ncol=3, frameon=False)
        fig.text(.08, .036, 'Both arms collect 50k primitive steps. Offline-only excludes every new transition from all training losses.', fontsize=9)
        fig.text(.08, .014, 'Same native QAM loop, full agent/optimizer initialization and 5k learning start. Final table uses 100 paired episodes.', fontsize=9)
        fig.subplots_adjust(left=.08, right=.98, top=.875, bottom=.155, hspace=.40, wspace=.24)
        name = 'update_curves' if by_updates else 'learning_curves'
        for ext in ('png', 'pdf'): fig.savefig(OUT / f'{name}.{ext}', dpi=180)
        plt.close(fig)


if __name__ == '__main__': main()
