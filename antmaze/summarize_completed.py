"""Validate the final source snapshot, sync records, and plot measured results."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/antmaze_large_task1_20260913'
snapshot = json.loads((OUT / 'final_complete_snapshot.json').read_text())
files = snapshot['files']
assert snapshot['full']
assert files['results/suite_status.json']['state'] == 'passed'
assert files['results/suite_status.json']['stage'] == 'complete'
assert not any(k.endswith('/FAILED.json') for k in files)
assert len(snapshot['checkpoint_audit']) == 3
assert all(v['matched'] for v in snapshot['checkpoint_audit'].values())

native = files['results/native/DONE.json']
dawn = files['results/dawn/DONE.json']
checks = files['results/dawn/CHECKS_PASSED.json']
offline = files['results/native/offline_checkpoint.json']
assert native['status'] == checks['status'] == 'passed'
assert (native['offline_updates'], native['online_env_steps'], native['online_updates']) == (500000, 50000, 45001)
assert native['official_main_byte_identical']
assert (dawn['steps'], dawn['updates']) == (150000, 17500)
assert (checks['steps'], checks['updates']) == (150000, 17500)
assert all(checks[k] for k in ['unchanged_historical_collection_and_update_loop', 'native_gate_passed',
                             'offline_checkpoint_matched', 'hard_TD_verified', 'frozen_base_verified'])
assert files['results/dawn/config.json']['offline_sha256'] == offline['sha256']
assert dawn['initial_hashes']['flow'] == dawn['final_flow_hash'] == offline['flow_hash']
assert dawn['initial_hashes']['critic'] == offline['q_hash']
assert dawn['initial_hashes']['target'] == offline['target_q_hash']
assert files['results/dawn/INITIAL_EVAL_MATCHED.json']['status'] == 'passed'

config = files['results/dawn/actual_agent_config.json']
expected = dict(inherited_Q_and_target=True, fresh_critic_optimizer=True, td_target='naive',
                actor_input='state+base_action', observation_dim=29, actor_input_dim=37, action_dim=8,
                horizon=1, residual_scale=0.1, utd=0.25, batch_size=256, warmup=80000,
                online_steps=150000, replay='online_only_uniform_growing_chunk_buffer',
                critic_ensemble=10, target_aggregation='minimum', actor_Q_aggregation='minimum',
                learning_rate=0.0001, target_tau=0.01, discount=0.99, gradient_clip_norm=50.0,
                initial_alpha=0.01, target_entropy=-8, actor_entropy_enabled=True,
                automatic_alpha_enabled=True, frozen_base_inv_temp=10.0, seed=0)
assert all(config[k] == v for k, v in expected.items())
native_config = files['results/native/actual_agent_config.json']
assert native_config['vanilla_QAM'] and native_config['seed'] == 0
for k, v in dict(inv_temp=10.0, edit_scale=0.0, fql_alpha=0.0, horizon_length=1,
                 batch_size=256, num_qs=10, rho=0.5, lr=0.0003, discount=0.99, tau=0.005).items():
    assert native_config['agent_config'][k] == v

def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    if isinstance(value, (float, int)): return math.isfinite(value)
    return True

assert finite(files['results/native/metrics.jsonl'])
assert finite(files['results/dawn/metrics.jsonl'])
for row in files['results/dawn/metrics.jsonl']:
    assert row['updates'] == max(0, row['step'] - 80000) // 4
online_rows = [r for r in files['results/native/metrics.jsonl'] if r['stage'] == 'native_online']
assert online_rows and online_rows[-1]['online_env_steps'] == 50000
for row in online_rows:
    assert row['online_updates'] == max(0, row['online_env_steps'] - 4999)

base = files['results/native/fixed_eval/eval_000000_100.json']
assert base['records'] == files['results/dawn/eval_000000_100.json']['records']
evaluations = {k: v for k, v in files.items() if k.endswith('.json') and
               (k.startswith('results/native/fixed_eval/eval_') or k.startswith('results/dawn/eval_'))}
summary = []
paired = {}
for name, evaluation in evaluations.items():
    records = evaluation['records']
    assert len(records) == evaluation['episodes'] == 100
    assert [r['episode'] for r in records] == list(range(100))
    assert all((a['reset_seed'], a['initial_hash']) == (b['reset_seed'], b['initial_hash'])
               for a, b in zip(records, base['records']))
    assert all(r['success'] in (0, 1) and r['return_'] == -r['length'] + r['success'] for r in records)
    success = sum(r['success'] for r in records) / 100
    mean_return = sum(r['return_'] for r in records) / 100
    assert math.isclose(success, evaluation['success'], abs_tol=1e-12)
    assert math.isclose(mean_return, evaluation['return_mean'], abs_tol=1e-9)
    method = ('QAM native' if 'native/' in name else
              'DAWN mean' if evaluation['deterministic_residual'] else 'DAWN sampled')
    gained = sum(r['success'] > b['success'] for r, b in zip(records, base['records']))
    lost = sum(r['success'] < b['success'] for r, b in zip(records, base['records']))
    paired[name] = dict(gained=gained, lost=lost)
    steps = evaluation['step']
    updates = (max(0, steps - 4999) if method == 'QAM native' else max(0, steps - 80000) // 4)
    summary.append(dict(method=method, online_env_steps=steps, online_updates=updates,
                        success_rate=success, avg_return=mean_return,
                        gain_vs_offline_pp=round(100 * (success - base['success']), 6),
                        gained_successes=gained, lost_successes=lost, source=name))

for name, value in files.items():
    if not name.startswith(('results/native/', 'results/dawn/', 'results/suite_status.json')):
        continue
    destination = OUT / 'synced_results' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith('.jsonl'):
        destination.write_text(''.join(json.dumps(row) + '\n' for row in value))
    else:
        destination.write_text(json.dumps(value, indent=2) + '\n')
(OUT / 'source_result_sha256.json').write_text(json.dumps(snapshot['file_sha256'], indent=2))
validation = dict(status='passed', utc=snapshot['utc'], episodes_per_evaluation=100,
                  evaluated_checkpoints=len(evaluations), initial_states_matched=True,
                  initial_records_exactly_matched=True, aggregates_recomputed=True,
                  finite_logged_metrics=True, update_schedules_verified=True,
                  frozen_base_verified=True, inherited_offline_critic_verified=True,
                  configurations_verified=True, checkpoint_audit=snapshot['checkpoint_audit'],
                  paired_outcomes=paired)
(OUT / 'final_completion_validation.json').write_text(json.dumps(validation, indent=2))
(OUT / 'comparison.json').write_text(json.dumps(summary, indent=2))
with (OUT / 'comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold'})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
colors = {'QAM native': '#1764AB', 'DAWN mean': '#D96022', 'DAWN sampled': '#777777'}
series = {}
for method in colors:
    rows = sorted((r for r in summary if r['method'] == method), key=lambda r: r['online_env_steps'])
    series[method] = rows
for ax, metric, title in zip(axes, ['success_rate', 'avg_return'], ['Success rate', 'Average return (higher is better)']):
    ax.axvspan(0, 80, color='#F1F1F1', zorder=0)
    ax.axvline(80, color='#AAAAAA', linewidth=1, linestyle=':')
    ax.text(40, 1.015, 'DAWN: base-only warmup', transform=ax.get_xaxis_transform(), ha='center', fontsize=9, color='#666666')
    for method, rows in series.items():
        if method.startswith('DAWN'):
            actual = [r for r in rows if r['online_env_steps'] > 0]
            initial_y = base['success'] if metric == 'success_rate' else base['return_mean']
            # Horizontal warmup segment denotes an unchanged policy, not an extra evaluation.
            xs = [80] + [r['online_env_steps'] / 1000 for r in actual]
            ys = [initial_y] + [r[metric] for r in actual]
            ax.plot(xs, ys, color=colors[method], linewidth=2 if method == 'DAWN mean' else 1.5,
                    linestyle='-' if method == 'DAWN mean' else '--', label=method)
            ax.scatter(xs[1:], ys[1:], color=colors[method], marker='o' if method == 'DAWN mean' else 's', s=33, zorder=4)
            ax.plot([0, 80], [initial_y] * 2, color=colors[method], linewidth=1.3, linestyle=':')
        else:
            ax.plot([r['online_env_steps'] / 1000 for r in rows], [r[metric] for r in rows],
                    color=colors[method], marker='o', linewidth=2.2, label=method, zorder=5)
    ax.set_title(title, pad=28)
    ax.set_xlabel('Online environment steps (thousands)')
    ax.set_xlim(-3, 155)
    ax.set_xticks([0, 50, 80, 100, 120, 150])
    ax.grid(axis='y', color='#E6E6E6', linewidth=0.8)
axes[0].yaxis.set_major_formatter(PercentFormatter(1))
axes[0].set_ylim(0.65, 1.05)
axes[1].set_ylim(-850, -335)
axes[0].legend(loc='lower left', frameon=False)
for step in [100000, 120000, 150000]:
    row = next(r for r in series['DAWN mean'] if r['online_env_steps'] == step)
    axes[0].annotate(f"{row['success_rate']:.0%}", (step / 1000, row['success_rate']),
                     xytext=(0, 9), textcoords='offset points', ha='center', color=colors['DAWN mean'], fontsize=10)
axes[0].annotate('100%', (50, 1), xytext=(0, 9), textcoords='offset points', ha='center', color=colors['QAM native'])
fig.suptitle('AntMaze-large · task 1 | Shared QAM offline checkpoint (500k updates)', y=0.98, fontsize=15, weight='bold')
fig.text(0.5, 0.045, 'Seed 0 · 100 paired episodes per evaluation · Mean refers to residual action; base policy remains stochastic.', ha='center', fontsize=9, color='#555555')
fig.text(0.5, 0.012, 'QAM: 50k env steps / 45,001 updates. DAWN: 150k / 17,500. Dots are evaluations; lines only guide the eye.', ha='center', fontsize=9, color='#555555')
fig.tight_layout(rect=[0, 0.08, 1, 0.92], w_pad=2.5)
fig.savefig(OUT / 'antmaze_task1_comparison.png', dpi=180, facecolor='white')
fig.savefig(OUT / 'antmaze_task1_comparison.pdf', facecolor='white')
plt.close(fig)

lines = ['# AntMaze-large task1: completed pilot', '',
         'Seed 0; shared QAM offline checkpoint after 500,000 gradient updates. All comparisons below use the same 100 reset seeds and verified initial states. DAWN mean refers only to the residual; the base policy remains stochastic.', '',
         '| Method | Online env steps | Online updates | Success | Avg return | Δ success vs offline |',
         '|---|---:|---:|---:|---:|---:|']
for row in summary:
    lines.append(f"| {row['method']} | {row['online_env_steps']:,} | {row['online_updates']:,} | {row['success_rate']:.0%} | {row['avg_return']:.2f} | {row['gain_vs_offline_pp']:+.0f} pp |")
lines += ['', 'The final DAWN mean-residual policy scores 71%, 13 percentage points below its offline base and 29 points below QAM native. Its measured success rose to 98% at 100k, then fell to 82% at 120k and 71% at 150k. This one-seed run demonstrates late degradation under these settings; it does not identify a cause or establish general algorithm superiority.', '',
          'Budgets differ: QAM has 50k online environment steps and 45,001 update cycles; DAWN has 150k environment steps, including 80k with no updates, and 17,500 update cycles. QAM was not evaluated beyond 50k. The warmup line denotes an unchanged policy, not additional evaluations.', '',
          'Verified: official vanilla QAM source, requested settings, completed update counts, three checkpoint SHA256 values, inherited offline Q/target-Q, frozen base, exact initial evaluation records, finite logged metrics, and all evaluation aggregates.', '',
          'Files: comparison.csv, comparison.json, antmaze_task1_comparison.png/.pdf, final_completion_validation.json, final_complete_snapshot.json, synced_results/.', '']
(OUT / 'final_result.md').write_text('\n'.join(lines))
print(json.dumps({'validation': 'passed', 'evaluations': len(evaluations),
                  'summary': summary, 'figure': str(OUT / 'antmaze_task1_comparison.png')}, indent=2))
