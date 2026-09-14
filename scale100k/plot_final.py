"""Export source-verified100k endpoint comparisons; never interpolate trajectories."""
import argparse, csv, hashlib, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / 'reports/100k_20260914'
parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type=Path, default=DEFAULT_OUT / 'final_complete_snapshot.json')
parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
snapshot_path = args.snapshot
snapshot = json.loads(snapshot_path.read_text())
validation = json.loads((OUT / 'validation_summary.json').read_text())
assert validation['all20_complete'] and not validation['errors']
assert validation['snapshot_utc'] == snapshot['utc']
assert all(q['files']['results/suite_status.json']['state'] == 'complete' for q in snapshot['queues'].values())
assert all(not p['alive'] for q in snapshot['queues'].values() for p in q['process_liveness'].values())
rows = validation['rows']
assert len(rows) == 30
BENCHMARKS = [('antmaze-large', 'AntMaze-large'), ('cube-double', 'Cube-double')]
TASKS = list(range(1, 6))
COLORS = {'native': '#2466A5', 'dawn': '#D07427'}

def select(benchmark, task, mode):
    found = [r for r in rows if (r['benchmark'], r['task'], r['evaluation_mode']) == (benchmark, task, mode)]
    assert len(found) == 1
    return found[0]

def original_eval(benchmark, task, mode):
    method = 'native' if mode == 'native' else 'dawn'
    q = snapshot['queues']['cube' if benchmark == 'cube-double' else 'antmaze_' + method]
    prefix = f'results/task{task}/{method}/'
    if benchmark == 'antmaze-large' and mode == 'native': prefix += 'fixed_eval/'
    suffix = '_mean_residual' if mode == 'mean' else ''
    return q['files'][prefix + f'eval_100000_100{suffix}.json']

wide, macros = [], {}
for benchmark, title in BENCHMARKS:
    macros[benchmark] = {}
    for task in TASKS:
        record = dict(benchmark=benchmark, task=task, online_env_steps=100000)
        for mode in ('native', 'mean', 'sampled'):
            row = select(benchmark, task, mode)
            e = original_eval(benchmark, task, mode)
            # Recompute displayed values from episode records, separately from validator output.
            success = sum(x['success'] for x in e['records']) / 100
            ret = sum(x['return_'] for x in e['records']) / 100
            assert math.isclose(success, row['success_rate'], abs_tol=1e-12)
            assert math.isclose(ret, row['avg_return'], abs_tol=1e-9)
            record[mode + '_success_rate'] = success
            record[mode + '_avg_return'] = ret
            record[mode + '_online_updates'] = row['online_gradient_updates']
        wide.append(record)
    for mode in ('native', 'mean', 'sampled'):
        # Equal weight for all5 tasks, including zero-success task4.
        records = [x for t in TASKS for x in original_eval(benchmark, t, mode)['records']]
        macros[benchmark][mode] = dict(success_rate=sum(x['success'] for x in records) / 500,
            avg_return=sum(x['return_'] for x in records) / 500, tasks=5, evaluation_episodes=500)

with (OUT / 'all_tasks_comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(wide[0])); writer.writeheader(); writer.writerows(wide)
summary = dict(snapshot_utc=snapshot['utc'], snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    seed=0, online_env_steps=100000, offline_updates=500000, native_online_updates=95001,
    dawn_online_updates=5625, macro_average_all5_tasks=macros, tasks=wide,
    uncertainty='One training seed. Episode samples do not estimate training-seed variability.')
(OUT / 'all_tasks_summary.json').write_text(json.dumps(summary, indent=2))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#26313A',
    'text.color': '#26313A', 'axes.titleweight': 'bold', 'axes.axisbelow': True,
    'pdf.fonttype': 42, 'svg.fonttype': 'none'})
with PdfPages(OUT / 'all_tasks_100k_comparison.pdf') as pdf:
    for mode in ('mean', 'sampled'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.8))
        for col, (benchmark, title) in enumerate(BENCHMARKS):
            for row_index, (metric, label) in enumerate([
                ('success_rate', 'Success rate'), ('avg_return', 'Average return (higher is better)')]):
                ax = axes[row_index, col]
                for method, eval_mode, offset, legend in [
                    ('native', 'native', -.19, 'QAM native | 95,001 updates'),
                    ('dawn', mode, .19, f'DAWN {mode} residual | 5,625 updates')]:
                    values = [select(benchmark, t, eval_mode)[metric] for t in TASKS]
                    bars = ax.bar(np.arange(5) + offset, values, width=.36, color=COLORS[method],
                        edgecolor='#FFFFFF', linewidth=.7, hatch='//' if method == 'dawn' else None, label=legend)
                    for bar, value in zip(bars, values):
                        text = f'{value:.0%}' if metric == 'success_rate' else f'{value:.1f}'
                        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 4 if metric == 'success_rate' else -5), textcoords='offset points',
                            ha='center', va='bottom' if metric == 'success_rate' else 'top', fontsize=10)
                ax.set_title(title + ' | ' + ('Success' if row_index == 0 else 'Return'), loc='left', fontsize=13)
                ax.set_xticks(range(5), [f'Task {t}' for t in TASKS])
                ax.set_ylabel(label); ax.grid(axis='y', color='#DCE2E6', linewidth=.65)
                ax.set_xlim(-.6, 4.6)
                if metric == 'success_rate':
                    ax.set_ylim(0, 1.12); ax.set_yticks(np.linspace(0, 1, 6)); ax.yaxis.set_major_formatter(PercentFormatter(1))
                else:
                    ax.set_ylim(-1100, 0); ax.set_yticks([-1000, -800, -600, -400, -200, 0])
        fig.suptitle('QAM native vs DAWN | 100k online environment steps', x=.075, ha='left', fontsize=19, weight='bold')
        fig.text(.075, .927, '500k offline checkpoint reused  ·  Seed 0  ·  100 paired evaluation episodes per task', fontsize=11)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.071, .91), ncol=2, frameon=False, fontsize=11)
        fig.text(.075, .028,
            'DAWN: 40k warmup; 40–50k mixed replay / UTD 0.25; 50–100k online-only / UTD 0.0625.\n'
            'Endpoints only. One training seed; training-seed uncertainty is unavailable. Native AntMaze starts a fresh online run; Cube resumes 50k.',
            fontsize=9, color='#53616C')
        fig.subplots_adjust(left=.075, right=.98, top=.83, bottom=.12, hspace=.4, wspace=.2)
        stem = 'comparison_primary' if mode == 'mean' else 'comparison_sampled'
        fig.savefig(OUT / (stem + '.png'), dpi=180)
        fig.savefig(OUT / (stem + '.svg'))
        pdf.savefig(fig); plt.close(fig)

lines = ['# 100k online comparison: all20 runs complete', '',
    'Both benchmarks use tasks1–5, seed0 and100 paired evaluation episodes per task. DAWN mean residual is the primary comparison.', '',
    '|Benchmark|Task|QAM native SR / return|DAWN mean SR / return|DAWN sampled SR / return|',
    '|---|---:|---:|---:|---:|']
for r in wide:
    cells = [f"{r[m+'_success_rate']:.0%} / {r[m+'_avg_return']:.2f}" for m in ('native','mean','sampled')]
    lines.append(f"|{r['benchmark']}|{r['task']}|" + '|'.join(cells) + '|')
lines += ['', 'Equal-weight macro averages across all5 tasks within each benchmark:', '',
    '|Benchmark|QAM native SR / return|DAWN mean SR / return|DAWN sampled SR / return|', '|---|---:|---:|---:|']
for b, title in BENCHMARKS:
    cells = [f"{macros[b][m]['success_rate']:.1%} / {macros[b][m]['avg_return']:.2f}" for m in ('native','mean','sampled')]
    lines.append('|' + title + '|' + '|'.join(cells) + '|')
lines += ['', 'Settings and interpretation:', '',
    '- Same100k online environment budget; QAM native95001 online updates versus DAWN5625. Update counts are not equal compute.',
    '- DAWN0–40k uses base-only rollout with no updates;40–50k uses128offline+128online atUTD0.25;50–100k uses256online atUTD0.0625.',
    '- Naive TD, state+base action, inherited Q/target, frozen base, scale0.1,10critics/minimum, actor entropy and autoalpha remain fixed.',
    '- Each task reuses its original offline500k checkpoint. Native Antmaze restarts online from that checkpoint; Cube native resumes the complete original50k state.',
    '- Antmaze native export has99999 online transitions and partial environment state, so it is not an exact resume checkpoint. Its100k environment steps and95001 updates are verified.',
    '- Return definitions differ: Antmaze counts unsuccessful steps; Cube sums per-object goal penalties. Do not combine raw returns across benchmarks.',
    '- One training seed;100 evaluation episodes do not establish stability across training seeds.', '',
    f"Source: final_complete_snapshot.json, SHA256 {summary['snapshot_sha256']}. Completed results passed source/checkpoint hashes, schedule, update, replay, aggregation and paired-state checks."]
(OUT / 'final_result.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(status='passed',macro_averages=macros,files=['comparison_primary.png','comparison_sampled.png',
    'all_tasks_100k_comparison.pdf','all_tasks_comparison.csv','all_tasks_summary.json','final_result.md']), indent=2))
