"""Compare fixed QAM native and DAWN UTD0.25 runs on environment and update axes."""
from pathlib import Path
import csv
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
REF = WORK / 'output/warmup80k_150k_task234_20260911'
OUT = WORK / 'output/qam_native_vs_dawn025_axes_20260912'
OUT.mkdir(parents=True, exist_ok=True)
sources = {}


def load(path):
    sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def validate_eval(e):
    assert 0 <= e['success'] <= 1 and math.isfinite(e['return_mean'])
    if 'records' in e:
        assert len(e['records']) == e['episodes']
        assert abs(sum(r['success'] for r in e['records']) / e['episodes'] - e['success']) < 1e-10
        assert abs(sum(r['return_'] for r in e['records']) / e['episodes'] - e['return_mean']) < 1e-8
    else:
        assert e['raw_record_aggregates_verified']


rows, checks = [], []
reference_csv = REF / 'all5_results.csv'
sources[str(reference_csv.resolve())] = hashlib.sha256(reference_csv.read_bytes()).hexdigest()
with reference_csv.open() as stream:
    dawn_reference_rows = list(csv.DictReader(stream))
for task in range(1, 6):
    native = ROOT / ('runs/native' if task == 1 else 'runs/task2/native' if task == 2 else f'runs/cube5/task{task}_native')
    dawn = REF / 'settings_audit' / f'task{task}_warm'
    qcfg, qdone = load(native / 'config.json'), load(native / 'DONE.json')
    dcfg, dactual, ddone = (load(dawn / name) for name in ('requested_config.json', 'actual_agent_config.json', 'DONE.json'))
    assert qcfg['offline_sha256'] == dcfg['offline_sha256']
    assert qcfg['seed'] == dcfg['seed'] == 0
    assert qcfg['offline_steps'] == dcfg['offline_steps'] == 500000
    assert qcfg['native_start'] == 5000
    assert qdone['steps'] == 50000 and qdone['updates'] == 45001
    assert ddone['steps'] == 150000 and ddone['updates'] == 17500
    assert dactual['utd'] == .25 and dactual['batch_size'] == 256 and dactual['warmup'] == 80000
    assert dactual['td_target'] == 'hard' and dactual['inherited_Q_and_target']
    # Use the original 50-episode QAM checkpoints; use the predesignated 100-episode final.
    native_evals = [native / f'eval_{step:06d}_050.json' for step in (0, 5000, 10000, 20000, 30000, 40000)]
    native_evals.append(native / 'eval_050000_100.json')
    qmetrics_path = native / 'metrics.jsonl'
    sources[str(qmetrics_path.resolve())] = hashlib.sha256(qmetrics_path.read_bytes()).hexdigest()
    qmetrics = {m['step']: m for s in qmetrics_path.read_text().splitlines() for m in [json.loads(s)]}
    for path in native_evals:
        e = load(path)
        validate_eval(e)
        updates = max(0, e['step'] - qcfg['native_start'] + 1)
        if e['step']:
            assert qmetrics[e['step']]['updates'] == updates
        rows.append(dict(task=task, method='QAM native', env_steps=e['step'], gradient_updates=updates,
            success=e['success'], return_mean=e['return_mean'], episodes=e['episodes'],
            evaluation_mode='sampled_base', source=str(path.resolve())))
    expected_final = load(REF / 'qam_comparison_sources' / f'task{task}_native' / 'eval_050000_100.json')
    qfinal = load(native / 'eval_050000_100.json')
    assert qfinal['records'] == expected_final['records']
    dawn_evals = [dawn / 'eval_000000_100.json', *sorted(dawn.glob('eval_*_100_mean_residual.json'))]
    assert len(dawn_evals) == 4
    dmetrics = load(dawn / 'metrics_verified.json')
    assert dmetrics['update_schedule_verified'] and dmetrics['all_finite']
    assert dmetrics['last']['updates'] == 17500
    for path in dawn_evals:
        e = load(path)
        validate_eval(e)
        assert e['episodes'] == 100
        updates = max(0, (e['step'] - 80000) // 4)
        reference_mode = 'offline' if e['step'] == 0 else 'mean_residual'
        reference_point = [r for r in dawn_reference_rows if int(r['task']) == task and
                           int(r['step']) == e['step'] and r['mode'] == reference_mode]
        assert len(reference_point) == 1 and int(reference_point[0]['updates']) == updates
        assert float(reference_point[0]['success']) == e['success']
        assert float(reference_point[0]['return_mean']) == e['return_mean']
        if e['step']:
            assert e['deterministic_residual']
        rows.append(dict(task=task, method='DAWN UTD0.25', env_steps=e['step'], gradient_updates=updates,
            success=e['success'], return_mean=e['return_mean'], episodes=100,
            evaluation_mode='offline_base' if e['step'] == 0 else 'mean_residual_sampled_base', source=str(path.resolve())))
    checks.append(dict(task=task, same_offline_checkpoint=True, native_updates_from_metrics=True,
        dawn_updates_from_audited_schedule_and_results=True, fixed_native_final100_matches_previous_report=True))

assert len(rows) == 55
with (OUT / 'plot_data.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.edgecolor':'#B8C2CD', 'axes.labelcolor':'#35495D',
    'text.color':'#172737', 'xtick.color':'#516170', 'ytick.color':'#516170',
    'pdf.fonttype':42, 'svg.fonttype':'none', 'savefig.facecolor':'white'})
STYLES = {'QAM native':('#C77524','s'), 'DAWN UTD0.25':('#2463A6','o')}
legend = [Line2D([0],[0],color=c,marker=m,lw=1.8,label=label) for label,(c,m) in STYLES.items()]


def plot(axis_key):
    env = axis_key == 'env_steps'
    fig, axes = plt.subplots(5,2,figsize=(13.8,15.3),sharex=True,sharey='col')
    fig.subplots_adjust(left=.09,right=.97,top=.865,bottom=.123,hspace=.32,wspace=.20)
    suffix = 'env_steps' if env else 'gradient_updates'
    title = 'Online environment steps' if env else 'Online gradient updates'
    fig.suptitle(f'QAM native vs. DAWN | {title}', x=.09,y=.977,ha='left',fontsize=21,weight='bold')
    fig.text(.09,.947,'Cube Double tasks 1–5 · Same task-specific QAM offline500k checkpoints · Seed 0',fontsize=12)
    fig.text(.09,.920,'QAM: 50k env / 45,001 updates   |   DAWN: 150k env / 17,500 updates, UTD 0.25',fontsize=11.5,color='#516170')
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.08,.900),ncol=2,frameon=False)
    for task in range(1,6):
        for j, metric in enumerate(('success','return_mean')):
            ax = axes[task-1,j]
            if env:
                ax.axvspan(0,80,color='#EFF4F9',zorder=0)
                ax.axvline(80,color='#8EAAC5',ls='--',lw=.8)
            for method,(color,marker) in STYLES.items():
                data = sorted([r for r in rows if r['task']==task and r['method']==method],key=lambda r:r[axis_key])
                # Initial evaluations are isolated; no synthetic observations during warmup.
                initial, learning = data[0], data[1:]
                ax.scatter(initial[axis_key]/1000,initial[metric],marker=marker,
                    facecolors='white' if initial['episodes']==50 else color,
                    edgecolors=color,s=30,zorder=4,linewidths=1.2)
                xs,ys = [r[axis_key]/1000 for r in learning], [r[metric] for r in learning]
                ax.plot(xs,ys,color=color,lw=1.6,zorder=2)
                for r in learning:
                    ax.scatter(r[axis_key]/1000,r[metric],marker=marker,
                        facecolors='white' if r['episodes']==50 else color,edgecolors=color,
                        s=28,zorder=3,linewidths=1.2)
            ax.set_title(f'Task {task}  |  '+('Success rate ↑' if j==0 else 'Average return ↑'),loc='left',fontsize=12,weight='bold')
            ax.grid(axis='y',color='#E6EBF0',lw=.7)
            ax.set_axisbelow(True)
            if j==0:
                ax.set_ylim(-.055,1.08)
                ax.set_yticks([0,.2,.4,.6,.8,1.])
                ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            else:
                ax.set_ylim(-1015,35)
                ax.set_yticks([-1000,-800,-600,-400,-200,0])
            ax.set_xlim((-4,155) if env else (-1.2,47))
            ax.set_xticks([0,25,50,80,100,125,150] if env else [0,5,10,17.5,25,35,45])
            if task==5:
                ax.set_xlabel('Online environment steps (k)' if env else 'Online optimizer updates (k)')
    if env:
        first='Blue shading: DAWN base-only warmup (0–80k); QAM starts updates at 5k. Warmup is included in the env budget.'
    else:
        first='Only online updates are counted. Equal update counts do not imply equal environment steps or equal compute.'
    fig.text(.09,.084,first,fontsize=10)
    fig.text(.09,.060,'Hollow markers: QAM 0–40k evaluations, 50 episodes. Filled markers: QAM final + all DAWN evaluations, 100 episodes.',fontsize=10)
    fig.text(.09,.036,'DAWN uses mean residual with sampled base. Initial estimates use different episode counts despite identical offline weights.',fontsize=10)
    fig.text(.09,.012,'Points are measured checkpoints; lines only guide the eye. No smoothing, no 80k evaluation, no extension beyond either run.',fontsize=10)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'qam_native_vs_dawn025_{suffix}.{ext}',dpi=190)
    plt.close(fig)


plot('env_steps')
plot('gradient_updates')
(OUT/'validation.json').write_text(json.dumps(dict(status='passed',tasks=checks,source_hashes=sources,
    points=len(rows),dawn_run='warmup80k_150k_20260911 + warmup80k_150k_task234_20260911',
    native_final_episodes=100,native_intermediate_episodes=50,dawn_episodes=100,
    training_seeds=[0],offline_updates_excluded_from_update_axis=True,
    native_update_formula='max(0, env_steps - 5000 + 1)',
    dawn_update_formula='max(0, floor((env_steps - 80000) / 4))',
    extrapolation=False,smoothed=False),indent=2)+'\n')
(OUT/'README.md').write_text('''# QAM native 与 DAWN UTD0.25 双横轴对照

展示 cube-double 五个任务的成功率和平均回报。DAWN 采用已完成的80k warmup /150k online实验，naive TD、batch256、UTD0.25、state+base action、继承critic、residual scale0.1。各任务两种方法使用同一offline500k checkpoint，seed0。

- 环境步横轴包含warmup。QAM到50k结束，共45001次更新；DAWN到150k结束，共17500次更新。蓝色背景仅表示DAWN的0–80k warmup；QAM从5k开始更新。
- 梯度更新横轴仅计算online，不计共同的offline500k。QAM的5k环境步评估对应第1次更新；DAWN约100k环境步评估对应第5000次更新。同更新次数不表示同环境交互预算或同计算量。
- DAWN使用均值residual，base仍随机采样。QAM保留其原始采样评估。
- QAM的0–40k评估点各50 episodes；QAM的50k最终点及DAWN所有点各100 episodes。空心/实心标记区分50/100。保留原始初始评估，不把不同数量episodes产生的初始差异改成同一个值。
- 使用预定的final100评估，没有选择最佳checkpoint。所有点来自存档评估，未插入80k评估，未平滑或外推。连接线仅为视觉引导。
- 数据见plot_data.csv；QAM更新次数与逐点日志核对，DAWN依据已审计的更新调度和结果记录核对。评估聚合来自原始records或已验证摘要，输入文件哈希见validation.json。
''')
print(json.dumps({'status':'passed','points':len(rows),'figures':[str(p) for p in sorted(OUT.glob('*.png'))]},indent=2))
