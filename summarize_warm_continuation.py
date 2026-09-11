"""Validate and plot the old native/DAWN runs and the DAWN continuation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT/'runs'
OUT = ROOT/'results_warm_continuation'
OUT.mkdir(exist_ok=True)
sources = {}


def load(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def evaluation(path):
    e = load(path)
    assert e['episodes'] == len(e['records'])
    assert len({r['reset_seed'] for r in e['records']}) == e['episodes']
    np.testing.assert_allclose(e['success'], np.mean([r['success'] for r in e['records']]), atol=1e-12)
    np.testing.assert_allclose(e['return_mean'], np.mean([r['return_'] for r in e['records']]), atol=1e-12)
    assert all(np.isfinite([r['return_'], r['length'], r['success']]).all() for r in e['records'])
    return e


def historical(arm):
    metrics = [json.loads(s) for s in (RUNS/arm/'metrics.jsonl').read_text().splitlines()]
    updates = {x['step']: x['updates'] for x in metrics}
    updates[0] = 0
    points = []
    for p in sorted((RUNS/arm).glob('eval_*_050.json')):
        e = evaluation(p)
        points.append(dict(e, updates=updates[e['step']]))
    return points


old_validation = load(ROOT/'results_policy_decorator/validation.json')
for name, expected in old_validation['sources'].items():
    assert hashlib.sha256((RUNS/name).read_bytes()).hexdigest() == expected, name
done = load(RUNS/'warm_extended/DONE.json')
assert done['updates'] == 60000
config = load(RUNS/'warm_extended/config.json')
assert config['utd'] == .25 and config['batch_size'] == 1024
assert config['warmup_counted_from_original_start'] == 20000
assert done['final_flow_hash'] == config['frozen_flow_hash']
checks = load(RUNS/'WARM_CONTINUATION_CHECKS_PASSED.json')
assert checks['status'] == 'passed'
native = historical('native')
warm = historical('warm')
extension = []
milestones = {}
for path in sorted((RUNS/'warm_extended').glob('milestone_u*.json')):
    m = load(path)
    assert m['updates'] not in milestones
    m['evaluations'] = [evaluation(RUNS/'warm_extended'/p) for p in m['evaluation_files']]
    e = m['evaluations'][0]
    assert e['step'] == m['step'] and e['episodes'] == 50
    assert m['updates'] <= (m['step']-20000)//4
    assert (m['step']-20000)//4 - m['updates'] <= 1
    milestones[m['updates']] = m
    if m['updates'] > 7500:
        extension.append(dict(e, updates=m['updates']))
    if m['updates'] in (45001,60000):
        sampled, mean = m['evaluations'][1:]
        assert sampled['episodes'] == mean['episodes'] == 100
        assert not sampled['deterministic_residual'] and mean['deterministic_residual']
        assert sampled['records'][:50] == e['records']
assert sorted(milestones) == [7500,15000,22500,30000,37500,45001,52500,60000]
for a,b in zip(warm[-1]['records'], milestones[7500]['evaluations'][0]['records']):
    for key in ('episode','reset_seed','success','return_','length','decisions'):
        assert a[key] == b[key], key

rows = []
full_evals = {}
cases = [('QAM native',45001,256,evaluation(RUNS/'native/eval_050000_100.json')),
         ('DAWN original',7500,1024,evaluation(RUNS/'warm/eval_050000_100.json')),
         ('DAWN matched updates',45001,1024,milestones[45001]['evaluations'][1]),
         ('DAWN extended',60000,1024,milestones[60000]['evaluations'][1])]
for name, updates, batch, e in cases:
    rows.append(dict(method=name,online_steps=e['step'],online_updates=updates,batch_size=batch,
        sampled_training_rows=updates*batch,eval_episodes=e['episodes'],
        success_percent=100*e['success'],return_mean=e['return_mean'],
        mean_episode_length=float(np.mean([r['length'] for r in e['records']]))))
    full_evals[name] = e
with (OUT/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

diagnostics={}
for name,e in full_evals.items():
    ref=full_evals['QAM native']['records']
    assert [r['reset_seed'] for r in ref] == [r['reset_seed'] for r in e['records']]
    common=[(a,b) for a,b in zip(ref,e['records']) if a['success'] and b['success']]
    diagnostics[name]=dict(both_success=len(common),
        native_only_success=sum(a['success'] and not b['success'] for a,b in zip(ref,e['records'])),
        method_only_success=sum(b['success'] and not a['success'] for a,b in zip(ref,e['records'])),
        native_length_on_common_success=float(np.mean([a['length'] for a,b in common])) if common else None,
        method_length_on_common_success=float(np.mean([b['length'] for a,b in common])) if common else None)
(OUT/'paired_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12.8,8.5))
for col,xkey,xlabel in [(0,'step','Primitive online steps (thousands)'),
                       (1,'updates','Cumulative online updates (thousands)')]:
    for row,ykey,factor,ylabel in [(0,'success',100,'Success rate (%)'),
                                  (1,'return_mean',1,'Mean episode return')]:
        ax=axes[row,col]
        for points,label,color,marker in [(native,'QAM native','#53575D','o'),
                                         (warm,'DAWN original 50k run','#2463A6','s'),
                                         ([warm[-1]]+extension,'DAWN continuation','#2463A6','D')]:
            ax.plot([p[xkey]/1000 for p in points],[p[ykey]*factor for p in points],
                    label=label,color=color,marker=marker,markersize=4,linewidth=1.8,
                    linestyle='--' if 'original' in label else '-')
        if row==0:
            ax.set_ylim(0,102)
        ax.set_ylabel(ylabel);ax.set_xlabel(xlabel);ax.grid(axis='y',color='#E8EAED')
        boundary=50 if col==0 else 45.001
        ax.axvline(boundary,color='#9DA3AA',linestyle=':',linewidth=1)
        ax.text(boundary,ax.get_ylim()[0], ' 50k resume' if col==0 else ' 45,001 updates',
                rotation=90,va='bottom',ha='right',fontsize=8,color='#666666')
fig.suptitle('Does longer DAWN training close the gap?',x=.075,ha='left',fontsize=16)
fig.text(.075,.923,'Cube-double task1 | Seed 0 | Shared 500k QAM pretraining | 50 paired evaluation episodes per point',fontsize=10)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.066),ncol=3,frameon=False)
fig.text(.075,.046,'DAWN retains UTD 0.25, batch 1024, online-only replay and all optimizer states. Native uses batch 256.',fontsize=9)
fig.text(.075,.024,'Matching updates uses more environment data. Curves show sampled actions; 100-episode checkpoint scores are reported separately.',fontsize=9)
fig.subplots_adjust(left=.075,right=.98,top=.87,bottom=.17,wspace=.22,hspace=.28)
fig.savefig(OUT/'learning_curves.png',dpi=180);fig.savefig(OUT/'learning_curves.pdf');plt.close(fig)

lines=['# DAWN naive continuation','',
       '同一 seed 0；QAM offline 500k；继承整个 50k online checkpoint；DAWN 超参数、online-only replay 和优化器状态保持不变。','',
       '| 方法 | Online 环境步 | 累计更新 | 成功率（100 eps） | 平均回报 |',
       '|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['method']} | {r['online_steps']:,} | {r['online_updates']:,} | {r['success_percent']:.0f}% | {r['return_mean']:.2f} |")
lines+=['','均值残差辅助评估（100 episodes；base proposal 仍随机）：']
for u in (45001,60000):
    e=milestones[u]['evaluations'][2]
    lines.append(f"- {u:,} updates: {100*e['success']:.0f}%，平均回报 {e['return_mean']:.2f}。")
lines+=['','这次只对齐更新次数；DAWN 使用更多环境交互，每次 batch 1024，而 native 为 256。',
        '旧 50k checkpoint 没有保存末尾 chunk 的剩余四个动作，因此从保存的环境状态重新采样 chunk；其余训练状态完整恢复。',
        '50k checkpoint 的 50-episode 结果复现通过；含 pending update 的断点恢复与连续运行完整状态一致。',
        '一个训练 seed 无法确定跨 seed 稳定性。图中每点评估 50 episodes，表中为 100 episodes。','',
        '![Learning curves](learning_curves.png)','']
(OUT/'summary.md').write_text('\n'.join(lines))
(OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'validation.json').write_text(json.dumps(dict(status='passed',sources=sources,
    report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(rows,indent=2))
