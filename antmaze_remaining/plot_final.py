"""Source-backed five-task tables and curves, with task1's existing 100k checkpoint."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/antmaze_large_tasks2_5_20260913'
snapshot = json.loads((OUT / 'full_snapshot.json').read_text())
assert snapshot['files']['results/suite_status.json']['state'] == 'passed'
assert snapshot['files']['results/suite_status.json']['completed_tasks'] == [2, 3, 4, 5]
assert len(snapshot['checkpoint_audit']) == 12
assert all(v['matched'] for v in snapshot['checkpoint_audit'].values())

rows = []
for task in (2, 3, 4, 5):
    audit = json.loads((OUT / f'task{task}_completion_validation.json').read_text())
    assert audit['status'] == 'passed' and audit['snapshot_utc'] == snapshot['utc']
    for row in audit['summary']:
        rows.append(dict(row, provenance='new 100k run'))

prior = json.loads((ROOT / 'output/antmaze_large_task1_20260913/final_complete_snapshot.json').read_text())
base = prior['files']['results/native/fixed_eval/eval_000000_100.json']['records']
for method, name in [
    ('Offline QAM', 'results/native/fixed_eval/eval_000000_100.json'),
    ('QAM native', 'results/native/fixed_eval/eval_050000_100.json'),
    ('DAWN mean', 'results/dawn/eval_100000_100_mean_residual.json'),
    ('DAWN sampled', 'results/dawn/eval_100000_100.json'),
]:
    e = prior['files'][name]
    records = e['records']
    assert e['episodes'] == len(records) == 100
    assert all((a['reset_seed'], a['initial_hash']) == (b['reset_seed'], b['initial_hash']) for a, b in zip(records, base))
    assert math.isclose(sum(x['success'] for x in records)/100, e['success'], abs_tol=1e-12)
    assert math.isclose(sum(x['return_'] for x in records)/100, e['return_mean'], abs_tol=1e-9)
    steps = e['step']
    updates = max(0, steps-4999) if method == 'QAM native' else max(0, steps-80000)//4
    rows.append(dict(task=1, method=method, online_env_steps=steps, online_updates=updates,
                     success_rate=e['success'], avg_return=e['return_mean'],
                     gained_vs_offline=sum(a['success'] > b['success'] for a,b in zip(records,base)),
                     lost_vs_offline=sum(a['success'] < b['success'] for a,b in zip(records,base)),
                     source='antmaze_large_task1_20260913/final_complete_snapshot.json#'+name,
                     provenance='existing task1 checkpoint; original run continued to150k'))
rows.sort(key=lambda r: (r['task'], r['online_env_steps'], r['method']))
(OUT / 'five_tasks_comparison.json').write_text(json.dumps(rows, indent=2))
with (OUT / 'five_tasks_comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def select(task, method, step=None):
    values = [r for r in rows if r['task']==task and r['method']==method
              and (step is None or r['online_env_steps']==step)]
    return sorted(values, key=lambda r:r['online_env_steps'])

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})
colors = {'QAM native':'#1764AB', 'DAWN mean':'#D96022', 'DAWN sampled':'#80858D'}
pdf = PdfPages(OUT / 'five_tasks_learning_curves.pdf')
for metric, ylabel, stem in [('success_rate','Success rate','success'),
                              ('avg_return','Average return (higher is better)','return')]:
    fig, axes = plt.subplots(2,3,figsize=(13.5,7.4),sharex=True,sharey=True)
    for task, ax in zip(range(1,6),axes.flat):
        b=select(task,'Offline QAM')[0]
        n=select(task,'QAM native')[0]
        ax.axvspan(0,80,color='#F2F3F5',zorder=0)
        ax.axvline(80,color='#A8ADB4',linewidth=.9,linestyle=':')
        ax.plot([0,50],[b[metric],n[metric]],color=colors['QAM native'],marker='o',
                linewidth=2,markersize=5,zorder=5)
        for method in ('DAWN mean','DAWN sampled'):
            values=select(task,method)
            ax.plot([0,80],[b[metric]]*2,color=colors[method],linewidth=1.2,linestyle=':')
            # 80k denotes the same frozen policy, not an extra evaluation.
            xs=[80]+[r['online_env_steps']/1000 for r in values]
            ys=[b[metric]]+[r[metric] for r in values]
            ax.plot(xs,ys,color=colors[method],linewidth=1.9 if method=='DAWN mean' else 1.4,
                    linestyle='-' if method=='DAWN mean' else '--')
            ax.scatter(xs[1:],ys[1:],color=colors[method],s=25,
                       marker='o' if method=='DAWN mean' else 's',zorder=4)
        ax.set_title(f'Task {task}'+(' *' if task==1 else ''),loc='left',weight='bold',pad=10)
        ax.grid(axis='y',color='#E0E3E7',linewidth=.7)
        ax.set_xlim(-4,111)
        ax.set_xticks([0,50,80,90,100])
        if metric=='success_rate':
            ax.set_ylim(-.055,1.14)
            ax.set_yticks([0,.25,.5,.75,1])
            ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            final=select(task,'DAWN mean',100000)[0]
            ax.annotate(f"{n[metric]:.0%}",(50,n[metric]),xytext=(0,8),textcoords='offset points',
                        ha='center',color=colors['QAM native'],fontsize=10)
            ax.annotate(f"{final[metric]:.0%}",(100,final[metric]),
                        xytext=(0,-16) if task==2 else (1,8),textcoords='offset points',
                        ha='center',color=colors['DAWN mean'],fontsize=10)
        else:
            ax.set_ylim(-1050,-210)
            ax.set_yticks([-1000,-800,-600,-400])
    panel=axes.flat[5]
    panel.set_axis_off()
    legend=[Line2D([0],[0],color=colors['QAM native'],marker='o',lw=2,label='QAM native'),
            Line2D([0],[0],color=colors['DAWN mean'],marker='o',lw=2,label='DAWN mean residual'),
            Line2D([0],[0],color=colors['DAWN sampled'],marker='s',ls='--',lw=1.4,label='DAWN sampled residual')]
    panel.legend(handles=legend,loc='upper left',frameon=False,fontsize=11)
    notes=('Shared offline pretraining: 500k updates\n'
           'QAM: 50k env steps / 45,001 updates\n'
           'DAWN: 100k env steps / 5,000 updates\n\n'
           'Gray region: DAWN base-only warmup.\n'
           'Dotted segment: unchanged base policy.\n'
           'Markers: actual 100-episode evaluations.\n'
           'Connecting lines only guide the eye.\n\n'
           '* Task1 uses its existing 100k checkpoint;\n'
           '  no 90k evaluation. Original run reached150k.')
    panel.text(.03,.64,notes,transform=panel.transAxes,va='top',fontsize=9.5,color='#4E5662',linespacing=1.45)
    fig.suptitle(f'AntMaze-large | {ylabel}',x=.06,ha='left',fontsize=17,weight='bold')
    fig.supxlabel('Online environment steps (thousands)',y=.057,fontsize=11)
    fig.supylabel(ylabel,x=.015,fontsize=11)
    fig.text(.5,.014,'Seed0 per task · 100 paired episodes per point · Mean refers only to residual; base policy remains stochastic.',
             ha='center',fontsize=9,color='#555F6B')
    fig.subplots_adjust(left=.07,right=.985,top=.89,bottom=.14,hspace=.32,wspace=.2)
    fig.savefig(OUT / f'five_tasks_{stem}_env_steps.png',dpi=170,facecolor='white')
    pdf.savefig(fig,facecolor='white')
    plt.close(fig)
pdf.close()

summary=[]
for task in range(1,6):
    record={'task':task}
    for label,method,step in [('offline','Offline QAM',0),('qam_50k','QAM native',50000),
                              ('dawn_90k','DAWN mean',90000),('dawn_100k','DAWN mean',100000)]:
        values=select(task,method,step)
        record[label]=dict(success=values[0]['success_rate'],avg_return=values[0]['avg_return']) if values else None
    summary.append(record)
macro={label:{metric:sum(r[label][metric] for r in summary)/5 for metric in ('success','avg_return')}
       for label in ('offline','qam_50k','dawn_100k')}
(OUT/'five_tasks_summary.json').write_text(json.dumps(dict(tasks=summary,macro_average_all_five=macro),indent=2))
lines=['# AntMaze-large 五个任务结果','','所有任务均使用seed0；每个评估点为固定100 episodes。DAWN mean仅指residual动作均值，base仍随机采样。',
       '', '单元格为success rate / avg return（越高越好）。','',
       '| Task | Offline500k | QAM native50k | DAWN mean90k | DAWN mean100k |',
       '|---|---:|---:|---:|---:|']
def fmt(value): return '未评估' if value is None else f"{value['success']:.0%} / {value['avg_return']:.2f}"
for r in summary:lines.append('| '+str(r['task'])+' | '+' | '.join(fmt(r[k]) for k in ('offline','qam_50k','dawn_90k','dawn_100k'))+' |')
lines+=['','Task1使用先前150k run的100k检查点，没有90k评估，本轮没有重跑Task1。Task2–5新run在100k停止。',
        '', 'QAM为50k online env steps、45,001次online更新；DAWN为100k online env steps（包含80k无更新warmup）、5,000次更新。每个task从对应的同一offline500k checkpoint开始；这不是相同环境预算或更新预算的比较。',
        '', '在100k检查点，DAWN mean的成功率在Task1/3/5距QAM分别为2/2/1个百分点，Task2为8个百分点，Task4为86个百分点。Task4的offline成功率为0%，QAM native提升至86%，DAWN在90k和100k评估仍为0%。单个seed及这些观测不足以确定差距原因。',
        '', '五个任务等权平均：'+', '.join(f"{k}={v['success']:.1%}, return={v['avg_return']:.2f}" for k,v in macro.items())+'。均值包含Task4；不对缺失Task1的90k结果计算五任务均值。',
        '', '完成检查：四组任务的12个模型文件SHA256、官方源码/参数、继承本task offline Q/target Q、frozen base、更新次数、evaluation RNG与完整episode聚合均通过。所有完整结果已同步至synced_results。',
        '', f'![成功率]({OUT / "five_tasks_success_env_steps.png"})',
        '', f'![平均回报]({OUT / "five_tasks_return_env_steps.png"})', '']
(OUT/'final_result.md').write_text('\n'.join(lines))
(OUT/'final_report_validation.json').write_text(json.dumps(dict(status='passed',snapshot_utc=snapshot['utc'],
    completed_tasks=[2,3,4,5],checkpoint_hashes_verified=12,task1_existing_records_recomputed=True,
    task1_90k_not_imputed=True,table_rows=len(rows),curves_use_only_measured_points_and_unchanged_warmup=True),indent=2))
print(json.dumps(dict(summary=summary,macro_average_all_five=macro),indent=2))
