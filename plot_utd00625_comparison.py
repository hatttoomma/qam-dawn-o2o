"""Produce the completed five-task UTD comparison from verified remote results."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

OUT=Path(__file__).resolve().parents[2]/'output/utd00625_warmup80k_150k_20260911'
PLOTS=OUT/'plots'
state=json.loads((OUT/'monitor_state.json').read_text())
assert state['suite_status.json']['event']=='all_training_complete'
assert all(t['complete'] and t['final_checkpoint_sha256_verified'] for t in state['tasks'].values())
with (OUT/'all_results.csv').open() as f:
    rows=list(csv.DictReader(f))
for r in rows:
    for k in ('task','step','updates'): r[k]=int(r[k])
    for k in ('utd','success','return_mean'): r[k]=float(r[k])
assert len(rows)==70
paired=json.loads((OUT/'paired_comparison.json').read_text())
assert len(paired)==10
def points(task,utd,mode='mean_residual'):
    return sorted([r for r in rows if r['task']==task and r['utd']==utd and r['mode']==mode],key=lambda r:r['step'])
for t in range(1,6):
    assert points(t,.25)[-1]['updates']==17500
    assert points(t,.0625)[-1]['updates']==4375
    assert points(t,.25,'offline')==[dict(points(t,.0625,'offline')[0],utd=.25)]
PLOTS.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'axes.edgecolor':'#B8C2CD','text.color':'#172737',
    'axes.labelcolor':'#35495D','pdf.fonttype':42,'svg.fonttype':'none'})
OLD,NEW='#D68826','#2374AB'
legend=[Line2D([0],[0],color=OLD,marker='s',lw=2,label='UTD 0.25 · 17,500 updates'),
        Line2D([0],[0],color=NEW,marker='o',lw=2,label='UTD 0.0625 · 4,375 updates')]
fig,axes=plt.subplots(1,2,figsize=(13,6.3))
fig.subplots_adjust(left=.085,right=.98,top=.73,bottom=.18,wspace=.21)
fig.suptitle('DAWN UTD ablation | 150k online steps',x=.085,y=.965,ha='left',fontsize=21,weight='bold')
fig.text(.085,.901,'Same 80k warmup data and initial state · Naive TD · batch 256 · state + base action',fontsize=11.5)
fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.075,.857),ncol=2,frameon=False)
for ax,metric,title in zip(axes,('success','return_mean'),('Final success rate ↑','Final average return ↑')):
    ax.set_title(title,loc='left',fontsize=14,weight='bold',pad=12)
    for task in range(1,6):
        a=points(task,.25)[-1][metric]
        b=points(task,.0625)[-1][metric]
        ax.plot([a,b],[task,task],color='#C2CBD4',lw=2,zorder=1)
        ax.scatter(a,task+.09,color=OLD,marker='s',s=58,zorder=3)
        ax.scatter(b,task-.09,color=NEW,marker='o',s=58,zorder=3)
        for v,y,c,dy in ((a,task+.09,OLD,-12),(b,task-.09,NEW,10)):
            label=f'{v:.0%}' if metric=='success' else f'{v:.1f}'
            ax.annotate(label,(v,y),xytext=(0,dy),textcoords='offset points',ha='center',va='center',color=c,fontsize=10)
    ax.set_yticks(range(1,6),[f'Task {t}' for t in range(1,6)])
    ax.set_ylim(5.55,.45)
    ax.grid(axis='x',color='#E8EDF1',lw=.7)
    ax.set_axisbelow(True)
    if metric=='success':
        ax.set_xlim(-.06,1.07)
        ax.set_xticks([0,.2,.4,.6,.8,1])
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    else: ax.set_xlim(-1030,45)
fig.text(.085,.07,'Mean residual evaluation; base still sampled. One training seed (0), 100 matched evaluation episodes per task.',fontsize=10.5)
for suffix in ('png','pdf'):
    fig.savefig(PLOTS/f'utd_final_comparison.{suffix}',dpi=200)
plt.close(fig)

fig,axes=plt.subplots(5,2,figsize=(13.5,16),sharex=True,sharey='col')
fig.subplots_adjust(left=.09,right=.97,top=.875,bottom=.095,hspace=.3,wspace=.2)
fig.suptitle('DAWN UTD ablation | Learning curves',x=.09,y=.977,ha='left',fontsize=22,weight='bold')
fig.text(.09,.951,'Same 150k environment budget and exact 80k warmup prefix · Tasks 1–5',fontsize=12.5)
fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.08,.938),ncol=2,frameon=False)
for i,task in enumerate(range(1,6)):
    for j,metric in enumerate(('success','return_mean')):
        ax=axes[i,j]
        ax.axvspan(0,80,color='#EEF1F4')
        ax.axvline(80,color='#9AA7B4',ls='--',lw=1)
        base=points(task,.25,'offline')[0][metric]
        ax.scatter(0,base,color='#58616B',marker='D',s=30,zorder=4)
        ax.axhline(base,color='#B7BFC7',lw=.8,ls=':')
        for utd,color,marker in ((.25,OLD,'s'),(.0625,NEW,'o')):
            p=points(task,utd)
            ax.plot([r['step']/1000 for r in p],[r[metric] for r in p],color=color,marker=marker,lw=1.8,ms=5)
        ax.set_title(f'Task {task} | '+('Success rate ↑' if j==0 else 'Average return ↑'),loc='left',fontsize=12,weight='bold')
        ax.set_xlim(-5,158)
        ax.set_xticks([0,40,80,100,120,150])
        ax.grid(axis='y',color='#E8EDF1',lw=.7)
        ax.set_axisbelow(True)
        if j==0:
            ax.set_ylim(-.04,1.08)
            ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        else: ax.set_ylim(-1000,30)
        if i==4: ax.set_xlabel('Online environment steps (k)')
fig.text(.09,.054,'Mean residual · Seed 0 · 100 matched episodes per point · Gray band: base-only warmup · Dotted line: offline reference.',fontsize=10)
fig.text(.09,.03,'Only measured checkpoints are plotted. Lines between ~100k, ~120k and 150k are visual guides; no 80k evaluation.',fontsize=10,color='#566475')
for suffix in ('png','pdf'):
    fig.savefig(PLOTS/f'utd_learning_curves.{suffix}',dpi=180)
plt.close(fig)

lines=['# DAWN UTD0.0625 vs UTD0.25','',
    '五个任务均已完成150k online环境步（包含80k warmup）。唯一训练设置变化为UTD从0.25降到0.0625，总更新次数从17500降到4375。复用相同warmup数据、初始agent和恢复状态；每个任务seed0，100条固定evaluation episodes。', '',
    '主结果：均值residual；base policy仍随机采样。', '',
    '| Task | Offline success | UTD0.25 success | UTD0.0625 success | Δ success (pp) | UTD0.25 return | UTD0.0625 return | Δ return |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for t in range(1,6):
    p=next(r for r in paired if r['task']==t and r['mode']=='mean_residual')
    baseline=points(t,.25,'offline')[0]['success']
    lines.append(f"| {t} | {baseline:.0%} | {p['old_success']:.0%} | {p['new_success']:.0%} | {p['delta_success_pp']:+.0f} | {p['old_return']:.2f} | {p['new_return']:.2f} | {p['delta_return']:+.2f} |")
lines+=['','随机residual评估和逐episode胜负变化见paired_comparison.csv。上述差异来自单个训练seed，不能据此认定跨训练seed的稳定性。', '',
    '共同设置：继承Q/target Q，naive TD，batch256，state+base action，residual scale0.1，online-only replay，10 critics/min聚合，horizon5，lr1e-4，discount0.99，tau0.01，actor entropy/自动alpha保留。', '',
    '只有开始学习之前的数据与状态严格相同。UTD改变后，策略、后续轨迹和replay会随之变化；这是完整online学习过程的UTD对照。', '',
    '验证：全部final checkpoint哈希、完整初始agent哈希、配置仅UTD变化、warmup replay哈希、精确恢复、更新次数、有限数值、每条episode初始状态和100条记录聚合均通过。']
(OUT/'comparison_report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'report':str(OUT/'comparison_report.md'),'plots':[str(p) for p in sorted(PLOTS.glob('*.png'))]},indent=2))
