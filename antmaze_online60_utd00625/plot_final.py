"""Plot verified matched60k UTD continuation comparisons; exclude task4."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/antmaze_large_online60_utd00625_20260914'
TASKS=(1,2,3,5)
snapshot=json.loads((OUT/'final_complete_snapshot.json').read_text())
assert snapshot['files']['results/suite_status.json']['completed_tasks']==list(TASKS)
assert snapshot['files']['results/suite_status.json']['state']=='passed'
assert len(snapshot['checkpoint_audit'])==4 and all(x['matched'] for x in snapshot['checkpoint_audit'].values())
rows=[]
for task in TASKS:
 v=json.loads((OUT/f'task{task}_completion_validation.json').read_text())
 assert v['status']=='passed' and v['snapshot_utc']==snapshot['utc']
 rows.extend(v['summary'])

def select(task,method,step):
 values=[r for r in rows if r['task']==task and r['method']==method and r['online_env_steps']==step]
 assert len(values)==1,(task,method,step)
 return values[0]

summary=[]
for task in TASKS:
 r={'task':task}
 for label,method,step in [('offline','Offline QAM',0),('qam50k','QAM native',50000),
     ('parent50k','Parent balanced mean',50000),('utd025_60k','Reference UTD0.25 mean',60000),
     ('utd00625_60k','Continuation UTD0.0625 mean',60000),
     ('utd025_sampled60k','Reference UTD0.25 sampled',60000),
     ('utd00625_sampled60k','Continuation UTD0.0625 sampled',60000)]:
  x=select(task,method,step);r[label]={k:x[k] for k in ('success_rate','avg_return')}
 r['wall_minutes']=snapshot['files'][f'results/task{task}/dawn/runtime.json']['wall_seconds']/60
 summary.append(r)
macro={label:{m:sum(x[label][m] for x in summary)/4 for m in ('success_rate','avg_return')}
       for label in ('offline','qam50k','parent50k','utd025_60k','utd00625_60k','utd025_sampled60k','utd00625_sampled60k')}
(OUT/'four_tasks_summary.json').write_text(json.dumps(dict(tasks=summary,macro_average_tasks1_2_3_5=macro),indent=2))
with (OUT/'four_tasks_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
with PdfPages(OUT/'four_tasks_comparison.pdf') as pdf:
 for mode in ('mean','sampled'):
  for metric,label,stem in [('success_rate','Success rate','success'),('avg_return','Average return','return')]:
   fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
   for task,ax in zip(TASKS,axes.flat):
    p=select(task,'Parent balanced '+mode,50000)
    endpoints=[]
    for method,color,legend in [('Reference UTD0.25 '+mode,'#D07427','UTD 0.25: +2,500 updates'),
        ('Continuation UTD0.0625 '+mode,'#2466A5','UTD 0.0625: +625 updates')]:
     x=select(task,method,60000)
     ax.plot([50,60],[p[metric],x[metric]],marker='o',color=color,lw=2,label=legend)
     endpoints.append((x[metric],color))
    for index,(value,color) in enumerate(sorted(endpoints)):
     ax.annotate(f"{value:.0%}" if metric=='success_rate' else f"{value:.1f}",
                 (60,value),xytext=(5,-12 if index==0 else 6),textcoords='offset points',color=color)
    native=select(task,'QAM native',50000)
    ax.axhline(native[metric],color='#737E8A',ls=':',lw=1.2,label='QAM native @50k (reference)')
    ax.set_title(f'Task {task}',loc='left',weight='bold');ax.set_xticks([50,60]);ax.set_xlim(49.5,62)
    ax.grid(axis='y',alpha=.25);ax.margins(y=.18)
    if metric=='success_rate':ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_ylim(.65,1.04)
   fig.suptitle(f'AntMaze-large | {mode.capitalize()} residual | {label}',x=.08,ha='left',weight='bold',fontsize=15)
   handles,labels=axes.flat[0].get_legend_handles_labels()
   fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
   fig.supxlabel('Online environment steps (thousands)',y=.16);fig.supylabel(label,x=.015)
   fig.text(.08,.025,'Same complete 50k state; online-only continuation. Seed0,100 paired episodes. Task4 excluded.\nOnly50k/60k endpoints shown; lines guide the eye. No intermediate evaluation for UTD0.0625.',fontsize=9,color='#4A535F')
   fig.subplots_adjust(left=.09,right=.94,top=.9,bottom=.24,hspace=.3)
   fig.savefig(OUT/f'four_tasks_{stem}_{mode}.png',dpi=180);pdf.savefig(fig);plt.close(fig)
lines=['AntMaze-large continuation UTD comparison (mean residual; seed0,100 paired episodes)','',
'|Task|Parent50k|UTD0.25 @60k|UTD0.0625 @60k|QAM native50k|','|---|---:|---:|---:|---:|']
for x in summary:
 cells=[f"{x[k]['success_rate']:.0%} / {x[k]['avg_return']:.2f}" for k in ('parent50k','utd025_60k','utd00625_60k','qam50k')]
 lines.append('|'+str(x['task'])+'|'+'|'.join(cells)+'|')
lines+=['','Each cell is success / average return. Both continuation arms resume identical50k checkpoints, optimizers, replay and RNGs. New arm uses625 added updates vs2500. No new warmup. One training seed; episode comparisons do not estimate training-seed uncertainty.']
lines+=['','Macro average across tasks1,2,3,5 only:','',
    '|Mode|Parent50k|UTD0.25 @60k|UTD0.0625 @60k|','|---|---:|---:|---:|',
    f"|Mean success|{macro['parent50k']['success_rate']:.2%}|{macro['utd025_60k']['success_rate']:.2%}|{macro['utd00625_60k']['success_rate']:.2%}|",
    f"|Mean return|{macro['parent50k']['avg_return']:.2f}|{macro['utd025_60k']['avg_return']:.2f}|{macro['utd00625_60k']['avg_return']:.2f}|",
    f"|Sampled success|—|{macro['utd025_sampled60k']['success_rate']:.2%}|{macro['utd00625_sampled60k']['success_rate']:.2%}|",
    f"|Sampled return|—|{macro['utd025_sampled60k']['avg_return']:.2f}|{macro['utd00625_sampled60k']['avg_return']:.2f}|",
    '', 'Lower continuation UTD improves mean-residual results primarily on tasks3 and5 in this seed. Sampled-residual averages worsen, so the improvement does not hold across both evaluation modes.',
    '',f"Total production continuation and final-evaluation wall time: {sum(x['wall_minutes'] for x in summary):.2f} minutes."]
(OUT/'final_result.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='passed',tasks=list(TASKS),macro_average_tasks1_2_3_5=macro),indent=2))
