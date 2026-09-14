"""Compare measured QAM and DAWN evaluations by actual online update counts."""
from pathlib import Path
import csv,hashlib,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter,FuncFormatter

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/antmaze_qam_vs_dawn00625_updates_20260914';OUT.mkdir(exist_ok=True)
TASKS=(1,2,3,5)
paths={name:ROOT/'output'/folder/'final_complete_snapshot.json' for name,folder in {
 'native1':'antmaze_large_task1_20260913','native_rest':'antmaze_large_tasks2_5_20260913',
 'parent':'antmaze_large_balanced_20260914','continued':'antmaze_large_online60_utd00625_20260914'}.items()}
sources={name:json.loads(path.read_text()) for name,path in paths.items()}
rows=[]
for task in TASKS:
 native_key='native1' if task==1 else 'native_rest';native=sources[native_key]['files']
 nprefix='results/native/' if task==1 else f'results/task{task}/native/'
 prefix=f'results/task{task}/dawn/';parent=sources['parent']['files'];new=sources['continued']['files']
 nd=native[nprefix+'DONE.json'];assert nd['offline_updates']==500000 and nd['online_updates']==45001
 assert nd['online_env_steps']==50000 and nd['native']['actual_gradient_updates']==545001
 assert parent[prefix+'DONE.json']['updates']==2500 and new[prefix+'DONE.json']['updates']==3125
 assert new[prefix+'CHECKS_PASSED.json']['status']=='passed'
 assert new[prefix+'resume_audit.json']['published_final_sha256']==parent[prefix+'DONE.json']['checkpoint_sha256']
 base=native[nprefix+'fixed_eval/eval_000000_100.json']
 candidates=[('QAM native',native_key,nprefix+'fixed_eval/eval_000000_100.json',0),
             ('QAM native',native_key,nprefix+'fixed_eval/eval_050000_100.json',45001),
             ('DAWN', 'parent',prefix+'eval_000000_100.json',0),
             ('DAWN', 'parent',prefix+'eval_045000_100_mean_residual.json',1250),
             ('DAWN', 'parent',prefix+'eval_050000_100_mean_residual.json',2500),
             ('DAWN', 'continued',prefix+'eval_060000_100_mean_residual.json',3125)]
 for method,key,rel,updates in candidates:
  e=sources[key]['files'][rel];records=e['records'];assert len(records)==e['episodes']==100
  assert [r['episode'] for r in records]==list(range(100))
  assert all((a['reset_seed'],a['initial_hash'])==(b['reset_seed'],b['initial_hash']) for a,b in zip(records,base['records']))
  success=sum(r['success'] for r in records)/100;ret=sum(r['return_'] for r in records)/100
  assert math.isclose(success,e['success'],abs_tol=1e-12) and math.isclose(ret,e['return_mean'],abs_tol=1e-10)
  if method=='DAWN' and updates:
   metric_rows=sources[key]['files'][prefix+'metrics.jsonl']
   matched=[r for r in metric_rows if r['step']==e['step']];assert len(matched)==1 and matched[0]['updates']==updates
  if not updates:assert e['records']==base['records']
  rows.append(dict(task=task,method=method,online_gradient_updates=updates,total_gradient_updates=500000+updates,
       online_env_steps=e['step'],success_rate=success,avg_return=ret,episodes=100,seed=0,
       evaluation_mode='mean residual' if method=='DAWN' and updates else 'base policy',
       source=str(paths[key])+'#'+rel))
assert len(rows)==24
with (OUT/'measured_points.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'measured_points.json').write_text(json.dumps(rows,indent=2))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'QAM native':'#D07427','DAWN':'#2466A5'}
with PdfPages(OUT/'qam_vs_dawn_training_steps.pdf') as pdf:
 for scale in ('linear','symlog'):
  for metric,ylabel,stem in [('success_rate','Success rate','success'),('avg_return','Average return (higher is better)','return')]:
   fig,axes=plt.subplots(2,2,figsize=(11.5,8),sharex=True)
   for task,ax in zip(TASKS,axes.flat):
    for method in ('QAM native','DAWN'):
     data=sorted([r for r in rows if r['task']==task and r['method']==method],key=lambda r:r['online_gradient_updates'])
     x=[r['online_gradient_updates'] for r in data];y=[r[metric] for r in data]
     label='QAM native (50k env steps)' if method=='QAM native' else 'DAWN (60k env steps; final-stage UTD 0.0625)'
     ax.plot(x,y,color=colors[method],marker='s' if method=='QAM native' else 'o',lw=1.8,
        ms=4.5,ls='--' if method=='QAM native' else '-',label=label,zorder=3 if method=='QAM native' else 4)
     final=data[-1];value=f"{final[metric]:.0%}" if metric=='success_rate' else f"{final[metric]:.1f}"
     ax.annotate(value,(x[-1],y[-1]),xytext=(-5,9) if method=='QAM native' else (8,-15),
         textcoords='offset points',ha='right' if method=='QAM native' else 'left',color=colors[method],fontsize=10,weight='bold')
    ax.set_title(f'Task {task}',loc='left',weight='bold');ax.grid(axis='y',color='#E3E7EB',lw=.7)
    ax.margins(y=.18)
    if metric=='success_rate':ax.set_ylim(.69,1.065);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    if scale=='linear':
     ax.set_xlim(-900,47800);ax.set_xticks([0,10000,20000,30000,40000,45001])
     ax.set_xticklabels(['0','10k','20k','30k','40k','45k'])
    else:
     ax.set_xscale('symlog',linthresh=1500,linscale=1.0)
     ax.set_xlim(-100,57000);ax.set_xticks([0,1250,2500,5000,10000,45001])
     ax.set_xticklabels(['0','1.25k','2.5k','5k','10k','45k'])
   fig.suptitle('AntMaze-large | QAM native vs DAWN | '+('Success rate' if metric=='success_rate' else 'Average return'),
                 x=.085,ha='left',weight='bold',fontsize=16)
   fig.supxlabel('Training steps = cumulative online gradient updates'+(' (symlog scale)' if scale=='symlog' else ' (linear scale)'),y=.195,fontsize=11)
   fig.supylabel(ylabel,x=.018,fontsize=11)
   handles,labels=axes.flat[0].get_legend_handles_labels()
   fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.126),frameon=False,ncol=1,fontsize=10)
   fig.text(.085,.055,'QAM endpoint: 45,001 updates. DAWN endpoint: 3,125 = 2,500 parent + 625 continuation updates.\n'
       'DAWN uses UTD 0.25 before 50k env steps, then UTD 0.0625 for 50k–60k. Common 500k offline updates excluded.\n'
       'Markers: 100 paired episodes, seed0; DAWN uses mean residual. Task4 excluded.',fontsize=9,color='#4C5662',linespacing=1.35)
   fig.text(.085,.022,'Native has no intermediate paired evaluations: dashed links do not show a measured learning trajectory.',
            fontsize=9,color='#4C5662')
   fig.subplots_adjust(left=.095,right=.96,bottom=.28,top=.9,hspace=.34,wspace=.24)
   fig.savefig(OUT/f'{stem}_training_steps_{scale}.png',dpi=180)
   fig.savefig(OUT/f'{stem}_training_steps_{scale}.svg')
   pdf.savefig(fig);plt.close(fig)
summary={'status':'passed','tasks':list(TASKS),'source_snapshots':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
 'plotted_rows':len(rows),'x_axis':'cumulative online agent gradient update iterations; excludes common500k offline',
 'qam_updates':[0,45001],'dawn_updates':[0,1250,2500,3125],'primary_evaluation':'mean residual;100 paired episodes;seed0',
 'native_intermediate_evaluations':'unavailable; endpoints only, dashed connection',
 'dawn_utd_schedule':'0.25 through50k environment steps;0.0625 from50k to60k',
 'linear_primary_symlog_supplement':True}
(OUT/'validation.json').write_text(json.dumps(summary,indent=2))
(OUT/'README.md').write_text('''# QAM native versus DAWN by training steps

Tasks1,2,3,5;100 paired evaluation episodes;seed0. Mean residual evaluation for DAWN.

The main figures use a linear x-axis: cumulative online gradient-update iterations, starting at the common offline500k checkpoint. Warmup consumes environment steps but adds no updates. Add500000 to every x-value if counting offline training too.

QAM native:0 and45001 measured online updates (50k online env steps). Intermediate paired evaluations were not collected; dashed lines connect endpoints only and must not be interpreted as measured learning curves.

DAWN:0,1250,2500,3125 measured updates (0,45k,50k,60k online env steps). The first2500 updates came from the UTD0.25 balanced-replay parent. Only the last625 use UTD0.0625 and online-only replay. Final cumulative updates are3125, not625. No55k evaluation was collected.

Linear figures are primary. The PDF also contains clearly labeled symlog versions to separate the early DAWN evaluation points. An update iteration is not a wall-time or FLOPs unit. The available evaluations do not establish QAM performance at3125 updates.
''')
print(json.dumps(summary,indent=2))
