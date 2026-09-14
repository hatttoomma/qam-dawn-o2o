"""Compare measured QAM and DAWN evaluations by actual online update counts."""
from pathlib import Path
import csv,hashlib,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter,FuncFormatter

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/antmaze_qam_vs_dawn00625_updates_20260914_all5';OUT.mkdir(exist_ok=True)
TASKS=(1,2,3,4,5)
paths={name:ROOT/'output'/folder/'final_complete_snapshot.json' for name,folder in {
 'native1':'antmaze_large_task1_20260913','native_rest':'antmaze_large_tasks2_5_20260913',
 'parent':'antmaze_large_balanced_20260914','continued':'antmaze_large_online60_utd00625_20260914','prior60':'antmaze_large_online60_20260914'}.items()}
sources={name:json.loads(path.read_text()) for name,path in paths.items()}
rows=[]
for task in TASKS:
 native_key='native1' if task==1 else 'native_rest';native=sources[native_key]['files']
 nprefix='results/native/' if task==1 else f'results/task{task}/native/'
 prefix=f'results/task{task}/dawn/';parent=sources['parent']['files'];new=sources['continued']['files']
 nd=native[nprefix+'DONE.json'];assert nd['offline_updates']==500000 and nd['online_updates']==45001
 assert nd['online_env_steps']==50000 and nd['native']['actual_gradient_updates']==545001
 assert parent[prefix+'DONE.json']['updates']==2500
 if task!=4:
  assert new[prefix+'DONE.json']['updates']==3125 and new[prefix+'CHECKS_PASSED.json']['status']=='passed'
  assert new[prefix+'resume_audit.json']['published_final_sha256']==parent[prefix+'DONE.json']['checkpoint_sha256']
 else:
  assert prefix+'DONE.json' not in new
  prior=sources['prior60']['files'];assert prior[prefix+'DONE.json']['updates']==5000
  assert prior[prefix+'CHECKS_PASSED.json']['status']=='passed'
  assert prior[prefix+'actual_agent_config.json']['utd']==.25
 base=native[nprefix+'fixed_eval/eval_000000_100.json']
 candidates=[('QAM native',native_key,nprefix+'fixed_eval/eval_000000_100.json',0),
             ('QAM native',native_key,nprefix+'fixed_eval/eval_050000_100.json',45001),
             ('DAWN', 'parent',prefix+'eval_000000_100.json',0),
             ('DAWN', 'parent',prefix+'eval_045000_100_mean_residual.json',1250),
             ('DAWN', 'parent',prefix+'eval_050000_100_mean_residual.json',2500),
             ('DAWN', 'continued',prefix+'eval_060000_100_mean_residual.json',3125)]
 if task==4:
  candidates=candidates[:-1]+[('DAWN','prior60',prefix+'eval_055000_100_mean_residual.json',3750),
                            ('DAWN','prior60',prefix+'eval_060000_100_mean_residual.json',5000)]
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
  if task==4 and method=='DAWN':assert success==0 and ret==-1000
  rows.append(dict(task=task,method=('DAWN reference (UTD0.25)' if task==4 and method=='DAWN' else method),online_gradient_updates=updates,total_gradient_updates=500000+updates,
       online_env_steps=e['step'],success_rate=success,avg_return=ret,episodes=100,seed=0,
       evaluation_mode='mean residual' if method=='DAWN' and updates else 'base policy',
       source=str(paths[key])+'#'+rel))
assert len(rows)==31
with (OUT/'measured_points.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'measured_points.json').write_text(json.dumps(rows,indent=2))

from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'QAM native':'#D07427','DAWN':'#2466A5','DAWN reference (UTD0.25)':'#737E8A'}
with PdfPages(OUT/'qam_vs_dawn_training_steps_all5.pdf') as pdf:
 for scale in ('linear','symlog'):
  for metric,ylabel,stem in [('success_rate','Success rate','success'),('avg_return','Average return (higher is better)','return')]:
   fig,axes=plt.subplots(2,3,figsize=(15,8.3))
   for task,ax in zip(TASKS,axes.flat):
    methods=('QAM native','DAWN reference (UTD0.25)' if task==4 else 'DAWN')
    for method in methods:
     data=sorted([r for r in rows if r['task']==task and r['method']==method],key=lambda r:r['online_gradient_updates'])
     x=[r['online_gradient_updates'] for r in data];y=[r[metric] for r in data]
     native=method=='QAM native';reference=method=='DAWN reference (UTD0.25)'
     ax.plot(x,y,color=colors[method],marker='s' if native else ('D' if reference else 'o'),lw=1.8,
        ms=4.5,ls='--' if native else (':' if reference else '-'),markerfacecolor='white' if reference else colors[method],
        zorder=3 if native else 4)
     value=f"{y[-1]:.0%}" if metric=='success_rate' else f"{y[-1]:.1f}"
     ax.annotate(value,(x[-1],y[-1]),xytext=(-5,9) if native else (7,8 if reference else -15),
         textcoords='offset points',ha='right' if native else 'left',color=colors[method],fontsize=10,weight='bold')
    ax.set_title(f'Task {task}'+(' | prior DAWN reference' if task==4 else ''),loc='left',weight='bold',fontsize=11)
    ax.grid(axis='y',color='#E3E7EB',lw=.7);ax.margins(y=.18)
    if metric=='success_rate':
     ax.set_ylim(-.07,1.1);ax.set_yticks([0,.25,.5,.75,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    if scale=='linear':
     ax.set_xlim(-900,47800);ax.set_xticks([0,10000,20000,30000,45001]);ax.set_xticklabels(['0','10k','20k','30k','45k'])
    else:
     ax.set_xscale('symlog',linthresh=1500,linscale=1.0);ax.set_xlim(-100,57000)
     ax.set_xticks([0,1250,2500,5000,45001]);ax.set_xticklabels(['0','1.25k','2.5k','5k','45k'])
   panel=axes.flat[5];panel.set_axis_off()
   handles=[Line2D([0],[0],color=colors['QAM native'],marker='s',ls='--',label='QAM native | 50k env steps'),
            Line2D([0],[0],color=colors['DAWN'],marker='o',label='DAWN | 60k env steps\nFinal-stage UTD 0.0625'),
            Line2D([0],[0],color=colors['DAWN reference (UTD0.25)'],marker='D',markerfacecolor='white',ls=':',
                   label='Task4: prior DAWN | 60k env steps\nUTD 0.25 reference')]
   panel.legend(handles=handles,labels=['QAM native (50k env steps)','DAWN UTD0.0625 (60k env steps)','Task4: prior DAWN UTD0.25'],loc='upper left',frameon=False,fontsize=10,labelspacing=1.2)
   panel.text(.02,.61,'Final online gradient updates\nQAM native: 45,001\nDAWN tasks1,2,3,5: 3,125\nTask4 prior DAWN: 5,000\n\nTask4 was skipped at UTD0.0625.\nIts 0% is from the prior UTD0.25 run.',transform=panel.transAxes,va='top',fontsize=10,color='#4C5662',linespacing=1.5)
   fig.suptitle('AntMaze-large | QAM native vs DAWN | '+('Success rate' if metric=='success_rate' else 'Average return'),
                 x=.065,ha='left',weight='bold',fontsize=17)
   fig.supxlabel('Training steps = cumulative online gradient updates'+(' (symlog scale)' if scale=='symlog' else ' (linear scale)'),y=.145,fontsize=12)
   fig.supylabel(ylabel,x=.015,fontsize=12)
   fig.text(.065,.074,'Markers: 100 paired episodes, seed0; mean residual evaluation. Common 500k offline updates excluded.\n'
       'Tasks1,2,3,5: first2,500 updates use UTD0.25; final625 use UTD0.0625. Task4 shows the prior UTD0.25 arm.',fontsize=10,color='#4C5662',linespacing=1.35)
   fig.text(.065,.03,'Native has no intermediate paired evaluations: dashed links are endpoint connections, not measured learning trajectories.',
            fontsize=10,color='#4C5662')
   fig.subplots_adjust(left=.07,right=.965,bottom=.23,top=.9,hspace=.34,wspace=.27)
   fig.savefig(OUT/f'{stem}_training_steps_{scale}_all5.png',dpi=180)
   fig.savefig(OUT/f'{stem}_training_steps_{scale}_all5.svg')
   pdf.savefig(fig);plt.close(fig)
summary={'status':'passed','tasks':list(TASKS),'plotted_rows':len(rows),'primary_axis':'cumulative online gradient updates; excludes500k offline',
 'new_arm_tasks':[1,2,3,5],'task4':dict(reference_arm='previous UTD0.25 online-only continuation',online_env_steps=60000,
     online_updates=5000,success_rate=0,avg_return=-1000,new_utd00625_run_exists=False),
 'source_snapshots':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
 'no_imputed_measurements':True,'no_mixed_arm_macro_averages':True}
(OUT/'validation.json').write_text(json.dumps(summary,indent=2))
(OUT/'README.md').write_text('''# QAM native vs DAWN: all five tasks

Horizontal axis: actual cumulative online gradient-update iterations, excluding common500k offline updates. Main plots use linear scale; supplementary PDF pages use symlog scale.

Tasks1,2,3,5: original UTD0.0625 continuation ending at60k environment steps /3125 updates. The first2500 updates used the UTD0.25 balanced parent, followed by625 online-only continuation updates.

Task4 was skipped in the UTD0.0625 arm. Its gray hollow diamonds show the earlier measured UTD0.25 continuation (mean residual0% success /−1000 return), at true update counts0,1250,2500,3750,5000. Its final point is at5000 updates, not3125. No result is fabricated for the unrun setting. Task4 QAM native reaches86% /−504.24 at45001 online updates.

All points use100 paired episodes andseed0. Native has only start/end paired evaluations; dashed links connect endpoints and do not provide intermediate learning trajectories. No five-task macro mean is computed across different DAWN arms.
''')
print(json.dumps(summary,indent=2))
