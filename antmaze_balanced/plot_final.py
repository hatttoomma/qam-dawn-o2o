"""Final measured curves and tables for the completed balanced replay suite."""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/antmaze_large_balanced_20260914'
snapshot=json.loads((OUT/'final_complete_snapshot.json').read_text())
assert snapshot['files']['results/suite_status.json']['completed_tasks']==list(range(1,6))
assert snapshot['files']['results/suite_status.json']['state']=='passed'
assert not snapshot['processes'].strip()
assert len(snapshot['checkpoint_audit'])==5 and all(x['matched'] for x in snapshot['checkpoint_audit'].values())
rows=[]
for task in range(1,6):
    validation=json.loads((OUT/f'task{task}_completion_validation.json').read_text())
    assert validation['status']=='passed' and validation['snapshot_utc']==snapshot['utc']
    for row in validation['summary']:
        row=dict(row)
        if row['source'].startswith('previous'):
            folder='antmaze_large_task1_20260913' if task==1 else 'antmaze_large_tasks2_5_20260913'
            prefix='results/' if task==1 else f'results/task{task}/'
            name={'previous native offline':'native/fixed_eval/eval_000000_100.json',
                  'previous native 50k':'native/fixed_eval/eval_050000_100.json',
                  'previous DAWN 100k':'dawn/eval_100000_100_mean_residual.json'}[row['source']]
            row['source']=str(ROOT/'output'/folder/'final_complete_snapshot.json')+'#'+prefix+name
        else:
            row['source']=str(OUT/'final_complete_snapshot.json')+'#'+row['source']
        rows.append(row)

# Old tasks2–5 also have a measured90k point. Task1 does not.
old=json.loads((ROOT/'output/antmaze_large_tasks2_5_20260913/final_complete_snapshot.json').read_text())['files']
for task in range(2,6):
    prefix=f'results/task{task}/'
    name=prefix+'dawn/eval_090000_100_mean_residual.json'
    e=old[name];base=old[prefix+'native/fixed_eval/eval_000000_100.json']['records']
    assert len(e['records'])==e['episodes']==100
    assert all((a['reset_seed'],a['initial_hash'])==(b['reset_seed'],b['initial_hash']) for a,b in zip(e['records'],base))
    assert math.isclose(sum(r['success'] for r in e['records'])/100,e['success'],abs_tol=1e-12)
    assert math.isclose(sum(r['return_'] for r in e['records'])/100,e['return_mean'],abs_tol=1e-9)
    rows.append(dict(task=task,method='DAWN online-only mean',online_env_steps=90000,online_updates=2500,
        success_rate=e['success'],avg_return=e['return_mean'],
        gained_vs_offline=sum(a['success']>b['success'] for a,b in zip(e['records'],base)),
        lost_vs_offline=sum(a['success']<b['success'] for a,b in zip(e['records'],base)),
        source=str(ROOT/'output/antmaze_large_tasks2_5_20260913/final_complete_snapshot.json')+'#'+name))
rows.sort(key=lambda x:(x['task'],x['method'],x['online_env_steps']))
with (OUT/'five_tasks_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'five_tasks_comparison.json').write_text(json.dumps(rows,indent=2))

def select(task,method,step=None):
    return sorted([r for r in rows if r['task']==task and r['method']==method
        and (step is None or r['online_env_steps']==step)],key=lambda r:r['online_env_steps'])

labels=[('offline','Offline QAM',0),('qam_50k','QAM native',50000),
        ('old_dawn_100k','DAWN online-only mean',100000),('new_dawn_45k','DAWN balanced mean',45000),
        ('new_dawn_50k','DAWN balanced mean',50000),('new_sampled_50k','DAWN balanced sampled',50000)]
summary=[]
for task in range(1,6):
    record={'task':task}
    for label,method,step in labels:
        r,=select(task,method,step)
        record[label]=dict(success=r['success_rate'],avg_return=r['avg_return'])
    record['wall_minutes']=snapshot['files'][f'results/task{task}/dawn/runtime.json']['wall_seconds']/60
    summary.append(record)
macro={label:{metric:sum(r[label][metric] for r in summary)/5 for metric in ('success','avg_return')}
       for label,_,_ in labels}
(OUT/'five_tasks_summary.json').write_text(json.dumps(dict(tasks=summary,macro_average_all_five=macro),indent=2))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False})
colors={'QAM native':'#2466A5','DAWN online-only mean':'#D07427','DAWN balanced mean':'#078579'}
with PdfPages(OUT/'five_tasks_learning_curves.pdf') as pdf:
    for metric,ylabel,stem in [('success_rate','Success rate','success'),('avg_return','Average return (higher is better)','return')]:
        fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),sharex=True,sharey=True)
        for task,ax in zip(range(1,6),axes.flat):
            base=select(task,'Offline QAM')[0]
            native=select(task,'QAM native')[0]
            ax.plot([0,50],[base[metric],native[metric]],color=colors['QAM native'],marker='s',
                    linewidth=1.9,markersize=4,zorder=5)
            for method,warmup in [('DAWN online-only mean',80),('DAWN balanced mean',40)]:
                color=colors[method];values=select(task,method)
                ax.plot([0,warmup],[base[metric]]*2,color=color,linestyle=':',linewidth=1.5)
                xs=[warmup]+[r['online_env_steps']/1000 for r in values]
                ys=[base[metric]]+[r[metric] for r in values]
                ax.plot(xs,ys,color=color,linewidth=2)
                ax.scatter(xs[1:],ys[1:],s=30,color=color,marker='o',zorder=6)
            ax.set_title(f'Task {task}',loc='left',weight='bold',pad=9)
            ax.grid(axis='y',color='#E2E6EA',linewidth=.7)
            ax.set_xlim(-3,104);ax.set_xticks([0,40,50,80,100])
            if metric=='success_rate':
                ax.set_ylim(-.04,1.075);ax.set_yticks([0,.25,.5,.75,1])
                ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            else:
                ax.set_ylim(-1040,-230);ax.set_yticks([-1000,-800,-600,-400])
        panel=axes.flat[5];panel.set_axis_off()
        handles=[Line2D([0],[0],color=colors[m],marker='s' if m=='QAM native' else 'o',lw=2,label=label)
          for m,label in [('QAM native','QAM native'),('DAWN online-only mean','Old DAWN: online-only replay'),
                          ('DAWN balanced mean','New DAWN: 50:50 replay')]]
        panel.legend(handles=handles,loc='upper left',frameon=False,fontsize=10.5)
        notes=('Shared starting checkpoint: QAM offline500k\n\n'
          'QAM: 50k env steps / 45,001 updates\n'
          'Old DAWN: 80k warmup +20k training\n'
          '                 100k total / 5,000 updates\n'
          'New DAWN: 40k warmup +10k training\n'
          '                  50k total / 2,500 updates\n\n'
          'Dotted lines: unchanged base-only warmup.\n'
          'Markers: measured100-episode evaluations.\n'
          'Connecting lines only guide the eye.\n'
          'Old task1 has no90k evaluation.\n\n'
          'Budget and replay both change in the new arm.')
        panel.text(.02,.70,notes,transform=panel.transAxes,va='top',fontsize=9.2,
                   color='#4A535F',linespacing=1.3)
        fig.suptitle(f'AntMaze-large | {ylabel}',x=.07,ha='left',fontsize=17,weight='bold')
        fig.supxlabel('Online environment steps (thousands)',y=.054,fontsize=11)
        fig.supylabel(ylabel,x=.016,fontsize=11)
        fig.text(.5,.012,'Seed0 per task | 100 paired episodes | DAWN curves use mean residual; the base policy remains stochastic.',
                 ha='center',fontsize=9,color='#535D69')
        fig.subplots_adjust(left=.073,right=.983,top=.90,bottom=.135,hspace=.32,wspace=.2)
        fig.savefig(OUT/f'five_tasks_{stem}_env_steps.png',dpi=170,facecolor='white')
        pdf.savefig(fig,facecolor='white');plt.close(fig)

def fmt(x):return f"{x['success']:.0%} / {x['avg_return']:.2f}"
lines=['# AntMaze-large：40k warmup +10k training，50:50 replay','',
    '全部5个task已完成，seed0；每个评估点为100条固定episodes。以下DAWN主结果使用均值residual，base policy仍随机。单元格为success rate / avg return。','',
    '|Task|Offline500k|QAM native50k|旧DAWN100k|新DAWN45k|新DAWN50k|',
    '|---|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append('|'+str(r['task'])+'|'+'|'.join(fmt(r[k]) for k in ('offline','qam_50k','old_dawn_100k','new_dawn_45k','new_dawn_50k'))+'|')
lines+=['','五任务等权平均（包含Task4）：',
    '', '|方法|平均成功率|平均回报|','|---|---:|---:|']
for label,method in [('offline','Offline QAM'),('qam_50k','QAM native50k'),('old_dawn_100k','旧DAWN100k'),('new_dawn_50k','新DAWN50k')]:
    lines.append(f"|{method}|{macro[label]['success']:.1%}|{macro[label]['avg_return']:.2f}|")
lines+=['','新组相对offline在Task1–5的成功率变化为+12、+1、0、0、+12个百分点；相对旧DAWN100k为−2、+4、−6、0、−1个百分点。Task4仍为0%，而QAM native为86%。',
    '', '随机residual的50k补充结果：', '', '|Task|Success / avg return|','|---|---:|']
for r in summary:lines.append(f"|{r['task']}|{fmt(r['new_sampled_50k'])}|")
lines+=['','Task5随机residual达到97%，均值residual为88%；两种评估均保留，不按task挑选较高值替换主结果。',
    '', '设置：每task使用原有QAM offline500k checkpoint，继承Q/targetQ并冻结base。40k无更新base-only warmup+10k训练环境步；batch256固定128offline+128online，保留warmup数据；UTD.25共2500次更新。Naive TD、state+base action、scale.1及其余已核对超参数不变。Offline critic使用数据集行为动作，base特征由冻结QAM生成。',
    '', '预算：QAM native50k/45001updates；旧DAWN100k（80k warmup）/5000updates；新DAWN50k（40k warmup）/2500updates。新组同时改变训练预算与replay，不能独立识别replay的因果影响。Task1旧组取原150k run的100k checkpoint；没有伪造90k评估。单seed结果不代表跨seed稳定性。',
    '', '包含评估的运行时长：'+', '.join(f"Task{r['task']} {r['wall_minutes']:.1f}分钟" for r in summary)+f"，5个正式run合计{sum(r['wall_minutes'] for r in summary):.1f}分钟。",
    '', '完成核验：5个最终模型SHA256、各task初始checkpoint、继承Q/target及冻结base、精确2500updates、每batch128/128、评估RNG、100episode初始状态配对和结果聚合、finite metrics均通过；远程训练进程已结束。原始记录已同步至synced_results。',
    '', f'![成功率]({OUT/"five_tasks_success_env_steps.png"})',
    '', f'![平均回报]({OUT/"five_tasks_return_env_steps.png"})', '']
(OUT/'final_result.md').write_text('\n'.join(lines))
(OUT/'final_report_validation.json').write_text(json.dumps(dict(status='passed',snapshot_utc=snapshot['utc'],
    completed_tasks=list(range(1,6)),checkpoint_hashes_verified=5,table_rows=len(rows),
    episode_aggregates_recomputed=True,task1_90k_not_imputed=True,
    curves_use_measured_points_and_unchanged_warmup=True,no_training_processes=True),indent=2))
artifacts=['final_complete_snapshot.json','five_tasks_comparison.csv','five_tasks_comparison.json',
    'five_tasks_summary.json','five_tasks_success_env_steps.png','five_tasks_return_env_steps.png',
    'five_tasks_learning_curves.pdf','final_result.md','final_report_validation.json']
(OUT/'final_artifact_manifest.json').write_text(json.dumps({name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
    for name in artifacts},indent=2))
print(json.dumps(dict(tasks=summary,macro_average_all_five=macro),indent=2))
