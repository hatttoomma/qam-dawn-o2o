"""Source-backed final50k→60k continuation comparisons and measured curves."""
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
OUT=ROOT/'output/antmaze_large_online60_20260914'
snapshot=json.loads((OUT/'final_complete_snapshot.json').read_text())
assert snapshot['files']['results/suite_status.json']['completed_tasks']==list(range(1,6))
assert snapshot['files']['results/suite_status.json']['state']=='passed'
assert not snapshot['processes'].strip()
assert len(snapshot['checkpoint_audit'])==5 and all(x['matched'] for x in snapshot['checkpoint_audit'].values())
rows=[]
for task in range(1,6):
    v=json.loads((OUT/f'task{task}_completion_validation.json').read_text())
    assert v['status']=='passed' and v['snapshot_utc']==snapshot['utc']
    for row in v['summary']:
        row=dict(row)
        if row['source'].startswith(('original','old DAWN')):
            folder='antmaze_large_task1_20260913' if task==1 else 'antmaze_large_tasks2_5_20260913'
            prefix='results/' if task==1 else f'results/task{task}/'
            name={'original offline500k':'native/fixed_eval/eval_000000_100.json',
                'original QAM native50k':'native/fixed_eval/eval_050000_100.json',
                'old DAWN100k':'dawn/eval_100000_100_mean_residual.json'}[row['source']]
            row['source']=str(ROOT/'output'/folder/'final_complete_snapshot.json')+'#'+prefix+name
        else:row['source']=str(OUT/'final_complete_snapshot.json')+'#'+row['source']
        rows.append(row)

parent_path=ROOT/'output/antmaze_large_balanced_20260914/final_complete_snapshot.json'
parents=json.loads(parent_path.read_text())['files']
old_path=ROOT/'output/antmaze_large_tasks2_5_20260913/final_complete_snapshot.json'
old=json.loads(old_path.read_text())['files']
for task in range(1,6):
    base=parents[f'results/task{task}/dawn/eval_050000_100.json']['records']
    extra=[('Parent balanced mean',parent_path,parents,f'results/task{task}/dawn/eval_045000_100_mean_residual.json',1250),
           ('Parent balanced sampled',parent_path,parents,f'results/task{task}/dawn/eval_045000_100.json',1250)]
    if task>1:extra.append(('Old DAWN online-only mean',old_path,old,f'results/task{task}/dawn/eval_090000_100_mean_residual.json',2500))
    for method,path,source,name,updates in extra:
        e=source[name];assert e['episodes']==len(e['records'])==100
        assert all((a['reset_seed'],a['initial_hash'])==(b['reset_seed'],b['initial_hash']) for a,b in zip(e['records'],base))
        assert math.isclose(sum(x['success'] for x in e['records'])/100,e['success'],abs_tol=1e-12)
        assert math.isclose(sum(x['return_'] for x in e['records'])/100,e['return_mean'],abs_tol=1e-9)
        rows.append(dict(task=task,method=method,online_env_steps=e['step'],online_updates=updates,
                        success_rate=e['success'],avg_return=e['return_mean'],source=str(path)+'#'+name))
rows.sort(key=lambda r:(r['task'],r['method'],r['online_env_steps']))
with (OUT/'five_tasks_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'five_tasks_comparison.json').write_text(json.dumps(rows,indent=2))

def select(task,method,step=None):
    return sorted([r for r in rows if r['task']==task and r['method']==method
        and (step is None or r['online_env_steps']==step)],key=lambda r:r['online_env_steps'])

labels=[('offline','Offline QAM',0),('qam_50k','QAM native',50000),
    ('old_dawn_100k','Old DAWN online-only mean',100000),('parent_50k','Parent balanced mean',50000),
    ('continued_55k','Continuation online-only mean',55000),('continued_60k','Continuation online-only mean',60000),
    ('sampled_50k','Parent balanced sampled',50000),('sampled_55k','Continuation online-only sampled',55000),
    ('sampled_60k','Continuation online-only sampled',60000)]
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
blue,orange,green,purple,gray='#2466A5','#D07427','#078579','#8A3DA6','#737E8A'
with PdfPages(OUT/'five_tasks_learning_curves.pdf') as pdf:
    for zoom in (False,True):
        for metric,ylabel,stem in [('success_rate','Success rate','success'),('avg_return','Average return (higher is better)','return')]:
            fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),sharex=True,sharey=True)
            for task,ax in zip(range(1,6),axes.flat):
                base=select(task,'Offline QAM')[0]
                parent=select(task,'Parent balanced mean',50000)[0]
                if not zoom:
                    native=select(task,'QAM native')[0]
                    ax.plot([0,50],[base[metric],native[metric]],color=blue,marker='s',lw=1.8,ms=4,zorder=5)
                    for method,warmup,color in [('Old DAWN online-only mean',80,orange),('Parent balanced mean',40,green)]:
                        values=select(task,method)
                        ax.plot([0,warmup],[base[metric]]*2,color=color,ls=':',lw=1.3)
                        xs=[warmup]+[r['online_env_steps']/1000 for r in values]
                        ys=[base[metric]]+[r[metric] for r in values]
                        ax.plot(xs,ys,color=color,lw=1.8)
                        ax.scatter(xs[1:],ys[1:],color=color,s=26,zorder=5)
                values=select(task,'Continuation online-only mean')
                xs=[50]+[r['online_env_steps']/1000 for r in values]
                ys=[parent[metric]]+[r[metric] for r in values]
                ax.plot(xs,ys,color=purple,lw=2.2,marker='o',ms=5,zorder=7)
                if zoom:
                    sampled=[select(task,'Parent balanced sampled',50000)[0]]+select(task,'Continuation online-only sampled')
                    ax.plot([r['online_env_steps']/1000 for r in sampled],[r[metric] for r in sampled],
                            color=gray,lw=1.5,ls='--',marker='s',ms=4,zorder=6)
                    ax.set_xlim(49.4,60.6);ax.set_xticks([50,55,60])
                else:
                    ax.set_xlim(-3,104);ax.set_xticks([0,40,60,80,100])
                ax.set_title(f'Task {task}',loc='left',weight='bold',pad=9)
                ax.grid(axis='y',color='#E2E6EA',lw=.7)
                if metric=='success_rate':
                    ax.set_ylim(-.04,1.075);ax.set_yticks([0,.25,.5,.75,1])
                    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
                else:
                    ax.set_ylim(-1040,-230);ax.set_yticks([-1000,-800,-600,-400])
            panel=axes.flat[5];panel.set_axis_off()
            if zoom:
                handles=[Line2D([0],[0],color=purple,marker='o',lw=2,label='Mean residual (primary)'),
                         Line2D([0],[0],color=gray,marker='s',lw=1.5,ls='--',label='Sampled residual')]
                notes=('Start: exact 50k checkpoint after balanced replay\n\n'
                    '50k: 2,500 cumulative updates\n55k: 3,750 cumulative updates\n60k: 5,000 cumulative updates\n\n'
                    'New stage: 10k env steps / 2,500 updates.\n'
                    'Each batch: 256 online / 0 offline samples.\n'
                    'Original online buffer and optimizers retained.\n\n'
                    'Each marker: 100 fixed paired episodes.\n'
                    'Lines connect measured points only.\n'
                    'No new warmup; base policy remains frozen.')
                top=.76;title='AntMaze-large | 50k to 60k continuation'
            else:
                handles=[Line2D([0],[0],color=c,marker=m,lw=2,label=label) for c,m,label in
                    [(blue,'s','QAM native'),(orange,'o','Old DAWN: online-only replay'),
                     (green,'o','Parent DAWN: 50:50 replay'),(purple,'o','Continuation: online-only (50k to 60k)')]]
                notes=('QAM: 50k env steps / 45,001 updates\n'
                    'Old DAWN: 100k / 5,000 updates\n'
                    'Parent: 50k / 2,500 updates\n'
                    'Continuation: 60k / 5,000 total updates\n\n'
                    'Dotted: unchanged base-only warmup.\n'
                    'Markers: measured 100-episode evaluations.\n'
                    'Connecting lines only guide the eye.\n'
                    'Old task1 has no 90k evaluation.\n\n'
                    'Continuation adds interaction and updates\n'
                    'while changing replay; it is not replay-only.')
                top=.62;title=f'AntMaze-large | {ylabel}'
            panel.legend(handles=handles,loc='upper left',frameon=False,fontsize=10)
            panel.text(.02,top,notes,transform=panel.transAxes,va='top',fontsize=9.2,color='#4A535F',linespacing=1.3)
            fig.suptitle(title,x=.073,ha='left',fontsize=17,weight='bold')
            fig.supxlabel('Online environment steps (thousands)',y=.054,fontsize=11)
            fig.supylabel(ylabel,x=.016,fontsize=11)
            fig.text(.5,.012,'Seed0 per task | 100 paired episodes | Mean refers only to residual; the frozen QAM base remains stochastic.',
                     ha='center',fontsize=9,color='#535D69')
            fig.subplots_adjust(left=.073,right=.983,top=.90,bottom=.135,hspace=.32,wspace=.2)
            name=f'five_tasks_{stem}_'+('continuation_50to60' if zoom else 'env_steps')+'.png'
            fig.savefig(OUT/name,dpi=170,facecolor='white');pdf.savefig(fig,facecolor='white');plt.close(fig)

def fmt(x):return f"{x['success']:.0%} / {x['avg_return']:.2f}"
lines=['# AntMaze-large：50k后追加10k online-only续训','',
    '五个task均已完成。Seed0；每个评估点100条固定配对episodes。主结果使用均值residual，base始终冻结且随机采样。单元格为success rate / avg return。','',
    '|Task|QAM native50k|旧DAWN100k|原balanced50k|续训55k|续训60k|',
    '|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append('|'+str(r['task'])+'|'+'|'.join(fmt(r[k]) for k in
    ('qam_50k','old_dawn_100k','parent_50k','continued_55k','continued_60k'))+'|')
lines+=['','五任务等权平均（含Task4）：','',
    '|方法|成功率|平均回报|','|---|---:|---:|']
for label,name in [('qam_50k','QAM native50k'),('old_dawn_100k','旧DAWN100k'),('parent_50k','原balanced50k'),
                   ('continued_55k','续训55k'),('continued_60k','续训60k')]:
    lines.append(f"|{name}|{macro[label]['success']:.1%}|{macro[label]['avg_return']:.2f}|")
lines+=['','均值residual的平均成功率为71.4%→74.0%→71.2%，没有持续的整体提升。Task1/2/3/5的55k平均回报均优于60k，Task4仍为0%。Task5成功率为88%→94%→85%。60k平均成功率比50k低0.2个百分点，平均回报也略低。',
    '', '随机residual的补充结果（每个task固定报告，不按task选择较高模式）：','',
    '|Task|原50k sampled|55k sampled|60k sampled|','|---|---:|---:|---:|']
for r in summary:lines.append('|'+str(r['task'])+'|'+'|'.join(fmt(r[k]) for k in ('sampled_50k','sampled_55k','sampled_60k'))+'|')
lines+=['','设置：从对应task的完整50k checkpoint恢复actor/critic/target/alpha及优化器、原50k online buffer、episode环境和随机数状态。新增10k env steps /2500次更新，不再warmup；新batch256全部来自online buffer，保留既有warmup数据。Naive TD、UTD.25、state+base action、scale.1、网络和其余超参数不变。',
    '', '累计更新次数：原50k为2500，55k为3750，60k为5000。旧DAWN100k同为5000，但其warmup80k，replay和数据历史不同；QAM native50k为45001次更新。续训同时增加交互与更新并切换replay，因此不能单独识别replay的影响。仅一个seed，以上不能证明跨seed稳定性或一般算法优劣。',
    '', '新增阶段含评估耗时：'+', '.join(f"Task{r['task']} {r['wall_minutes']:.1f}分钟" for r in summary)+
        f"；五个正式续训共{sum(r['wall_minutes'] for r in summary):.1f}分钟，不包含旧50k训练。",
    '', '完成核验：五个原50k完整状态及final对应、所有优化器继承、online replay恢复、环境恢复、冻结base、每batch256online/0offline、额外10000steps/2500updates、640000新增online draws、55k/60k评估RNG、100episode初始状态配对与聚合、有限训练指标、五个最终模型SHA256均通过。远程训练进程已结束，原始结果已同步至synced_results。',
    '', f'![续训成功率]({OUT/"five_tasks_success_continuation_50to60.png"})',
    '', f'![续训平均回报]({OUT/"five_tasks_return_continuation_50to60.png"})',
    '', f'![完整环境步数对比]({OUT/"five_tasks_success_env_steps.png"})',
    '', f'![完整平均回报对比]({OUT/"five_tasks_return_env_steps.png"})', '']
(OUT/'final_result.md').write_text('\n'.join(lines))
(OUT/'final_report_validation.json').write_text(json.dumps(dict(status='passed',snapshot_utc=snapshot['utc'],
    completed_tasks=list(range(1,6)),checkpoint_hashes_verified=5,table_rows=len(rows),
    episode_aggregates_recomputed=True,task1_old90k_not_imputed=True,
    only_measured_points_and_unchanged_warmup_plotted=True,no_training_processes=True),indent=2))
artifacts=['final_complete_snapshot.json','five_tasks_comparison.csv','five_tasks_comparison.json',
    'five_tasks_summary.json','five_tasks_success_env_steps.png','five_tasks_return_env_steps.png',
    'five_tasks_success_continuation_50to60.png','five_tasks_return_continuation_50to60.png',
    'five_tasks_learning_curves.pdf','final_result.md','final_report_validation.json']
(OUT/'final_artifact_manifest.json').write_text(json.dumps({name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
    for name in artifacts},indent=2))
print(json.dumps(dict(tasks=summary,macro_average_all_five=macro),indent=2))
