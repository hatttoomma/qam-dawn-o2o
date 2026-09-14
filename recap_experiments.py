"""Reprint locally verified results without needing the training environment."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_recap_20260908';OUT.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text())
validation=load(ROOT/'results_policy_decorator/validation.json')
for p,h in validation['sources'].items():
    assert hashlib.sha256((ROOT/'runs'/p).read_bytes()).hexdigest()==h,p
summary=load(ROOT/'results_policy_decorator/summary.json')
styles={
    'native':('QAM native (edit=0)','#53575D','o','-'),
    'warm':('DAWN / pretrained Q / online-only','#2463A6','s','--'),
    'random':('DAWN / random Q / online-only','#B68420','^','--'),
    'warm_qam_replay':('DAWN / pretrained Q / QAM replay','#2463A6','D','-'),
    'random_qam_replay':('DAWN / random Q / QAM replay','#B68420','P','-'),
    'policy_decorator_qam_replay':('Policy Decorator / pretrained Q / QAM replay','#2463A6','*',':'),
}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(14,6.8))
for s in summary:
    arm=s['arm'];label,color,marker,style=styles[arm]
    points=[load(p) for p in sorted((ROOT/'runs'/arm).glob('eval_*_050.json'))]
    assert len(points)==7
    for p in points:
        assert p['episodes']==len(p['records'])==50
        assert abs(np.mean([r['success'] for r in p['records']])-p['success'])<1e-12
        assert abs(np.mean([r['return_'] for r in p['records']])-p['return_mean'])<1e-10
    for ax,key,multiplier in [(axes[0],'success',100),(axes[1],'return_mean',1)]:
        ax.plot([p['step']/1000 for p in points],[p[key]*multiplier for p in points],
            label=label,color=color,marker=marker,linestyle=style,
            linewidth=2.5 if arm.startswith('policy') else 1.8,
            markersize=9 if marker=='*' else 5,
            markerfacecolor='white' if arm in ['warm','random'] else color)
axes[0].set_ylabel('Success rate (%)');axes[0].set_ylim(-3,103)
axes[1].set_ylabel('Mean episode return')
for ax in axes:
    ax.set_xlabel('Primitive online steps (thousands)');ax.set_xlim(-.8,50.8)
    ax.set_xticks([0,10,20,30,40,50]);ax.grid(axis='y',color='#E8EAED')
fig.suptitle('Cube-double task1: six verified QAM-pretrained experiments',x=.07,ha='left',fontsize=16)
fig.text(.07,.89,'Seed 0 | Shared 500k offline checkpoint | 50k online steps | 50 evaluation episodes per curve point',fontsize=10)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.078),ncol=2,frameon=False,fontsize=9)
fig.text(.07,.045,'DAWN: 20k base-only warmup. Policy Decorator: 8k learning starts; progressive exploration reaches p=1 at 30k.',fontsize=9)
fig.text(.07,.02,'Residual evaluations use sampled actions. Final 100-episode scores are in the table. No BC online results are inferred.',fontsize=9)
fig.subplots_adjust(left=.07,right=.98,top=.82,bottom=.30,wspace=.24)
fig.savefig(OUT/'learning_curves_six.png',dpi=180);fig.savefig(OUT/'learning_curves_six.pdf');plt.close(fig)
with (OUT/'performance_six.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
lines=['# 已核验的六组实验结果','',
    '任务 cube-double-play-singletask-task1-v0；seed 0；500k offline updates；50k primitive online steps；action chunk=5。', '',
    '| Online 方法 | Critic | Replay | 最终成功率（100 eps） | 平均回报 | 归一化成功率 AUC |',
    '|---|---|---|---:|---:|---:|']
for s in summary:
    lines.append(f"| {s['label']} | {s['critic_initialization']} | {s['replay']} | {100*s['final_success_100']:.0f}% | {s['final_return_100']:.2f} | {100*s['success_auc_50']:.2f}% |")
lines+=['','AUC 按每点 50 episodes 的七个预设检查点计算。主表的 residual 方法均使用随机残差评估。',
    '', '![Six learning curves](learning_curves_six.png)','',
    'QAM + PD 的均值残差辅助评估为 90% / -185.45（100 episodes），与主表口径分开。', '',
    'BC 的 500k offline checkpoint 已在上一轮实测得到 4%（50 episodes），四组 online 结果仍需从远程机器取回；截至本次重印，旧 SSH 地址在认证前重置连接，因此未填入 BC online performance。','']
(OUT/'summary.md').write_text('\n'.join(lines))
(OUT/'validation.json').write_text(json.dumps(dict(status='passed',evaluation_files=len(validation['sources']),
    sources=validation['sources'],report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print(OUT)
