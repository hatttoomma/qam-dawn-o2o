"""Validate the five fixed task pairs and report complete endpoint/curve evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import summarize_task2_scale as evidence
from summarize_td_ablation import arrays, check_mc, check_probe
from audit_cube5_checkpoints import locations

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/cube5'
OUT=ROOT/'results_cube5'
load,checked,track,digest=evidence.load,evidence.checked,evidence.track,evidence.digest
LABELS={'qam_edit':'QAM-EDIT (official edit=0)','dawn':'DAWN (state + base action)'}
STYLE={'qam_edit':('#C77524','s'),'dawn':('#2463A6','o')}


def main():
    OUT.mkdir(exist_ok=True)
    frozen=load(RUNS/'source_manifest.json');assert len(frozen)==58
    for n,h in frozen.items(): assert digest(ROOT/n)==h,n
    assert load(RUNS/'CHECKS_PASSED.json')['status']=='passed'
    task_audit=load(RUNS/'TASK_AUDIT_PASSED.json');assert task_audit['status']=='passed'
    reused=load(RUNS/'reused_runs.json');assert reused['status']=='passed' and len(reused['entries'])==4
    states=load(RUNS/'checkpoint_audit/audit.json');assert states['status']=='passed' and len(states['entries'])==10
    for r in states['entries']:
        assert r['before_final_evaluation_state_hash']==r['after_final_evaluation_state_hash']
        if r['method']=='dawn':
            p=ROOT/r['warmup_snapshot'];track(p);assert digest(p)==r['warmup_snapshot_sha256']
    for p in [RUNS/'pip_freeze.txt',ROOT/'runs/actor_input_ablation/pip_freeze.txt']:
        track(p)
    assert (RUNS/'pip_freeze.txt').read_bytes()==(ROOT/'runs/actor_input_ablation/pip_freeze.txt').read_bytes()
    rows,evals,episodes,pairs,differences,probes,mc_rows,offline_rows=[],[],[],[],[],[],[],[]
    curves={}
    for task in range(1,6):
        off,native,dawn=locations(task)
        od,oc=load(off/'DONE.json'),load(off/'config.json')
        assert od['step']==500000 and oc['seed']==0
        assert oc['qam_config']['edit_scale']==0 and oc['qam_config']['inv_temp']==1
        base=checked(off/'eval_500000_050.json')
        offline_rows.append(dict(task=task,seed=0,updates=500000,episodes=50,
            success_percent=100*base['success'],return_mean=base['return_mean'],
            checkpoint_sha256=od['checkpoint_sha256']))
        manifest=task_audit['tasks'][task-1]['manifest']
        if task!=1:assert load(off/'task_manifest.json')==manifest
        curves[task]={};finals={}
        for method,folder in [('qam_edit',native),('dawn',dawn)]:
            cfg,done=load(folder/'config.json'),load(folder/'DONE.json')
            expected_updates=7500 if method=='dawn' else 45001
            assert cfg['seed']==0 and cfg['offline_steps']==500000
            assert cfg['online_steps']==done['steps']==50000 and done['updates']==expected_updates
            assert cfg['offline_sha256']==od['checkpoint_sha256']
            if task>=3: assert load(folder/'CHECKS_PASSED.json')['status']=='passed'
            if method=='dawn':
                actual=load(folder/'actual_agent_config.json');initial=load(folder/'initial_hashes.json')
                warm=load(folder/'warmup.json')
                assert cfg['td_target']=='hard' and cfg['dawn_batch']==256 and cfg['warmup']==20000
                assert actual['actor_input']=='base_action' and actual['actor_input_dim']==62
                assert actual['gaussian_output'] and actual['actor_entropy_enabled'] and actual['automatic_alpha_enabled']
                assert actual['utd']==.25 and actual['target_entropy']==-25
                assert actual['residual_scale']==.1 and actual['target_tau']==.01
                assert actual['hidden_dims']==[256]*3 and actual['critic_ensemble']==10
                assert actual['target_aggregation']==actual['actor_Q_aggregation']=='minimum'
                assert initial['critic']==od['q_hash'] and initial['target']==od['target_q_hash']
                assert initial['flow']==done['final_flow_hash']==od['flow_hash']
                assert load(folder/'task_manifest.json')==manifest
                assert load(folder/'actor_input_checks.json')['status']=='passed'
                assert load(folder/'backup_math.json')['status']=='passed'
                fixed=arrays(folder/'fixed_probe.npz');fm=load(folder/'fixed_probe.json')
                assert digest(folder/'fixed_probe.npz')==fm['sha256']
                ps=[check_probe(folder,p,fixed) for p in sorted(folder.glob('probe_*.json'))]
                assert len(ps)==4 and ps[0]['updates']==0 and ps[-1]['updates']==7500
                probes.extend(dict(task=task,**p) for p in ps)
                mean=checked(folder/'eval_050000_050_mean_residual.json')
                assert mean['deterministic_residual'] and mean['residual_enabled']
            else:
                assert cfg['native_start']==5000 and done['initial_flow_hash']==od['flow_hash']
                if task!=1:assert load(folder/'task_manifest.json')==manifest
                mean=None
            points=[checked(p) for p in sorted(folder.glob('eval_*_050.json'))]
            assert len(points)==7
            for nominal,e in zip([0,5000,10000,20000,30000,40000,50000],points):
                assert nominal<=e['step']<=min(nominal+4,50000)
                assert not e['deterministic_residual']
                assert [r['initial_hash'] for r in e['records']]==[r['initial_hash'] for r in base['records']]
                if method=='dawn':
                    mc=check_mc(folder,e,'hard')
                    mc_rows.append(dict(task=task,**{k:v for k,v in mc.items() if k!='records'}))
            final=checked(folder/'eval_050000_100.json')
            assert final['step']==50000 and not final['deterministic_residual']
            finals[method]=final;curves[task][method]=points
            for role,e in [('initial',points[0]),('endpoint_repeat',points[-1])]:
                reference=base['records'] if role=='initial' else final['records'][:50]
                for a,b in zip(e['records'],reference):
                    assert a['initial_hash']==b['initial_hash'] and a['reset_seed']==b['reset_seed']
                    if a!=b:differences.append(dict(task=task,method=method,role=role,episode=a['episode'],observed=a,reference=b))
            mp=folder/'metrics.jsonl';track(mp)
            metrics=[json.loads(s) for s in mp.read_text().splitlines()]
            assert metrics[-1]['step']==50000 and metrics[-1]['updates']==expected_updates
            assert all(np.isfinite(v) for m in metrics for v in m.values() if isinstance(v,(int,float)))
            if method=='dawn':
                assert all(m['updates']==int((m['step']-20000)*.25) for m in metrics if m['step']>warm['steps'])
            if task>=3:
                p=folder/'evaluation_integrity.jsonl';track(p)
                checks=[json.loads(s) for s in p.read_text().splitlines()]
                assert all(r['before']==r['after'] for r in checks)
                assert any(r['step']==50000 and r['episodes']==100 for r in checks)
            rows.append(dict(task=task,method=method,seed=0,reused=task<=2,
                offline_updates=500000,online_steps=50000,batch_size=256,
                utd=.25 if method=='dawn' else 1.,updates=expected_updates,
                online_update_start=20000 if method=='dawn' else 5000,
                actor_input='state + base action' if method=='dawn' else 'QAM flow (edit_scale=0)',
                td_target='ordinary_min10' if method=='dawn' else 'ordinary_mean_minus_0.5std',
                replay='online_only' if method=='dawn' else 'uniform_offline_plus_online',
                final_episodes=100,success_percent=100*final['success'],return_mean=final['return_mean'],
                initial50_success_percent=100*points[0]['success'],initial50_return_mean=points[0]['return_mean'],
                mean_residual50_success_percent=100*mean['success'] if mean else None,
                mean_residual50_return_mean=mean['return_mean'] if mean else None,
                success_auc_percent=float(np.trapz([p['success'] for p in points],[p['step'] for p in points])/500),
                source_path=str(folder.relative_to(ROOT)),offline_sha256=od['checkpoint_sha256'],
                final_checkpoint_sha256=done['checkpoint_sha256']))
            groups=[('curve50',points),('final100',[final])]+([('mean_residual50',[mean])] if mean else [])
            for role,es in groups:
                for e in es:
                    evals.append(dict(task=task,method=method,role=role,step=e['step'],episodes=e['episodes'],
                        success_percent=100*e['success'],return_mean=e['return_mean']))
                    episodes.extend(dict(task=task,method=method,role=role,step=e['step'],**r) for r in e['records'])
        a,b=finals['qam_edit']['records'],finals['dawn']['records']
        assert all(x['reset_seed']==y['reset_seed'] and x['initial_hash']==y['initial_hash'] for x,y in zip(a,b))
        pairs.append(dict(task=task,episodes=100,
            qam_only_success=sum(bool(x['success']) and not y['success'] for x,y in zip(a,b)),
            dawn_only_success=sum(bool(y['success']) and not x['success'] for x,y in zip(a,b)),
            both_success=sum(bool(x['success']) and bool(y['success']) for x,y in zip(a,b)),
            neither_success=sum(not x['success'] and not y['success'] for x,y in zip(a,b)),
            dawn_minus_qam_success_pp=100*(finals['dawn']['success']-finals['qam_edit']['success']),
            dawn_minus_qam_return=finals['dawn']['return_mean']-finals['qam_edit']['return_mean']))
    aggregate=[]
    for method in LABELS:
        selected=[r for r in rows if r['method']==method]
        aggregate.append(dict(method=method,tasks=5,training_seeds_per_task=1,episodes_per_task=100,
            success_percent=float(np.mean([r['success_percent'] for r in selected])),
            return_mean=float(np.mean([r['return_mean'] for r in selected])),
            success_auc_percent=float(np.mean([r['success_auc_percent'] for r in selected]))))
    for filename,data in [('comparison.csv',rows),('aggregate.csv',aggregate),('offline.csv',offline_rows),
        ('evaluations.csv',evals),('episodes.csv',episodes),('paired_outcomes.csv',pairs),
        ('fixed_probe.csv',probes),('mc_summary.csv',mc_rows)]:evidence.write_csv(OUT/filename,data)
    (OUT/'summary.json').write_text(json.dumps(dict(rows=rows,aggregate=aggregate),indent=2)+'\n')
    (OUT/'evaluation_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    plot_endpoints(rows,aggregate);plot_curves(curves);report(rows,aggregate,differences)
    for n in ['summarize_cube5.py','audit_cube5_checkpoints.py','audit_td_warmup.py','summarize_td_ablation.py',
              'summarize_task2_scale.py','CHART_CONTRACT_CUBE5.md','export_cube5.py']:track(ROOT/n)
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',sources=evidence.sources,
        checks=['58 frozen training/protocol files','5 distinct task reward relabelings','4 reused endpoints verified',
                '10 final checkpoint and evaluation state hashes','5 full warmup snapshots','task-specific inherited critics',
                '7500 DAWN and 45001 QAM online updates','fixed probe and MC arithmetic','raw episode aggregation',
                'paired reset states and recorded repeated-evaluation differences'],evaluation_differences=len(differences)),indent=2)+'\n')
    print(json.dumps(dict(rows=rows,aggregate=aggregate),indent=2),flush=True)


def plot_endpoints(rows,aggregate):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for col,(key,title) in enumerate([('success_percent','Success rate (%)'),('return_mean','Mean episode return (higher is better)')]):
        ax=axes[col]
        for method in LABELS:
            selected=[r for r in rows if r['method']==method]+[r for r in aggregate if r['method']==method]
            ys=np.arange(6)+(-.12 if method=='qam_edit' else .12)
            color,marker=STYLE[method];xs=[r[key] for r in selected]
            ax.scatter(xs,ys,c=color,marker=marker,s=45,label=LABELS[method],zorder=3)
            for x,y in zip(xs,ys):
                ax.annotate(f'{x:.1f}' if key=='success_percent' else f'{x:.2f}',(x,y),
                    textcoords='offset points',xytext=(0,-12 if method=='dawn' else 7),ha='center',color=color,fontsize=9)
        ax.set_yticks(range(6),[f'Task{i}' for i in range(1,6)]+['Five-task mean'])
        ax.invert_yaxis();ax.set_ylim(5.55,-.55);ax.axhline(4.5,color='#AEB4BA',linestyle=':',linewidth=1)
        ax.set_xlabel(title);ax.grid(axis='x',color='#E6E9ED');ax.set_axisbelow(True)
        if key=='success_percent':ax.set_xlim(-5,105);ax.set_xticks([0,20,40,60,80,100])
        else:ax.margins(x=.14)
    fig.suptitle('Cube-double: five-task offline-to-online comparison',x=.08,ha='left',fontsize=16)
    fig.text(.08,.91,'Offline 500k | Online 50k | Seed 0 | Final 100 episodes per task and method',fontsize=11)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.105),ncol=2,frameon=False)
    fig.text(.08,.066,'DAWN: plain TD, batch 256, UTD 0.25, inherited Q, frozen base, state + base action, scale 0.1.',fontsize=9)
    fig.text(.08,.039,'QAM-EDIT uses the paper-selected cube-double edit scale 0. Task1/2 reuse matching runs; Task3–5 are new.',fontsize=9)
    fig.text(.08,.012,'Five-task mean gives each task equal weight. One training seed per task; no across-seed uncertainty estimate.',fontsize=9)
    fig.subplots_adjust(left=.12,right=.98,top=.84,bottom=.23,wspace=.34)
    for ext in ('png','pdf'):fig.savefig(OUT/f'final_performance.{ext}',dpi=180)
    plt.close(fig)


def plot_curves(curves):
    fig,axes=plt.subplots(5,2,figsize=(13,18))
    for task in range(1,6):
        for col,key,factor,label in [(0,'success',100,'Success rate (%)'),(1,'return_mean',1,'Mean episode return')]:
            ax=axes[task-1,col]
            for method in LABELS:
                points=curves[task][method];color,marker=STYLE[method]
                ax.plot([p['step']/1000 for p in points],[p[key]*factor for p in points],
                    linestyle='none',marker=marker,color=color,markersize=5,label=LABELS[method])
            ax.set_title(f'Task{task}',loc='left');ax.set_ylabel(label)
            ax.set_xlabel('Primitive online steps (thousands)');ax.set_xticks([0,5,10,20,30,40,50])
            ax.set_xlim(-1,51);ax.grid(axis='y',color='#E6E9ED')
            if col==0:ax.set_ylim(0,103)
    fig.suptitle('Cube-double: observed online evaluation checkpoints',x=.08,ha='left',fontsize=16)
    fig.text(.08,.955,'Offline 500k | Online 50k | Seed 0 | Seven 50-episode checkpoints per run',fontsize=11)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.048),ncol=2,frameon=False)
    fig.text(.08,.033,'QAM-EDIT: official edit=0, batch 256, UTD 1, updates from 5k. DAWN: plain TD, batch 256, UTD .25, 20k warmup.',fontsize=9)
    fig.text(.08,.019,'DAWN actor sees state + base action; inherited Q, frozen base, scale .1. Task1/2 reused; Task3–5 new.',fontsize=9)
    fig.text(.08,.006,'Points are observed evaluations, with no smoothing or interpolation. Final endpoint table uses 100 episodes.',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.925,bottom=.095,hspace=.5,wspace=.24)
    for ext in ('png','pdf'):fig.savefig(OUT/f'evaluation_checkpoints.{ext}',dpi=180)
    plt.close(fig)


def report(rows,aggregate,differences):
    lines=['# Cube-double 五任务对照','',
        '每个任务使用 seed0、500k offline updates 和 50k primitive online env steps。主表为最终 100 episodes 的随机采样评估，平均回报越高越好。','',
        'QAM-EDIT 使用论文为 cube-double 选出的配置 inv_temp=1、edit_scale=0，因此实际退化为 QAM。论文为同一域的五个任务使用相同配置；这里没有重新调参，也没有按最终评估成绩挑选 checkpoint。[论文超参数表](https://arxiv.org/html/2601.14234v1#A5)','',
        '| Task | QAM-EDIT 成功率 | DAWN 成功率 | QAM-EDIT 平均回报 | DAWN 平均回报 |',
        '|---|---:|---:|---:|---:|']
    for task in range(1,6):
        a,b=[next(r for r in rows if r['task']==task and r['method']==m) for m in LABELS]
        lines.append(f"| Task{task} | {a['success_percent']:.0f}% | {b['success_percent']:.0f}% | {a['return_mean']:.2f} | {b['return_mean']:.2f} |")
    a,b=aggregate
    lines.append(f"| 五任务等权平均 | {a['success_percent']:.1f}% | {b['success_percent']:.1f}% | {a['return_mean']:.2f} | {b['return_mean']:.2f} |")
    lines+=['','固定设置：','',
        '| 项目 | QAM-EDIT 官方 cube-double 配置 | DAWN 派生方法 |',
        '|---|---|---|',
        '| Offline | 各任务 QAM 500k | 同一任务的同一个 checkpoint |',
        '| Online 更新 | prior、flow actor、critic 继续训练 | base 冻结，继承 Q/target，训练 Gaussian residual 和 critic |',
        '| Critic TD target | 普通 TD，mean−0.5std | 普通 TD，min10 |',
        '| Batch / UTD | 256 / 1 | 256 / 0.25 |',
        '| 开始更新 / 更新次数 | 5k / 45,001 | 20k / 7,500 |',
        '| Replay | offline+online 均匀采样 | online-only chunk replay |',
        '| Residual input | editor 关闭 | state37＋base action chunk25 |',
        '| Residual / entropy | edit_scale=0 | scale=.1，Gaussian、actor entropy、自动 α 保留 |',
        '| Action chunking | 5 | 5 |','',
        '辅助均值残差评估（各 50 episodes，base 仍随机采样）：','',
        '| Task | DAWN 均值残差成功率 | 平均回报 |','|---|---:|---:|']
    for r in rows:
        if r['method']=='dawn':lines.append(f"| Task{r['task']} | {r['mean_residual50_success_percent']:.0f}% | {r['mean_residual50_return_mean']:.2f} |")
    lines+=['','可比性和验证：','',
        '- Task1/2 复用已完成且符合本轮配置的 native 与 base-action DAWN 结果；Task3–5 新训练。各任务的两种方法从同一 offline500k 模型开始，任务之间分别训练。',
        '- Task1/2 native 来自较早运行，保留其来源；DAWN Task1/2 和新任务在当前机器运行。复用不是新增 seed。',
        '- 此比较固定了两种方法各自的 replay 和更新日程，未匹配梯度更新次数，因此不是仅替换 actor 结构的单因素消融。',
        '- 只有一个 training seed。每任务 100 个 evaluation episodes 不能替代多个训练 seed，不能据此建立跨 seed 稳定优劣。',
        '- 58 个训练/协议文件冻结；五任务奖励标签、继承 Q、冻结 base、完整 warmup、checkpoint 哈希、评估前后 agent 状态、逐 episode 聚合和诊断计算已核验。',
        f'- 初始或终点重复评估共记录 {len(differences)} 条差异，保存在 evaluation_differences.json；主结果始终使用预定 final100。',
        '- 均值残差只改变评估时动作选择，不是移除 std 的训练实验。曲线每点使用 50 episodes，主表使用 100 episodes。','',
        '![最终结果](final_performance.png)','', '![评估点](evaluation_checkpoints.png)']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
