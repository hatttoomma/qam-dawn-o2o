"""Report the fixed-batch UTD comparison with source and episode validation."""
import ast
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import summarize_task2_scale as evidence
from summarize_td_ablation import arrays,check_mc,check_probe
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/utd_ablation'
OUT=ROOT/'results_utd_ablation'
load,checked,track,digest=evidence.load,evidence.checked,evidence.track,evidence.digest


def main():
    OUT.mkdir(exist_ok=True)
    frozen=load(RUNS/'source_manifest.json');assert len(frozen)==51
    for name,h in frozen.items():assert digest(ROOT/name)==h,name
    smoke=load(RUNS/'CHECKS_PASSED.json');assert smoke['status']=='passed'
    for test in smoke['tasks']:
        ref=ROOT/test['reference'];assert test['original_full_checkpoint_parity']
        assert load(ref/'DONE.json')['checkpoint_sha256']==load(RUNS/f"smoke_task{test['task']}_utd025/DONE.json")['checkpoint_sha256']
    audit=load(RUNS/'warmup_audit/audit.json');assert audit['status']=='passed'
    for v in audit['tasks'].values():
        for pk,hk in [('snapshot','snapshot_sha256'),('baseline_snapshot','baseline_sha256')]:
            track(ROOT/v[pk]);assert digest(ROOT/v[pk])==v[hk]
    assert load(RUNS/'runtime_setup.json')['probe_exit_code']==0
    assert (RUNS/'pip_freeze.txt').read_bytes()==(ROOT/'runs/td_ablation/batch_comparison/pip_freeze.txt').read_bytes()
    track(RUNS/'pip_freeze.txt');track(ROOT/'runs/td_ablation/batch_comparison/pip_freeze.txt')
    source=(ROOT/'run.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='residual')
    loop='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    old='target_updates=int((step-args.warmup)*.25)';new='target_updates=int((step-args.warmup)*args.utd)'
    assert loop.count(old)==1
    rows,evals,episodes,pairs,probes,mc_rows,mc_episodes,initial_diffs,endpoint_diffs=[],[],[],[],[],[],[],[],[]
    curves={};context=[]
    for task in (1,2):
        root=ROOT/('runs' if task==1 else 'runs/task2')
        baseline=ROOT/f'runs/td_ablation/batch_comparison/task{task}_b256'
        cfg0=load(baseline/'config.json');actual0=load(baseline/'actual_agent_config.json');init0=load(baseline/'initial_hashes.json')
        offline=load(root/'offline/DONE.json');base_eval=checked(root/'offline/eval_500000_050.json')
        fixed0=arrays(baseline/'fixed_probe.npz');warm0=load(baseline/'warmup.json')
        assert offline['step']==500000
        native=checked(root/'native/eval_050000_100.json')
        large=checked(ROOT/f'runs/td_ablation/batch_comparison/task{task}_b1024/eval_050000_100.json')
        context.append(dict(task=task,native_success=100*native['success'],native_return=native['return_mean'],
            batch1024_utd025_success=100*large['success'],batch1024_utd025_return=large['return_mean']))
        finals={};curves[task]={}
        for utd,folder in [(.25,baseline),(1.,RUNS/f'task{task}_utd1')]:
            cfg=load(folder/'config.json');actual=load(folder/'actual_agent_config.json');init=load(folder/'initial_hashes.json')
            done=load(folder/'DONE.json');selected=load(folder/'td_config.json')
            assert {k:v for k,v in cfg.items() if k not in ('out','utd')}=={k:v for k,v in cfg0.items() if k!='out'}
            assert cfg.get('utd',.25)==selected.get('utd',.25)==utd
            assert cfg['dawn_batch']==256 and cfg['warmup']==20000 and cfg['td_target']=='hard' and cfg['seed']==0
            assert cfg['offline_sha256']==selected['offline_sha256']==offline['checkpoint_sha256']
            assert cfg['online_steps']==done['steps']==50000
            expected=int(30000*utd);assert done['updates']==expected
            assert actual==actual0 and init==init0 and init['critic']==offline['q_hash'] and init['target']==offline['target_q_hash']
            assert actual['actor_entropy_enabled'] and actual['automatic_alpha_enabled'] and actual['residual_scale']==.1
            assert actual['critic_ensemble']==10 and actual['actor_Q_aggregation']==actual['target_aggregation']=='minimum'
            assert done['final_flow_hash']==init['flow']==offline['flow_hash']
            assert done['final_actor_hash']!=init['actor'] and done['final_critic_hash']!=init['critic']
            assert load(folder/'CHECKS_PASSED.json')['status']=='passed' and load(folder/'backup_math.json')['status']=='passed'
            assert load(folder/'task_manifest.json')==load(baseline/'task_manifest.json')
            warm=load(folder/'warmup.json')
            assert {k:v for k,v in warm.items() if k!='replay_hash'}=={k:v for k,v in warm0.items() if k!='replay_hash'}
            if utd==1.:
                intervention=load(folder/'utd_intervention.json')
                assert intervention['status']=='passed' and intervention['utd']==utd
                assert intervention['old']==old and intervention['new']==new
                assert intervention['original_loop_sha256']==hashlib.sha256(loop.encode()).hexdigest()
                assert intervention['adapted_loop_sha256']==hashlib.sha256(loop.replace(old,new).encode()).hexdigest()
            if (folder/'final.pkl').exists():
                assert digest(folder/'final.pkl')==done['checkpoint_sha256']
                with (folder/'final.pkl').open('rb') as f:cp=pickle.load(f)
                assert cp['step']==50000 and cp['updates']==int(cp['agent']['updates'])==expected
                assert int(cp['agent']['actor']['step'])==int(cp['agent']['critic']['step'])==expected
                del cp
            fixed=arrays(folder/'fixed_probe.npz');fm=load(folder/'fixed_probe.json')
            assert fm['samples']==256 and fm['sha256']==digest(folder/'fixed_probe.npz')
            for k,a in fixed.items():
                if k in ('observations','next_observations'):np.testing.assert_allclose(a,fixed0[k],rtol=0,atol=1e-12)
                else:np.testing.assert_array_equal(a,fixed0[k])
            points=[checked(p) for p in sorted(folder.glob('eval_*_050.json'))];assert len(points)==7
            for nominal,e in zip([0,5000,10000,20000,30000,40000,50000],points):
                assert nominal<=e['step']<=min(nominal+4,50000) and not e['deterministic_residual']
                assert [r['initial_hash'] for r in e['records']]==[r['initial_hash'] for r in base_eval['records']]
                mc=check_mc(folder,e,'hard');mc_rows.append(dict(task=task,utd=utd,**{k:v for k,v in mc.items() if k!='records'}))
                mc_episodes.extend(dict(task=task,utd=utd,step=e['step'],**r) for r in mc['records'])
            final=checked(folder/'eval_050000_100.json');mean=checked(folder/'eval_050000_050_mean_residual.json')
            state=load(folder/'evaluation_state_check.json')
            assert state['agent_unchanged_during_final_evaluations'] and state['before_eval_hash']==state['after_eval_hash']
            for a,b in zip(points[-1]['records'],final['records'][:50]):
                for k in ('episode','reset_seed','initial_hash'):assert a[k]==b[k]
                if a!=b:endpoint_diffs.append(dict(task=task,utd=utd,episode=a['episode'],curve50=a,final100=b))
            assert not final['deterministic_residual'] and mean['deterministic_residual']
            metrics_path=folder/'metrics.jsonl';track(metrics_path);metrics=[json.loads(s) for s in metrics_path.read_text().splitlines()]
            assert metrics[-1]['step']==50000 and metrics[-1]['updates']==expected
            for m in metrics:
                assert all(np.isfinite(v) for v in m.values() if isinstance(v,(int,float)))
                if m['step']>warm['steps']:assert m['updates']==int((m['step']-20000)*utd)
            ps=[check_probe(folder,p,fixed) for p in sorted(folder.glob('probe_*.json'))]
            assert len(ps)==4 and ps[0]['updates']==0 and ps[-1]['updates']==expected
            probes.extend(dict(task=task,utd=utd,**p) for p in ps)
            rows.append(dict(task=task,utd=utd,batch_size=256,td_target='hard',seed=0,offline_updates=500000,
                online_steps=50000,warmup_steps=warm['steps'],updates=expected,sampled_rows=expected*256,
                residual_scale=.1,replay='online_only',final_episodes=100,success_percent=100*final['success'],
                return_mean=final['return_mean'],initial50_success=100*points[0]['success'],curve_final50_success=100*points[-1]['success'],
                mean_residual50_success=100*mean['success'],mean_residual50_return=mean['return_mean'],
                repeated_final_first50_exact=points[-1]['records']==final['records'][:50],alpha_final=ps[-1]['alpha'],
                success_auc_percent=float(np.trapz([e['success'] for e in points],[e['step'] for e in points])/500)))
            curves[task][utd]=points;finals[utd]=final
            for role,es in [('curve50',points),('final100',[final]),('mean_residual50',[mean])]:
                for e in es:
                    evals.append(dict(task=task,utd=utd,role=role,step=e['step'],episodes=e['episodes'],success_percent=100*e['success'],return_mean=e['return_mean']))
                    episodes.extend(dict(task=task,utd=utd,role=role,step=e['step'],**r) for r in e['records'])
        a,b=finals[1.]['records'],finals[.25]['records']
        for x,y in zip(a,b):assert x['reset_seed']==y['reset_seed'] and x['initial_hash']==y['initial_hash']
        pairs.append(dict(task=task,episodes=100,both_success=sum(bool(x['success']) and bool(y['success']) for x,y in zip(a,b)),
            utd1_only_success=sum(bool(x['success']) and not bool(y['success']) for x,y in zip(a,b)),
            utd025_only_success=sum(not bool(x['success']) and bool(y['success']) for x,y in zip(a,b)),
            neither_success=sum(not bool(x['success']) and not bool(y['success']) for x,y in zip(a,b)),
            success_difference_pp=100*(finals[1.]['success']-finals[.25]['success']),return_difference=finals[1.]['return_mean']-finals[.25]['return_mean']))
        for x,y in zip(curves[task][1.][0]['records'],curves[task][.25][0]['records']):
            assert x['reset_seed']==y['reset_seed'] and x['initial_hash']==y['initial_hash']
            if x!=y:initial_diffs.append(dict(task=task,episode=x['episode'],utd1=x,utd025=y))
    for filename,data in [('comparison.csv',rows),('evaluations.csv',evals),('episodes.csv',episodes),('paired_outcomes.csv',pairs),
        ('fixed_probe.csv',probes),('mc_summary.csv',mc_rows),('mc_episodes.csv',mc_episodes),('historical_context.csv',context)]:
        evidence.write_csv(OUT/filename,data)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'initial_evaluation_differences.json').write_text(json.dumps(initial_diffs,indent=2)+'\n')
    (OUT/'repeated_endpoint_differences.json').write_text(json.dumps(endpoint_diffs,indent=2)+'\n')
    plot(curves);report(rows,pairs,initial_diffs,endpoint_diffs,context)
    for name in ['summarize_utd_ablation.py','audit_utd_warmup.py','summarize_td_ablation.py','summarize_task2_scale.py',
                 'audit_td_warmup.py','audit_td_batch_warmup.py','CHART_CONTRACT_UTD_ABLATION.md']:track(ROOT/name)
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',sources=evidence.sources,
        checks=['51 frozen sources','original UTD .25 short checkpoint parity','only UTD differs in config and loop',
        'correct task checkpoints and full initial states','full warmup arrays and fixed probes','50k steps / 7500 or 30000 updates',
        'finite training state','frozen base','inherited current and target critics','raw evaluation aggregates',
        'paired reset states','repeated evaluation full state unchanged','all episode differences disclosed'],
        repeated_endpoint_differences=len(endpoint_diffs)),indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


def plot(curves):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for task in (1,2):
        for col,key,factor,label in [(0,'success',100,'Success rate (%)'),(1,'return_mean',1,'Mean episode return')]:
            ax=axes[task-1,col]
            for u,color,marker,fill in [(.25,'#82A9D0','s','white'),(1.,'#2463A6','o','#2463A6')]:
                ps=curves[task][u]
                ax.plot([p['step']/1000 for p in ps],[p[key]*factor for p in ps],linestyle='none',marker=marker,
                    markersize=7 if u==.25 else 5,markerfacecolor=fill,markeredgecolor=color,markeredgewidth=1.5,label=f'UTD {u:g}')
            ax.axvline(20,color='#ADB2B8',linestyle=':',linewidth=1)
            ax.set_title(f'Task{task}',loc='left',fontsize=12);ax.set_xlabel('Primitive online steps (thousands)');ax.set_ylabel(label)
            ax.set_xlim(-1,51);ax.set_xticks([0,5,10,20,30,40,50]);ax.grid(axis='y',color='#E8EAED')
            if col==0:ax.set_ylim(0,103)
    fig.suptitle('Plain TD: update-to-data ratio comparison',x=.08,ha='left',fontsize=16)
    fig.text(.08,.93,'Seed 0 | Batch 256 | Inherited Q | Frozen base | Scale 0.1 | Online-only replay | 50 episodes per point',fontsize=10)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
    fig.text(.08,.037,'50k online steps including 20k warmup. UTD 0.25: 7,500 updates; UTD 1: 30,000 updates. Actor entropy retained.',fontsize=9)
    fig.text(.08,.014,'Seven measured checkpoints; no interpolation or across-seed uncertainty bands. Final table uses 100 episodes.',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.875,bottom=.155,hspace=.40,wspace=.24)
    for ext in ('png','pdf'):fig.savefig(OUT/f'evaluation_checkpoints.{ext}',dpi=180)
    plt.close(fig)


def report(rows,pairs,initial_diffs,endpoint_diffs,context):
    lines=['# 普通 TD、batch 256 下的 UTD 对照','',
        'Task1 / Task2，seed 0。两组均继承各任务 QAM offline 500k critic / target 参数，冻结 base，online-only replay，scale=.1。普通 TD 保留 actor entropy 和自动 alpha，仅 critic target 不含 entropy。',
        'UTD=1 为本轮新增；UTD=.25 是刚在同一机器完成的 batch=256 对照。总环境预算仍为 50k，包含 20k base-only warmup。学习率、target tau、actor/critic/alpha 的每更新步规则均不变。','',
        '| Task | UTD | Online steps | Updates | 成功率（100 eps） | 平均回报 ↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {r['utd']:g} | 50,000 | {r['updates']:,} | {r['success_percent']:.0f}% | {r['return_mean']:.2f} |")
    lines+=['','UTD 1 − UTD .25：','']
    for p in pairs:lines.append(f"- Task{p['task']}：成功率 {p['success_difference_pp']:+.0f} pp，平均回报 {p['return_difference']:+.2f}。")
    lines+=['','本轮提高 UTD 的效果因任务而异：Task1 没有改善，Task2 的成功率提高、平均回报小幅改善。增加到四倍更新次数没有使两个任务都接近 QAM native；仍只有一个训练 seed，不能推断稳定或普遍的优劣。']
    lines+=['','相同 50 episodes 的起终点及均值残差诊断：','',
        '| Task | UTD | 初始成功率 | 终点随机残差 | 终点均值残差 | 曲线 AUC / 50k |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {r['utd']:g} | {r['initial50_success']:.0f}% | {r['curve_final50_success']:.0f}% | {r['mean_residual50_success']:.0f}% | {r['success_auc_percent']:.2f}% |")
    lines+=['','设置与解释：','',
        '- 本轮只把 UTD 从 .25 改为 1，batch 保持 256。因此更新次数和累计训练抽样量均变为四倍：7,500→30,000 次，192 万→768 万行。有放回抽样，累计行数不代表独立新数据量。',
        '- 训练入口只替换原 loop 中的一处 UTD 常量，并逐次保存准确 diff 和源代码哈希。UTD=.25 的短程完整 checkpoint 在两个任务都与原入口一致，UTD=1 短程完成 40 次更新。',
        '- Task1 最初与历史短程参考的权重有细小差异；补跑原入口后，它与新 UTD=.25 入口的完整 checkpoint 精确一致。历史差异及排除的参考运行保留在实验记录中，没有修改正式对照数据。',
        '- 固定 20k warmup；QAM native 使用 5k warmup，50k 环境预算下仍有 45,001 次更新，因此本轮并未把整个日程完全对齐 QAM。',
        '- 其余参数：10 critics，actor/target minimum 聚合，gamma=.99，horizon=5，lr=1e-4，tau=.01，gradient clip=50，alpha 初始 .01，目标残差熵=-25。',
        '- 单 training seed。先前同设置复跑存在数值/轨迹分叉，Task2 控制组曾有一条终点评估轨迹未精确复现；本报告沿用预定最终 100-episode 结果，不选更好的重复评估。',
        f'- 当前配对初始评估有 {len(initial_diffs)} 条记录差异，终点 50 与 100 episodes 的重叠部分有 {len(endpoint_diffs)} 条差异。完整模型状态在终点评估前后保持一致，所有差异均保留在 JSON 中。',
        '- 图展示七个实际评估点，每点 50 episodes；AUC 为这些离散点的梯形面积除以 50k，不代表未评估区间内没有波动。','',
        '其他已完成设置，仅作背景：','']
    for r in context:lines.append(f"- Task{r['task']}：本机普通 TD / batch1024 / UTD.25 为 {r['batch1024_utd025_success']:.0f}% / {r['batch1024_utd025_return']:.2f}；历史 QAM native 为 {r['native_success']:.0f}% / {r['native_return']:.2f}。")
    lines+=['','![Evaluation checkpoints](evaluation_checkpoints.png)','']
    (OUT/'summary.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
