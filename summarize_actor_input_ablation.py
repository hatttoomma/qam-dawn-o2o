"""Validate and report paired actor-input experiments, preserving historical controls."""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import summarize_task2_scale as evidence
from summarize_td_ablation import arrays, check_mc, check_probe

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/actor_input_ablation'
OUT=ROOT/'results_actor_input_ablation'
load,checked,track,digest=evidence.load,evidence.checked,evidence.track,evidence.digest
MODES=('masked','base_action')
LABELS={'masked':'State only (masked)','base_action':'State + base action'}


def main():
    OUT.mkdir(exist_ok=True)
    frozen=load(RUNS/'source_manifest.json');assert len(frozen)==54
    for n,h in frozen.items():assert digest(ROOT/n)==h,n
    patch=load(ROOT/'actor_input_agent_patch.json')
    source=(ROOT/patch['original']).read_text();track(ROOT/patch['original'])
    assert hashlib.sha256(source.encode()).hexdigest()==patch['original_sha256']
    for old,new in patch['replacements']:
        assert source.count(old)==1;source=source.replace(old,new)
    source=source.replace('\n\nclass ResidualActor',patch['added_helper']+'\n\nclass ResidualActor',1)
    assert source==(ROOT/patch['adapted']).read_text();track(ROOT/patch['adapted'])
    assert digest(ROOT/patch['adapted'])==patch['adapted_sha256']
    assert load(RUNS/'CHECKS_PASSED.json')['status']=='passed'
    assert load(RUNS/'runtime_setup.json')['probe_exit_code']==0
    assert (RUNS/'pip_freeze.txt').read_bytes()==(ROOT/'runs/td_ablation/batch_comparison/pip_freeze.txt').read_bytes()
    track(RUNS/'pip_freeze.txt');track(ROOT/'runs/td_ablation/batch_comparison/pip_freeze.txt')
    audit=load(RUNS/'warmup_audit/audit.json');assert audit['status']=='passed'
    for v in audit['arms'].values():
        p=ROOT/v['snapshot'];track(p);assert digest(p)==v['snapshot_sha256']
        if 'historical_snapshot' in v:
            p=ROOT/v['historical_snapshot'];track(p);assert digest(p)==v['historical_snapshot_sha256']
    rows,evals,episodes,pairs,probes,mc_rows,mc_episodes,initial_diffs,endpoint_diffs,context=[],[],[],[],[],[],[],[],[],[]
    curves={}
    for task in (1,2):
        historical=ROOT/('runs' if task==1 else 'runs/task2')
        offline=load(historical/'offline/DONE.json');assert offline['step']==500000
        base=checked(historical/'offline/eval_500000_050.json')
        native=checked(historical/'native/eval_050000_100.json')
        previous=ROOT/f'runs/td_ablation/batch_comparison/task{task}_b256'
        prior=checked(previous/'eval_050000_100.json');prior_cfg=load(previous/'config.json')
        context.append(dict(task=task,native_success_percent=100*native['success'],native_return=native['return_mean'],
            historical_37input_success_percent=100*prior['success'],historical_37input_return=prior['return_mean']))
        first=None;finals={};curves[task]={}
        for mode in MODES:
            folder=RUNS/f'task{task}_{mode}'
            cfg=load(folder/'config.json');selected=load(folder/'td_config.json');actual=load(folder/'actual_agent_config.json')
            init=load(folder/'initial_hashes.json');warm=load(folder/'warmup.json');done=load(folder/'DONE.json')
            assert {k:v for k,v in cfg.items() if k not in ('out','actor_input')}=={k:v for k,v in prior_cfg.items() if k!='out'}
            assert cfg['actor_input']==selected['actor_input']==actual['actor_input']==mode
            assert cfg['td_target']=='hard' and cfg['dawn_batch']==256 and cfg['warmup']==20000 and cfg['seed']==0
            assert cfg['offline_sha256']==selected['offline_sha256']==offline['checkpoint_sha256']
            assert done['steps']==cfg['online_steps']==50000 and done['updates']==7500
            assert init['critic']==offline['q_hash'] and init['target']==offline['target_q_hash']
            assert init['flow']==done['final_flow_hash']==offline['flow_hash']
            assert done['final_actor_hash']!=init['actor'] and done['final_critic_hash']!=init['critic']
            assert actual['actor_input_dim']==62 and actual['action_dim']==25 and actual['target_entropy']==-25
            assert actual['hidden_dims']==[256,256,256] and actual['activation']=='relu' and actual['gaussian_output']
            assert actual['residual_scale']==.1 and actual['target_tau']==.01 and actual['utd']==.25
            assert actual['actor_entropy_enabled'] and actual['automatic_alpha_enabled'] and actual['critic_ensemble']==10
            assert actual['actor_Q_aggregation']==actual['target_aggregation']=='minimum'
            assert load(folder/'actor_input_checks.json')['actor_parameter_count']==160562
            for n in ['CHECKS_PASSED.json','backup_math.json','actor_input_checks.json']:
                assert load(folder/n)['status']=='passed'
            assert load(folder/'task_manifest.json')==load(previous/'task_manifest.json')
            fixed=arrays(folder/'fixed_probe.npz');fm=load(folder/'fixed_probe.json')
            assert digest(folder/'fixed_probe.npz')==fm['sha256'] and fm['samples']==256
            if first is None:first=(init,actual,warm,fixed)
            else:
                fi,fa,fw,ff=first
                assert init==fi
                assert {k:v for k,v in actual.items() if k!='actor_input'}=={k:v for k,v in fa.items() if k!='actor_input'}
                assert {k:v for k,v in warm.items() if k!='replay_hash'}=={k:v for k,v in fw.items() if k!='replay_hash'}
                for k,a in fixed.items():
                    if k in ('observations','next_observations'):np.testing.assert_allclose(a,ff[k],rtol=0,atol=1e-12)
                    else:np.testing.assert_array_equal(a,ff[k])
            if (folder/'final.pkl').exists():
                assert digest(folder/'final.pkl')==done['checkpoint_sha256']
                with (folder/'final.pkl').open('rb') as f:cp=pickle.load(f)
                assert cp['step']==50000 and cp['updates']==int(cp['agent']['updates'])==7500
                assert int(cp['agent']['actor']['step'])==int(cp['agent']['critic']['step'])==7500
                assert cp['agent']['actor']['params']['Dense_0']['kernel'].shape==(62,256)
                del cp
            state=load(folder/'evaluation_state_check.json')
            assert state['agent_unchanged_during_final_evaluations'] and state['before_eval_hash']==state['after_eval_hash']
            points=[checked(p) for p in sorted(folder.glob('eval_*_050.json'))];assert len(points)==7
            for nominal,e in zip([0,5000,10000,20000,30000,40000,50000],points):
                assert nominal<=e['step']<=min(nominal+4,50000) and not e['deterministic_residual']
                assert [r['initial_hash'] for r in e['records']]==[r['initial_hash'] for r in base['records']]
                mc=check_mc(folder,e,'hard')
                mc_rows.append(dict(task=task,actor_input=mode,**{k:v for k,v in mc.items() if k!='records'}))
                mc_episodes.extend(dict(task=task,actor_input=mode,step=e['step'],**r) for r in mc['records'])
            final=checked(folder/'eval_050000_100.json');mean=checked(folder/'eval_050000_050_mean_residual.json')
            assert not final['deterministic_residual'] and mean['deterministic_residual']
            for a,b in zip(points[-1]['records'],final['records'][:50]):
                for k in ('episode','reset_seed','initial_hash'):assert a[k]==b[k]
                if a!=b:endpoint_diffs.append(dict(task=task,actor_input=mode,episode=a['episode'],curve50=a,final100=b))
            metrics_path=folder/'metrics.jsonl';track(metrics_path)
            metrics=[json.loads(s) for s in metrics_path.read_text().splitlines()]
            assert metrics[-1]['step']==50000 and metrics[-1]['updates']==7500
            for m in metrics:
                assert all(np.isfinite(v) for v in m.values() if isinstance(v,(int,float)))
                if m['step']>warm['steps']:assert m['updates']==int((m['step']-20000)*.25)
            ps=[check_probe(folder,p,fixed) for p in sorted(folder.glob('probe_*.json'))]
            assert len(ps)==4 and ps[0]['updates']==0 and ps[-1]['updates']==7500
            probes.extend(dict(task=task,actor_input=mode,**p) for p in ps)
            rows.append(dict(task=task,actor_input=mode,input_width=62,actor_parameters=160562,seed=0,
                td_target='hard',batch_size=256,utd=.25,offline_updates=500000,online_steps=50000,updates=7500,
                warmup_steps=warm['steps'],residual_scale=.1,replay='online_only',final_episodes=100,
                success_percent=100*final['success'],return_mean=final['return_mean'],
                initial50_success=100*points[0]['success'],curve_final50_success=100*points[-1]['success'],
                mean_residual50_success=100*mean['success'],mean_residual50_return=mean['return_mean'],
                final_alpha=ps[-1]['alpha'],repeated_final_first50_exact=points[-1]['records']==final['records'][:50],
                success_auc_percent=float(np.trapz([e['success'] for e in points],[e['step'] for e in points])/500)))
            curves[task][mode]=points;finals[mode]=final
            for role,es in [('curve50',points),('final100',[final]),('mean_residual50',[mean])]:
                for e in es:
                    evals.append(dict(task=task,actor_input=mode,role=role,step=e['step'],episodes=e['episodes'],
                        success_percent=100*e['success'],return_mean=e['return_mean']))
                    episodes.extend(dict(task=task,actor_input=mode,role=role,step=e['step'],**r) for r in e['records'])
        control,treatment=finals['masked'],finals['base_action']
        for a,b in zip(control['records'],treatment['records']):
            assert a['reset_seed']==b['reset_seed'] and a['initial_hash']==b['initial_hash']
        pairs.append(dict(task=task,episodes=100,
            both_success=sum(a['success'] and b['success'] for a,b in zip(control['records'],treatment['records'])),
            base_input_only_success=sum(not a['success'] and b['success'] for a,b in zip(control['records'],treatment['records'])),
            masked_only_success=sum(a['success'] and not b['success'] for a,b in zip(control['records'],treatment['records'])),
            neither_success=sum(not a['success'] and not b['success'] for a,b in zip(control['records'],treatment['records'])),
            success_difference_pp=100*(treatment['success']-control['success']),return_difference=treatment['return_mean']-control['return_mean']))
        for a,b in zip(curves[task]['masked'][0]['records'],curves[task]['base_action'][0]['records']):
            assert a['initial_hash']==b['initial_hash']
            if a!=b:initial_diffs.append(dict(task=task,episode=a['episode'],masked=a,base_action=b))
    for filename,data in [('comparison.csv',rows),('evaluations.csv',evals),('episodes.csv',episodes),('paired_outcomes.csv',pairs),
                          ('fixed_probe.csv',probes),('mc_summary.csv',mc_rows),('mc_episodes.csv',mc_episodes),('historical_context.csv',context)]:
        evidence.write_csv(OUT/filename,data)
    for filename,data in [('summary.json',rows),('initial_evaluation_differences.json',initial_diffs),('repeated_endpoint_differences.json',endpoint_diffs)]:
        (OUT/filename).write_text(json.dumps(data,indent=2)+'\n')
    plot(curves);report(rows,pairs,context,initial_diffs,endpoint_diffs)
    for n in ['summarize_actor_input_ablation.py','audit_actor_input_warmup.py','summarize_td_ablation.py',
              'summarize_task2_scale.py','audit_td_warmup.py','audit_td_batch_warmup.py','CHART_CONTRACT_ACTOR_INPUT.md']:track(ROOT/n)
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',sources=evidence.sources,
        checks=['54 frozen training/protocol files','exact agent source patch','input masking and feature gradients',
                'identical full initial paired states','correct inherited task critics','full paired warmup arrays',
                'fixed probes and MC arithmetic','7500 updates / 50k primitive steps','frozen base',
                'unchanged agent across final evaluations','raw episode aggregation and paired reset states'],
        initial_evaluation_differences=len(initial_diffs),repeated_endpoint_differences=len(endpoint_diffs)),indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


def plot(curves):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for task in (1,2):
        for col,key,factor,label in [(0,'success',100,'Success rate (%)'),(1,'return_mean',1,'Mean episode return')]:
            ax=axes[task-1,col]
            for mode,color,marker,fill in [('masked','#82A9D0','s','white'),('base_action','#2463A6','o','#2463A6')]:
                ps=curves[task][mode]
                ax.plot([p['step']/1000 for p in ps],[p[key]*factor for p in ps],linestyle='none',marker=marker,
                    markersize=7 if mode=='masked' else 5,markerfacecolor=fill,markeredgecolor=color,markeredgewidth=1.5,label=LABELS[mode])
            ax.axvline(20,color='#ADB2B8',linestyle=':',linewidth=1)
            ax.set_title(f'Task{task}',loc='left',fontsize=12);ax.set_xlabel('Primitive online steps (thousands)');ax.set_ylabel(label)
            ax.set_xlim(-1,51);ax.set_xticks([0,5,10,20,30,40,50]);ax.grid(axis='y',color='#E8EAED')
            if col==0:ax.set_ylim(0,103)
    fig.suptitle('Residual actor: does the sampled base action help?',x=.08,ha='left',fontsize=16)
    fig.text(.08,.93,'Seed 0 | Plain TD | Batch 256 | UTD 0.25 | Inherited Q | Frozen base | Scale 0.1 | Online-only replay',fontsize=10)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
    fig.text(.08,.037,'Both actors: 62 inputs, same initial weights. Control masks 25 base features. Gaussian output and actor entropy retained.',fontsize=9)
    fig.text(.08,.014,'50k steps including 20k warmup; 7,500 updates. Seven 50-episode checkpoints; final table uses 100 episodes.',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.875,bottom=.155,hspace=.40,wspace=.24)
    for ext in ('png','pdf'):fig.savefig(OUT/f'evaluation_checkpoints.{ext}',dpi=180)
    plt.close(fig)


def report(rows,pairs,context,initial_diffs,endpoint_diffs):
    lines=['# Residual actor 输入对照','',
        'Task1 / Task2，seed 0。普通 TD、batch256、UTD=.25；offline500k，继承 current/target critic，冻结 QAM base，online-only replay，scale=.1，20k warmup，50k primitive online steps，7,500 次更新。Gaussian residual、actor entropy 和自动 alpha 保留。','',
        '两组均新训练；同一任务的完整初始参数、优化器和 RNG 一致。都使用 62 维输入、3×256 ReLU 和两个 25 维输出头。对照组屏蔽额外 25 维 base features，实验组输入实际 sampled base action chunk。Critic 输入保持 state 与最终相加后的动作。','',
        '| Task | Actor 输入 | 成功率（100 eps） | 平均回报 ↑ |',
        '|---|---|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {'state（额外输入填零）' if r['actor_input']=='masked' else 'state + base action'} | {r['success_percent']:.0f}% | {r['return_mean']:.2f} |")
    lines+=['','加入 base action − 屏蔽 base action：','']
    for p in pairs:lines.append(f"- Task{p['task']}：成功率 {p['success_difference_pp']:+.0f} pp；平均回报 {p['return_difference']:+.2f}。")
    lines+=['','辅助评估与历史背景：','',
        '| Task | 输入 | 随机残差终点（50 eps） | 均值残差（50 eps） | 成功率 AUC / 50k |',
        '|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {r['actor_input']} | {r['curve_final50_success']:.0f}% | {r['mean_residual50_success']:.0f}% | {r['success_auc_percent']:.2f}% |")
    lines+=['']
    for c in context:lines.append(f"- Task{c['task']} 历史 37-input / batch256 / UTD.25：{c['historical_37input_success_percent']:.0f}% / {c['historical_37input_return']:.2f}；QAM native：{c['native_success_percent']:.0f}% / {c['native_return']:.2f}。")
    lines+=['','解释与验证：','',
        '- 主对照是本次配对重跑的 masked control。两个新网络均为 160,562 参数，比旧 37 维 actor 多 6,400 个第一层权重；初始化因输入宽度而变化，因此不能把新 treatment 与历史 control 的差值全归因于 conditioning。',
        '- 网络宽度、随机种子、完整初始状态、critic/target、学习率、熵目标 -25 和训练日程在配对内一致；唯一配置差异为额外 base 特征是否可见。',
        '- 54 个训练/协议文件冻结，输入梯度检查、完整 warmup replay、固定 probe、更新计数、冻结 base、终点评估前后完整 agent 状态与原始 episode 聚合均通过。',
        f'- 初始配对评估有 {len(initial_diffs)} 条轨迹记录差异，最终 50/100 episodes 重叠部分有 {len(endpoint_diffs)} 条差异；全部保留，主表使用预定最终 100 episodes，不挑选更好重复评估。',
        '- 曲线是七个实际评估点，每点 50 episodes；AUC 为梯形积分，不能代表未评估区间内没有波动。',
        '- 只有一个 training seed。结果仅支持本次固定设置下的观察，不能确定跨 seed 稳定优势，也不确定各输入形式调参后的最优性能。','',
        '![Evaluation checkpoints](evaluation_checkpoints.png)']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
