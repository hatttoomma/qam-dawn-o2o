"""Validate and report target-entropy ablations with original-data provenance."""
import json
import pickle
from pathlib import Path
import numpy as np
import summarize_task2_scale as evidence
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/td_ablation'
OUT=ROOT/'results_td_ablation'
load,checked,track,digest=evidence.load,evidence.checked,evidence.track,evidence.digest

def arrays(path):
    track(path)
    with np.load(path) as z:return {k:z[k] for k in z.files}

def check_mc(folder,e,kind):
    p=folder/f'mc_{e["step"]:06d}_050.json';m=load(p)
    a=arrays(p.with_suffix('.npz'))
    assert digest(p.with_suffix('.npz'))==m['array_sha256']
    assert m['episodes']==50 and m['step']==e['step'] and m['td_target']==kind
    assert a['qs'].shape==(10,50) and all(np.isfinite(v).all() for v in a.values())
    for i,(x,y) in enumerate(zip(m['records'],e['records'])):
        for k in ('episode','reset_seed','initial_hash','success','length'):assert x[k]==y[k]
        np.testing.assert_allclose(x['soft_mc'],x['reward_mc']+x['future_entropy_mc'],rtol=0,atol=1e-10)
        np.testing.assert_allclose(x['q_min'],a['qs'][:,i].min(),rtol=0,atol=1e-10)
        np.testing.assert_allclose(x['q_mean'],a['qs'][:,i].mean(),rtol=0,atol=1e-10)
    target=np.array([x['soft_mc' if kind=='soft' and m['residual_enabled'] else 'reward_mc'] for x in m['records']])
    error=a['qs'].min(axis=0)-target;terminal=np.array([x['terminated'] for x in m['records']])
    np.testing.assert_allclose(np.abs(error).mean(),m['all_horizon_limited_mae'],rtol=0,atol=1e-10)
    assert int(terminal.sum())==m['terminal_episodes']
    if terminal.any():np.testing.assert_allclose(np.abs(error[terminal]).mean(),m['terminal_mae'],rtol=0,atol=1e-10)
    else:assert m['terminal_mae'] is None
    return m

def check_probe(folder,p,fixed):
    m=load(p);a=arrays(p.with_suffix('.npz'))
    assert m['array_sha256']==digest(p.with_suffix('.npz'))
    assert m['fixed_probe_sha256']==digest(folder/'fixed_probe.npz') and m['samples']==256
    assert all(np.isfinite(v).all() for v in a.values())
    np.testing.assert_array_equal(a['rewards'],fixed['rewards']);np.testing.assert_array_equal(a['discounts'],fixed['discounts'])
    np.testing.assert_allclose(a['hard_target'],a['rewards']+a['discounts']*a['next_min_q'],rtol=0,atol=3e-5)
    np.testing.assert_allclose(a['entropy_target_term'],-a['discounts']*m['alpha']*a['next_logp'],rtol=0,atol=1e-6)
    np.testing.assert_allclose(a['soft_target'],a['hard_target']+a['entropy_target_term'],rtol=0,atol=3e-5)
    np.testing.assert_allclose(a['residual_q_difference'],a['residual_q']-a['base_q'],rtol=0,atol=1e-6)
    for field,value in [('entropy_target_mean',a['entropy_target_term'].mean()),
        ('entropy_target_abs_mean',np.abs(a['entropy_target_term']).mean()),('q_mean',a['current_qs'].mean()),
        ('hard_td_mse',np.square(a['current_qs']-a['hard_target']).mean()),
        ('soft_td_mse',np.square(a['current_qs']-a['soft_target']).mean()),
        ('residual_q_difference_mean',a['residual_q_difference'].mean()),
        ('residual_gradient_norm_mean',a['residual_gradient_norm'].mean())]:
        np.testing.assert_allclose(m[field],value,rtol=0,atol=1e-6)
    return m

def main():
    OUT.mkdir(exist_ok=True)
    for n,h in load(RUNS/'source_manifest.json').items():assert digest(ROOT/n)==h,n
    smoke=load(RUNS/'CHECKS_PASSED.json');assert smoke['status']=='passed' and smoke['original_soft_full_checkpoint_parity']
    audit=load(RUNS/'warmup_audit/audit.json');assert audit['status']=='passed'
    for v in audit['arms'].values():
        p=ROOT/v['snapshot'];track(p);assert digest(p)==v['snapshot_sha256']
    assert load(RUNS/'runtime_setup.json')['probe_exit_code']==0
    reproduction=load(RUNS/'task2_soft_historical_reproduction.json')
    assert reproduction['initial_hashes_identical'] and reproduction['warmup_hashes_identical']
    rows,eval_rows,episode_rows,probe_rows,mc_rows,mc_episodes,pairs,initial_diffs=[],[],[],[],[],[],[],[]
    curves,probes,native={},{},{}
    for task in (1,2):
        hist=ROOT/('runs' if task==1 else 'runs/task2')
        offline=load(hist/'offline/DONE.json');assert offline['step']==500000
        base=checked(hist/'offline/eval_500000_050.json')
        old=checked(hist/'warm/eval_050000_100.json');old_done=load(hist/'warm/DONE.json')
        old_cfg=load(hist/'warm/config.json');native_final=checked(hist/'native/eval_050000_100.json')
        native[task]=[checked(p) for p in sorted((hist/'native').glob('eval_*_050.json'))]
        curves[task],probes[task]={},{}
        previous_init,previous_actual,previous_fixed,previous_warm,finals=None,None,None,None,{}
        for kind in ('soft','hard'):
            folder=RUNS/f'task{task}_{kind}'
            cfg=load(folder/'config.json');selected=load(folder/'td_config.json');actual=load(folder/'actual_agent_config.json')
            for k,v in old_cfg.items():
                if k!='out':assert cfg[k]==v,(task,kind,k)
            assert set(cfg)-set(old_cfg)<={'task','td_target','environment'}
            assert cfg['offline_sha256']==selected['offline_sha256']==offline['checkpoint_sha256']
            assert cfg['task']==task and cfg['td_target']==kind and cfg['seed']==0
            assert cfg['online_steps']==50000 and cfg['warmup']==20000 and cfg['dawn_batch']==1024
            assert actual['residual_scale']==.1 and actual['target_tau']==.01 and actual['action_dim']==25
            assert actual['target_aggregation']==actual['actor_Q_aggregation']=='minimum'
            assert actual['critic_ensemble']==10 and actual['actor_entropy_enabled'] and actual['automatic_alpha_enabled']
            init=load(folder/'initial_hashes.json');warm=load(folder/'warmup.json')
            assert init['critic']==offline['q_hash'] and init['target']==offline['target_q_hash']
            assert init['flow']==offline['flow_hash']
            task_manifest=load(folder/'task_manifest.json');assert task_manifest['reward_task_id']==task
            done=load(folder/'DONE.json');checks=load(folder/'CHECKS_PASSED.json')
            assert checks['status']=='passed' and done['steps']==50000 and done['updates']==7500
            assert done['final_flow_hash']==init['flow']
            assert done['final_critic_hash']!=init['critic'] and done['final_actor_hash']!=init['actor']
            assert load(folder/'backup_math.json')['status']=='passed'
            if (folder/'final.pkl').exists():
                assert digest(folder/'final.pkl')==done['checkpoint_sha256']
                with (folder/'final.pkl').open('rb') as f:cp=pickle.load(f)
                assert cp['step']==50000 and cp['updates']==int(cp['agent']['updates'])==7500
                assert int(cp['agent']['actor']['step'])==int(cp['agent']['critic']['step'])==7500
                assert float(cp['agent']['log_alpha'])!=np.log(init['alpha']);del cp
            fixed=arrays(folder/'fixed_probe.npz');fm=load(folder/'fixed_probe.json')
            assert digest(folder/'fixed_probe.npz')==fm['sha256'] and fm['samples']==256
            if previous_init is None:
                previous_init,previous_actual,previous_fixed,previous_warm=init,actual,fixed,warm
            else:
                assert init==previous_init
                assert {k:v for k,v in actual.items() if k!='td_target'}=={k:v for k,v in previous_actual.items() if k!='td_target'}
                assert {k:v for k,v in warm.items() if k!='replay_hash'}=={k:v for k,v in previous_warm.items() if k!='replay_hash'}
                assert fixed.keys()==previous_fixed.keys()
                for k,a in fixed.items():
                    if k in ('observations','next_observations'):np.testing.assert_allclose(a,previous_fixed[k],rtol=0,atol=1e-12)
                    else:np.testing.assert_array_equal(a,previous_fixed[k])
            pts=[checked(p) for p in sorted(folder.glob('eval_*_050.json'))];assert len(pts)==7
            for nominal,e in zip([0,5000,10000,20000,30000,40000,50000],pts):
                assert nominal<=e['step']<=min(nominal+4,50000) and not e['deterministic_residual']
                assert [r['initial_hash'] for r in e['records']]==[r['initial_hash'] for r in base['records']]
                mc=check_mc(folder,e,kind)
                mc_rows.append(dict(task=task,**{k:v for k,v in mc.items() if k!='records'}))
                mc_episodes.extend([dict(task=task,td_target=kind,step=e['step'],**r) for r in mc['records']])
            final=checked(folder/'eval_050000_100.json');mean=checked(folder/'eval_050000_050_mean_residual.json')
            assert final['records'][:50]==pts[-1]['records']
            assert final['residual_enabled'] and not final['deterministic_residual'] and mean['deterministic_residual']
            assert [r['initial_hash'] for r in final['records']]==[r['initial_hash'] for r in old['records']]
            path=folder/'metrics.jsonl';track(path);metrics=[json.loads(s) for s in path.read_text().splitlines()]
            assert metrics[-1]['step']==50000 and metrics[-1]['updates']==7500
            for m in metrics:
                for v in m.values():
                    if isinstance(v,(int,float)):assert np.isfinite(v)
                if m['step']>20004:assert m['updates']==int((m['step']-20000)*.25)
            ps=[check_probe(folder,p,fixed) for p in sorted(folder.glob('probe_*.json'))]
            assert len(ps)==4 and [p['updates'] for p in ps][0]==0 and ps[-1]['updates']==7500
            probes[task][kind],curves[task][kind],finals[kind]=ps,pts,final
            probe_rows.extend([dict(task=task,td_target=kind,**p) for p in ps])
            rows.append(dict(task=task,td_target=kind,seed=0,offline_updates=500000,online_steps=50000,
                online_updates=7500,batch_size=1024,residual_scale=.1,replay='online_only',actor_entropy=True,
                final_episodes=100,success_percent=100*final['success'],return_mean=final['return_mean'],
                curve_final50_success_percent=100*pts[-1]['success'],mean_residual50_success_percent=100*mean['success'],
                initial50_success_percent=100*pts[0]['success'],alpha_final=ps[-1]['alpha'],
                inherited_Q=True,native_context_success_percent=100*native_final['success'],native_context_updates=45001,
                historical_soft_success_percent=100*old['success'],
                historical_soft_return_mean=old['return_mean'],
                exact_historical_soft_checkpoint=done['checkpoint_sha256']==old_done['checkpoint_sha256'] if kind=='soft' else None))
            for role,es in [('curve50',pts),('final100',[final]),('mean_residual50',[mean])]:
                for e in es:
                    eval_rows.append(dict(task=task,td_target=kind,role=role,step=e['step'],episodes=e['episodes'],
                        success_percent=100*e['success'],return_mean=e['return_mean']))
                    episode_rows.extend([dict(task=task,td_target=kind,role=role,step=e['step'],**r) for r in e['records']])
        a,b=finals['hard']['records'],finals['soft']['records']
        pairs.append(dict(task=task,episodes=100,both_success=sum(bool(x['success']) and bool(y['success']) for x,y in zip(a,b)),
            hard_only_success=sum(bool(x['success']) and not bool(y['success']) for x,y in zip(a,b)),
            soft_only_success=sum(not bool(x['success']) and bool(y['success']) for x,y in zip(a,b)),
            neither_success=sum(not bool(x['success']) and not bool(y['success']) for x,y in zip(a,b)),
            hard_minus_soft_success_pp=100*(finals['hard']['success']-finals['soft']['success']),
            hard_minus_soft_return=finals['hard']['return_mean']-finals['soft']['return_mean']))
        for x,y in zip(curves[task]['soft'][0]['records'],curves[task]['hard'][0]['records']):
            for k in ('episode','reset_seed','initial_hash','success','length','decisions'):assert x[k]==y[k]
            if x!=y:initial_diffs.append(dict(task=task,episode=x['episode'],soft=x,hard=y))
    for filename,values in [('comparison.csv',rows),('evaluations.csv',eval_rows),('episodes.csv',episode_rows),
        ('fixed_probe.csv',probe_rows),('mc_summary.csv',mc_rows),('mc_episodes.csv',mc_episodes),('paired_outcomes.csv',pairs)]:
        evidence.write_csv(OUT/filename,values)
    (OUT/'initial_evaluation_differences.json').write_text(json.dumps(initial_diffs,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    plot(curves,probes,native)
    write_report(rows,pairs,probe_rows,initial_diffs)
    track(Path(evidence.__file__).resolve())
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',sources=evidence.sources,
        report_source_sha256=digest(Path(__file__)),checks=['frozen source hashes','correct task checkpoints',
        'full paired initial states','unchanged actor entropy and alpha update','warmup replay arrays',
        'same fixed diagnostic batches','counterfactual entropy target arithmetic','original evaluation MC capture',
        'terminal and truncated MC separated','paired reset states','episode aggregates recomputed',
        'final100 first50 matches endpoint','50k steps and 7500 updates']),indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


def plot(curves,probes,native):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False})
    styles=[('soft','#2463A6','o','-','Soft TD'),('hard','#82A9D0','s','--','Plain TD (actor entropy retained)')]
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for task in (1,2):
        for col,key,factor,label in [(0,'success',100,'Success rate (%)'),(1,'return_mean',1,'Mean episode return')]:
            ax=axes[task-1,col]
            for kind,color,marker,ls,title in styles:
                pts=curves[task][kind]
                ax.plot([p['step']/1000 for p in pts],[p[key]*factor for p in pts],color=color,
                    marker=marker,linestyle=ls,linewidth=2,markersize=4.5,
                    markerfacecolor=color if kind=='soft' else 'white',label=title)
            pts=native[task]
            ax.plot([p['step']/1000 for p in pts],[p[key]*factor for p in pts],color='#8B9096',
                linestyle=':',linewidth=1.4,label='QAM native context (45,001 updates)')
            ax.axvline(20,color='#ADB2B8',linestyle=':',linewidth=1)
            ax.set_title(f'Task{task}',loc='left',fontsize=12);ax.set_ylabel(label)
            ax.set_xlabel('Primitive online steps (thousands)');ax.set_xlim(-.5,50.5)
            if col==0:ax.set_ylim(0,103)
            ax.grid(axis='y',color='#E8EAED')
    fig.suptitle('Cube-double: critic target entropy ablation',x=.08,ha='left',fontsize=15)
    fig.text(.08,.93,'Seed 0 | Inherited Q | Frozen base | Scale 0.1 | Online-only replay | 50 episodes per point',fontsize=10)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.06),ncol=3,frameon=False)
    fig.text(.08,.036,'Both residual arms: 20k warmup, 50k online steps, 7,500 updates, batch 1024. Only target entropy changes.',fontsize=9)
    fig.text(.08,.014,'Actor entropy, automatic alpha and minimum aggregation are retained. Final table uses 100 paired episodes.',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.875,bottom=.155,hspace=.40,wspace=.24)
    for ext in ('png','pdf'):fig.savefig(OUT/f'learning_curves.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    specs=[('entropy_target_mean','Mean entropy contribution to target'),
        ('residual_q_difference_mean','Mean Q(residual action) - Q(base action)'),
        ('residual_gradient_norm_mean','Mean norm of dQ / du')]
    for task in (1,2):
        for col,(key,label) in enumerate(specs):
            ax=axes[task-1,col]
            for kind,color,marker,ls,title in styles:
                ps=probes[task][kind]
                ax.plot([p['step']/1000 for p in ps],[p[key] for p in ps],color=color,
                    marker=marker,linestyle=ls,linewidth=2,markerfacecolor=color if kind=='soft' else 'white',label=title)
            ax.set_title(f'Task{task}',loc='left',fontsize=12);ax.set_ylabel(label)
            ax.set_xlabel('Primitive online steps (thousands)');ax.set_xticks([20,30,40,50])
            if col<2:ax.axhline(0,color='#ADB2B8',linewidth=1,linestyle=':')
            ax.grid(axis='y',color='#E8EAED')
    fig.suptitle('Critic diagnostics on fixed warmup transitions',x=.07,ha='left',fontsize=15)
    fig.text(.07,.93,'Same 256 transitions and probe RNG within task | Four measured checkpoints | Unscaled residual u',fontsize=10)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.06),ncol=2,frameon=False)
    fig.text(.07,.036,'Entropy contribution for Plain TD is counterfactual: it is measured but never added to its training target.',fontsize=9)
    fig.text(.07,.014,'Q differences and gradients are critic estimates, not measured return improvements. No smoothing or seed uncertainty bands.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,top=.875,bottom=.155,hspace=.40,wspace=.30)
    for ext in ('png','pdf'):fig.savefig(OUT/f'critic_diagnostics.{ext}',dpi=180)
    plt.close(fig)


def write_report(rows,pairs,probe_rows,initial_diffs):
    lines=['# DAWN critic target entropy 消融','',
        'Task1 / Task2，seed 0，共享各任务 offline 500k checkpoint；继承 Q 与 target Q，冻结 base，scale=.1，online-only replay。两模式均为 20k warmup、50k primitive online steps、7,500 updates、batch 1024。',
        '只删除 critic target 的 entropy 项；actor 的 SAC entropy loss 和自动 alpha 调节保持不变。两处 Q 聚合均继续取 10 个 critic 的 minimum。', '',
        '| Task | Critic target | 最终成功率（100 eps） | 平均回报 | 最终 alpha |',
        '|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {r['td_target']} | {r['success_percent']:.0f}% | {r['return_mean']:.2f} | {r['alpha_final']:.6f} |")
    lines+=['','普通 TD − Soft TD 的配对差值：','']
    for p in pairs:lines.append(f"- Task{p['task']}：成功率 {p['hard_minus_soft_success_pp']:+.0f} pp；平均回报 {p['hard_minus_soft_return']:+.2f}。")
    lines+=['','历史 Soft TD 的复现情况（主对照仍使用本轮实测值）：','']
    for r in rows:
        if r['td_target']=='soft':lines.append(f"- Task{r['task']}：历史 {r['historical_soft_success_percent']:.0f}% / {r['historical_soft_return_mean']:.2f}，本轮 {r['success_percent']:.0f}% / {r['return_mean']:.2f}；最终 checkpoint 与历史是否精确一致：{r['exact_historical_soft_checkpoint']}。")
    lines+=['Task2 新旧 Soft 初始化、完整 warmup 和前 25k 训练指标一致；首次检测到的 replay 差异位于 index 5041 的 next observation，原始最大差约 5.55e-16，float32 后约 4.08e-17，随后轨迹和训练分叉。具体数值差异来源尚未定位，不能仅凭此记录断言完整因果链。证据见 runs/td_ablation/task2_soft_historical_reproduction.json。']
    lines+=['','固定 warmup batch 上，尚未更新时的 entropy target 项：','']
    for task in (1,2):
        p=next(p for p in probe_rows if p['task']==task and p['td_target']=='soft' and p['updates']==0)
        lines.append(f"- Task{task}：均值 {p['entropy_target_mean']:.4f}；平均绝对值 {p['entropy_target_abs_mean']:.4f}；chunk reward 平均绝对值 {p['reward_abs_mean']:.4f}。")
    lines+=['','Fairness / implementation：',
        '- 两任务新 Soft TD 入口的短程完整 checkpoint 与原入口精确一致，证明诊断未改变短程训练状态；hard 与 soft 全部初始状态及 warmup 配对核验。',
        '- 继承 QAM critic / target 参数，但按照原 DAWN 初始化新的 Adam 状态；没有改成继承 QAM optimizer。actor/critic lr=1e-4，gradient norm clip=50，target tau=.01，alpha 初始化 .01，目标残差熵 -25。',
        '- source_manifest 固定 46 个训练/诊断源文件。原始 DawnAgent.update、actor loss、alpha update 与 rollout/replay 循环保持不变；hard 仅替换 backup 函数。',
        '- 主对照使用新机器配对复跑；历史 Soft TD 和 native 仅作为复现/背景参考。native 有 45,001 updates，不是匹配更新预算的第三个消融组。',
        '- 固定 probe 来自首次 base-only warmup，独立 RNG 采样，不进入训练。完整 warmup replay 的动作、奖励、折扣与 base proposals 要求完全一致；观测允许已声明的模拟器数值容差，原哈希和差异全部保留。',
        '- MC 诊断复用原有评估轨迹，未增加训练或评估 episodes。Soft MC 包含未来残差熵，排除首个动作的熵。time-limit 截断的有限轨迹回报不含截断后的 bootstrap，不能当作无偏 continuing-value 真值；terminal 子集单列。不同模式的当前策略不同，不应把不同轨迹的 calibration MAE 当作同一分布上的排名。',
        '- Hard TD + actor entropy 是刻意设计的 target-only 消融，不是标准 SAC；也未恢复 QAM 的 mean−.5std target 聚合。',
        '- 该消融检验 target entropy 对当前设置的总体影响。即使普通 TD 更好，也不能单独证明收益全部来自继承 critic 时的目标切换不一致：它同时改变了整个 online 阶段的价值定义，尚未用随机初始化 Q 的交互对照分离这两种机制。',
        '- 曲线每点 50 episodes，最终主表 100 episodes，mean-residual 50-episode 结果单列。只有一个 training seed，不支持跨 seed 稳定性判断。',
        f'- 配对初始评估存在 {len(initial_diffs)} 条 episode 回报差异；成功率、reset 状态、模型/优化器/RNG 一致。具体原始记录见 initial_evaluation_differences.json。', '',
        '![Learning curves](learning_curves.png)','', '![Fixed probe](critic_diagnostics.png)','']
    (OUT/'summary.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
