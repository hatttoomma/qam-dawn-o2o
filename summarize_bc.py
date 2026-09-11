"""Validate all four BC transfers and preserve the six QAM-pretrained arms."""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize import load,digest,checked_eval
from summarize_five import write_json,write_csv

METHODS={'native':('BC → QAM native','native',45001),
    'dawn_online':('BC → DAWN / online-only','random',7500),
    'dawn_qam':('BC → DAWN / QAM replay','random_qam_replay',7500),
    'policy_decorator':('BC → Policy Decorator / QAM replay','policy_decorator_qam_replay',10500)}


def main(runs,out):
    runs,out=Path(runs),Path(out);root=runs.parent;out.mkdir(parents=True,exist_ok=True)
    assert load(runs/'BC_PRETRAIN_CHECKS_PASSED.json')['status']=='passed'
    assert load(runs/'BC_ONLINE_CHECKS_PASSED.json')['status']=='passed'
    for name in ['source_manifest.json','replay_source_manifest.json','policy_decorator_source_manifest.json',
                 'bc_pretrain_source_manifest.json','bc_online_source_manifest.json']:
        for p,h in load(runs/name).items():assert digest(root/p)==h,p
    sources=load(root/'results_policy_decorator/validation.json')['sources']
    for p,h in sources.items():assert digest(runs/p)==h,p
    offline=load(runs/'bc_offline/DONE.json');offline_cfg=load(runs/'bc_offline/config.json')
    assert offline['step']==500000 and not offline['critic_trained'] and not offline_cfg['critic_training']
    assert digest(runs/'bc_offline/final.pkl')==offline['checkpoint_sha256']
    assert offline['q_hash']==offline['target_q_hash']==load(runs/'random/initial_hashes.json')['critic']
    assert offline['bc_actor_hash']!=offline['initial_actor_hash']
    assert offline['checkpoint_sha256']!=load(runs/'offline/DONE.json')['checkpoint_sha256']
    original_data=load(runs/'offline/dataset_manifest.json');bc_data=load(runs/'bc_offline/dataset_manifest.json')
    for name,item in bc_data.items():assert item['sha256']==original_data[name]['sha256']
    cache=load(root/'data/bc_base_cache_seed0/manifest.json')
    assert cache['offline_sha256']==offline['checkpoint_sha256'] and cache['dataset_size']==1000000
    for p,h in cache['files'].items():assert digest(root/'data/bc_base_cache_seed0'/p)==h,p
    write_json(out/'bc_proposal_cache_manifest.json',cache)
    curves={};finals={};means={};rows=[];new_summary=[];deltas=[];paired={};configs={}
    for method,(label,old,updates) in METHODS.items():
        arm='bc_'+method;folder=runs/arm
        cfg=load(folder/'config.json');configs[arm]=cfg;done=load(folder/'DONE.json');transfer=load(folder/'BC_TRANSFER.json')
        assert cfg['seed']==0 and cfg['offline_steps']==500000 and done['steps']==cfg['online_steps']==50000
        assert cfg['pretraining']=='pure BC' and cfg['critic_initialization']==done['critic_initialization']=='random'
        assert cfg['offline_sha256']==transfer['shared_bc_checkpoint_sha256']==offline['checkpoint_sha256']
        assert transfer['critic_hash']==transfer['target_hash']==offline['q_hash']
        assert transfer['flow_hash']==offline['flow_hash'] and transfer['bc_actor_hash']==offline['bc_actor_hash']
        assert done['updates']==updates and digest(folder/'final.pkl')==done['checkpoint_sha256']
        if method!='native':
            init=load(folder/'initial_hashes.json')
            assert init['critic']==init['target']==offline['q_hash']
            assert done['final_flow_hash']==offline['flow_hash']
            assert done['final_critic_hash']!=init['critic'] and done['final_actor_hash']!=init['actor']
            assert init['actor']==load(runs/'random/initial_hashes.json')['actor']
        old_cfg=load(runs/old/'config.json')
        for k in ['seed','online_steps','offline_steps','dawn_batch','eval_episodes','final_episodes']:
            assert cfg[k]==old_cfg[k],(method,k)
        if method=='native':assert cfg['native_start']==old_cfg['native_start']==5000
        else:assert cfg['warmup']==old_cfg['warmup']==(8000 if method=='policy_decorator' else 20000)
        if method=='policy_decorator':
            assert cfg['policy_decorator_hyperparameters']==old_cfg['policy_decorator_hyperparameters']
        if method in ['dawn_qam','policy_decorator']:
            assert cfg['cache_manifest_sha256']==digest(root/'data/bc_base_cache_seed0/manifest.json')
            stats=done['replay_stats']
            assert stats['replay_size']==1050000 and stats['sampled_rows']==updates*1024
            assert .007<stats['sampled_online_fraction']<.048
        metrics=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        assert all(np.isfinite(v) for m in metrics for v in m.values() if isinstance(v,(int,float)))
        points=[checked_eval(p) for p in sorted(folder.glob('eval_*_050.json'))]
        assert len(points)==7
        for nominal,p in zip([0,5000,10000,20000,30000,40000,50000],points):assert nominal<=p['step']<=min(50000,nominal+4)
        curves[arm]=points
        final=checked_eval(folder/'eval_050000_100.json');finals[arm]=final
        assert final['records'][:50]==points[-1]['records']
        for p in folder.glob('eval_*.json'):
            x=checked_eval(p);rel=str(p.relative_to(runs));sources[rel]=digest(p)
            rows.append(dict(arm=arm,step=x['step'],episodes=x['episodes'],mean_residual=x['deterministic_residual'],
                success_percent=100*x['success'],return_mean=x['return_mean'],source=rel))
        if method!='native':
            means[arm]=checked_eval(folder/'eval_050000_050_mean_residual.json')
        auc=float(np.trapz([p['success'] for p in points],[p['step'] for p in points])/50000)
        s=dict(arm=arm,label=label,pretraining='BC',critic_initialization='随机',
            replay='online-only' if method=='dawn_online' else 'QAM replay',initial_success_50=points[0]['success'],
            final_success_50=points[-1]['success'],final_success_100=final['success'],final_return_100=final['return_mean'],
            success_auc_50=auc,online_updates=updates)
        new_summary.append(s)
        reference=checked_eval(runs/old/'eval_050000_100.json')
        assert [(x['reset_seed'],x['initial_hash']) for x in final['records']]==[(x['reset_seed'],x['initial_hash']) for x in reference['records']]
        a=np.array([x['success'] for x in reference['records']]);b=np.array([x['success'] for x in final['records']])
        delta=dict(method=method,old_arm=old,new_arm=arm,old_success_percent=100*a.mean(),bc_success_percent=100*b.mean(),
            difference_percentage_points=float(100*(b-a).mean()),return_difference=final['return_mean']-reference['return_mean'],
            interpretation='same random Q; pretraining replacement' if method.startswith('dawn') else 'critic initialization also changes; native optimizer also resets')
        deltas.append(delta)
        paired[method]=dict(both_success=int(((a==1)&(b==1)).sum()),bc_only=int(((a==0)&(b==1)).sum()),
            qam_only=int(((a==1)&(b==0)).sum()),neither_success=int(((a==0)&(b==0)).sum()))
    offline_evals=[checked_eval(p) for p in sorted((runs/'bc_offline').glob('eval_*_050.json'))]
    assert [x['step'] for x in offline_evals]==[0,100000,250000,500000]
    first=offline_evals[-1]['records']
    assert all(points[0]['records']==first for points in curves.values())
    for p in (runs/'bc_offline').glob('eval_*_050.json'):sources[str(p.relative_to(runs))]=digest(p)
    prefixes={};warmup={}
    for method in ['dawn_online','dawn_qam']:
        folder=runs/('bc_'+method);meta=load(folder/'warmup.json');warmup[method]=meta
        with (folder/'latest.pkl').open('rb') as f:prefixes[method]=pickle.load(f)['replay'][:meta['chunks']]
    assert warmup['dawn_online']==warmup['dawn_qam'] or (warmup['dawn_online']['steps']==warmup['dawn_qam']['steps'] and warmup['dawn_online']['chunks']==warmup['dawn_qam']['chunks'])
    warm_audit={}
    for k in prefixes['dawn_online'][0]:
        x=np.asarray([r[k] for r in prefixes['dawn_online']]);y=np.asarray([r[k] for r in prefixes['dawn_qam']])
        tolerance=1e-10 if k in ['observations','next_observations'] else 0
        np.testing.assert_allclose(x,y,atol=tolerance,rtol=0)
        warm_audit[k]=dict(max_absolute_difference=float(np.max(np.abs(x-y))),absolute_tolerance=tolerance)
    pdmean=checked_eval(runs/'bc_policy_decorator/eval_050000_100_mean_residual.json')
    assert pdmean['records'][:50]==means['bc_policy_decorator']['records']
    old_summary=load(root/'results_policy_decorator/summary.json')
    summaries=[dict(s,pretraining='QAM') for s in old_summary]+new_summary
    write_csv(out/'comparison_ten.csv',summaries);write_json(out/'summary.json',summaries)
    write_csv(out/'bc_evaluations.csv',rows);write_csv(out/'pretraining_effects.csv',deltas)
    write_json(out/'paired_final_episodes.json',paired);write_json(out/'bc_warmup_audit.json',warm_audit)
    write_json(out/'online_configs.json',configs);write_json(out/'offline_config.json',offline_cfg)
    write_json(out/'validation.json',dict(status='passed',sources=sources,report_source_sha256=digest(Path(__file__)),checks=[
        'all previous source and evaluation hashes unchanged','pure BC objective unit checks and four online smoke checks passed',
        'same offline dataset; final fixed 500k BC checkpoint','all four share identical untrained random critics and initial BC policy',
        'all four online schedules and losses match corresponding old arms; fresh transfer optimizers documented',
        'exact 50k budgets and update counts; finite metrics; final checkpoint hashes verified',
        'BC-specific proposal cache hashes and replay statistics checked','residual arms preserve frozen BC flows',
        'raw evaluation aggregates and final paired episode records verified','two DAWN warmup prefixes match within existing observation tolerance']))
    shutil.copyfile(root/'BC_PROTOCOL.md',out/'protocol.md')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    titles=['Native QAM online\n(critic initialization also changes)','DAWN / online-only replay\n(random Q in both)',
        'DAWN / QAM replay\n(random Q in both)','Policy Decorator / QAM replay\n(critic initialization also changes)']
    for ax,(method,(_,old,_)),title in zip(axes.flat,METHODS.items(),titles):
        for label,points,color,marker,style in [('QAM pretraining',[checked_eval(p) for p in sorted((runs/old).glob('eval_*_050.json'))],'#2463A6','o','-'),
            ('BC pretraining',curves['bc_'+method],'#B68420','^','--')]:
            ax.plot([p['step']/1000 for p in points],[100*p['success'] for p in points],color=color,marker=marker,linestyle=style,label=label,linewidth=2,markersize=5)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Primitive online steps (thousands)');ax.set_ylabel('Success rate (%)')
        ax.set_xlim(-.7,50.7);ax.set_ylim(-3,103);ax.set_xticks([0,10,20,30,40,50]);ax.grid(axis='y',color='#E8EAED')
    fig.suptitle('Cube-double task1: replacing QAM pretraining with BC',x=.075,ha='left',fontsize=16)
    fig.text(.075,.94,'Seed 0 | 500k offline updates | 50k online steps | 50 episodes per measured curve point',fontsize=10)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.072),ncol=2,frameon=False)
    fig.text(.075,.043,'All BC arms start with the same random critics. DAWN: 20k warmup; PD: 8k learning starts / 30k progressive exploration.',fontsize=8.5)
    fig.text(.075,.019,'Native QAM: 5k learning starts. No smoothing or seed confidence bands. Final 100-episode scores are in the table.',fontsize=8.5)
    fig.subplots_adjust(left=.075,right=.98,top=.88,bottom=.17,hspace=.38,wspace=.23)
    fig.savefig(out/'learning_curves_bc.png',dpi=180);fig.savefig(out/'learning_curves_bc.pdf');plt.close(fig)

    lines=['# BC 替换 QAM offline pretraining：四组对照','',
        '完成 seed 0，纯 flow-matching BC 500k offline 更新，再分别执行四组 50k primitive online steps。四组 critic 均随机初始化，旧六组结果保留。','',
        f"BC 500k checkpoint 的初始成功率为 **{100*offline_evals[-1]['success']:.0f}%（50 episodes）**；旧 QAM checkpoint 为 76%。四组 BC online 的起始评估原始记录与 BC checkpoint 一致。",'',
        '| Pretraining | 方法 | Critic | Replay | 最终成功率（100 eps） | 平均回报 | AUC / 50k（50 eps 曲线） |',
        '|---|---|---|---|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['pretraining']} | {s['label']} | {s['critic_initialization']} | {s['replay']} | {100*s['final_success_100']:.0f}% | {s['final_return_100']:.2f} | {100*s['success_auc_50']:.2f}% |")
    lines+=['','## 对应设置的变化','',
        '| Online 设置 | QAM pretraining | BC pretraining | 成功率差值 | 平均回报差值 |','|---|---:|---:|---:|---:|']
    for d in deltas:lines.append(f"| {d['method']} | {d['old_success_percent']:.0f}% | {d['bc_success_percent']:.0f}% | {d['difference_percentage_points']:+.0f} pp | {d['return_difference']:+.2f} |")
    lines+=['','DAWN 两组对照使用旧的 random-critic 结果，随机 critic 初值、online losses、replay 和训练日程均一致。原生 QAM 与 Policy Decorator 的旧组使用 pretrained critic，新组使用 random critic；原生 QAM 的 combined optimizer 也在 BC 迁移时重新初始化。因此后两组的差异包含这些迁移设置，不能全部归因于 actor 的预训练目标。','',
        '![BC pretraining comparison](learning_curves_bc.png)','',
        '主表沿用随机残差评估。均值残差诊断如下；QAM/BC base 自身仍然随机，评估中不应用 PD 的训练 gate。','',
        '| BC online 方法 | 均值残差成功率（50 eps） |','|---|---:|']
    for arm,x in means.items():lines.append(f"| {arm} | {100*x['success']:.0f}% |")
    lines += ['',f"BC → Policy Decorator 的均值残差最终 100-episode 成功率为 **{100*pdmean['success']:.0f}%**，平均回报 {pdmean['return_mean']:.2f}。",'',
        '## BC 预训练过程','', '| Offline updates | 成功率（50 eps） | 平均回报 |','|---:|---:|---:|']
    for x in offline_evals:lines.append(f"| {x['step']:,} | {100*x['success']:.0f}% | {x['return_mean']:.2f} |")
    lines+=['','固定采用最后的 500k checkpoint，没有按评估挑选模型。BC 使用原 QAM 的 4×512 GELU flow 网络，batch 256、Adam lr 3e-4、梯度裁剪 1、10 步 Euler 采样、action horizon 5；只优化带原始 valid mask 的 flow-matching BC loss。','',
        '四组 online 超参数及迁移协议见 [protocol.md](protocol.md)，实际配置见 online_configs.json。DAWN 保留 20k warmup / 7,500 updates；PD 保留 8k learning starts、30k progressive exploration、alpha0=1 / 10,500 updates；native 保留 5k learning starts / 45,001 updates。','',
        '所有正式训练结束，源文件、checkpoint、BC proposal cache、逐 episode 汇总及共享初始策略已校验。两组 DAWN 的 base-only warmup 动作/回报/proposals 完全一致，观测沿用原来的 1e-10 绝对容差。单个训练 seed 的结果用于流程与趋势观察。','']
    (out/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(results=new_summary,differences=deltas,validation='passed'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();main(args.runs,args.out)
