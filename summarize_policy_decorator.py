"""Validate and report the PD pilot alongside the completed five arms."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize import load,digest,checked_eval
from summarize_five import write_json,write_csv

ARM='policy_decorator_qam_replay'


def main(runs,out):
    runs,out=Path(runs),Path(out);root=runs.parent;folder=runs/ARM
    out.mkdir(parents=True,exist_ok=True)
    assert load(runs/'POLICY_DECORATOR_CHECKS_PASSED.json')['status']=='passed'
    for manifest in ['source_manifest.json','replay_source_manifest.json','policy_decorator_source_manifest.json']:
        for p,h in load(runs/manifest).items():assert digest(root/p)==h,p
    sources=load(root/'results_five/validation.json')['sources']
    for p,h in sources.items():assert digest(runs/p)==h,p
    offline=load(runs/'offline/DONE.json')
    cfg=load(folder/'config.json');initial=load(folder/'initial_hashes.json');done=load(folder/'DONE.json')
    assert cfg['offline_steps']==offline['step']==500000
    assert cfg['offline_sha256']==offline['checkpoint_sha256']
    assert done['steps']==cfg['online_steps']==50000 and done['updates']==10500
    assert cfg['seed']==0 and cfg['warmup']==8000 and cfg['prog_explore']==30000 and cfg['dawn_batch']==1024
    assert cfg['replay_mode']=='qam_uniform_offline_online_sequences' and cfg['critic_boundary_mask']
    assert initial['critic']==offline['q_hash'] and initial['target']==offline['target_q_hash']
    reference=load(runs/'warm_qam_replay/initial_hashes.json')
    for key in ['flow','critic','target','actor','critic_optimizer','actor_optimizer']:assert initial[key]==reference[key],key
    assert initial['alpha']==1.
    assert done['final_flow_hash']==initial['flow']==offline['flow_hash']
    assert done['final_critic_hash']!=initial['critic'] and done['final_actor_hash']!=initial['actor']
    assert digest(folder/'final.pkl')==done['checkpoint_sha256']
    cache=load(root/'data/qam_base_cache_seed0/manifest.json')
    assert cfg['cache_manifest_sha256']==digest(root/'data/qam_base_cache_seed0/manifest.json')
    assert cache['offline_sha256']==cfg['offline_sha256']
    for p,h in cache['files'].items():assert digest(root/'data/qam_base_cache_seed0'/p)==h,p
    stats=done['replay_stats']
    assert stats['replay_size']==1050000 and stats['sampled_rows']==10500*1024
    assert stats['offline_transitions']==1000000 and stats['online_transitions']==50000
    assert .007<stats['sampled_online_fraction']<.048
    warmup=load(folder/'warmup.json')
    assert 8000<=warmup['steps']<=8004 and warmup['critic_hash']==initial['critic'] and warmup['actor_hash']==initial['actor']
    metrics=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
    assert all(np.isfinite(v) for m in metrics for v in m.values() if isinstance(v,(float,int)))
    for m in metrics:
        if m['step']<=8000:assert m['updates']==0
        if m['step']>8004:assert m['updates']==int((m['step']-8000)*.25)
    post=[m for m in metrics if m['step']>=31000]
    assert all(m['residual_probability']==1 for m in post)
    assert len({m['chunks']-m['enabled_chunks'] for m in post})==1
    assert done['gate_counts']['random_enabled_chunks']>0 and done['gate_counts']['learned_enabled_chunks']>0

    files=sorted(folder.glob('eval_*_050.json'));curve=[checked_eval(p) for p in files]
    mean_files=sorted(folder.glob('eval_*_050_mean_residual.json'));means=[checked_eval(p) for p in mean_files]
    assert len(curve)==len(means)==7
    for expected,a,b in zip([0,5000,10000,20000,30000,40000,50000],curve,means):
        assert expected<=a['step']<=min(50000,expected+4) and b['step']==a['step']
        assert b['deterministic_residual'] and b['residual_enabled']
        if a['step']>0:assert a['residual_enabled'] and not a['deterministic_residual']
    final_path=folder/'eval_050000_100.json';mean_path=folder/'eval_050000_100_mean_residual.json'
    final,mean_final=checked_eval(final_path),checked_eval(mean_path)
    assert final['records'][:50]==curve[-1]['records']
    assert mean_final['records'][:50]==means[-1]['records']
    assert curve[0]['records']==checked_eval(runs/'native/eval_000000_050.json')['records']
    old_final=checked_eval(runs/'warm_qam_replay/eval_050000_100.json')
    assert [r['reset_seed'] for r in final['records']]==list(range(500000,500100))
    assert [(r['reset_seed'],r['initial_hash']) for r in final['records']]==[(r['reset_seed'],r['initial_hash']) for r in old_final['records']]
    a=np.array([r['success'] for r in old_final['records']]);b=np.array([r['success'] for r in final['records']])
    write_json(out/'paired_vs_dawn_qam_replay.json',dict(episodes=100,
        both_success=int(((a==1)&(b==1)).sum()),pd_only=int(((a==0)&(b==1)).sum()),
        dawn_only=int(((a==1)&(b==0)).sum()),neither_success=int(((a==0)&(b==0)).sum()),
        success_difference_percentage_points=float(100*(b-a).mean()),return_difference=final['return_mean']-old_final['return_mean']))
    summary=load(root/'results_five/summary.json')
    entry=dict(arm=ARM,label='Policy Decorator pretrained Q / QAM replay',critic_initialization='继承',replay='QAM replay',
        initial_success_50=curve[0]['success'],final_success_50=curve[-1]['success'],final_success_100=final['success'],
        final_return_100=final['return_mean'],success_auc_50=float(np.trapz([p['success'] for p in curve],[p['step'] for p in curve])/50000),online_updates=done['updates'])
    summary.append(entry)
    write_csv(out/'comparison_six.csv',summary);write_json(out/'summary.json',summary)
    rows=[]
    for path in files+mean_files+[final_path,mean_path]:
        x=checked_eval(path);relative=str(path.relative_to(runs));sources[relative]=digest(path)
        rows.append(dict(arm=ARM,step=x['step'],episodes=x['episodes'],mean_residual=x['deterministic_residual'],
            residual_enabled=x['residual_enabled'],success_percent=100*x['success'],return_mean=x['return_mean'],source=relative))
    write_csv(out/'policy_decorator_evaluations.csv',rows)
    write_json(out/'hyperparameters.json',cfg)
    write_json(out/'diagnostics.json',dict(replay=stats,gate_counts=done['gate_counts'],warmup=warmup,
        mean_residual_final_100={k:v for k,v in mean_final.items() if k!='records'},final_metrics=metrics[-1]))
    checks=['all previous five evaluations and source files unchanged','shared 500k checkpoint, exact current and target critic inheritance',
        'same actor initialization as DAWN; initial alpha 1; frozen QAM unchanged','exactly 50k primitive steps and 10500 updates',
        'no updates before 8k; progressive gate always on after 30k; uniform residual samples collected before learning',
        'QAM replay size, cached proposal hashes, sampling counts and finite training metrics checked',
        'raw evaluation aggregates recalculated; paired initial base reference identical',
        'both final 100-episode evaluations contain exact corresponding final 50 records',
        'final comparison uses the same 100 reset seeds and initial observation hashes as DAWN+QAM replay']
    write_json(out/'validation.json',dict(status='passed',checks=checks,sources=sources,report_source_sha256=digest(Path(__file__))))
    shutil.copyfile(root/'POLICY_DECORATOR_PROTOCOL.md',out/'protocol.md')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    for arm,label,color,marker in [('native','QAM native (edit=0)','#53575D','o'),
        ('warm_qam_replay','DAWN + QAM replay (inherited Q)','#2463A6','D'),
        (ARM,'Policy Decorator + QAM replay (inherited Q)','#B68420','^')]:
        pts=curve if arm==ARM else [checked_eval(p) for p in sorted((runs/arm).glob('eval_*_050.json'))]
        for ax,key,factor in [(axes[0],'success',100),(axes[1],'return_mean',1)]:
            ax.plot([p['step']/1000 for p in pts],[p[key]*factor for p in pts],label=label,color=color,marker=marker,linewidth=2,markersize=5)
    axes[0].set_ylabel('Success rate (%)');axes[0].set_ylim(-3,103)
    axes[1].set_ylabel('Mean episode return')
    for ax in axes:
        ax.set_xlabel('Primitive online steps (thousands)');ax.set_xlim(-.7,50.7);ax.set_xticks([0,10,20,30,40,50]);ax.grid(axis='y',color='#E8EAED')
    fig.suptitle('Cube-double task1: Policy Decorator online pilot',x=.075,ha='left',fontsize=16)
    fig.text(.075,.89,'Seed 0 | Shared 500k offline checkpoint | 50 episodes per curve point | Sampled residual',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.115),frameon=False,ncol=1,fontsize=9)
    fig.text(.075,.064,'PD: learning starts at 8k, progressive exploration reaches p=1 at 30k. DAWN: 20k base-only warmup.',fontsize=8.5)
    fig.text(.075,.032,'Step 0 is the shared base-only reference. No gate at later evaluations. Final 100-episode scores are reported separately.',fontsize=8.5)
    fig.subplots_adjust(left=.075,right=.98,top=.82,bottom=.32,wspace=.28)
    fig.savefig(out/'learning_curves_policy_decorator.png',dpi=180);fig.savefig(out/'learning_curves_policy_decorator.pdf');plt.close(fig)

    old=next(x for x in summary if x['arm']=='warm_qam_replay')
    delta=100*(entry['final_success_100']-old['final_success_100'])
    lines=['# Policy Decorator online 试跑：继承 QAM critic','',
        f"完成 seed 0、500k offline + 50k primitive online steps。Policy Decorator 主评估成功率为 **{100*final['success']:.0f}%（100 episodes）**，平均回报 **{final['return_mean']:.2f}**；相对同为继承 critic + QAM replay 的 DAWN 为 **{delta:+.0f} 个百分点**。",'',
        '这是 Policy Decorator 的 QAM/OGBench 适配；同时改变了探索方式、learning starts 和熵系数初值，不是单变量消融。全部超参数在训练前固定，仅一个训练 seed，不作方法优劣的统计结论。','',
        '| 方法 | Critic | Replay | 最终成功率（100 eps） | 平均回报 | 曲线 AUC / 50k | 更新次数 |',
        '|---|---|---|---:|---:|---:|---:|']
    names={'native':'QAM（edit=0）','warm':'QAM + DAWN','random':'QAM + DAWN','warm_qam_replay':'QAM + DAWN','random_qam_replay':'QAM + DAWN',ARM:'QAM + Policy Decorator'}
    for x in summary:
        lines.append(f"| {names[x['arm']]} | {x['critic_initialization']} | {x['replay']} | {100*x['final_success_100']:.0f}% | {x['final_return_100']:.2f} | {100*x['success_auc_50']:.2f}% | {x['online_updates']:,} |")
    lines+=['', 'AUC 是七个预设时间点、每点 50 episodes 的成功率梯形面积除以 50k；最终表格使用 100 episodes。',
        '', '![Policy Decorator learning curves](learning_curves_policy_decorator.png)','',
        'Policy Decorator 官方评估使用 residual 均值动作。为与前五组一致，上面的主结果继续使用 sampled residual；另报均值残差评估（frozen QAM 本身仍随机）：','',
        '| 实际 online steps | Sampled residual（50 eps） | Mean residual（50 eps） |','|---:|---:|---:|']
    for a,b in zip(curve,means):lines.append(f"| {a['step']:,} | {100*a['success']:.0f}% | {100*b['success']:.0f}% |")
    lines += ['',f"最终 mean-residual 评估为 **{100*mean_final['success']:.0f}%（100 episodes）**，平均回报 **{mean_final['return_mean']:.2f}**。step 0 的 sampled 列是所有组共享的 base-only 参照，mean 列是初始化后的完整 PD 策略。",'',
        '## 本次实际超参数','',
        (root/'POLICY_DECORATOR_PROTOCOL.md').read_text().split('## Actual settings\n\n')[1].split('\n## Explicit adaptations')[0],
        '## 与官方实现及 DAWN 的差别','',
        (root/'POLICY_DECORATOR_PROTOCOL.md').read_text().split('## Explicit adaptations and interpretation\n\n')[1],
        '', '## 执行校验','',
        f"完成 {done['updates']:,} 次更新，QAM replay 累计采样 {stats['sampled_rows']:,} 行；online 占 {100*stats['sampled_online_fraction']:.3f}%，跨 episode 被 mask 的序列占 {100*stats['masked_sequence_fraction']:.3f}%。",'',
        f"共 {done['gate_counts']['chunks']:,} 次 chunk 决策，启用残差 {done['gate_counts']['enabled_chunks']:,} 次，其中学习前随机残差 {done['gate_counts']['random_enabled_chunks']:,} 次，学习后策略残差 {done['gate_counts']['learned_enabled_chunks']:,} 次。实际学习前采集结束于 {warmup['steps']:,} primitive steps。",'',
        'critic 和 target 的初始权重哈希与离线 checkpoint 完全一致；QAM actor 全程冻结；训练指标均有限值；主评估和均值评估的 final-100 均包含对应 final-50 的完全相同原始记录。历史五组训练源文件和原始评估未改动。','',
        '参考：[官方代码固定版本](https://github.com/tongzhoumu/policy_decorator/blob/92ba9ba442587ae5989c286355ace2200a0537fb/online/pi_dec_diffusion_maniskill2.py)、[Policy Decorator 论文](https://arxiv.org/abs/2412.13630)。', '',
        '机器可读配置：hyperparameters.json；校验：validation.json；完整六组结果：comparison_six.csv。原始 JSON 和 checkpoint 保存在远程 runs/policy_decorator_qam_replay；轻量原始记录同时导出到本地。','']
    (out/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(result=entry,mean_residual_success_100=mean_final['success'],validation='passed'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();main(args.runs,args.out)
