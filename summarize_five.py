"""Validated five-arm report, retaining the original three experiment results."""
import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize import load, digest, checked_eval

ARMS={
    'native': ('QAM native (edit=0)', '原生 QAM', 'offline+online', '#53575D', 'o', '-'),
    'warm': ('DAWN pretrained Q / online only', '继承', 'online-only', '#2463A6', 's', '--'),
    'random': ('DAWN random Q / online only', '随机', 'online-only', '#B68420', '^', '--'),
    'warm_qam_replay': ('DAWN pretrained Q / QAM replay', '继承', 'QAM replay', '#2463A6', 'D', '-'),
    'random_qam_replay': ('DAWN random Q / QAM replay', '随机', 'QAM replay', '#B68420', 'P', '-'),
}


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def warmup_audit(runs):
    prefixes={};meta={}
    for arm in list(ARMS)[1:]:
        meta[arm]=load(runs/arm/'warmup.json')
        with (runs/arm/'latest.pkl').open('rb') as f:
            prefixes[arm]=pickle.load(f)['replay'][:meta[arm]['chunks']]
    audits={}
    for old,new in [('warm','warm_qam_replay'),('random','random_qam_replay'),
                    ('warm_qam_replay','random_qam_replay')]:
        a,b=meta[old],meta[new]
        assert a['steps']==b['steps']==20004 and a['chunks']==b['chunks']==4041
        assert a['actor_hash']==b['actor_hash']
        fields={}
        for key in prefixes[old][0]:
            x=np.asarray([r[key] for r in prefixes[old]])
            y=np.asarray([r[key] for r in prefixes[new]])
            tolerance=1e-10 if key in ('observations','next_observations') else 0.
            np.testing.assert_allclose(x,y,atol=tolerance,rtol=0,err_msg=f'{old}/{new}/{key}')
            fields[key]=dict(exact=bool(np.array_equal(x,y)),
                             max_absolute_difference=float(np.max(np.abs(x-y))),absolute_tolerance=tolerance)
        audits[old+'_vs_'+new]=dict(steps=a['steps'],chunks=a['chunks'],fields=fields,
            raw_hash_identical=a['replay_hash']==b['replay_hash'],old_raw_hash=a['replay_hash'],new_raw_hash=b['replay_hash'])
    return audits


def main(runs,out):
    runs,out=Path(runs),Path(out)
    root=runs.parent
    out.mkdir(parents=True,exist_ok=True)
    assert load(runs/'REPLAY_CHECKS_PASSED.json')['status']=='passed'
    for name in ['source_manifest.json','replay_source_manifest.json']:
        for p,h in load(runs/name).items():assert digest(root/p)==h,p
    for p,h in load(root/'results/validation.json')['sources'].items():
        assert digest(runs/p)==h,'Original evaluation changed: '+p
    offline=load(runs/'offline/DONE.json')
    assert offline['step']==500000
    cache=load(root/'data/qam_base_cache_seed0/manifest.json')
    assert cache['offline_sha256']==offline['checkpoint_sha256'] and cache['dataset_size']==1000000
    for p,h in cache['files'].items():assert digest(root/'data/qam_base_cache_seed0'/p)==h,p
    write_json(out/'offline_proposal_cache_manifest.json',cache)
    curves={};summaries=[];rows=[];sources={};finals={};initial={};secondary={}
    for arm,(label,init,replay,*_) in ARMS.items():
        folder=runs/arm;cfg=load(folder/'config.json');done=load(folder/'DONE.json')
        assert cfg['seed']==0 and cfg['online_steps']==done['steps']==50000
        assert cfg['offline_steps']==500000 and cfg['offline_sha256']==offline['checkpoint_sha256']
        files=sorted(folder.glob('eval_*_050.json'))
        points=[checked_eval(p) for p in files]
        assert len(points)==7
        for nominal,p in zip([0,5000,10000,20000,30000,40000,50000],points):
            assert nominal<=p['step']<=min(50000,nominal+4)
        curves[arm]=points
        final_path=folder/'eval_050000_100.json'
        final=checked_eval(final_path);finals[arm]=final
        assert final['records'][:50]==points[-1]['records'],arm
        if arm!='native':
            initial[arm]=load(folder/'initial_hashes.json')
            assert done['updates']==7500 and done['final_flow_hash']==offline['flow_hash']
            assert done['final_critic_hash']!=initial[arm]['critic']
            assert done['final_actor_hash']!=initial[arm]['actor']
            mean_path=folder/'eval_050000_050_mean_residual.json'
            secondary[arm]=checked_eval(mean_path)
        else:
            assert done['updates']==45001
        auc=float(np.trapz([p['success'] for p in points],[p['step'] for p in points])/50000)
        summaries.append(dict(arm=arm,label=label,critic_initialization=init,replay=replay,
            initial_success_50=points[0]['success'],final_success_50=points[-1]['success'],
            final_success_100=final['success'],final_return_100=final['return_mean'],
            success_auc_50=auc,online_updates=done['updates']))
        entries=list(zip(files,points,['primary_curve']*7))+[(final_path,final,'final_100')]
        if arm!='native':entries.append((mean_path,secondary[arm],'mean_residual_diagnostic'))
        for path,p,role in entries:
            rel=str(path.relative_to(runs));sources[rel]=digest(path)
            rows.append(dict(arm=arm,seed=0,evaluation_role=role,online_steps=p['step'],episodes=p['episodes'],
                successes=sum(x['success'] for x in p['records']),success_percent=100*p['success'],
                return_mean=p['return_mean'],source=rel))
    first=curves['native'][0]['records']
    assert all(curves[a][0]['records']==first for a in ARMS)
    replay_stats={}
    for kind in ['warm','random']:
        new=kind+'_qam_replay'
        assert initial[kind]==initial[new],new
        old_cfg=load(runs/kind/'config.json');cfg=load(runs/new/'config.json')
        for key in ['seed','offline_steps','online_steps','warmup','dawn_batch','eval_episodes','final_episodes']:
            assert cfg[key]==old_cfg[key],key
        assert cfg['warmup']==20000 and cfg['dawn_batch']==1024
        assert cfg['replay_mode']=='qam_uniform_offline_online_sequences' and cfg['critic_boundary_mask']
        assert cfg['cache_manifest_sha256']==digest(root/'data/qam_base_cache_seed0/manifest.json')
        stats=load(runs/new/'DONE.json')['replay_stats'];replay_stats[new]=stats
        assert stats['replay_size']==1050000 and stats['offline_transitions']==1000000
        assert stats['online_transitions']==50000 and stats['sampled_rows']==7500*1024
        assert .019<stats['sampled_online_fraction']<.048
    assert initial['warm']['critic']==offline['q_hash'] and initial['warm']['target']==offline['target_q_hash']
    assert initial['random']['critic']==initial['random']['target']!=offline['q_hash']
    audit=warmup_audit(runs)
    effects=[];paired={}
    for kind in ['warm','random']:
        new=kind+'_qam_replay'
        a,b=finals[kind]['records'],finals[new]['records']
        assert [r['reset_seed'] for r in a]==[r['reset_seed'] for r in b]
        old_success=np.array([r['success'] for r in a]);new_success=np.array([r['success'] for r in b])
        effects.append(dict(critic_initialization=ARMS[kind][1],old_online_only_success_percent=100*old_success.mean(),
            qam_replay_success_percent=100*new_success.mean(),difference_percentage_points=100*(new_success-old_success).mean(),
            mean_return_difference=finals[new]['return_mean']-finals[kind]['return_mean']))
        paired[kind]=dict(episodes=100,both_success=int(((old_success==1)&(new_success==1)).sum()),
            new_only=int(((old_success==0)&(new_success==1)).sum()),old_only=int(((old_success==1)&(new_success==0)).sum()),
            neither_success=int(((old_success==0)&(new_success==0)).sum()))
    assert len(rows)==44
    write_csv(out/'evaluation_table.csv',rows);write_csv(out/'comparison_five.csv',summaries)
    write_csv(out/'replay_effects.csv',effects)
    write_json(out/'summary.json',summaries);write_json(out/'replay_effects.json',effects)
    write_json(out/'warmup_comparison.json',audit);write_json(out/'paired_final_episodes.json',paired)
    write_json(out/'replay_sampling.json',replay_stats)
    write_json(out/'validation.json',dict(status='passed',sources=sources,checks=[
        'original evaluation files unchanged','original and new training source manifests unchanged',
        'one shared 500k checkpoint; all five runs have exactly 50k primitive steps',
        'new DAWN initialization and schedule equal corresponding original DAWN',
        'all five paired initial evaluations identical','final 100 episodes contain exact primary final 50',
        'frozen QAM unchanged; residual and critic updated','shared offline proposal cache hashes verified',
        'QAM buffer contains 1m offline + 50k online transitions; empirical sampling fraction checked',
        'warmup data match: exact actions/rewards/discounts/proposals; observations within 1e-10'],
        report_source_sha256=digest(Path(__file__))))

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'text.color':'#24282D','axes.labelcolor':'#24282D',
        'axes.edgecolor':'#9DA3AA','xtick.color':'#555B63','ytick.color':'#555B63'})
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.8))
    for arm,(label,init,replay,color,marker,style) in ARMS.items():
        pts=curves[arm]
        for ax,key,factor in [(axes[0],'success',100),(axes[1],'return_mean',1)]:
            ax.plot([p['step']/1000 for p in pts],[factor*p[key] for p in pts],label=label,
                color=color,marker=marker,linestyle=style,linewidth=2,markersize=5,
                markerfacecolor='white' if arm in ['warm','random'] else color)
    axes[0].set_ylabel('Success rate (%)');axes[0].set_ylim(-3,103)
    axes[1].set_ylabel('Mean episode return')
    for ax in axes:
        ax.set_xlabel('Online environment steps (thousands)');ax.set_xlim(-.7,50.7)
        ax.set_xticks([0,10,20,30,40,50]);ax.grid(axis='y',color='#E8EAED',linewidth=.7)
        ax.axvline(20,color='#9299A1',linestyle=':',linewidth=1.2)
    fig.suptitle('Cube-double task1: replay comparison across five methods',x=.07,ha='left',fontsize=16)
    fig.text(.07,.89,'Seed 0 | Shared 500k offline updates | 50 evaluation episodes per curve point',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.067),ncol=3,frameon=False,fontsize=9)
    fig.text(.07,.038,'All DAWN arms include 20k base-only warmup in the 50k budget. Dashed/open: online-only replay; solid/filled: QAM replay.',fontsize=8.5)
    fig.text(.07,.016,'Markers are measured evaluations; no seed confidence bands. Final 100-episode scores are reported separately in the table.',fontsize=8.5)
    fig.subplots_adjust(left=.07,right=.98,top=.82,bottom=.28,wspace=.25)
    fig.savefig(out/'learning_curves_five.png',dpi=180);fig.savefig(out/'learning_curves_five.pdf');plt.close(fig)

    lines=['# Cube-double task1：五组 replay 对比','',
        'seed 0；共享同一个 500k offline QAM checkpoint；每组 50k 个原始环境步。保留原三组结果，新增两组 QAM replay 消融。','',
        '| 方法 | Critic 初始化 | Replay | 终点成功率（50 eps） | 最终成功率（100 eps） | 最终平均回报 | 成功率 AUC / 50k |',
        '|---|---|---|---:|---:|---:|---:|']
    for s in summaries:
        method='QAM（edit=0）' if s['arm']=='native' else 'QAM + DAWN'
        lines.append(f"| {method} | {s['critic_initialization']} | {s['replay']} | {100*s['final_success_50']:.1f}% | {100*s['final_success_100']:.1f}% | {s['final_return_100']:.2f} | {100*s['success_auc_50']:.2f}% |")
    lines+=['','## 切换 replay 的影响','',
        '| Critic 初始化 | 原 online-only | QAM replay | 成功率变化 | 平均回报变化 |',
        '|---|---:|---:|---:|---:|']
    for e in effects:
        lines.append(f"| {e['critic_initialization']} | {e['old_online_only_success_percent']:.1f}% | {e['qam_replay_success_percent']:.1f}% | {e['difference_percentage_points']:+.1f} pp | {e['mean_return_difference']:+.2f} |")
    by_arm={s['arm']:s for s in summaries}
    lines+=['',f"随机 critic + QAM replay 在 30k 时成功率为 {100*curves['random_qam_replay'][4]['success']:.0f}%，随后恢复；其 AUC 为 {100*by_arm['random_qam_replay']['success_auc_50']:.2f}%，低于原 online-only 的 {100*by_arm['random']['success_auc_50']:.2f}%。本轮终点提高伴随明显的中途退化，不能仅用最终成功率概括学习过程。"]
    lines+=['','上述变化来自相同 100 个 reset seed 的最终评估，不能代替多个独立训练 seed 的统计验证。','',
        '![Five-method learning curves](learning_curves_five.png)','',
        '曲线每点评估 50 episodes，七个预设时间点之间用直线连接；AUC 为梯形面积除以 50k。主表同时列出终点 50-episode 与额外 100-episode 结果，避免混淆。','',
        '## 本次保持与改变的设置','',
        '四组 DAWN 都保留：frozen QAM、state-only 3×256 residual actor、scale=0.1、10 个 QAM critic backbone、SAC entropy TD、target/actor Q 均对全部 10 个 critic 取 minimum、batch=1024、UTD=0.25、lr=1e-4、tau=0.01、alpha 初始 0.01 并自动调整，以及约 20k base-only warmup。warmup 包含在 50k 总预算中，各组恰好 7,500 次 online 更新。','',
        '新增 replay 使用 QAM 的原始 transition buffer 与 sequence sampler：1,000,000 条 offline transitions 加全部新增 online transitions，均匀抽取 5 步滑动序列。终点 online 数据占比约 4.76%，并非固定 50:50。与原 DAWN 只采样实际执行的 decision chunks 相比，这同时改变 replay 的数据来源与序列采样方式。','',
        '跨 episode 的序列沿用 QAM valid mask 屏蔽 critic MSE；actor/温度更新保持 DAWN。有效 batch 上与原 DAWN 更新的数值对照通过。offline 数据的 frozen base/next-base proposals 预计算并缓存，两个新组共享相同缓存；online decision 起止状态沿用实际采集 proposal，中间状态按固定独立 key 补齐缓存。','']
    for arm,stats in replay_stats.items():
        lines.append(f"- {ARMS[arm][0]}：累计抽样 {stats['sampled_rows']:,} 行，online 占 {100*stats['sampled_online_fraction']:.3f}%，跨 episode mask 占 {100*stats['masked_sequence_fraction']:.3f}%。")
    max_roundoff=max(f['max_absolute_difference'] for a in audit.values() for f in a['fields'].values())
    lines+=['','## 校验与辅助评估','',
        f'新旧组初始化参数、训练预算和 frozen QAM 均已核对。四组 DAWN warmup 均为 20,004 步；动作、回报、discount 和 base proposals 逐位一致。观测最大差异 {max_roundoff:.3g}，在原实验已采用的 1e-10 绝对容差内；原始哈希与逐字段比较保存在 warmup_comparison.json。','',
        '| DAWN 组 | residual 均值动作成功率（50 eps；base 仍随机） |','|---|---:|']
    for arm,p in secondary.items():lines.append(f"| {ARMS[arm][0]} | {100*p['success']:.1f}% |")
    lines+=['','均值动作评估仅作辅助诊断；主结果继续使用 sampled residual。所有原始评估聚合重新计算，文件哈希及检查见 validation.json。','',
        '完整补充协议见 [REPLAY_ABLATION.md](../REPLAY_ABLATION.md)，原始三组报告见 [原报告](../results/summary.md)。','']
    (out/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(summary=summaries,effects=effects),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.runs,a.out)
