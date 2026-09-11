"""Validate raw evaluations and export the single-seed scientific comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = {
    'native': ('QAM native (edit=0)', '#53575D', 'o', '-'),
    'warm': ('QAM + DAWN, pretrained Q', '#2463A6', 's', '-'),
    'random': ('QAM + DAWN, random Q', '#B68420', '^', '--'),
}


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_eval(path):
    x=load(path)
    records=x['records']
    assert len(records)==x['episodes'],path
    assert len({r['reset_seed'] for r in records})==len(records),path
    success=np.array([r['success'] for r in records])
    returns=np.array([r['return_'] for r in records])
    assert np.all(np.isin(success,[0.,1.])) and np.all(np.isfinite(returns)),path
    np.testing.assert_allclose(success.mean(),x['success'],atol=1e-12)
    np.testing.assert_allclose(returns.mean(),x['return_mean'],atol=1e-10)
    return x


def check_warmup(runs):
    """Audit actual replay, allowing only tiny simulator observation roundoff."""
    w=load(runs/'warm/warmup.json');c=load(runs/'random/warmup.json')
    assert w['steps']==c['steps'] and w['chunks']==c['chunks']
    assert w['actor_hash']==c['actor_hash']
    # These are trusted checkpoints produced by this experiment, not external pickle files.
    with (runs/'warm/latest.pkl').open('rb') as f:
        a=pickle.load(f)['replay'][:w['chunks']]
    with (runs/'random/latest.pkl').open('rb') as f:
        b=pickle.load(f)['replay'][:c['chunks']]
    assert len(a)==len(b)==w['chunks']
    fields={}
    for key in a[0]:
        x=np.asarray([r[key] for r in a]);y=np.asarray([r[key] for r in b])
        assert x.shape==y.shape
        exact=bool(np.array_equal(x,y))
        tolerance=1e-10 if key in ('observations','next_observations') else 0.
        np.testing.assert_allclose(x,y,atol=tolerance,rtol=0,err_msg=key)
        fields[key]=dict(exact=exact,max_absolute_difference=float(np.max(np.abs(x-y))),
                         unequal_values=int((x!=y).sum()),absolute_tolerance=tolerance)
    return dict(steps=w['steps'],chunks=w['chunks'],raw_hash_identical=w['replay_hash']==c['replay_hash'],
                warm_raw_hash=w['replay_hash'],random_raw_hash=c['replay_hash'],fields=fields,
                note='Observation tolerance was introduced after a raw-hash mismatch was investigated. '
                     'Actions, rewards, discounts, base actions and next base actions must remain bit-identical.')


def main(runs,out):
    runs,out=Path(runs),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    assert (runs/'CHECKS_PASSED.json').exists()
    source_manifest=load(runs/'source_manifest.json')
    for relative,expected in source_manifest.items():
        assert digest(runs.parent/relative)==expected,relative
    offline=load(runs/'offline/DONE.json')
    assert offline['step']==500000
    rows,curve_data,summary=[],{},[]
    provenance={}
    for arm,(label,color,marker,style) in ARMS.items():
        done=load(runs/arm/'DONE.json')
        cfg=load(runs/arm/'config.json')
        assert done['steps']==50000
        assert cfg['offline_sha256']==offline['checkpoint_sha256']
        files=sorted((runs/arm).glob('eval_*_050.json'))
        points=[checked_eval(p) for p in files]
        assert len(points)==7,(arm,len(points))
        assert points[0]['step']==0 and points[-1]['step']==50000
        for nominal,point in zip([0,5000,10000,20000,30000,40000,50000],points):
            assert nominal<=point['step']<=min(nominal+4,50000)
        curve_data[arm]=points
        final_path=runs/arm/'eval_050000_100.json'
        final=checked_eval(final_path)
        # Final 100-episode evaluation must contain the exact primary 50 episodes.
        assert final['records'][:50]==points[-1]['records'],arm
        auc=float(np.trapz([p['success'] for p in points],[p['step'] for p in points])/50000.)
        summary.append(dict(arm=arm,label=label,initial_success=points[0]['success'],
                            final_success_100=final['success'],final_return_100=final['return_mean'],
                            success_auc_50=auc,updates=done['updates'],train_seconds=done['elapsed_seconds']))
        for p,x in zip(files,points):
            provenance[str(p.relative_to(runs))]=digest(p)
            rows.append(dict(arm=arm,seed=0,evaluation_role='primary_curve',online_steps=x['step'],episodes=x['episodes'],
                             successes=sum(r['success'] for r in x['records']),
                             success_percent=100*x['success'],return_mean=x['return_mean'],
                             evaluation_seconds=x['duration_seconds'],source=str(p.relative_to(runs))))
        provenance[str(final_path.relative_to(runs))]=digest(final_path)
        rows.append(dict(arm=arm,seed=0,evaluation_role='final_100',online_steps=final['step'],episodes=final['episodes'],
                         successes=sum(r['success'] for r in final['records']),success_percent=100*final['success'],
                         return_mean=final['return_mean'],evaluation_seconds=final['duration_seconds'],
                         source=str(final_path.relative_to(runs))))
    initial_records=[curve_data[arm][0]['records'] for arm in ARMS]
    assert all(x==initial_records[0] for x in initial_records[1:])
    warm,cold=load(runs/'warm/initial_hashes.json'),load(runs/'random/initial_hashes.json')
    assert warm['actor']==cold['actor'] and warm['critic']!=cold['critic']
    assert warm['actor_optimizer']==cold['actor_optimizer']
    assert warm['critic_optimizer']==cold['critic_optimizer'] and warm['alpha']==cold['alpha']
    assert warm['critic']==offline['q_hash'] and warm['target']==offline['target_q_hash']
    assert cold['critic']==cold['target']
    warmup_audit=check_warmup(runs)
    (out/'warmup_comparison.json').write_text(json.dumps(warmup_audit,indent=2)+'\n')
    for arm in ['warm','random']:
        d=load(runs/arm/'DONE.json')
        assert d['updates']==7500 and d['final_flow_hash']==offline['flow_hash']
        assert d['final_critic_hash']!=d['initial_hashes']['critic']
        assert d['final_actor_hash']!=d['initial_hashes']['actor']
    native=load(runs/'native/DONE.json')
    assert native['updates']==45001 and native['final_flow_hash']!=native['initial_flow_hash']
    pairs={}
    for left,right in [('warm','random'),('warm','native'),('random','native')]:
        a=checked_eval(runs/left/'eval_050000_100.json')
        b=checked_eval(runs/right/'eval_050000_100.json')
        assert [r['reset_seed'] for r in a['records']]==[r['reset_seed'] for r in b['records']]
        sa=np.array([r['success'] for r in a['records']])
        sb=np.array([r['success'] for r in b['records']])
        pairs[left+'_vs_'+right]=dict(episodes=len(sa),both_success=int(((sa==1)&(sb==1)).sum()),
            left_only=int(((sa==1)&(sb==0)).sum()),right_only=int(((sa==0)&(sb==1)).sum()),
            neither_success=int(((sa==0)&(sb==0)).sum()),success_difference_pp=float(100*(sa-sb).mean()))
    secondary={arm:checked_eval(runs/arm/'eval_050000_050_mean_residual.json')['success']
               for arm in ['warm','random']}
    for arm in ['warm','random']:
        path=runs/arm/'eval_050000_050_mean_residual.json'
        point=checked_eval(path)
        provenance[str(path.relative_to(runs))]=digest(path)
        rows.append(dict(arm=arm,seed=0,evaluation_role='mean_residual_diagnostic',online_steps=point['step'],
                         episodes=point['episodes'],successes=sum(r['success'] for r in point['records']),
                         success_percent=100*point['success'],return_mean=point['return_mean'],
                         evaluation_seconds=point['duration_seconds'],source=str(path.relative_to(runs))))
    with (out/'evaluation_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'paired_final_episodes.json').write_text(json.dumps(pairs,indent=2)+'\n')
    (out/'secondary_evaluation.json').write_text(json.dumps(dict(
        description='Mean residual action; base QAM remains stochastic. 50 paired evaluation episodes.',
        success=secondary),indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps(dict(status='passed',checks=[
        'training source unchanged since suite launch','raw episode means recomputed',
        'all arms use same offline checkpoint','paired initial evaluation identical',
        'warm/random residual and optimizer initialization identical',
        'warmup actions/rewards/discounts/base proposals bit-identical; observations within 1e-10',
        'frozen flows unchanged','critic and residual updated','exact 500k offline / 50k online budgets',
        'final 100-episode evaluation contains primary 50 episodes'],
        warmup_raw_hash_identical=warmup_audit['raw_hash_identical'],
        warmup_observation_max_difference=max(warmup_audit['fields'][k]['max_absolute_difference']
                                             for k in ['observations','next_observations']),
        sources=provenance),indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#24282D','text.color':'#24282D',
                         'axes.edgecolor':'#9DA3AA','xtick.color':'#555B63','ytick.color':'#555B63'})
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
    for arm,(label,color,marker,style) in ARMS.items():
        pts=curve_data[arm]
        x=[p['step']/1000 for p in pts]
        for ax,key,multiplier in [(axes[0],'success',100),(axes[1],'return_mean',1)]:
            ax.plot(x,[p[key]*multiplier for p in pts],label=label,color=color,marker=marker,
                    linestyle=style,linewidth=2,markersize=5,markerfacecolor='white' if arm=='random' else color)
    axes[0].set_ylabel('Success rate (%)'); axes[0].set_ylim(-3,103)
    axes[1].set_ylabel('Mean episode return')
    for ax in axes:
        ax.set_xlabel('Online environment steps (thousands)')
        ax.set_xlim(-.7,50.7);ax.set_xticks([0,10,20,30,40,50])
        ax.grid(axis='y',color='#E8EAED',linewidth=.7)
        ax.axvline(20,color='#9299A1',linestyle=':',linewidth=1.2)
    fig.suptitle('Cube-double task1: offline-to-online comparison',x=.07,ha='left',fontsize=16)
    fig.text(.07,.88,'Seed 0 | Shared 500k-update QAM checkpoint | 50 evaluation episodes per point',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,fontsize=9)
    fig.text(.07,.015,'DAWN base-only warmup is included in the 50k budget; dotted line marks 20k. Markers are measured evaluations; no seed confidence bands.',fontsize=8.5)
    fig.subplots_adjust(left=.07,right=.98,top=.81,bottom=.25,wspace=.25)
    fig.savefig(out/'learning_curves.png',dpi=180)
    fig.savefig(out/'learning_curves.pdf')
    plt.close(fig)

    lines=['# Cube-double task1：seed 0 实验结果','',
           '三组共享 500,000 次更新的 QAM offline checkpoint；各自执行 50,000 个原始环境步。',
           '按用户指定保留官方 cube-double QAM-edit 配置 edit_scale=0，因此 baseline 实际为 QAM 原生 online continuation。',
           '', '| 方法 | 起点成功率（50 episodes） | 最终成功率（100 episodes） | 最终平均回报 | 成功率曲线面积 / 50k | Online 更新次数 |',
           '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['label']} | {100*s['initial_success']:.1f}% | {100*s['final_success_100']:.1f}% | {s['final_return_100']:.2f} | {100*s['success_auc_50']:.2f}% | {s['updates']} |")
    lines += ['', '![Learning curves](learning_curves.png)','',
              '曲线每个点为 50 个固定 reset seed 的评估 episode。AUC 使用七个预先指定评估点之间的梯形面积除以 50k，最终 100-episode 指标另外列出。',
              '本轮只有一个训练 seed，用于验证流程和观察趋势。评估 episode 数量不能替代多个独立训练 seed，因此不据此断言统计意义上的优劣。',
              '', '## 实际采用的设计', '',
              '| 设置 | QAM 原生 continuation | QAM + DAWN 两组 |',
              '|---|---|---|',
              '| Base policy | 继续更新 QAM | 冻结同一 QAM checkpoint |',
              '| Critic | 保留 QAM 及 optimizer/target | 10 个 QAM backbone；fresh optimizer |',
              '| TD target | 原生 QAM，无 entropy；mean − 0.5 std | SAC entropy bonus；minimum over all 10 |',
              '| Actor Q 聚合 | mean | minimum over all 10 |',
              '| Replay | offline + online 均匀采样 | online-only，包含 warmup 采集 |',
              '| 开始学习 | 第 5,000 个原始环境步 | 约 20,000 步 base-only warmup 后 |',
              '| UTD / batch size | 1 / 256 | 0.25 / 1024 |',
              '| Learning rate / target tau | 3e-4 / 0.005 | 1e-4 / 0.01 |',
              '| Action chunk | 5 步，完整执行 | 5 步，完整执行 |',
              '| Discount | 每原始步 0.99 | 每原始步 0.99 |',
              '',
              'DAWN residual 为 state-only 3×256 ReLU tanh-Gaussian，scale=0.1；alpha 初始值 0.01 并自动调节，target entropy=-25（未乘 scale 的完整 action chunk）。current/target critic 的初始化是两组 DAWN 的唯一初始设置差别；residual、optimizer 与温度初始值已逐项检查一致。',
              f"两组 warmup 均为 {warmup_audit['steps']:,} 个原始环境步、{warmup_audit['chunks']:,} 个 chunk。动作、回报、bootstrap discount、base/next-base action 完全一致；观测最大绝对差异为 {max(warmup_audit['fields'][k]['max_absolute_difference'] for k in ['observations','next_observations']):.3g}，因此原始 replay 哈希不相同。发现哈希差异后逐字段核查，观测改用绝对容差 1e-10、相对容差 0 验证，其余字段仍要求逐位一致；检查值和原始哈希见 warmup_comparison.json。训练代码与数据未作修改。",
              '为直接继承 offline 权重，DAWN 组保留 QAM 的 4×512 GELU + LayerNorm critic backbone；这是一项适配 QAM/OGBench 的 DAWN 实现，不能称为原论文实现的逐项复现。正式训练源码和依赖版本均已记录。',
              '', '## 辅助诊断', '',
              f"最终使用 residual 均值动作（base 仍随机采样，50 episodes）时，pretrained Q 成功率为 {100*secondary['warm']:.1f}%，random Q 为 {100*secondary['random']:.1f}%。此项只作诊断，主结果始终使用预先确定的 sampled-residual 评估。",
              'paired_final_episodes.json 保存最终 100 个配对 reset seed 上的共同成功、单边成功和共同失败计数；这些仍然来自同一个训练 seed。',
              '', '所有一致性检查已通过；原始数值见 evaluation_table.csv，文件校验记录见 validation.json。',
              '', '参考：[QAM](https://arxiv.org/abs/2601.14234) · [DAWN](https://arxiv.org/abs/2602.10539) · [QAM 官方代码](https://github.com/ColinQiyangLi/qam) · [DAWN 官方代码](https://github.com/Guozheng-Ma/DAWN)', '']
    (out/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runs',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args();main(args.runs,args.out)
