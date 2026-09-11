"""Validate task2's two original-budget arms and produce plots and tables."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'runs/task2'
OUT=ROOT/'results_task2'
ENV='cube-double-play-singletask-task2-v0'
sources={}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    sources[str(p.relative_to(ROOT))]=digest(p)
    return json.loads(p.read_text())
def checked(p):
    e=load(p)
    assert e['episodes']==len(e['records'])
    assert len({r['reset_seed'] for r in e['records']})==e['episodes']
    success=np.array([r['success'] for r in e['records']])
    returns=np.array([r['return_'] for r in e['records']])
    assert np.isin(success,[0,1]).all() and np.isfinite(returns).all()
    np.testing.assert_allclose(success.mean(),e['success'],atol=1e-12)
    np.testing.assert_allclose(returns.mean(),e['return_mean'],atol=1e-10)
    return e


def main():
    OUT.mkdir(exist_ok=True)
    manifest=load(RUNS/'source_manifest.json')
    for n,h in manifest.items():assert digest(ROOT/n)==h,n
    assert load(RUNS/'CHECKS_PASSED.json')['status']=='passed'
    audit=load(RUNS/'TASK_AUDIT_PASSED.json')
    assert audit['status']=='passed' and audit['different_reward_rows']>0
    offline=load(RUNS/'offline/DONE.json')
    offline_cfg=load(RUNS/'offline/config.json')
    task=load(RUNS/'offline/task_manifest.json')
    assert offline['step']==500000 and offline_cfg['environment']==ENV
    assert task['environment']==ENV and task['reward_task_id']==2
    assert task['rewards_hash']==audit['tasks'][1]['rewards_hash']
    assert task['masks_hash']==audit['tasks'][1]['masks_hash']
    assert task['rewards_hash']!=audit['tasks'][0]['rewards_hash']
    assert offline['checkpoint_sha256']!=load(ROOT/'runs/offline/DONE.json')['checkpoint_sha256']
    # Remote model files may be omitted from the portable evidence bundle.
    if (RUNS/'offline/final.pkl').exists():assert digest(RUNS/'offline/final.pkl')==offline['checkpoint_sha256']
    offline_evals=[checked(p) for p in sorted((RUNS/'offline').glob('eval_*_050.json'))]
    assert [e['step'] for e in offline_evals]==[0,100000,250000,500000]
    curves,finals,rows,eval_rows={},{},[],[]
    for arm in ('native','warm'):
        done=load(RUNS/arm/'DONE.json');cfg=load(RUNS/arm/'config.json')
        assert cfg['environment']==ENV and cfg['seed']==0
        assert cfg['offline_sha256']==offline['checkpoint_sha256']
        assert cfg['offline_steps']==500000 and cfg['online_steps']==done['steps']==50000
        assert load(RUNS/arm/'task_manifest.json')==task
        assert done['updates']==(45001 if arm=='native' else 7500)
        metrics_path=RUNS/arm/'metrics.jsonl'
        sources[str(metrics_path.relative_to(ROOT))]=digest(metrics_path)
        for line in metrics_path.read_text().splitlines():
            metric=json.loads(line)
            for value in metric.values():
                if isinstance(value,(int,float)):assert np.isfinite(value)
        points=[checked(p) for p in sorted((RUNS/arm).glob('eval_*_050.json'))]
        assert len(points)==7
        for nominal,e in zip([0,5000,10000,20000,30000,40000,50000],points):
            assert nominal<=e['step']<=min(nominal+4,50000)
        final=checked(RUNS/arm/'eval_050000_100.json')
        assert final['records'][:50]==points[-1]['records']
        curves[arm]=points;finals[arm]=final
        rows.append(dict(task='task2',method='QAM native' if arm=='native' else 'DAWN / pretrained Q / online-only',
            seed=0,offline_updates=500000,online_environment_steps=50000,online_updates=done['updates'],
            batch_size=256 if arm=='native' else 1024,initial_success_percent=100*points[0]['success'],
            final_episodes=100,final_success_percent=100*final['success'],final_return_mean=final['return_mean'],
            success_auc_percent=100*float(np.trapz([e['success'] for e in points],[e['step'] for e in points])/50000),
            mean_episode_length=float(np.mean([r['length'] for r in final['records']]))))
        for role,es in [('curve',points),('final100',[final])]:
            for e in es:eval_rows.append(dict(arm=arm,role=role,steps=e['step'],episodes=e['episodes'],
                success_percent=100*e['success'],return_mean=e['return_mean']))
    assert curves['native'][0]['records']==curves['warm'][0]['records']==offline_evals[-1]['records']
    warm=load(RUNS/'warm/DONE.json');init=load(RUNS/'warm/initial_hashes.json')
    assert init['critic']==offline['q_hash'] and init['target']==offline['target_q_hash']
    assert warm['final_flow_hash']==init['flow']==offline['flow_hash']
    assert warm['final_critic_hash']!=init['critic'] and warm['final_actor_hash']!=init['actor']
    native=load(RUNS/'native/DONE.json')
    assert native['initial_flow_hash']==offline['flow_hash']!=native['final_flow_hash']
    warmup=load(RUNS/'warm/warmup.json')
    assert 20000<=warmup['steps']<=20004
    assert warmup['critic_hash']==init['critic'] and warmup['actor_hash']==init['actor']
    secondary=checked(RUNS/'warm/eval_050000_050_mean_residual.json')
    assert secondary['deterministic_residual'] and secondary['episodes']==50
    a,b=finals['native']['records'],finals['warm']['records']
    assert [r['reset_seed'] for r in a]==[r['reset_seed'] for r in b]
    common=[(x,y) for x,y in zip(a,b) if x['success'] and y['success']]
    pair=dict(both_success=len(common),native_only=sum(bool(x['success']) and not bool(y['success']) for x,y in zip(a,b)),
        dawn_only=sum(bool(y['success']) and not bool(x['success']) for x,y in zip(a,b)),
        neither=sum(not x['success'] and not y['success'] for x,y in zip(a,b)),
        native_length_common_success=float(np.mean([x['length'] for x,y in common])) if common else None,
        dawn_length_common_success=float(np.mean([y['length'] for x,y in common])) if common else None)
    historical=[]
    for arm in ('native','warm'):
        e=checked(ROOT/'runs'/arm/'eval_050000_100.json')
        historical.append(dict(task='task1',method=rows[0 if arm=='native' else 1]['method'],
            online_environment_steps=50000,online_updates=45001 if arm=='native' else 7500,
            final_success_percent=100*e['success'],final_return_mean=e['return_mean']))
    for filename,data in [('comparison.csv',rows),('evaluations.csv',eval_rows),('task1_reference.csv',historical)]:
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'paired_final_episodes.json').write_text(json.dumps(pair,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.5))
    for arm,label,color,marker in [('native','QAM native (edit=0)','#53575D','o'),
                                   ('warm','DAWN / pretrained Q / online-only','#2463A6','s')]:
        pts=curves[arm]
        for ax,key,scale in [(axes[0],'success',100),(axes[1],'return_mean',1)]:
            ax.plot([e['step']/1000 for e in pts],[e[key]*scale for e in pts],label=label,
                    color=color,marker=marker,linewidth=2.4 if arm=='native' else 1.8,
                    markersize=6 if arm=='native' else 4,linestyle='-' if arm=='native' else '--')
    axes[0].set_ylim(0,103);axes[0].set_ylabel('Success rate (%)')
    axes[1].set_ylabel('Mean episode return')
    for ax in axes:
        ax.set_xlabel('Primitive online steps (thousands)');ax.set_xlim(-.5,50.5)
        ax.set_xticks([0,10,20,30,40,50]);ax.grid(axis='y',color='#E8EAED')
        ax.axvline(20,color='#ADB2B8',linestyle=':',linewidth=1)
    fig.suptitle('Cube-double task2: original 50k online budget',x=.075,ha='left',fontsize=15)
    fig.text(.075,.88,'Seed 0 | New shared 500k task2 offline checkpoint | Action chunk 5 | 50 evaluation episodes per curve point',fontsize=9)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False,fontsize=10)
    fig.text(.075,.045,'Dotted line: DAWN 20k base-only warmup ends. Native starts updating at 5k; DAWN keeps UTD 0.25 and batch 1024.',fontsize=9)
    fig.text(.075,.017,'Curves use sampled actions. The final table uses 100 paired episodes. One training seed; no extended online training.',fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,top=.79,bottom=.25,wspace=.24)
    fig.savefig(OUT/'learning_curves.png',dpi=180);fig.savefig(OUT/'learning_curves.pdf');plt.close(fig)
    lines=['# Task2: QAM native 与 DAWN pretrained Q','',
        '任务 `cube-double-play-singletask-task2-v0`；seed 0；新训练的共享 QAM offline 500k checkpoint；online 50k primitive steps；chunk=5。','',
        '| 方法 | 初始成功率（50 eps） | Online 更新 | 最终成功率（100 eps） | 平均回报 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {r['initial_success_percent']:.0f}% | {r['online_updates']:,} | {r['final_success_percent']:.0f}% | {r['final_return_mean']:.2f} |")
    lines+=['',f"DAWN 均值残差辅助评估：{100*secondary['success']:.0f}% / {secondary['return_mean']:.2f}（50 episodes；base 仍随机）。",'',
        'Native：继承完整 offline agent 与优化器，offline+online 均匀 replay，batch 256，UTD 1，5k learning starts。',
        'DAWN：继承 task2 critic/target，冻结 QAM flow，online-only replay，20k base-only warmup，batch 1024，UTD 0.25，SAC entropy TD，10 critic 全 minimum，residual scale 0.1。其余设置见 TASK2_PROTOCOL.md。','',
        '## Task1 的原始 50k 结果，仅供跨任务参考','',
        '| 方法 | Task1 成功率（100 eps） | Task1 平均回报 |','|---|---:|---:|']
    for r in historical:lines.append(f"| {r['method']} | {r['final_success_percent']:.0f}% | {r['final_return_mean']:.2f} |")
    lines+=['','没有复用 task1 policy/critic；底层 play 轨迹相同，但按 task2 重新标注奖励并从头预训练。',
        '同环境交互预算不等于同更新次数。单 seed 结果只支持流程验证和趋势比较。','',
        '![Task2 learning curves](learning_curves.png)','']
    (OUT/'summary.md').write_text('\n'.join(lines))
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',sources=sources,
        report_source_sha256=digest(Path(__file__)),checks=['task2 relabeling differs from task1',
        'new shared task2 offline checkpoint','training source hashes unchanged','finite training metrics',
        'raw episode means recomputed','paired initial records identical','final100 contains first50',
        'pretrained critic and target inherited','frozen flow unchanged','exact training budgets']),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
