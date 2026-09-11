"""Validate raw experiment evidence and compare ordinary-TD batch sizes."""
import json
from pathlib import Path
import pickle
import numpy as np
import summarize_task2_scale as evidence
from summarize_td_ablation import check_mc, check_probe, arrays
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/td_ablation/batch_comparison'
OUT = ROOT / 'results_td_batch_ablation'
load, checked, track, digest = evidence.load, evidence.checked, evidence.track, evidence.digest


def main():
    OUT.mkdir(exist_ok=True)
    for n, h in load(RUNS/'source_manifest.json').items(): assert digest(ROOT/n) == h, n
    assert load(RUNS/'CHECKS_PASSED.json')['status'] == 'passed'
    assert load(RUNS/'runtime_setup.json')['probe_exit_code'] == 0
    audit = load(RUNS/'warmup_audit/audit.json'); assert audit['status'] == 'passed'
    for v in audit['arms'].values():
        for pk, hk in [('snapshot','snapshot_sha256'), ('historical_snapshot','historical_snapshot_sha256')]:
            if pk in v:
                track(ROOT/v[pk]); assert digest(ROOT/v[pk]) == v[hk]
    machine = load(RUNS/'machine.json')
    rows, evaluations, episodes, pairs, initial_diffs, endpoint_diffs, probes, mc_rows, mc_episodes = [],[],[],[],[],[],[],[],[]
    curves = {}
    for task in (1,2):
        reference = ROOT / ('runs' if task == 1 else 'runs/task2')
        offline = load(reference/'offline/DONE.json')
        assert offline['step'] == 500000
        original = ROOT / f'runs/td_ablation/task{task}_hard'
        old_cfg = load(original/'config.json'); old_final = checked(original/'eval_050000_100.json')
        old_done = load(original/'DONE.json'); old_actual = load(original/'actual_agent_config.json')
        native = checked(reference/'native/eval_050000_100.json')
        base = checked(reference/'offline/eval_500000_050.json')
        previous_cfg = previous_init = previous_actual = previous_fixed = None
        finals = {}; curves[task] = {}
        for batch in (1024,256):
            folder = RUNS / f'task{task}_b{batch}'
            cfg = load(folder/'config.json'); selected = load(folder/'td_config.json')
            actual = load(folder/'actual_agent_config.json'); init = load(folder/'initial_hashes.json')
            done = load(folder/'DONE.json'); checks = load(folder/'CHECKS_PASSED.json')
            exclude = ('out','dawn_batch')
            assert {k:v for k,v in cfg.items() if k not in exclude} == {k:v for k,v in old_cfg.items() if k not in exclude}
            assert cfg['dawn_batch'] == selected['dawn_batch'] == batch
            assert cfg['offline_sha256'] == selected['offline_sha256'] == offline['checkpoint_sha256']
            assert cfg['online_steps'] == done['steps'] == 50000 and done['updates'] == 7500
            assert cfg['warmup'] == 20000 and cfg['td_target'] == 'hard' and cfg['seed'] == 0
            assert actual['residual_scale'] == .1 and actual['target_tau'] == .01 and actual['action_dim'] == 25
            assert actual['actor_entropy_enabled'] and actual['automatic_alpha_enabled']
            assert actual['target_aggregation'] == actual['actor_Q_aggregation'] == 'minimum'
            assert actual['critic_ensemble'] == 10 and actual['inherited_Q_and_target']
            assert checks['status'] == 'passed' and load(folder/'backup_math.json')['status'] == 'passed'
            assert init['critic'] == offline['q_hash'] and init['target'] == offline['target_q_hash']
            assert init['flow'] == done['final_flow_hash'] == offline['flow_hash']
            assert done['initial_hashes'] == init
            assert done['final_critic_hash'] != init['critic'] and done['final_actor_hash'] != init['actor']
            assert load(folder/'task_manifest.json') == load(original/'task_manifest.json')
            warm = load(folder/'warmup.json'); old_warm = load(original/'warmup.json')
            assert warm['steps'] == old_warm['steps'] and 20000 <= warm['steps'] <= 20004
            assert warm['requested_steps'] == 20000 and warm['chunks'] == old_warm['chunks']
            if (folder/'final.pkl').exists():
                assert digest(folder/'final.pkl') == done['checkpoint_sha256']
                with (folder/'final.pkl').open('rb') as f: cp = pickle.load(f)
                assert cp['step'] == 50000 and cp['updates'] == int(cp['agent']['updates']) == 7500
                assert int(cp['agent']['actor']['step']) == int(cp['agent']['critic']['step']) == 7500
                del cp
            fixed = arrays(folder/'fixed_probe.npz'); fm = load(folder/'fixed_probe.json')
            assert fm['samples'] == 256 and fm['sha256'] == digest(folder/'fixed_probe.npz')
            if previous_cfg is None:
                previous_cfg, previous_init, previous_actual, previous_fixed = cfg, init, actual, fixed
            else:
                assert {k:v for k,v in cfg.items() if k not in exclude} == {k:v for k,v in previous_cfg.items() if k not in exclude}
                assert init == previous_init and actual == previous_actual
                for k,a in fixed.items():
                    if k in ('observations','next_observations'): np.testing.assert_allclose(a,previous_fixed[k],rtol=0,atol=1e-12)
                    else: np.testing.assert_array_equal(a,previous_fixed[k])
            pts = [checked(p) for p in sorted(folder.glob('eval_*_050.json'))]; assert len(pts) == 7
            for nominal,e in zip([0,5000,10000,20000,30000,40000,50000],pts):
                assert nominal <= e['step'] <= min(nominal+4,50000)
                assert not e['deterministic_residual']
                assert [r['initial_hash'] for r in e['records']] == [r['initial_hash'] for r in base['records']]
                mc = check_mc(folder,e,'hard')
                mc_rows.append(dict(task=task,batch_size=batch,**{k:v for k,v in mc.items() if k!='records'}))
                mc_episodes.extend(dict(task=task,batch_size=batch,step=e['step'],**r) for r in mc['records'])
            final = checked(folder/'eval_050000_100.json')
            mean = checked(folder/'eval_050000_050_mean_residual.json')
            endpoint_equal = final['records'][:50] == pts[-1]['records']
            state_check = load(folder/'evaluation_state_check.json')
            assert state_check['agent_unchanged_during_final_evaluations']
            assert state_check['before_eval_hash'] == state_check['after_eval_hash']
            for x,y in zip(pts[-1]['records'],final['records'][:50]):
                for k in ('episode','reset_seed','initial_hash'): assert x[k] == y[k]
                if x != y: endpoint_diffs.append(dict(task=task,batch_size=batch,episode=x['episode'],curve50=x,final100=y))
            assert not final['deterministic_residual'] and mean['deterministic_residual']
            assert final['residual_enabled'] and mean['residual_enabled']
            assert [r['initial_hash'] for r in final['records']] == [r['initial_hash'] for r in old_final['records']]
            metrics_path = folder/'metrics.jsonl'; track(metrics_path)
            metrics = [json.loads(s) for s in metrics_path.read_text().splitlines()]
            assert metrics[-1]['step'] == 50000 and metrics[-1]['updates'] == 7500
            for m in metrics:
                assert all(np.isfinite(v) for v in m.values() if isinstance(v,(int,float)))
                if m['step'] > warm['steps']: assert m['updates'] == int((m['step']-20000)*.25)
            ps = [check_probe(folder,p,fixed) for p in sorted(folder.glob('probe_*.json'))]
            assert len(ps) == 4 and ps[0]['updates'] == 0 and ps[-1]['updates'] == 7500
            probes.extend(dict(task=task,batch_size=batch,**p) for p in ps)
            rows.append(dict(task=task,batch_size=batch,td_target='hard',seed=0,offline_updates=500000,
                online_steps=50000,online_updates=7500,actual_warmup_steps=warm['steps'],sampled_training_rows=7500*batch,
                replay='online_only',residual_scale=.1,actor_entropy=True,final_episodes=100,
                success_percent=100*final['success'],return_mean=final['return_mean'],
                initial50_success_percent=100*pts[0]['success'],curve_final50_success_percent=100*pts[-1]['success'],
                repeated_final_first50_exact=endpoint_equal,final100_first50_success_percent=2*sum(r['success'] for r in final['records'][:50]),
                mean_residual50_success_percent=100*mean['success'],mean_residual50_return_mean=mean['return_mean'],
                success_auc_percent=float(np.trapz([p['success'] for p in pts],[p['step'] for p in pts])/500),
                alpha_final=ps[-1]['alpha'],historical1024_success_percent=100*old_final['success'],
                historical1024_return_mean=old_final['return_mean'],native_context_success_percent=100*native['success'],
                exact_historical_initial_agent=actual['full_initial_agent_hash']==old_actual['full_initial_agent_hash'],
                exact_historical_final_checkpoint=done['checkpoint_sha256']==old_done['checkpoint_sha256'] if batch==1024 else None))
            curves[task][batch] = pts; finals[batch] = final
            for role, es in [('curve50',pts),('final100',[final]),('mean_residual50',[mean])]:
                for e in es:
                    evaluations.append(dict(task=task,batch_size=batch,role=role,step=e['step'],episodes=e['episodes'],
                        success_percent=100*e['success'],return_mean=e['return_mean']))
                    episodes.extend(dict(task=task,batch_size=batch,role=role,step=e['step'],**r) for r in e['records'])
        a,b = finals[256]['records'],finals[1024]['records']
        pairs.append(dict(task=task,episodes=100,
            both_success=sum(bool(x['success']) and bool(y['success']) for x,y in zip(a,b)),
            batch256_only_success=sum(bool(x['success']) and not bool(y['success']) for x,y in zip(a,b)),
            batch1024_only_success=sum(not bool(x['success']) and bool(y['success']) for x,y in zip(a,b)),
            neither_success=sum(not bool(x['success']) and not bool(y['success']) for x,y in zip(a,b)),
            batch256_minus1024_success_pp=100*(finals[256]['success']-finals[1024]['success']),
            batch256_minus1024_return=finals[256]['return_mean']-finals[1024]['return_mean']))
        for x,y in zip(curves[task][256][0]['records'],curves[task][1024][0]['records']):
            assert x['initial_hash'] == y['initial_hash'] and x['reset_seed'] == y['reset_seed']
            if x != y: initial_diffs.append(dict(task=task,episode=x['episode'],batch256=x,batch1024=y))
    for name,data in [('comparison.csv',rows),('evaluations.csv',evaluations),('episodes.csv',episodes),
        ('paired_outcomes.csv',pairs),('fixed_probe.csv',probes),('mc_summary.csv',mc_rows),('mc_episodes.csv',mc_episodes)]:
        evidence.write_csv(OUT/name,data)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'initial_evaluation_differences.json').write_text(json.dumps(initial_diffs,indent=2)+'\n')
    (OUT/'repeated_endpoint_differences.json').write_text(json.dumps(endpoint_diffs,indent=2)+'\n')
    plot(curves)
    write_report(rows,pairs,initial_diffs,endpoint_diffs,machine)
    for name in ['summarize_td_batch_ablation.py','audit_td_batch_warmup.py','summarize_td_ablation.py',
                 'summarize_task2_scale.py','audit_td_warmup.py','CHART_CONTRACT_TD_BATCH_ABLATION.md']:
        track(ROOT/name)
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',sources=evidence.sources,
        checks=['48 frozen training/protocol source files','task-specific offline checkpoint',
                'only batch and output path differ in configuration','paired full initial states',
                'full warmup replay arrays','fixed probe arrays and target math','MC episode records',
                '50k primitive steps and 7500 updates','finite training metrics','frozen base',
                'inherited Q and target','raw episode aggregates','repeated endpoint reset states and unchanged full agent',
                'all repeated endpoint episode differences preserved'],
        repeated_endpoint_difference_count=len(endpoint_diffs)),indent=2)+'\n')
    print(json.dumps(rows,indent=2),flush=True)


def plot(curves):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(13,9))
    for task in (1,2):
        for col,key,factor,label in [(0,'success',100,'Success rate (%)'),(1,'return_mean',1,'Mean episode return')]:
            ax=axes[task-1,col]
            for batch,color,marker,fill in [(1024,'#82A9D0','s','white'),(256,'#2463A6','o','#2463A6')]:
                ps=curves[task][batch]
                ax.plot([p['step']/1000 for p in ps],[p[key]*factor for p in ps],linestyle='none',
                    marker=marker,markersize=7 if batch==1024 else 5,markerfacecolor=fill,
                    markeredgecolor=color,markeredgewidth=1.5,label=f'Batch {batch}')
            ax.axvline(20,color='#ADB2B8',linestyle=':',linewidth=1)
            ax.set_title(f'Task{task}',loc='left',fontsize=12)
            ax.set_xlabel('Primitive online steps (thousands)');ax.set_ylabel(label)
            ax.set_xlim(-1,51);ax.set_xticks([0,5,10,20,30,40,50])
            if col==0:ax.set_ylim(0,103)
            ax.grid(axis='y',color='#E8EAED')
    fig.suptitle('Plain TD: batch-size comparison',x=.08,ha='left',fontsize=16)
    fig.text(.08,.93,'Seed 0 | Inherited Q | Frozen base | Scale 0.1 | Online-only replay | 50 episodes per point',fontsize=10)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
    fig.text(.08,.037,'Both batches: 20k warmup, 50k online steps, 7,500 updates. Actor entropy and automatic alpha are retained.',fontsize=9)
    fig.text(.08,.014,'Dots show the seven measured checkpoints; no interpolation or seed uncertainty bands. Final table uses 100 episodes.',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.875,bottom=.155,hspace=.40,wspace=.24)
    for ext in ('png','pdf'):fig.savefig(OUT/f'evaluation_checkpoints.{ext}',dpi=180)
    plt.close(fig)


def write_report(rows,pairs,initial_diffs,endpoint_diffs,machine):
    lines=['# 普通 TD 下 DAWN batch size 对照','',
        'Task1 / Task2，各 seed 0；从各自 QAM offline 500k checkpoint 开始 online training。继承 critic 和 target 参数，冻结 base，online-only replay，residual scale=0.1。',
        '普通 TD 只移除 critic target 的 entropy 项；actor 的 SAC entropy loss 和自动 alpha 保留。每组包含 20k warmup，共 50k primitive online steps、7,500 次更新。', '',
        '| Task | Batch | 最终成功率（100 eps） | 平均回报 ↑ | 训练累计抽样行数 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {r['batch_size']} | {r['success_percent']:.0f}% | {r['return_mean']:.2f} | {r['sampled_training_rows']:,} |")
    lines+=['','本机配对结果（batch 256 − batch 1024）：','']
    for p in pairs:lines.append(f"- Task{p['task']}：成功率 {p['batch256_minus1024_success_pp']:+.0f} pp；平均回报 {p['batch256_minus1024_return']:+.2f}。")
    lines+=['','本次同机、单训练 seed 对照中，batch 256 在两个任务的终点成功率和回报均更好。Task2 的 batch 1024 复跑相对历史结果波动较大，因此这只能作为继续验证小 batch 的趋势证据，不能据此认定稳定或普遍的优势。']
    lines+=['','历史 batch 1024 复现情况：','']
    for r in rows:
        if r['batch_size']==1024:
            lines.append(f"- Task{r['task']}：历史 {r['historical1024_success_percent']:.0f}% / {r['historical1024_return_mean']:.2f}；本次 {r['success_percent']:.0f}% / {r['return_mean']:.2f}。初始化完整 agent 是否精确一致：{r['exact_historical_initial_agent']}；最终 checkpoint 是否精确一致：{r['exact_historical_final_checkpoint']}。")
    lines+=['',f"当前机器：{machine['gpu']}。之前 TD 对照的驱动版本为 550.120，Python 依赖一致。虽然完整初始化一致，两个 batch 1024 的最终 checkpoint 都未精确复现历史权重。具体分叉来源尚未定位，不能仅凭版本差异将其归因于驱动；主比较使用当前机器上的两组，历史值仅供复现参考。",'',
        '| Task | Batch | 初始成功率（50 eps） | 终点随机残差（50 eps） | 终点均值残差（50 eps） | 曲线 AUC / 50k |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['task']} | {r['batch_size']} | {r['initial50_success_percent']:.0f}% | {r['curve_final50_success_percent']:.0f}% | {r['mean_residual50_success_percent']:.0f}% | {r['success_auc_percent']:.2f}% |")
    lines+=['','评估图只画预先固定的七个测量点；AUC 使用这些点的梯形面积除以 50k，是离散评估的汇总，不能反映点间波动。', '',
        f'终点重复评估检查：50-episode 评估与随后 100-episode 评估中的前 50 个 reset seed 完全对应，但共有 {len(endpoint_diffs)} 条轨迹记录未精确复现。所有组在这些评估前后的完整 agent 状态哈希一致。两套原始结果均保留，主表始终使用预先指定的 100-episode 评估，未选择更好的那次结果。具体差异见 repeated_endpoint_differences.json。', '',
        '固定设置与检查：',
        '- 两组唯一学习配置差异为 batch size。UTD=0.25、7,500 次更新保持不变；batch 256 的累计抽样量是 1024 的四分之一，没有通过额外更新进行补偿。抽样有放回，累计抽样行数不等于独立数据量。',
        '- 原训练入口、rollout、replay、actor/critic/alpha 更新代码全部保留，48 个训练/协议源文件已冻结并验证。',
        '- Critic ensemble=10，actor/target 聚合都取 minimum；gamma=.99，horizon=5，tau=.01；actor/critic/alpha lr=1e-4，梯度范数裁剪=50，初始 alpha=.01，目标残差熵=-25。继承 Q 和 target 权重，按原 DAWN 初始化新的 Adam 状态。',
        '- 核验同任务完整初始 agent、整个 warmup replay 和独立固定诊断 batch。固定 probe 仍有 256 行，它不参与训练，与本轮训练 batch 参数相互独立。',
        '- 报表由逐 episode 原始记录重新聚合；主表为最终 100 episodes，图为每点 50 episodes，均值残差另列。评估交互不进入训练 replay。',
        '- 仅一个训练 seed。独立训练重复与不同 batch 的最佳学习率/UTD 尚未测试；不能从本次对照推断普遍最优 batch size。',
        f'- 本机两组初始评估有 {len(initial_diffs)} 条 episode 记录差异；具体内容保留在 initial_evaluation_differences.json，初始化权重和 reset 状态另行核验。', '',
        '![Evaluation checkpoints](evaluation_checkpoints.png)','']
    (OUT/'summary.md').write_text('\n'.join(lines))


if __name__ == '__main__': main()
