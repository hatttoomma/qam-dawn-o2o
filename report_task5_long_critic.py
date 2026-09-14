"""Recompute Task5 long-budget results from exported episode records and plot them."""
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2]/'output/task5_long_critic_shared_20260910'
RAW = OUT/'raw'
RUNS = RAW/'runs/task5_long_critic_shared_20260910'


def read(path):
    return json.loads(path.read_text())


def main():
    RAW.mkdir(exist_ok=True)
    with tarfile.open(OUT/'results_export_20260910.tar.gz') as archive:
        archive.extractall(RAW, filter='data')
    baseline = read(RUNS/'warm/eval_000000_100.json')
    initial_states = [(r['episode'], r['reset_seed'], r['initial_hash']) for r in baseline['records']]
    assert [r[0] for r in initial_states] == list(range(100))
    rows, source_hashes = [], {}

    def verify_eval(path):
        result = read(path)
        records = result['records']
        assert result['episodes'] == len(records) == 100
        assert [(r['episode'],r['reset_seed'],r['initial_hash']) for r in records] == initial_states
        success = math.fsum(r['success'] for r in records)/100
        ret = math.fsum(r['return_'] for r in records)/100
        assert math.isclose(success,result['success'],abs_tol=1e-12)
        assert math.isclose(ret,result['return_mean'],abs_tol=1e-10)
        assert all(r['success'] in (0,1) and 0 < r['length'] <= 500 for r in records)
        source_hashes[str(path.relative_to(RAW))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    for arm in ('warm','random'):
        directory = RUNS/arm
        done = read(directory/'DONE.json')
        assert done['steps'] == 500000 and done['updates'] == 120000
        assert read(directory/'CHECKS_PASSED.json')['status'] == 'passed'
        for path in sorted(directory.glob('eval_*.json')):
            result = verify_eval(path)
            step = result['step']
            mode = 'mean_residual' if result['deterministic_residual'] else 'sampled'
            gained = sum(a['success']==0 and b['success']==1 for a,b in zip(baseline['records'],result['records']))
            lost = sum(a['success']==1 and b['success']==0 for a,b in zip(baseline['records'],result['records']))
            if step:
                pair = read(directory/f'paired_{step:06d}.json')[mode]
                assert pair['gained']==gained and pair['lost']==lost
                assert math.isclose(pair['success_delta'],(gained-lost)/100,abs_tol=1e-12)
            rows.append(dict(arm=arm,mode=mode,step=step,nominal_step=round(step/1000)*1000,
                updates=max(0,(step-20000)//4),episodes=100,success=result['success'],
                return_mean=result['return_mean'],gained_vs_offline=gained,lost_vs_offline=lost,
                source=str(path.relative_to(RAW))))
    native = verify_eval(RAW/'runs/cube5/task5_native/eval_050000_100.json')
    historical = verify_eval(RAW/'runs/cube5/task5_warm/eval_050000_100.json')
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'validation.json').write_text(json.dumps(dict(
        status='passed', evaluation_files_recomputed=len(source_hashes), episodes_per_evaluation=100,
        identical_reset_seeds_and_initial_observations=True, paired_counts_recomputed=True,
        final_steps_per_arm=500000, final_updates_per_arm=120000,
        source_sha256=source_hashes),indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,
        'axes.titlesize':13,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#858585','text.color':'#222222','axes.labelcolor':'#333333',
        'xtick.color':'#444444','ytick.color':'#444444','savefig.facecolor':'white'})
    fig, axes = plt.subplots(1,2,figsize=(12.5,6.5))
    fig.subplots_adjust(top=.70,bottom=.20,left=.075,right=.97,wspace=.20)
    colors={'warm':'#176B9F','random':'#C65D28'}
    names={'warm':'Inherited Q','random':'Random Q'}
    handles=[]
    for arm in ('warm','random'):
        for mode in ('sampled','mean_residual'):
            group=[r for r in rows if r['arm']==arm and r['mode']==mode]
            group.sort(key=lambda r:r['step'])
            xs=[r['step']/1000 for r in group]
            style=dict(color=colors[arm],linewidth=2.1,linestyle='-' if mode=='sampled' else '--',
                marker='o' if mode=='sampled' else 's',markersize=5,
                markerfacecolor=colors[arm] if mode=='sampled' else 'white',markeredgewidth=1.2)
            label=names[arm]+(' / sampled residual' if mode=='sampled' else ' / mean residual')
            line,=axes[0].plot(xs,[r['success']*100 for r in group],label=label,**style)
            axes[1].plot(xs,[r['return_mean'] for r in group],**style)
            handles.append(line)
    for ax in axes:
        ax.set_xlim(-8,555)
        ax.set_xticks([0,50,100,200,300,400,500])
        ax.set_xlabel('Online environment steps (thousands)')
        ax.grid(axis='y',color='#E6E6E6',linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylim(45,104);axes[0].set_yticks([50,60,70,80,90,100])
    axes[0].set_ylabel('Success rate (%)');axes[0].set_title('Success rate',loc='left',pad=12)
    axes[1].set_ylim(-585,-230)
    axes[1].set_ylabel('Average episode return');axes[1].set_title('Average return (higher is better)',loc='left',pad=12)
    for ax,key,mult in [(axes[0],'success',100),(axes[1],'return_mean',1)]:
        base=baseline[key]*mult
        ax.axhline(base,color='#888888',linewidth=1.1,linestyle=':',zorder=1)
        base_label=f'Offline: {base:.0f}%' if key=='success' else f'Offline: {base:.2f}'
        ax.annotate(base_label,(345,base),xytext=(0,-16),textcoords='offset points',color='#666666',fontsize=9)
        ref=native[key]*mult
        ax.scatter([50],[ref],marker='D',s=48,color='#333333',zorder=5)
        ref_label=f'QAM native @50k: {ref:.0f}%' if key=='success' else f'QAM native @50k: {ref:.2f}'
        ax.annotate(ref_label,(50,ref),xytext=(12,1),textcoords='offset points',fontsize=9,color='#333333',va='center')
    for arm,dy in [('warm',8),('random',-9)]:
        last=next(r for r in rows if r['arm']==arm and r['mode']=='sampled' and r['step']==500000)
        axes[0].annotate(f"{last['success']*100:.0f}%",(500,last['success']*100),xytext=(8,dy),
            textcoords='offset points',color=colors[arm],fontsize=10,fontweight='bold')
    fig.text(.075,.955,'Task5: critic initialization and longer online training',fontsize=17,fontweight='bold')
    fig.text(.075,.911,'Frozen QAM 500k  |  Ordinary TD  |  Batch 256  |  UTD 0.25  |  Residual scale 0.1',fontsize=10,color='#555555')
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.068,.88),ncol=2,frameon=False,fontsize=10,columnspacing=2)
    fig.text(.075,.105,'One training seed; 100 fixed evaluation episodes per point. No across-seed uncertainty estimate.',fontsize=9,color='#555555')
    fig.text(.075,.072,'Base actions are sampled in both modes. Lines connect measured checkpoints only; QAM native is a historical 50k reference.',fontsize=9,color='#555555')
    fig.savefig(OUT/'training_curves.png',dpi=180)
    fig.savefig(OUT/'training_curves.svg')
    plt.close(fig)

    def at(arm,step,mode='sampled'):
        return next(r for r in rows if r['arm']==arm and r['nominal_step']==step and r['mode']==mode)
    table=['| Online steps | 更新次数 | 继承 Q 成功率 | 继承 Q return | 随机 Q 成功率 | 随机 Q return |',
           '|---:|---:|---:|---:|---:|---:|']
    for step in (0,50000,100000,200000,500000):
        a,b=at('warm',step),at('random',step)
        table.append(f"| {step:,} | {a['updates']:,} | {a['success']:.0%} | {a['return_mean']:.2f} | {b['success']:.0%} | {b['return_mean']:.2f} |")
    mean_table=['| Online steps | 继承 Q 成功率 / return | 随机 Q 成功率 / return |','|---:|---:|---:|']
    for step in (50000,100000,200000,500000):
        a,b=at('warm',step,'mean_residual'),at('random',step,'mean_residual')
        mean_table.append(f"| {step:,} | {a['success']:.0%} / {a['return_mean']:.2f} | {b['success']:.0%} / {b['return_mean']:.2f} |")
    report='\n'.join([
        '# Task5：critic 初始化与延长 online 训练', '',
        '两组均完成 500,000 primitive online steps、120,000 次更新。最终 checkpoint 的 SHA256 已在远端核对。', '',
        '共享同一份 QAM 500k offline checkpoint、seed=0；ordinary TD、batch256、UTD=0.25、state+base action、scale=0.1、online-only replay。两组共享无更新的 warmup 数据和恢复状态，实际 warmup 边界为 20,003 步，数据哈希与历史 Task5 一致。', '',
        '以下为采样 residual 的评估；每个点均使用相同的 100 个 reset seeds。Return 越大越好。', '',
        *table, '',
        f"历史 QAM native 在 50k online steps：成功率 {native['success']:.0%}，平均 return {native['return_mean']:.2f}。这是较小预算的参考点，不是 QAM 的 500k 结果。", '',
        '均值 residual 评估（使用 tanh(mean)，base action 仍采样）：', '', *mean_table, '',
        '本轮观察：随机 Q 从 50k 的 59% 逐步恢复到 500k 的 83%；继承 Q 为 84%→84%→81%→86%，增长不单调。两组最终都超过 offline 的 76%，但仍低于 QAM native 50k 的 98%。随机 Q 在更长预算下有明显恢复，未观察到随机初始化最终优于继承 Q。', '',
        '最终采样评估相对 offline：继承 Q 新增成功 20 条、损失原有成功 10 条；随机 Q 新增成功 19 条、损失原有成功 12 条。因此这里存在学到新成功路径同时破坏旧成功路径的现象。', '',
        '只有一个训练 seed；最终 86% 与 83% 的差异不能直接视为稳定优势。更多步数同时增加了新数据和更新次数，本实验不能区分两者的单独贡献，也不能据此证明 residual 的表达上限。', '',
        f"跨运行说明：本轮继承 Q 在约 50k 的成功率为 84%；此前独立 50k run 为 {historical['success']:.0%}、return {historical['return_mean']:.2f}。本轮从共享 warmup checkpoint 重新启动，不是续训旧 50k checkpoint；因此延长训练的增益应按本轮内部曲线比较。独立进程的数值轨迹差异尚未完全定位。", '',
        '步数显示为名义里程碑；实际评估位于完整 action chunk 边界，个别点多 1–3 步。CSV 保留实际步数。', '',
        '图：training_curves.png / training_curves.svg；数值：results.csv；原始逐 episode 记录：raw/；核验：validation.json。', ''
    ])
    (OUT/'report.md').write_text(report)
    print(json.dumps(dict(output=str(OUT),evaluations_verified=len(source_hashes),rows=len(rows),
                         final_sampled={a:at(a,500000) for a in ('warm','random')}),indent=2))


if __name__=='__main__':
    main()
