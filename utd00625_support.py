"""Auditable UTD-only adaptation of the immutable historical residual loop."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/utd00625_warmup80k_150k_20260911'
UTD = 0.0625
REFERENCE_RUN_HASH = '28eb970ed70773d744a072460a4595b8fe26285812464b13a6a302bb25f043fd'


def reference_dir(task):
    suite = 'warmup80k_150k_20260911' if task in (1, 5) else 'warmup80k_150k_task234_20260911'
    return ROOT / 'runs' / suite / f'task{task}_warm'


def collector_dir(task, smoke=False):
    if smoke:
        return RUNS / f'smoke_task{task}_collector'
    suite = 'warmup80k_150k_20260911' if task in (1, 5) else 'warmup80k_150k_task234_20260911'
    return ROOT / 'runs' / suite / f'{"smoke_" if smoke else ""}task{task}_collector'


def schedule_source(source):
    old = 'target_updates=int((step-args.warmup)*.25)'
    new = 'target_updates=int((step-args.warmup)*.0625)'
    assert source.count(old) == 1
    modified = source.replace(old, new)
    modified = modified.replace('Primitive-step UTD .25;', 'Primitive-step UTD .0625;')
    assert modified.replace(new, old).replace('Primitive-step UTD .0625;', 'Primitive-step UTD .25;') == source
    # Prove that the executable AST differs only in this numerical constant.
    original_ast, modified_ast = ast.parse(source), ast.parse(modified)
    original_nodes, modified_nodes = list(ast.walk(original_ast)), list(ast.walk(modified_ast))
    differences = []
    assert len(original_nodes) == len(modified_nodes)
    for i, (a, b) in enumerate(zip(original_nodes, modified_nodes)):
        if isinstance(a, ast.Constant) and isinstance(b, ast.Constant) and a.value != b.value:
            differences.append((a.value, b.value))
            b.value = a.value
    assert differences == [(0.25, 0.0625)]
    assert ast.dump(original_ast) == ast.dump(modified_ast)
    return modified


def install_utd_schedule(common, out):
    assert common.file_hash(ROOT / 'run.py') == REFERENCE_RUN_HASH
    source = inspect.getsource(common.residual)
    modified = schedule_source(source)
    namespace = {}
    exec(compile(modified, str(ROOT / 'utd00625_residual.generated.py'), 'exec'), common.__dict__, namespace)
    common.residual = namespace['residual']
    common.atomic_json(out / 'utd_schedule_audit.json', dict(
        status='passed', original_utd=.25, actual_utd=UTD,
        only_executable_change='target_updates multiplier: 0.25 -> 0.0625',
        original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        modified_function_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        ast_single_constant_change_verified=True,
        final_updates=4375, updates_at_100k=1250, updates_at_120k=2500))
    (out / 'residual_loop_executed.py').write_text(modified)


def verify_matching_configuration(common, out, task, smoke=False):
    reference = reference_dir(task)
    load = lambda p: json.loads(p.read_text())
    old, new = load(reference / 'actual_agent_config.json'), load(out / 'actual_agent_config.json')
    expected = dict(old, utd=UTD)
    if smoke:
        expected['warmup'] = 80
    assert new == expected, {k: (old.get(k), new.get(k)) for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    requested = load(out / 'requested_config.json')
    old_requested = load(reference / 'requested_config.json')
    operational = {'out', 'source_hashes', 'utd'}
    if smoke:
        operational |= {'smoke', 'online_steps', 'warmup', 'grid', 'eval_episodes', 'final_episodes', 'shared_warmup', 'shared_warmup_sha256', 'collect_warmup'}
    assert {k:v for k,v in requested.items() if k not in operational} == {k:v for k,v in old_requested.items() if k not in operational}
    assert requested['utd'] == UTD
    collecting_smoke = smoke and requested['collect_warmup']
    assert requested['shared_warmup'] == (None if collecting_smoke else str(collector_dir(task, smoke)))
    if not smoke:
        assert requested['shared_warmup_sha256'] == load(reference / 'shared_initialization.json')['collector_checkpoint_sha256']
    common.atomic_json(out / 'MATCHED_CONFIG_PASSED.json', dict(status='passed', task=task,
        reference=str(reference), old_utd=.25, new_utd=UTD, smoke=smoke,
        actual_agent_config_only_utd_changed=not smoke,
        initial_agent_identical=True, offline_checkpoint_identical=True,
        same_warmup_checkpoint_and_initial_evaluation=not smoke,
        shared_warmup_sha256=requested.get('shared_warmup_sha256')))
