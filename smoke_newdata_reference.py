"""Independent original-native and zero-interaction offline-update references."""
import argparse
import fcntl
from pathlib import Path
import sys
import numpy as np
import run as common


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--task', type=int, choices=[1, 2], required=True)
    p.add_argument('--mode', choices=['native', 'pure_offline'], required=True)
    extra, rest = p.parse_known_args();sys.argv = [sys.argv[0], *rest]
    args = common.parse();assert args.stage == 'native' and args.online_steps == 60 and args.native_start == 10
    assert args.offline_steps == 500000 and args.seed == 0
    common.ENV = f'cube-double-play-singletask-task{extra.task}-v0'
    root = common.ROOT / ('runs' if extra.task == 1 else 'runs/task2')
    args.offline_checkpoint = str(root / 'offline/final.pkl')
    out = Path(args.out);assert out.resolve().is_relative_to((common.ROOT / 'runs/newdata_ablation').resolve())
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('w');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'DONE.json').exists(): return
    if extra.mode == 'native':
        common.native(args)
    else:
        _, env, ds, _, qam = common.prepare_online(args)
        np.random.seed(31000 + args.seed)
        common.evaluate(qam, out, 0, args)
        count = args.online_steps - args.native_start + 1
        for _ in range(count): qam, _ = qam.batch_update(common.seq_batch(ds, 1, 256))
        common.evaluate(qam, out, args.online_steps, args)
        common.save_checkpoint(out / 'final.pkl', qam, step=args.online_steps, updates=count)
        common.atomic_json(out / 'DONE.json', dict(mode=extra.mode, updates=count, actual_training_env_steps=0,
            checkpoint_sha256=common.file_hash(out / 'final.pkl')))
        env.close()


if __name__ == '__main__': main()
