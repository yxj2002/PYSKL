#!/usr/bin/env python
"""Test augmentation-baseline work-1 checkpoints on the frozen 16-condition
matrix (inner_val).

Run on ONE machine (e.g. the 4090) after the 7 training checkpoints have been
synced to it.  For each A group it locates ``best_top1_acc_epoch_*.pth`` under
``work_dirs/aug_baseline/stgcnpp_j/train/{group}_{tag}/`` and evaluates it on the
16 conditions (1 clean + 5 degradations x 3 severities), writing each result to
``work_dirs/aug_baseline/stgcnpp_j/results/{group}_{tag}/{condition}/result.pkl``.

Resumable via a manifest; supports --dry-run / --groups / --max-runs.
Uses ``torch.distributed.run --standalone`` (single GPU), matching stage 3.
"""
import argparse
import json
import os
import subprocess
import sys
import time

# A group -> training degradation tag (also the training work_dir suffix).
A_GROUPS = {
    'A0': 'clean',
    'A1': 'joint_missing',
    'A2': 'limb_occlusion',
    'A3': 'coord_noise',
    'A4': 'frame_missing',
    'A5': 'random_single',
    'A6': 'mixed',
}

DEGRADE_TYPES = (
    'joint_missing', 'limb_occlusion', 'coord_noise',
    'frame_missing', 'mixed')
SEVERITIES = ('mild', 'moderate', 'severe')

ANN_FILE = os.path.join('data', 'nturgbd', 'ntu60_inner_split.pkl')
CONFIG_ROOT = os.path.join('configs', 'aug_baseline', 'stgcnpp_j')
WORK_ROOT = os.path.join('work_dirs', 'aug_baseline', 'stgcnpp_j')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--groups', nargs='+', choices=list(A_GROUPS.keys()),
        default=list(A_GROUPS.keys()),
        help='A groups to test. Default: all 7.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate files and print commands without running.')
    parser.add_argument(
        '--max-runs', type=int, default=None,
        help='Stop after launching this many new runs.')
    return parser.parse_args()


def group_dir(group):
    return '{}_{}'.format(group, A_GROUPS[group])


def _train_workdir(group):
    return os.path.join(WORK_ROOT, 'train', group_dir(group))


def _test_config(condition):
    if condition == 'clean':
        return os.path.join(CONFIG_ROOT, 'test', 'clean.py')
    degrade_type, severity = condition.rsplit('_', 1)
    return os.path.join(
        CONFIG_ROOT, 'test', severity, '{}.py'.format(degrade_type))


def condition_list():
    conds = ['clean']
    for severity in SEVERITIES:
        for degrade_type in DEGRADE_TYPES:
            conds.append('{}_{}'.format(degrade_type, severity))
    return conds


def _find_checkpoint(group):
    ckpt_dir = _train_workdir(group)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            'Training workdir missing: {} (sync the checkpoint first)'.format(
                os.path.abspath(ckpt_dir)))
    candidates = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith('best_top1_acc_epoch_') and f.endswith('.pth')]
    if not candidates:
        raise FileNotFoundError(
            'No best_top1_acc checkpoint in {}'.format(os.path.abspath(ckpt_dir)))
    return os.path.join(ckpt_dir, candidates[0])


def build_runs(groups, allow_missing_ckpt=False):
    runs = []
    for group in groups:
        if allow_missing_ckpt:
            checkpoint = os.path.join(
                _train_workdir(group), 'best_top1_acc_epoch_<N>.pth')
        else:
            checkpoint = _find_checkpoint(group)
        for condition in condition_list():
            config = _test_config(condition)
            output_dir = os.path.join(
                WORK_ROOT, 'results', group_dir(group), condition)
            output = os.path.join(output_dir, 'result.pkl')
            log = os.path.join(output_dir, 'test.log')
            command = [
                sys.executable, '-m', 'torch.distributed.run',
                '--standalone', '--nproc_per_node=1',
                os.path.join('tools', 'test.py'), config,
                '-C', checkpoint,
                '--out', output,
                '--eval', 'top_k_accuracy', 'mean_class_accuracy']
            runs.append(dict(
                group=group, condition=condition, config=config,
                checkpoint=checkpoint, output=output, log=log,
                command=command))
    return runs


def _preflight(runs, skip_ckpt=False):
    missing = []
    if not os.path.isfile(ANN_FILE):
        missing.append(os.path.abspath(ANN_FILE))
    seen = set()
    for run in runs:
        for path in (run['config'], run['checkpoint']):
            if skip_ckpt and path == run['checkpoint']:
                continue
            if path not in seen and not os.path.isfile(path):
                seen.add(path)
                missing.append(os.path.abspath(path))
    if missing:
        raise FileNotFoundError(
            'Missing required artifacts:\n  ' + '\n  '.join(missing))


def _load_manifest(path):
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None


def _dump_manifest(path, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)


def main():
    args = parse_args()
    runs = build_runs(args.groups, allow_missing_ckpt=args.dry_run)
    _preflight(runs, skip_ckpt=args.dry_run)

    manifest_path = os.path.join(WORK_ROOT, 'aug_baseline_work1_test_manifest.json')
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        manifest = dict(
            status='pending',
            ann_file=os.path.abspath(ANN_FILE),
            runs=[])

    existing = {
        (r['group'], r['condition']): r for r in manifest.get('runs', [])}
    for r in runs:
        key = (r['group'], r['condition'])
        if key in existing and existing[key].get('returncode') == 0:
            r.update(existing[key])
    manifest['runs'] = runs
    _dump_manifest(manifest_path, manifest)

    print('\n=== Augmentation-baseline work-1 test ({} groups x {} conditions = {} runs) ==='.format(
        len(args.groups), len(condition_list()), len(runs)))

    if args.dry_run:
        print('[dry-run] {} commands would be launched:'.format(len(runs)))
        for r in runs:
            print('  {} {}: {}'.format(
                r['group'], r['condition'],
                ' '.join('"{}"'.format(p) if ' ' in p else p
                         for p in r['command'])))
        manifest['status'] = 'dry_run'
        _dump_manifest(manifest_path, manifest)
        print('\n[dry-run] manifest: {}'.format(manifest_path))
        return

    launched = 0
    for index, run in enumerate(runs, 1):
        os.makedirs(os.path.dirname(run['log']), exist_ok=True)
        if run.get('returncode') == 0 and os.path.isfile(run['output']):
            print('[{}/{}] SKIP (done): {} {}'.format(
                index, len(runs), run['group'], run['condition']))
            continue
        if args.max_runs is not None and launched >= args.max_runs:
            print('[{}/{}] STOP: reached --max-runs {}'.format(
                index, len(runs), args.max_runs))
            break
        print('[{}/{}] {} {}'.format(
            index, len(runs), run['group'], run['condition']))
        start = time.time()
        with open(run['log'], 'w', encoding='utf-8') as log_file:
            completed = subprocess.run(
                run['command'], stdout=log_file, stderr=subprocess.STDOUT)
        run['returncode'] = completed.returncode
        run['elapsed_seconds'] = time.time() - start
        _dump_manifest(manifest_path, manifest)
        launched += 1
        if completed.returncode != 0:
            manifest['status'] = 'failed'
            _dump_manifest(manifest_path, manifest)
            raise RuntimeError(
                '{} {} failed (rc={}); inspect {}'.format(
                    run['group'], run['condition'],
                    completed.returncode, run['log']))

    manifest['status'] = 'completed'
    _dump_manifest(manifest_path, manifest)
    print('\nPASS: completed {} runs; manifest: {}'.format(
        len(runs), manifest_path))


if __name__ == '__main__':
    main()
