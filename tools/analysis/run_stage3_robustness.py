#!/usr/bin/env python
"""Run the full NTU60 xsub 3D Joint robustness benchmark with multiple seeds.

Stage 3 protocol: 3 models x 16 conditions (1 clean + 5 degradation x 3
severity) x 3 training seeds = 144 inference runs.  Each run loads the clean
checkpoint trained with the matching seed and evaluates it on ``xsub_val``
under the requested degradation condition.  Degradation is applied in the test
pipeline (seed 255, sample-level deterministic), so only the *checkpoint*
varies across the seed dimension.

This script replaces the earlier single-seed ``run_moderate_robustness.py``
(now removed) with a full seed loop and a resumable manifest.
"""

import argparse
import json
import os
import subprocess
import sys
import time


MODELS = ('stgcn', 'stgcnpp', 'ctrgcn')
SEEDS = (255, 2026, 3407)
ANN_FILE = os.path.join('data', 'nturgbd', 'ntu60_3danno.pkl')
DEGRADE_TYPES = (
    'joint_missing',
    'limb_occlusion',
    'coord_noise',
    'frame_missing',
    'mixed')
SEVERITIES = ('mild', 'moderate', 'severe')

# Per-model checkpoint directory prefix.  All three models now share the same
# work_dir layout: work_dirs/{model}/{model}_pyskl_ntu60_xsub_3dkp/.
CKPT_DIRS = {
    'stgcn': os.path.join(
        'work_dirs', 'stgcn', 'stgcn_pyskl_ntu60_xsub_3dkp'),
    'stgcnpp': os.path.join(
        'work_dirs', 'stgcn++', 'stgcnpp_pyskl_ntu60_xsub_3dkp'),
    'ctrgcn': os.path.join(
        'work_dirs', 'ctrgcn', 'ctrgcn_pyskl_ntu60_xsub_3dkp'),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root', default='work_dirs/robustness_benchmark')
    parser.add_argument(
        '--severities', nargs='+', choices=SEVERITIES,
        default=['mild', 'moderate', 'severe'],
        help='Degradation severities to run. Default: all three.')
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=list(SEEDS),
        help='Training seeds whose checkpoints to evaluate. '
             'Default: 255 2026 3407.')
    parser.add_argument(
        '--models', nargs='+', choices=MODELS, default=list(MODELS),
        help='Models to evaluate. Default: all three.')
    parser.add_argument(
        '--skip-clean', action='store_true',
        help='Do not repeat the Clean condition (useful when clean '
             'results already exist from a prior run).')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate files and print commands without launching inference.')
    parser.add_argument(
        '--max-runs', type=int, default=None,
        help='Stop after launching this many new runs (for testing or '
             'GPU-time-budgeted sessions). Existing completed runs still '
             'count toward the manifest.')
    return parser.parse_args()


def _normalized(path):
    return os.path.abspath(os.path.normpath(path))


def _find_checkpoint(model, seed):
    """Locate the best-accuracy checkpoint for a model/seed pair."""
    ckpt_dir = os.path.join(CKPT_DIRS[model], 'j_single_gpu_seed{}'.format(seed))
    candidates = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith('best_top1_acc_epoch_') and f.endswith('.pth')]
    if not candidates:
        raise FileNotFoundError(
            'No best_top1_acc checkpoint in {}'.format(_normalized(ckpt_dir)))
    # Exactly one best checkpoint is expected per seed dir.
    return os.path.join(ckpt_dir, candidates[0])


def _config_path(config_root, model, condition):
    if condition == 'clean':
        return os.path.join(config_root, model, 'clean.py')
    degrade_type, severity = condition.rsplit('_', 1)
    return os.path.join(config_root, model, severity, '{}.py'.format(degrade_type))


def _condition_name(degrade_type, severity):
    return '{}_{}'.format(degrade_type, severity)


def _build_runs(args):
    config_root = os.path.join(
        'configs', 'robust_skeleton', 'ntu60_xsub_3dkp')
    conditions = [] if args.skip_clean else ['clean']
    for severity in args.severities:
        for degrade_type in DEGRADE_TYPES:
            conditions.append(_condition_name(degrade_type, severity))

    runs = []
    for model in args.models:
        for seed in args.seeds:
            checkpoint = _find_checkpoint(model, seed)
            for condition in conditions:
                config = _config_path(config_root, model, condition)
                output_dir = os.path.join(
                    args.output_root, model, condition, 'seed{}'.format(seed))
                output = os.path.join(output_dir, 'result.pkl')
                log = os.path.join(output_dir, 'test.log')
                command = [
                    sys.executable,
                    '-m', 'torch.distributed.run',
                    '--standalone',
                    '--nproc_per_node=1',
                    os.path.join('tools', 'test.py'),
                    config,
                    '-C', checkpoint,
                    '--out', output,
                    '--eval', 'top_k_accuracy', 'mean_class_accuracy']
                runs.append(dict(
                    model=model,
                    seed=seed,
                    condition=condition,
                    config=config,
                    checkpoint=checkpoint,
                    output=output,
                    log=log,
                    command=command))
    return runs


def _preflight(runs):
    missing = []
    if not os.path.isfile(ANN_FILE):
        missing.append(_normalized(ANN_FILE))
    seen_ckpt = set()
    for run in runs:
        for path in (run['checkpoint'], run['config']):
            if path not in seen_ckpt and not os.path.isfile(path):
                missing.append(_normalized(path))
                seen_ckpt.add(path)
    if missing:
        raise FileNotFoundError(
            'Missing required artifacts:\n  ' + '\n  '.join(missing))


def _validate_configs(runs):
    """Assert every config matches the frozen stage-3 protocol."""
    try:
        from mmcv import Config
    except ImportError as error:
        raise RuntimeError(
            'MMCV is unavailable. Run this matrix in the same environment '
            'used for clean baseline testing.') from error

    expected_backbone = {
        'stgcn': 'STGCN',
        'stgcnpp': 'STGCN',
        'ctrgcn': 'CTRGCN',
    }
    clean_types = [
        'PreNormalize3D', 'GenSkeFeat', 'UniformSample', 'PoseDecode',
        'FormatGCNInput', 'Collect', 'ToTensor']
    degraded_types = [
        'PreNormalize3D', 'RandomSkeletonDegrade', 'GenSkeFeat',
        'UniformSample', 'PoseDecode', 'FormatGCNInput', 'Collect',
        'ToTensor']

    checked = set()
    for run in runs:
        if run['config'] in checked:
            continue  # config is identical across seeds for a given condition
        cfg = Config.fromfile(run['config'])
        test = cfg.data.test
        assert os.path.normpath(test.ann_file) == os.path.normpath(ANN_FILE), \
            run['config']
        assert test.split == 'xsub_val', run['config']
        assert cfg.model.backbone.type == expected_backbone[run['model']], \
            run['config']
        assert cfg.model.cls_head.num_classes == 60, run['config']
        pipeline = test.pipeline
        pipeline_types = [item['type'] for item in pipeline]
        expected_types = (
            clean_types if run['condition'] == 'clean' else degraded_types)
        assert pipeline_types == expected_types, (run['config'], pipeline_types)
        sample = next(
            item for item in pipeline if item['type'] == 'UniformSample')
        assert sample['clip_len'] == 100, run['config']
        assert sample['num_clips'] == 10, run['config']
        assert sample['seed'] == 255, run['config']
        feature = next(
            item for item in pipeline if item['type'] == 'GenSkeFeat')
        assert feature['dataset'] == 'nturgb+d', run['config']
        assert feature['feats'] == ['j'], run['config']
        if run['condition'] != 'clean':
            degrade_type, severity = run['condition'].rsplit('_', 1)
            degrade = next(
                item for item in pipeline
                if item['type'] == 'RandomSkeletonDegrade')
            assert degrade['degrade_type'] == degrade_type, run['config']
            assert degrade['severity'] == severity, run['config']
            assert degrade['dataset'] == 'nturgb+d', run['config']
            assert degrade['prob'] == 1.0, run['config']
            assert degrade['seed'] == 255, run['config']
        checked.add(run['config'])
    print('PASS: validated {} distinct configs against stage-3 protocol'.format(
        len(checked)))


def _load_manifest(path):
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    args = parse_args()
    runs = _build_runs(args)
    _preflight(runs)
    _validate_configs(runs)

    expected_runs = len(args.models) * len(args.seeds) * (
        (0 if args.skip_clean else 1) +
        len(DEGRADE_TYPES) * len(args.severities))
    assert len(runs) == expected_runs, (len(runs), expected_runs)

    os.makedirs(args.output_root, exist_ok=True)
    manifest_name = 'stage3_manifest.json'
    manifest_path = os.path.join(args.output_root, manifest_name)
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        manifest = dict(
            status='pending',
            ann_file=_normalized(ANN_FILE),
            seeds=list(args.seeds),
            models=list(args.models),
            severities=list(args.severities),
            runs=runs)
        # Index runs by a stable key for resumable status updates.
        manifest['run_index'] = {
            '{}_{}_seed{}'.format(r['model'], r['condition'], r['seed']): i
            for i, r in enumerate(runs)}
    else:
        # Merge run list into existing manifest (preserve completed status).
        existing = {r['output']: r for r in manifest.get('runs', [])}
        for r in runs:
            if r['output'] in existing and existing[r['output']].get('returncode') == 0:
                r.update(existing[r['output']])
        manifest['runs'] = runs
        manifest['run_index'] = {
            '{}_{}_seed{}'.format(r['model'], r['condition'], r['seed']): i
            for i, r in enumerate(runs)}

    print('\n=== Stage 3 robustness benchmark ===')
    print('Models: {} | Seeds: {} | Conditions: {} | Total runs: {}'.format(
        ', '.join(args.models), ', '.join(str(s) for s in args.seeds),
        (0 if args.skip_clean else 1) + len(DEGRADE_TYPES) * len(args.severities),
        len(runs)))

    if args.dry_run:
        print('\n[dry-run] {} commands would be launched:'.format(len(runs)))
        for run in runs:
            print('  {} seed{} {}: {}'.format(
                run['model'], run['seed'], run['condition'],
                ' '.join('"{}"'.format(p) if ' ' in p else p
                         for p in run['command'])))
        manifest['status'] = 'dry_run'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)
        print('\n[dry-run] manifest written: {}'.format(manifest_path))
        return

    launched = 0
    for index, run in enumerate(runs, 1):
        output_dir = os.path.dirname(run['output'])
        os.makedirs(output_dir, exist_ok=True)
        if os.path.isfile(run['output']) and run.get('returncode') == 0:
            print('[{}/{}] SKIP (done): {} seed{} {}'.format(
                index, len(runs), run['model'], run['seed'], run['condition']))
            continue
        if args.max_runs is not None and launched >= args.max_runs:
            print('[{}/{}] STOP: reached --max-runs {}'.format(
                index, len(runs), args.max_runs))
            break
        print('[{}/{}] {} seed{} {}'.format(
            index, len(runs), run['model'], run['seed'], run['condition']))
        start = time.time()
        with open(run['log'], 'w', encoding='utf-8') as log_file:
            completed = subprocess.run(
                run['command'], stdout=log_file, stderr=subprocess.STDOUT)
        run['returncode'] = completed.returncode
        run['elapsed_seconds'] = time.time() - start
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)
        launched += 1
        if completed.returncode != 0:
            manifest['status'] = 'failed'
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=True)
            raise RuntimeError(
                '{} seed{} {} failed (rc={}); inspect {}'.format(
                    run['model'], run['seed'], run['condition'],
                    completed.returncode, run['log']))

    manifest['status'] = 'completed'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
    print('\nPASS: completed {} runs; manifest: {}'.format(
        len(runs), manifest_path))


if __name__ == '__main__':
    main()
