#!/usr/bin/env python
"""Run selected NTU60 xsub 3D Joint robustness conditions."""

import argparse
import json
import os
import subprocess
import sys
import time


MODELS = ('stgcn', 'stgcnpp', 'ctrgcn')
ANN_FILE = os.path.join('data', 'nturgbd', 'ntu60_3danno.pkl')
DEGRADE_TYPES = (
    'joint_missing',
    'limb_occlusion',
    'coord_noise',
    'frame_missing',
    'mixed')
SEVERITIES = ('mild', 'moderate', 'severe')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stgcn-checkpoint', required=True)
    parser.add_argument('--stgcnpp-checkpoint', required=True)
    parser.add_argument('--ctrgcn-checkpoint', required=True)
    parser.add_argument(
        '--output-root', default='work_dirs/robustness_benchmark')
    parser.add_argument(
        '--severities', nargs='+', choices=SEVERITIES, default=['moderate'],
        help='Degradation severities to run. Default: moderate.')
    parser.add_argument(
        '--skip-clean', action='store_true',
        help='Do not repeat the Clean condition for each model.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate files and print commands without launching inference.')
    return parser.parse_args()


def _normalized(path):
    return os.path.abspath(os.path.normpath(path))


def _preflight(checkpoints, runs):
    missing = []
    if not os.path.isfile(ANN_FILE):
        missing.append(_normalized(ANN_FILE))
    for checkpoint in checkpoints.values():
        if not os.path.isfile(checkpoint):
            missing.append(_normalized(checkpoint))
    for run in runs:
        if not os.path.isfile(run['config']):
            missing.append(_normalized(run['config']))
    if missing:
        raise FileNotFoundError('Missing required artifacts:\n  ' + '\n  '.join(missing))


def _validate_configs(runs):
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

    for run in runs:
        cfg = Config.fromfile(run['config'])
        test = cfg.data.test
        assert os.path.normpath(test.ann_file) == os.path.normpath(ANN_FILE)
        assert test.split == 'xsub_val'
        assert cfg.model.backbone.type == expected_backbone[run['model']]
        assert cfg.model.cls_head.num_classes == 60
        pipeline = test.pipeline
        pipeline_types = [item['type'] for item in pipeline]
        expected_types = clean_types if run['condition'] == 'clean' else degraded_types
        assert pipeline_types == expected_types, (run['config'], pipeline_types)
        sample = next(item for item in pipeline if item['type'] == 'UniformSample')
        assert sample['clip_len'] == 100
        assert sample['num_clips'] == 10
        assert sample['seed'] == 255
        feature = next(item for item in pipeline if item['type'] == 'GenSkeFeat')
        assert feature['dataset'] == 'nturgb+d'
        assert feature['feats'] == ['j']
        if run['condition'] != 'clean':
            degrade_type, severity = run['condition'].rsplit('_', 1)
            degrade = next(
                item for item in pipeline
                if item['type'] == 'RandomSkeletonDegrade')
            assert degrade['degrade_type'] == degrade_type
            assert degrade['severity'] == severity
            assert degrade['dataset'] == 'nturgb+d'
            assert degrade['prob'] == 1.0
            assert degrade['seed'] == 255


def _build_runs(args, checkpoints):
    config_root = os.path.join(
        'configs', 'robust_skeleton', 'ntu60_xsub_3dkp')
    runs = []
    for model in MODELS:
        conditions = [] if args.skip_clean else ['clean']
        conditions += [
            '{}_{}'.format(degrade_type, severity)
            for severity in args.severities
            for degrade_type in DEGRADE_TYPES]
        for condition in conditions:
            if condition == 'clean':
                config = os.path.join(config_root, model, 'clean.py')
            else:
                degrade_type, severity = condition.rsplit('_', 1)
                config = os.path.join(
                    config_root, model, severity,
                    '{}.py'.format(degrade_type))
            output_dir = os.path.join(args.output_root, model, condition)
            output = os.path.join(output_dir, 'result.pkl')
            log = os.path.join(output_dir, 'test.log')
            command = [
                sys.executable,
                '-m', 'torch.distributed.run',
                '--standalone',
                '--nproc_per_node=1',
                os.path.join('tools', 'test.py'),
                config,
                '-C', checkpoints[model],
                '--out', output,
                '--eval', 'top_k_accuracy', 'mean_class_accuracy']
            runs.append(dict(
                model=model,
                condition=condition,
                config=config,
                checkpoint=checkpoints[model],
                output=output,
                log=log,
                command=command))
    return runs


def main():
    args = parse_args()
    checkpoints = {
        'stgcn': args.stgcn_checkpoint,
        'stgcnpp': args.stgcnpp_checkpoint,
        'ctrgcn': args.ctrgcn_checkpoint,
    }
    runs = _build_runs(args, checkpoints)
    expected_runs = len(MODELS) * (
        (0 if args.skip_clean else 1) +
        len(DEGRADE_TYPES) * len(args.severities))
    assert len(runs) == expected_runs
    _preflight(checkpoints, runs)
    _validate_configs(runs)

    manifest = dict(
        status='dry_run' if args.dry_run else 'running',
        ann_file=_normalized(ANN_FILE),
        runs=runs)
    for run in runs:
        print(' '.join('"{}"'.format(part) if ' ' in part else part
                       for part in run['command']))
    if args.dry_run:
        return

    os.makedirs(args.output_root, exist_ok=True)
    manifest_name = '{}_manifest.json'.format('_'.join(args.severities))
    manifest_path = os.path.join(args.output_root, manifest_name)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    for index, run in enumerate(runs, 1):
        output_dir = os.path.dirname(run['output'])
        os.makedirs(output_dir, exist_ok=True)
        print('[{}/{}] {} {}'.format(
            index, len(runs), run['model'], run['condition']))
        start = time.time()
        with open(run['log'], 'w', encoding='utf-8') as log_file:
            completed = subprocess.run(
                run['command'], stdout=log_file, stderr=subprocess.STDOUT)
        run['returncode'] = completed.returncode
        run['elapsed_seconds'] = time.time() - start
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)
        if completed.returncode != 0:
            manifest['status'] = 'failed'
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=True)
            raise RuntimeError(
                '{} {} failed; inspect {}'.format(
                    run['model'], run['condition'], run['log']))

    manifest['status'] = 'completed'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
    print('PASS: completed all {} runs; manifest: {}'.format(
        len(runs), manifest_path))


if __name__ == '__main__':
    main()
