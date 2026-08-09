#!/usr/bin/env python
"""Run the frozen 18-condition NTU60 xsub 3D Joint pilot matrix."""

import argparse
import json
import os
import subprocess
import sys
import time


MODELS = ('stgcn', 'stgcnpp', 'ctrgcn')
ANN_FILE = os.path.join('data', 'nturgbd', 'ntu60_3danno.pkl')
CONDITIONS = (
    'clean',
    'joint_missing_moderate',
    'limb_occlusion_moderate',
    'coord_noise_moderate',
    'frame_missing_moderate',
    'mixed_moderate')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stgcn-checkpoint', required=True)
    parser.add_argument('--stgcnpp-checkpoint', required=True)
    parser.add_argument('--ctrgcn-checkpoint', required=True)
    parser.add_argument(
        '--output-root', default='work_dirs/robustness_benchmark')
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
    expected_degrade = {
        'joint_missing_moderate': 'joint_missing',
        'limb_occlusion_moderate': 'limb_occlusion',
        'coord_noise_moderate': 'coord_noise',
        'frame_missing_moderate': 'frame_missing',
        'mixed_moderate': 'mixed',
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
            degrade = next(
                item for item in pipeline
                if item['type'] == 'RandomSkeletonDegrade')
            assert degrade['degrade_type'] == expected_degrade[run['condition']]
            assert degrade['severity'] == 'moderate'
            assert degrade['dataset'] == 'nturgb+d'
            assert degrade['prob'] == 1.0
            assert degrade['seed'] == 255


def _build_runs(args, checkpoints):
    config_root = os.path.join(
        'configs', 'robust_skeleton', 'ntu60_xsub_3dkp')
    runs = []
    for model in MODELS:
        for condition in CONDITIONS:
            config = os.path.join(
                config_root, '{}_{}.py'.format(model, condition))
            output_dir = os.path.join(args.output_root, model, condition)
            output = os.path.join(output_dir, 'result.pkl')
            log = os.path.join(output_dir, 'test.log')
            command = [
                sys.executable,
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
    assert len(runs) == 18
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
    manifest_path = os.path.join(args.output_root, 'moderate_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    for index, run in enumerate(runs, 1):
        output_dir = os.path.dirname(run['output'])
        os.makedirs(output_dir, exist_ok=True)
        print('[{}/18] {} {}'.format(index, run['model'], run['condition']))
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
    print('PASS: completed all 18 runs; manifest: {}'.format(manifest_path))


if __name__ == '__main__':
    main()
