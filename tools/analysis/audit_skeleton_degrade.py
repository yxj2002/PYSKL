#!/usr/bin/env python
"""Audit the NTU 3D skeleton degradation protocol on a small sample."""

import argparse
import copy
import importlib.util
import json
import os
import pickle
import sys
import types

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_pipeline_classes():
    try:
        from pyskl.datasets.pipelines.pose_related import (
            PoseDecode, PreNormalize3D, RandomSkeletonDegrade)
        from pyskl.datasets.pipelines.sampling import UniformSample
        return PoseDecode, PreNormalize3D, RandomSkeletonDegrade, UniformSample
    except ModuleNotFoundError as error:
        if error.name != 'mmcv':
            raise

    # Auditing the NumPy transforms does not require MMCV or Torch. These
    # minimal modules let the exact repository source load in a CPU-only
    # environment; formal inference must still use the full training env.
    class RegistryStub:
        def register_module(self, module=None, **kwargs):
            del kwargs
            if module is not None:
                return module

            def decorator(cls):
                return cls
            return decorator

    class ComposeStub:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, results):
            for transform in self.transforms:
                results = transform(results)
            return results

    class RenameStub:
        def __init__(self, mapping):
            self.mapping = mapping

        def __call__(self, results):
            for source, target in self.mapping.items():
                results[target] = results.pop(source)
            return results

    for name in list(sys.modules):
        if name == 'pyskl' or name.startswith('pyskl.'):
            del sys.modules[name]
    package_names = ['pyskl', 'pyskl.datasets', 'pyskl.datasets.pipelines']
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    builder = types.ModuleType('pyskl.datasets.builder')
    builder.PIPELINES = RegistryStub()
    sys.modules[builder.__name__] = builder
    compose = types.ModuleType('pyskl.datasets.pipelines.compose')
    compose.Compose = ComposeStub
    sys.modules[compose.__name__] = compose
    formatting = types.ModuleType('pyskl.datasets.pipelines.formatting')
    formatting.Rename = RenameStub
    sys.modules[formatting.__name__] = formatting
    utils = types.ModuleType('pyskl.utils')
    utils.warning_r0 = lambda message: None
    sys.modules[utils.__name__] = utils

    def load_module(name, relative_path):
        path = os.path.join(REPO_ROOT, relative_path)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    pose_related = load_module(
        'pyskl.datasets.pipelines.pose_related',
        os.path.join('pyskl', 'datasets', 'pipelines', 'pose_related.py'))
    sampling = load_module(
        'pyskl.datasets.pipelines.sampling',
        os.path.join('pyskl', 'datasets', 'pipelines', 'sampling.py'))
    return (pose_related.PoseDecode, pose_related.PreNormalize3D,
            pose_related.RandomSkeletonDegrade, sampling.UniformSample)


(PoseDecode, PreNormalize3D,
 RandomSkeletonDegrade, UniformSample) = _load_pipeline_classes()


DEGRADE_TYPES = (
    'joint_missing', 'limb_occlusion', 'coord_noise',
    'frame_missing', 'mixed')
SEVERITIES = ('mild', 'moderate', 'severe')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit RandomSkeletonDegrade on 20-30 NTU xsub_val samples.')
    parser.add_argument(
        '--ann-file', default='data/nturgbd/ntu60_3danno.pkl')
    parser.add_argument('--split', default='xsub_val')
    parser.add_argument('--num-samples', type=int, default=25)
    parser.add_argument('--seed', type=int, default=255)
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use deterministic NTU-shaped samples when the annotation is unavailable.')
    parser.add_argument(
        '--out', default='work_dirs/robustness_benchmark/degrade_audit.json')
    return parser.parse_args()


def _sample_id(sample, index):
    return str(sample.get('frame_dir', sample.get('filename', index)))


def _load_real_samples(ann_file, split_name, num_samples, seed):
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    annotations = data['annotations']
    identifier = 'filename' if 'filename' in annotations[0] else 'frame_dir'
    split = set(data['split'][split_name])
    annotations = [item for item in annotations if item[identifier] in split]
    if num_samples > len(annotations):
        raise ValueError('Requested {} samples, but {} are available'.format(
            num_samples, len(annotations)))
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(annotations), size=num_samples, replace=False)
    return [copy.deepcopy(annotations[index]) for index in indices]


def _make_synthetic_samples(num_samples, seed):
    rng = np.random.RandomState(seed)
    samples = []
    for index in range(num_samples):
        num_frames = 48 + index * 3
        keypoint = np.zeros((2, num_frames, 25, 3), dtype=np.float32)
        base = rng.uniform(-0.8, 0.8, size=(25, 3)).astype(np.float32)
        base[0] = [0.0, 0.0, 0.2]
        base[1] = [0.0, 0.1, 0.6]
        base[4] = [-0.3, 0.1, 0.8]
        base[8] = [0.3, 0.1, 0.8]
        time = np.linspace(0, 2 * np.pi, num_frames, dtype=np.float32)
        motion = np.stack(
            [0.05 * np.sin(time), 0.03 * np.cos(time), 0.02 * np.sin(2 * time)],
            axis=-1)
        keypoint[0] = base[None, :, :] + motion[:, None, :]
        if index % 3:
            visible = num_frames if index % 2 else num_frames // 2
            keypoint[1, :visible] = keypoint[0, :visible] + np.array(
                [0.6, 0.0, 0.0], dtype=np.float32)
        samples.append(dict(
            frame_dir='synthetic_{:03d}'.format(index),
            label=index % 60,
            total_frames=num_frames,
            keypoint=keypoint))
    return samples


def _prepare(sample):
    sample = copy.deepcopy(sample)
    sample.setdefault('start_index', 0)
    sample['keypoint'] = np.asarray(sample['keypoint'], dtype=np.float32)
    return PreNormalize3D()(sample)


def _valid_mask(keypoint):
    return np.abs(keypoint).sum(axis=-1) > 1e-6


def _assert_invariants(before, after):
    assert after['label'] == before['label']
    assert after['total_frames'] == before['total_frames']
    assert after['keypoint'].shape == before['keypoint'].shape


def _assert_padding_unchanged(before_keypoint, after_keypoint):
    padding = np.all(before_keypoint == 0, axis=-1)
    assert np.array_equal(after_keypoint[padding], before_keypoint[padding])


def _apply(before, degrade_type, severity, seed):
    transform = RandomSkeletonDegrade(
        degrade_type=degrade_type,
        severity=severity,
        prob=1.0,
        dataset='nturgb+d',
        seed=seed,
        return_mask=True)
    return transform(copy.deepcopy(before))


def _joint_missing_metrics(before, after, severity):
    valid_before = _valid_mask(before['keypoint'])
    valid_after = _valid_mask(after['keypoint'])
    removed = valid_before & ~valid_after
    assert np.array_equal(removed, after['degrade_mask'])
    ratio = float(removed.sum()) / max(int(valid_before.sum()), 1)
    target = RandomSkeletonDegrade._SEVERITY_PRESETS[severity][
        'joint_missing_ratio']
    count = max(int(valid_before.sum()), 1)
    tolerance = max(0.04, 4.0 * np.sqrt(target * (1.0 - target) / count))
    assert abs(ratio - target) <= tolerance, (ratio, target, tolerance)
    return dict(effective_missing_ratio=ratio, target_ratio=target)


def _limb_metrics(before, after):
    parts = RandomSkeletonDegrade._PARTS['nturgb+d']
    selected = after['degrade_parts']
    assert selected
    selected_joints = sorted({joint for part in selected for joint in parts[part]})
    valid_before = _valid_mask(before['keypoint'])
    expected = np.zeros(valid_before.shape, dtype=bool)
    expected[:, :, selected_joints] = True
    expected &= valid_before
    assert np.array_equal(after['degrade_mask'], expected)
    assert np.all(after['keypoint'][expected] == 0)
    assert np.array_equal(after['keypoint'][~expected], before['keypoint'][~expected])
    return dict(parts=selected, joint_indices=selected_joints)


def _noise_metrics(before, after, severity):
    valid = _valid_mask(before['keypoint'])
    assert np.array_equal(valid, after['degrade_mask'])
    delta = after['keypoint'][valid] - before['keypoint'][valid]
    scale = float(after['degrade_coord_scale'])
    normalized_rms = float(np.sqrt(np.mean(delta ** 2)) / scale)
    target = RandomSkeletonDegrade._SEVERITY_PRESETS[severity][
        'coord_noise_sigma']
    assert abs(normalized_rms - target) <= max(0.005, target * 0.2)
    return dict(
        skeleton_scale=scale,
        normalized_noise_rms=normalized_rms,
        target_sigma=target)


def _frame_metrics(before, after, severity):
    start, end = after['degrade_frame_range']
    total_frames = before['keypoint'].shape[1]
    target = RandomSkeletonDegrade._SEVERITY_PRESETS[severity][
        'frame_missing_ratio']
    expected_len = int(round(total_frames * target))
    assert 0 <= start < end <= total_frames
    assert end - start == expected_len
    expected = np.zeros(_valid_mask(before['keypoint']).shape, dtype=bool)
    expected[:, start:end, :] = True
    expected &= _valid_mask(before['keypoint'])
    assert np.array_equal(after['degrade_mask'], expected)
    assert np.all(after['keypoint'][:, start:end] == 0)
    outside = np.ones(total_frames, dtype=bool)
    outside[start:end] = False
    assert np.array_equal(
        after['keypoint'][:, outside], before['keypoint'][:, outside])
    return dict(
        frame_range=[start, end],
        nominal_missing_ratio=float(end - start) / total_frames,
        target_ratio=target)


def _mixed_metrics(before, after, severity, seed):
    repeated = _apply(before, 'mixed', severity, seed)
    assert after['degrade_types']
    assert after['degrade_types'] == repeated['degrade_types']
    assert np.array_equal(after['keypoint'], repeated['keypoint'])
    assert np.array_equal(after['degrade_mask'], repeated['degrade_mask'])
    return dict(applied_types=after['degrade_types'])


def _audit_part_definitions(before, seed):
    parts = RandomSkeletonDegrade._PARTS['nturgb+d']
    all_joints = [joint for joints in parts.values() for joint in joints]
    assert len(all_joints) == len(set(all_joints)) == 25
    assert sorted(all_joints) == list(range(25))
    checks = {}
    for part, expected_joints in parts.items():
        transform = RandomSkeletonDegrade(
            degrade_type='limb_occlusion',
            severity='moderate',
            prob=1.0,
            dataset='nturgb+d',
            occluded_parts=part,
            seed=seed,
            return_mask=True)
        after = transform(copy.deepcopy(before))
        metrics = _limb_metrics(before, after)
        assert metrics['parts'] == [part]
        assert metrics['joint_indices'] == expected_joints
        checks[part] = expected_joints
    return checks


def _audit_multiclip(before, seed):
    degraded = _apply(before, 'frame_missing', 'moderate', seed)
    sample_input = copy.deepcopy(degraded)
    sample_input['test_mode'] = True
    sampled = UniformSample(clip_len=100, num_clips=10, seed=seed)(sample_input)
    frame_inds = sampled['frame_inds'].copy()
    decoded = PoseDecode()(sampled)
    expected = degraded['keypoint'][:, frame_inds].astype(np.float32)
    assert frame_inds.shape == (1000, )
    assert np.array_equal(decoded['keypoint'], expected)
    start, end = degraded['degrade_frame_range']
    sampled_missing = (frame_inds >= start) & (frame_inds < end)
    return dict(
        original_frame_range=[start, end],
        sampled_missing_ratio=float(sampled_missing.mean()),
        num_clips=10,
        clip_len=100)


def audit_sample(sample, index, seed):
    before = _prepare(sample)
    record = dict(
        sample_id=_sample_id(before, index),
        label=int(before['label']),
        num_person=int(before['keypoint'].shape[0]),
        num_frames=int(before['keypoint'].shape[1]),
        num_joints=int(before['keypoint'].shape[2]),
        conditions={})
    assert before['keypoint'].shape[2:] == (25, 3)

    noise_rms = []
    for degrade_type in DEGRADE_TYPES:
        for severity in SEVERITIES:
            after = _apply(before, degrade_type, severity, seed)
            _assert_invariants(before, after)
            _assert_padding_unchanged(before['keypoint'], after['keypoint'])
            if degrade_type == 'joint_missing':
                metrics = _joint_missing_metrics(before, after, severity)
            elif degrade_type == 'limb_occlusion':
                metrics = _limb_metrics(before, after)
            elif degrade_type == 'coord_noise':
                metrics = _noise_metrics(before, after, severity)
                noise_rms.append(metrics['normalized_noise_rms'])
            elif degrade_type == 'frame_missing':
                metrics = _frame_metrics(before, after, severity)
            else:
                metrics = _mixed_metrics(before, after, severity, seed)
            record['conditions']['{}_{}'.format(degrade_type, severity)] = metrics

    assert noise_rms[0] < noise_rms[1] < noise_rms[2]
    record['multiclip'] = _audit_multiclip(before, seed)
    return record


def summarize(records):
    summary = {}
    for severity in SEVERITIES:
        key = 'joint_missing_{}'.format(severity)
        values = [item['conditions'][key]['effective_missing_ratio']
                  for item in records]
        summary[key] = dict(mean_effective_missing_ratio=float(np.mean(values)))
        key = 'coord_noise_{}'.format(severity)
        values = [item['conditions'][key]['normalized_noise_rms']
                  for item in records]
        summary[key] = dict(mean_normalized_noise_rms=float(np.mean(values)))
        key = 'frame_missing_{}'.format(severity)
        values = [item['conditions'][key]['nominal_missing_ratio']
                  for item in records]
        summary[key] = dict(mean_nominal_missing_ratio=float(np.mean(values)))
    return summary


def main():
    args = parse_args()
    if not 20 <= args.num_samples <= 30:
        raise ValueError('--num-samples must be between 20 and 30')
    if args.synthetic:
        samples = _make_synthetic_samples(args.num_samples, args.seed)
        source = 'synthetic'
    else:
        if not os.path.isfile(args.ann_file):
            raise FileNotFoundError(
                '{} is unavailable; restore the NTU 3D annotation or use '
                '--synthetic for an implementation-only audit'.format(args.ann_file))
        samples = _load_real_samples(
            args.ann_file, args.split, args.num_samples, args.seed)
        source = os.path.abspath(args.ann_file)

    records = [audit_sample(sample, index, args.seed)
               for index, sample in enumerate(samples)]
    part_definitions = _audit_part_definitions(_prepare(samples[0]), args.seed)
    report = dict(
        status='passed',
        source=source,
        split=args.split,
        seed=args.seed,
        num_samples=len(records),
        pipeline_order=[
            'PreNormalize3D', 'RandomSkeletonDegrade', 'GenSkeFeat',
            'UniformSample(num_clips=10)', 'PoseDecode', 'FormatGCNInput'],
        part_definitions=part_definitions,
        summary=summarize(records),
        samples=records)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print('PASS: audited {} {} samples; report: {}'.format(
        len(records), source, args.out))


if __name__ == '__main__':
    main()
