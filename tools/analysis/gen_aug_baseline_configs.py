#!/usr/bin/env python
"""Generate stage-4 augmentation-baseline configs (work 1: A0-A6 + test matrix).

Stage 4 trains on ``inner_train`` and tunes on ``inner_val`` (never xsub_val).
Work 1 produces 7 training configs (A0 clean, A1-A4 single degradations,
A5 random_single, A6 mixed) plus the frozen 16-condition test matrix
(1 clean + 5 degradations x 3 severities) evaluated on ``inner_val``.

The base protocol for work 1 is: prob=1.0, severity_sampling=S3 (uniform
mild/moderate/severe), 16 epochs.  Works 2/3/4 later adjust a single variable
(prob / severity sampling / epochs) and can reuse this generator via
``--prob``, ``--severity-sampling`` and ``--epochs``; non-default values are
encoded into the training-config filename so variants never overwrite the
work-1 baseline files.

Only standard-library modules are required, so this runs on the dev host.
"""
import argparse
import os

# A-group -> training degradation type (None = clean, no augmentation).
AUG_GROUPS = {
    'A0': None,
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

ANN_FILE = 'data/nturgbd/ntu60_inner_split.pkl'
MODEL_DEF = """model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out-root', default='configs/aug_baseline/stgcnpp_j')
    parser.add_argument(
        '--ann-file', default=ANN_FILE)
    parser.add_argument(
        '--prob', type=float, default=1.0,
        help='Training degradation probability (work 2 varies this).')
    parser.add_argument(
        '--severity-sampling', default='S3', choices=['fixed', 'S3', 'S5'],
        help='Training severity sampling mode (work 3 varies this).')
    parser.add_argument(
        '--epochs', type=int, default=16,
        help='Training epochs (work 4 varies this).')
    parser.add_argument(
        '--train-configs', nargs='+', default=list(AUG_GROUPS.keys()),
        help='A groups to emit training configs for (default: all 7).')
    parser.add_argument(
        '--skip-train', action='store_true',
        help='Only emit the 16-condition test matrix.')
    parser.add_argument(
        '--skip-test', action='store_true',
        help='Only emit training configs.')
    return parser.parse_args()


def degrade_step(degrade_type, prob, sampling):
    """Build the train-pipeline RandomSkeletonDegrade dict (or None if clean)."""
    if degrade_type is None:
        return None
    step = dict(
        type='RandomSkeletonDegrade',
        degrade_type=degrade_type,
        prob=prob,
        dataset='nturgb+d')
    if sampling == 'fixed':
        step['severity'] = 'moderate'
    else:
        step['severity_sampling'] = sampling
    if degrade_type == 'mixed':
        step['mixed_apply_prob'] = 0.5
    return step


def train_filename(group, tag, prob, sampling, epochs):
    base = '{}_{}'.format(group, tag)
    suffix = ''
    if prob != 1.0:
        suffix += '_p{}'.format(prob)
    if sampling != 'S3':
        suffix += '_{}'.format(sampling.lower())
    if epochs != 16:
        suffix += '_e{}'.format(epochs)
    return base + suffix + '.py'


def render_train_config(group, degrade_type, prob, sampling, epochs, ann_file,
                        work_dir):
    degrade = degrade_step(degrade_type, prob, sampling)
    if degrade is None:
        train_aug = ''
    else:
        args = ', '.join(
            "{}={}".format(k, repr(v)) for k, v in degrade.items()
            if k != 'type')
        train_aug = "    dict(type='{}', {}),\n".format(degrade['type'], args)

    tag = group.replace('A', '').lower()
    if degrade_type is None:
        tag = 'clean'
    header = (
        '# {}: {} (stage-4 augmentation baseline)\n'
        '# protocol: prob={}, severity_sampling={}, {} epochs, '
        'inner_train/inner_val\n'.format(
            group, degrade_type or 'clean', prob, sampling, epochs))

    train_pipeline = (
        "train_pipeline = [\n"
        "    dict(type='PreNormalize3D'),\n"
        + train_aug +
        "    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),\n"
        "    dict(type='UniformSample', clip_len=100),\n"
        "    dict(type='PoseDecode'),\n"
        "    dict(type='FormatGCNInput', num_person=2),\n"
        "    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),\n"
        "    dict(type='ToTensor', keys=['keypoint'])\n"
        "]\n")

    config = (
        header + '\n' + MODEL_DEF +
        "dataset_type = 'PoseDataset'\n"
        "ann_file = '{ann_file}'\n\n"
        "{train_pipeline}\n"
        "val_pipeline = [\n"
        "    dict(type='PreNormalize3D'),\n"
        "    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),\n"
        "    dict(type='UniformSample', clip_len=100, num_clips=1),\n"
        "    dict(type='PoseDecode'),\n"
        "    dict(type='FormatGCNInput', num_person=2),\n"
        "    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),\n"
        "    dict(type='ToTensor', keys=['keypoint'])\n"
        "]\n"
        "test_pipeline = [\n"
        "    dict(type='PreNormalize3D'),\n"
        "    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),\n"
        "    dict(type='UniformSample', clip_len=100, num_clips=10, seed=255),\n"
        "    dict(type='PoseDecode'),\n"
        "    dict(type='FormatGCNInput', num_person=2),\n"
        "    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),\n"
        "    dict(type='ToTensor', keys=['keypoint'])\n"
        "]\n"
        "data = dict(\n"
        "    videos_per_gpu=32,\n"
        "    workers_per_gpu=2,\n"
        "    test_dataloader=dict(videos_per_gpu=1),\n"
        "    train=dict(\n"
        "        type='RepeatDataset',\n"
        "        times=5,\n"
        "        dataset=dict(\n"
        "            type='PoseDataset',\n"
        "            ann_file=ann_file,\n"
        "            pipeline=train_pipeline,\n"
        "            split='inner_train')),\n"
        "    val=dict(\n"
        "        type='PoseDataset',\n"
        "        ann_file=ann_file,\n"
        "        pipeline=val_pipeline,\n"
        "        split='inner_val'),\n"
        "    test=dict(\n"
        "        type='PoseDataset',\n"
        "        ann_file=ann_file,\n"
        "        pipeline=test_pipeline,\n"
        "        split='inner_val'))\n"
        "optimizer = dict(\n"
        "    type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, "
        "nesterov=True)\n"
        "optimizer_config = dict(grad_clip=None)\n"
        "lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)\n"
        "total_epochs = {epochs}\n"
        "checkpoint_config = dict(interval=1)\n"
        "evaluation = dict(interval=1, metrics=['top_k_accuracy'])\n"
        "log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])\n"
        "log_level = 'INFO'\n"
        "work_dir = '{work_dir}'\n"
        "dist_params = dict(backend='gloo')\n"
        "gpu_ids = range(0, 1)\n"
    ).format(
        ann_file=ann_file,
        train_pipeline=train_pipeline,
        epochs=epochs,
        work_dir=work_dir)

    return config


def render_test_config(degrade_type, severity, ann_file, work_dir):
    if severity is None:
        # clean condition
        degrade_line = ''
        condition = 'clean'
        header = '# clean (no degradation) test on inner_val\n'
    else:
        degrade = dict(
            type='RandomSkeletonDegrade',
            degrade_type=degrade_type,
            severity=severity,
            prob=1.0,
            dataset='nturgb+d',
            seed=255)
        if degrade_type == 'mixed':
            degrade['mixed_apply_prob'] = 0.5
        args = ', '.join(
            "{}={}".format(k, repr(v)) for k, v in degrade.items()
            if k != 'type')
        degrade_line = "    dict(type='{}', {}),\n".format(
            degrade['type'], args)
        condition = '{}_{}'.format(degrade_type, severity)
        header = '# {} test on inner_val\n'.format(condition)

    config = (
        header +
        MODEL_DEF +
        "dataset_type = 'PoseDataset'\n"
        "ann_file = '{ann_file}'\n"
        "test_pipeline = [\n"
        "    dict(type='PreNormalize3D'),\n"
        + degrade_line +
        "    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),\n"
        "    dict(type='UniformSample', clip_len=100, num_clips=10, seed=255),\n"
        "    dict(type='PoseDecode'),\n"
        "    dict(type='FormatGCNInput', num_person=2),\n"
        "    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),\n"
        "    dict(type='ToTensor', keys=['keypoint'])\n"
        "]\n"
        "data = dict(\n"
        "    videos_per_gpu=16,\n"
        "    workers_per_gpu=2,\n"
        "    test_dataloader=dict(videos_per_gpu=1),\n"
        "    test=dict(\n"
        "        type=dataset_type,\n"
        "        ann_file=ann_file,\n"
        "        pipeline=test_pipeline,\n"
        "        split='inner_val'))\n"
        "evaluation = dict(metrics=['top_k_accuracy', 'mean_class_accuracy'])\n"
        "dist_params = dict(backend='gloo')\n"
        "log_level = 'INFO'\n"
        "work_dir = '{work_dir}'\n"
    ).format(ann_file=ann_file, work_dir=work_dir)

    return config


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('  wrote {}'.format(path))


def main():
    args = parse_args()

    if not args.skip_train:
        train_root = os.path.join(args.out_root, 'train')
        for group in args.train_configs:
            if group not in AUG_GROUPS:
                raise ValueError('Unknown group {}'.format(group))
            degrade_type = AUG_GROUPS[group]
            tag = degrade_type or 'clean'
            fname = train_filename(
                group, tag, args.prob, args.severity_sampling, args.epochs)
            # work_dir must match the config filename (including prob/sampling/
            # epoch suffixes) so variants never overwrite each other's
            # checkpoints.
            dir_name = os.path.splitext(fname)[0]
            work_dir = './work_dirs/aug_baseline/stgcnpp_j/train/{}'.format(
                dir_name)
            content = render_train_config(
                group, degrade_type, args.prob, args.severity_sampling,
                args.epochs, args.ann_file, work_dir)
            write(os.path.join(train_root, fname), content)

    if not args.skip_test:
        test_root = os.path.join(args.out_root, 'test')
        write(
            os.path.join(test_root, 'clean.py'),
            render_test_config(
                None, None, args.ann_file,
                './work_dirs/aug_baseline/stgcnpp_j/test/clean'))
        for severity in SEVERITIES:
            for degrade_type in DEGRADE_TYPES:
                fname = '{}.py'.format(degrade_type)
                content = render_test_config(
                    degrade_type, severity, args.ann_file,
                    './work_dirs/aug_baseline/stgcnpp_j/test/{}/{}'.format(
                        severity, degrade_type))
                write(os.path.join(test_root, severity, fname), content)

    print('Done. train root: {} | test root: {}'.format(
        os.path.join(args.out_root, 'train'),
        os.path.join(args.out_root, 'test')))


if __name__ == '__main__':
    main()
