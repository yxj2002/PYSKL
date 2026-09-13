#!/usr/bin/env python
"""Generate work-5 B_aug final training configs (full xsub_train protocol).

B_aug (frozen from work 1-4):
    degrade_type=mixed, prob=0.3, severity_sampling='S3', 16 epochs.

Work 5 trains 4 models on the FULL xsub_train and evaluates on xsub_val:
    - ST-GCN++ B_aug x seeds {255, 2026, 3407}
    - CTR-GCN  B_aug x seed 255

Configs follow the stage-2 clean-baseline protocol exactly (batch 32, lr 0.025,
16 epoch CosineAnnealing, RepeatDataset x5, GenSkeFeat j) and only add
``RandomSkeletonDegrade`` into the train pipeline (before UniformSample).

Outputs:
    configs/aug_baseline/b_aug/stgcnpp_j/b_aug_seed{255,2026,3407}.py
    configs/aug_baseline/b_aug/ctrgcn_j/b_aug_seed255.py
"""
import os

ANN_FILE = 'data/nturgbd/ntu60_3danno.pkl'

STGCNPP_MODEL = """model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))"""

CTRGCN_MODEL = """model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN',
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))"""

# B_aug augmentation block: outer (train_pipeline) uses 4-space indent,
# nested (dataset.pipeline) uses 16-space indent.
DEGRADE_LINE_OUTER = ("    dict(type='RandomSkeletonDegrade', degrade_type='mixed', "
                      "prob=0.3, dataset='nturgb+d', severity_sampling='S3', "
                      "mixed_apply_prob=0.5),")
DEGRADE_LINE_INNER = ("                dict(type='RandomSkeletonDegrade', "
                      "degrade_type='mixed', prob=0.3, dataset='nturgb+d', "
                      "severity_sampling='S3', mixed_apply_prob=0.5),")


def render(model_def, work_dir, comment):
    return (
        comment + '\n' + model_def + '\n' +
        "dataset_type = 'PoseDataset'\n"
        "ann_file = '{}'\n".format(ANN_FILE) +
        "train_pipeline = [\n"
        "    dict(type='PreNormalize3D'),\n"
        + DEGRADE_LINE_OUTER + '\n' +
        "    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),\n"
        "    dict(type='UniformSample', clip_len=100),\n"
        "    dict(type='PoseDecode'),\n"
        "    dict(type='FormatGCNInput', num_person=2),\n"
        "    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),\n"
        "    dict(type='ToTensor', keys=['keypoint'])\n"
        "]\n"
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
        "            pipeline=[\n"
        "                dict(type='PreNormalize3D'),\n"
        + DEGRADE_LINE_INNER + '\n' +
        "                dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),\n"
        "                dict(type='UniformSample', clip_len=100),\n"
        "                dict(type='PoseDecode'),\n"
        "                dict(type='FormatGCNInput', num_person=2),\n"
        "                dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),\n"
        "                dict(type='ToTensor', keys=['keypoint'])\n"
        "            ],\n"
        "            split='xsub_train')),\n"
        "    val=dict(\n"
        "        type='PoseDataset',\n"
        "        ann_file=ann_file,\n"
        "        pipeline=val_pipeline,\n"
        "        split='xsub_val'),\n"
        "    test=dict(\n"
        "        type='PoseDataset',\n"
        "        ann_file=ann_file,\n"
        "        pipeline=test_pipeline,\n"
        "        split='xsub_val'))\n"
        "optimizer = dict(\n"
        "    type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, "
        "nesterov=True)\n"
        "optimizer_config = dict(grad_clip=None)\n"
        "lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)\n"
        "total_epochs = 16\n"
        "checkpoint_config = dict(interval=1)\n"
        "evaluation = dict(interval=1, metrics=['top_k_accuracy'])\n"
        "log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])\n"
        "log_level = 'INFO'\n"
        "work_dir = '{}'\n".format(work_dir) +
        "dist_params = dict(backend='gloo')\n"
        "gpu_ids = range(0, 1)\n")


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('  wrote {}'.format(path))


def main():
    out_root = os.path.join('configs', 'aug_baseline', 'b_aug')

    for seed in (255, 2026, 3407):
        fname = os.path.join(out_root, 'stgcnpp_j', 'b_aug_seed{}.py'.format(seed))
        write(fname, render(
            STGCNPP_MODEL,
            './work_dirs/aug_baseline/b_aug/stgcnpp_j/seed{}'.format(seed),
            '# B_aug (mixed + prob=0.3 + S3 + 16 epoch), ST-GCN++ J, seed {}'.format(seed)))

    write(os.path.join(out_root, 'ctrgcn_j', 'b_aug_seed255.py'), render(
        CTRGCN_MODEL,
        './work_dirs/aug_baseline/b_aug/ctrgcn_j/seed255',
        '# B_aug (mixed + prob=0.3 + S3 + 16 epoch), CTR-GCN J, seed 255'))


if __name__ == '__main__':
    main()
