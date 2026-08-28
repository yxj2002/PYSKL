model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_inner_split.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(
        type='RandomSkeletonDegrade',
        degrade_type='frame_missing',
        prob=1.0,
        dataset='nturgb+d',
        severity_sampling='S3'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10, seed=255),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='PoseDataset',
            ann_file='data/nturgbd/ntu60_inner_split.pkl',
            pipeline=[
                dict(type='PreNormalize3D'),
                dict(
                    type='RandomSkeletonDegrade',
                    degrade_type='frame_missing',
                    prob=1.0,
                    dataset='nturgb+d',
                    severity_sampling='S3'),
                dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
                dict(type='UniformSample', clip_len=100),
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=2),
                dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['keypoint'])
            ],
            split='inner_train')),
    val=dict(
        type='PoseDataset',
        ann_file='data/nturgbd/ntu60_inner_split.pkl',
        pipeline=[
            dict(type='PreNormalize3D'),
            dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
            dict(type='UniformSample', clip_len=100, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='inner_val'),
    test=dict(
        type='PoseDataset',
        ann_file='data/nturgbd/ntu60_inner_split.pkl',
        pipeline=[
            dict(type='PreNormalize3D'),
            dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
            dict(type='UniformSample', clip_len=100, num_clips=10, seed=255),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='inner_val'))
optimizer = dict(
    type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/aug_baseline/stgcnpp_j/train/A4_frame_missing'
dist_params = dict(backend='gloo')
gpu_ids = range(0, 1)
