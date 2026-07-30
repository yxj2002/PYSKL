# 本文件用于 NTU RGB+D 60 XSub 官方 3D 骨架的 ST-GCN++ Joint 鲁棒训练与测试示例。
# 训练集采用混合退化增强，验证集保持干净，测试集使用固定种子的混合退化。
# 运行前需要准备 data/nturgbd/ntu60_3danno.pkl，当前工作区的 HRNet 2D 数据不能直接使用本配置。

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
ann_file = 'data/nturgbd/ntu60_3danno.pkl'

train_degrade_type = 'mixed'
train_severity = 'moderate'
test_degrade_type = 'mixed'
test_severity = 'moderate'

train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(
        type='RandomSkeletonDegrade',
        degrade_type=train_degrade_type,
        severity=train_severity,
        prob=0.8,
        dataset='nturgb+d',
        mixed_apply_prob=0.5),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(
        type='RandomSkeletonDegrade',
        degrade_type=test_degrade_type,
        severity=test_severity,
        prob=1.0,
        dataset='nturgb+d',
        mixed_apply_prob=0.5,
        seed=255),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

log_level = 'INFO'
work_dir = './work_dirs/robust_skeleton/stgcnpp_ntu60_xsub_3dkp_robust'
