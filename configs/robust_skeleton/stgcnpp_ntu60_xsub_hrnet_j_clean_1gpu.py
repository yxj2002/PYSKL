# 本文件用于 NTU RGB+D 60 XSub 的 HRNet 2D 骨架 ST-GCN++ Joint（j）干净基线。
# 面向 Windows 单张 GPU：每卡 batch size 为 16，学习率按官方 8 卡方案线性缩放为 0.0125。
# 它既是干净模型的训练配置，也是所有低质量骨架测试配置的公共基础配置。

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN', gcn_adaptive='init', gcn_with_res=True,
        tcn_type='mstcn', graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_hrnet.pkl'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=0,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type='RepeatDataset', times=5, dataset=dict(
        type=dataset_type, ann_file=ann_file, pipeline=train_pipeline,
        split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline,
             split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline,
              split='xsub_val'))

optimizer = dict(type='SGD', lr=0.0125, momentum=0.9, weight_decay=0.0005,
                 nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='gloo')
log_level = 'INFO'
work_dir = './work_dirs/robust_skeleton/stgcnpp_ntu60_xsub_hrnet_j_clean_1gpu'
