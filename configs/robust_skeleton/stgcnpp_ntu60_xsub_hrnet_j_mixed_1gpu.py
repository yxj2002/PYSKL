# 本文件用于 NTU RGB+D 60 XSub 的 HRNet 2D 骨架 ST-GCN++ Joint（j）鲁棒训练。
# 它继承干净单卡基线配置，仅在训练流水线中以 80% 概率施加中等强度的混合退化。
# 验证和测试流水线仍继承干净配置；鲁棒性评测请使用对应的 *_test.py 文件。

_base_ = './stgcnpp_ntu60_xsub_hrnet_j_clean_1gpu.py'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='RandomSkeletonDegrade', degrade_type='mixed',
         severity='moderate', prob=0.8, dataset='coco',
         mixed_apply_prob=0.5),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(train=dict(type='RepeatDataset', times=5, dataset=dict(
    type='PoseDataset', ann_file='data/nturgbd/ntu60_hrnet.pkl',
    pipeline=train_pipeline, split='xsub_train')))

work_dir = './work_dirs/robust_skeleton/stgcnpp_ntu60_xsub_hrnet_j_mixed_1gpu'
