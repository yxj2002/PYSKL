# 本文件用于评测 ST-GCN++ Joint（j）在 HRNet 2D 骨架“随机关节缺失”下的鲁棒性。
# 测试集每个样本都施加中等强度退化，固定 seed=255，保证不同 checkpoint 的结果可公平比较。
# 可分别加载干净模型或 mixed 增强模型的 checkpoint，比较退化前后的准确率。

_base_ = './stgcnpp_ntu60_xsub_hrnet_j_clean_1gpu.py'

test_pipeline = [
    dict(type='PreNormalize2D'), dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'), dict(type='RandomSkeletonDegrade',
        degrade_type='joint_missing', severity='moderate', prob=1.0,
        dataset='coco', seed=255), dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]), dict(type='ToTensor', keys=['keypoint'])]
data = dict(test=dict(type='PoseDataset', ann_file='data/nturgbd/ntu60_hrnet.pkl', pipeline=test_pipeline, split='xsub_val'))
