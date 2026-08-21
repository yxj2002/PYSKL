# coord_noise_moderate test on inner_val
dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_inner_split.pkl'
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomSkeletonDegrade', degrade_type='coord_noise', severity='moderate', prob=1.0, dataset='nturgb+d', seed=255),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10, seed=255),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='inner_val'))
evaluation = dict(metrics=['top_k_accuracy', 'mean_class_accuracy'])
dist_params = dict(backend='gloo')
log_level = 'INFO'
work_dir = './work_dirs/aug_baseline/stgcnpp_j/test/moderate/coord_noise'
