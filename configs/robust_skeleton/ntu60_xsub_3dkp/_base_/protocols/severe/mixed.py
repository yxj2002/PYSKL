dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'

test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomSkeletonDegrade', degrade_type='mixed', severity='severe',
         prob=1.0, dataset='nturgb+d', mixed_apply_prob=0.5, seed=255),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10, seed=255),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(videos_per_gpu=16, workers_per_gpu=2,
            test_dataloader=dict(videos_per_gpu=1),
            test=dict(type=dataset_type, ann_file=ann_file,
                      pipeline=test_pipeline, split='xsub_val'))
evaluation = dict(metrics=['top_k_accuracy', 'mean_class_accuracy'])
dist_params = dict(backend='gloo')
log_level = 'INFO'
