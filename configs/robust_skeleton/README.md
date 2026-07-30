# Robust Skeleton Degradation

This folder contains starter configs for low-quality skeleton action recognition.

The degradation transform is `RandomSkeletonDegrade`, implemented in
`pyskl/datasets/pipelines/pose_related.py`. It supports:

- `joint_missing`
- `limb_occlusion`
- `coord_noise`
- `frame_missing`
- `mixed`

Recommended usage:

1. Train with `mixed` degradation to improve robustness.
2. Keep validation clean to monitor normal recognition accuracy.
3. Test with one degradation type at a time and fixed `seed` for reproducible robustness evaluation.

Change these fields in the config to build different test protocols:

```python
test_degrade_type = 'mixed'
test_severity = 'moderate'
```

Available severity levels are `mild`, `moderate`, and `severe`.

## NTU60 HRNet 2D protocol

The `stgcnpp_ntu60_xsub_hrnet_j_clean_1gpu.py` configuration is a clean
single-GPU ST-GCN++ Joint baseline for the included `ntu60_hrnet.pkl` file.
`stgcnpp_ntu60_xsub_hrnet_j_mixed_1gpu.py` uses the same model and training
schedule, but applies mixed moderate degradation to 80 percent of training
clips. Validation remains clean in both configurations.

The four `*_moderate_test.py` files evaluate a checkpoint under exactly one
moderate degradation. They use `dataset='coco'`, matching HRNet's 17 COCO
keypoints, and `seed=255` so every checkpoint sees the same corruption.
To evaluate mild or severe conditions, replace `severity='moderate'` in the
corresponding test configuration. The clean training configuration itself is
also the clean test protocol.

The transform is deliberately placed after `PoseDecode` and before
`GenSkeFeat`: corrupt raw joint detections first, then derive the Joint, Bone,
Joint Motion, or Bone Motion representation. This ordering is required when
the protocol is extended beyond the current Joint (`j`) configurations.
