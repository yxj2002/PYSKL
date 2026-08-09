# NTU60 XSub 3D Joint Robustness Pilot

This directory defines the 18-run Moderate pilot for the clean ST-GCN,
ST-GCN++, and CTR-GCN checkpoints. It is evaluation-only: no config in this
directory defines a training dataset, optimizer, or training schedule.

## Frozen protocol

- Dataset: `data/nturgbd/ntu60_3danno.pkl`
- Split: `xsub_val`
- Modality: NTU 25-joint 3D Joint stream
- Sampling: `UniformSample(clip_len=100, num_clips=10, seed=255)`
- Degradation seed: `255`
- Conditions: Clean plus five Moderate degradations
- Metrics: top-k accuracy and mean class accuracy

The degraded pipeline order is:

```text
PreNormalize3D
RandomSkeletonDegrade
GenSkeFeat(j)
UniformSample(100 x 10)
PoseDecode
FormatGCNInput(2 persons)
```

Degradation therefore acts once on the normalized full video sequence. Every
clip that samples the same physical frame observes the same corruption. The
Clean pipeline remains behavior-equivalent to each model's baseline pipeline.

## Config matrix

Each top-level `*.py` file is a directly runnable test config. The names are
the Cartesian product of:

```text
models:     stgcn, stgcnpp, ctrgcn
conditions: clean, joint_missing_moderate, limb_occlusion_moderate,
            coord_noise_moderate, frame_missing_moderate, mixed_moderate
```

Example:

```powershell
python tools/test.py `
  configs/robust_skeleton/ntu60_xsub_3dkp/stgcn_frame_missing_moderate.py `
  -C CHECKPOINT `
  --out work_dirs/robustness_benchmark/stgcn/frame_missing_moderate/result.pkl `
  --eval top_k_accuracy mean_class_accuracy
```

The three files under `_base_/models` reproduce the model definitions in the
completed clean baseline configs. The six files under `_base_/protocols` are
shared by all models, which prevents condition-specific data or sampling drift.

## Preflight audit

Run the 25-sample audit before inference:

```powershell
python tools/analysis/audit_skeleton_degrade.py `
  --ann-file data/nturgbd/ntu60_3danno.pkl `
  --split xsub_val `
  --num-samples 25 `
  --seed 255 `
  --out work_dirs/robustness_benchmark/degrade_audit_real.json
```

`--synthetic` is an implementation-only fallback when the annotation file is
not present. A synthetic pass does not replace the required real-data audit.

Do not use `checkpoints/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/j.pth` for this
matrix. It is a COCO 17-joint 2D checkpoint and is incompatible with this
protocol.
