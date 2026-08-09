# NTU60 XSub 3D Joint Robustness Benchmark

This directory defines the 48-run robustness benchmark for the clean ST-GCN,
ST-GCN++, and CTR-GCN checkpoints. It is evaluation-only: no config in this
directory defines a training dataset, optimizer, or training schedule.

## Frozen protocol

- Dataset: `data/nturgbd/ntu60_3danno.pkl`
- Split: `xsub_val`
- Modality: NTU 25-joint 3D Joint stream
- Sampling: `UniformSample(clip_len=100, num_clips=10, seed=255)`
- Degradation seed: `255`
- Conditions: Clean plus five degradations at Mild, Moderate, and Severe
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

Each leaf `*.py` file is a directly runnable test config. The hierarchy is:

```text
_base_/models/                 shared model structure definitions
_base_/protocols/              shared Clean and degradation data pipelines
stgcn|stgcnpp|ctrgcn/          one directory per clean checkpoint model
  clean.py                     Clean condition
  mild|moderate|severe/        severity level
    joint_missing.py           degradation type
    limb_occlusion.py
    coord_noise.py
    frame_missing.py
    mixed.py
```

Example:

```powershell
python -m torch.distributed.run --standalone --nproc_per_node=1 tools/test.py `
  configs/robust_skeleton/ntu60_xsub_3dkp/stgcn/moderate/frame_missing.py `
  -C CHECKPOINT `
  --out work_dirs/robustness_benchmark/stgcn/frame_missing_moderate/result.pkl `
  --eval top_k_accuracy mean_class_accuracy
```

The three files under `_base_/models` reproduce the model definitions in the
completed clean baseline configs. The sixteen files under `_base_/protocols`
are shared by all models, which prevents condition-specific data or sampling
drift.

## Completing the matrix

After the 18-run Clean + Moderate pilot has completed, run the remaining 30
Mild and Severe conditions without repeating Clean or Moderate:

```powershell
python tools/analysis/run_moderate_robustness.py `
  --stgcn-checkpoint work_dirs/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/j_single_gpu/best_top1_acc_epoch_15.pth `
  --stgcnpp-checkpoint work_dirs/stgcn++/ntu60_xsub_3dkp/j_single_gpu/best_top1_acc_epoch_15.pth `
  --ctrgcn-checkpoint work_dirs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/j_single_gpu/best_top1_acc_epoch_16.pth `
  --severities mild severe `
  --skip-clean
```

The runner records return codes, elapsed time, commands, and result paths in
`work_dirs/robustness_benchmark/mild_severe_manifest.json`.

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
