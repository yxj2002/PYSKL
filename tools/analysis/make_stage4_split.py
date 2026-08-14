"""Generate inner_val split from xsub_train for stage-4 tuning.

Why: stages 4-8 need a held-out tuning set (augmentation probability, severity
distribution, epoch selection). Using xsub_val for tuning would leak test-set
information into the final reported numbers. We therefore draw a subject-based
inner_val from xsub_train and verify full class coverage.

The output is a pkl compatible with PYSKL's PoseDataset:
    {'split': {'inner_train': [...], 'inner_val': [...]}}
so downstream configs can use `ann_file=<out>, split='inner_train'`.

Protocol:
    - subjects are sampled (not instances), seed fixed for reproducibility;
    - inner_val subjects must be disjoint from xsub_val subjects;
    - all 60 classes must appear in inner_val; if not, the subject ratio is
      enlarged in 0.05 steps until coverage is satisfied.

Usage:
    python tools/analysis/make_stage4_split.py
"""
import argparse
import json
import os
import os.path as osp
import random
import re
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Generate inner_val split.')
    parser.add_argument('--ann-file', default='data/nturgbd/ntu60_3danno.pkl')
    parser.add_argument('--split', default='xsub_train')
    parser.add_argument('--val-split', default='xsub_val')
    parser.add_argument('--ratio', type=float, default=0.10,
                        help='initial fraction of xsub_train subjects for inner_val')
    parser.add_argument('--ratio-step', type=float, default=0.05)
    parser.add_argument('--ratio-max', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--min-per-class', type=int, default=1)
    parser.add_argument('--out', default='data/nturgbd/ntu60_inner_split.pkl')
    parser.add_argument('--report',
                        default='work_dirs/stage1_inner_val/coverage.json')
    return parser.parse_args()


def subject_of(frame_dir):
    """Extract performer id (e.g. P001) from NTU frame_dir name."""
    m = re.search(r'P(\d{3})', frame_dir)
    if m is None:
        raise ValueError(f'Cannot parse subject from frame_dir: {frame_dir}')
    return int(m.group(1))


def main():
    args = parse_args()
    t0 = time.time()

    print(f'[1/5] Loading annotation file (this may take a while): {args.ann_file}')
    import mmcv
    data = mmcv.load(args.ann_file)
    print(f'      loaded in {time.time() - t0:.1f}s')

    split_map = data['split']
    annotations = data['annotations']
    identifier = 'filename' if 'filename' in annotations[0] else 'frame_dir'
    print(f'      identifier: {identifier}')

    train_ids = set(split_map[args.split])
    val_ids = set(split_map[args.val_split])

    # frame_dir -> label
    label_of = {}
    for item in annotations:
        fid = item[identifier]
        if fid in train_ids or fid in val_ids:
            label_of[fid] = item['label']

    print(f'[2/5] {args.split}: {len(train_ids)} samples, '
          f'{args.val_split}: {len(val_ids)} samples')

    # subject sets
    train_subjects = sorted({subject_of(f) for f in train_ids})
    val_subjects = sorted({subject_of(f) for f in val_ids})
    overlap = set(train_subjects) & set(val_subjects)
    assert not overlap, f'Subjects overlap between splits: {overlap}'
    print(f'      {args.split}: {len(train_subjects)} subjects, '
          f'{args.val_split}: {len(val_subjects)} subjects, no overlap')

    # all labels present in train
    all_labels = sorted({label_of[f] for f in train_ids})
    num_classes = len(all_labels)
    print(f'      classes in {args.split}: {num_classes} '
          f'(expect 60 for NTU-60)')

    rng = random.Random(args.seed)

    chosen = None
    ratio = args.ratio
    while True:
        n = max(1, int(round(len(train_subjects) * ratio)))
        n = min(n, len(train_subjects) - 1)
        sampled = rng.sample(train_subjects, n)
        inner_val_ids = {f for f in train_ids if subject_of(f) in sampled}
        label_counts = {}
        for f in inner_val_ids:
            label_counts[label_of[f]] = label_counts.get(label_of[f], 0) + 1
        missing = [c for c in all_labels if label_counts.get(c, 0) < args.min_per_class]
        if not missing:
            chosen = (ratio, sampled, inner_val_ids, label_counts)
            break
        print(f'      ratio={ratio:.2f} -> {len(inner_val_ids)} samples, '
              f'missing {len(missing)} classes (e.g. {missing[:5]}), enlarging')
        ratio += args.ratio_step
        if ratio > args.ratio_max:
            raise RuntimeError(
                f'Cannot achieve class coverage within ratio_max={args.ratio_max}. '
                f'Missing classes: {missing}')

    ratio, sampled, inner_val_ids, label_counts = chosen
    inner_train_ids = train_ids - inner_val_ids
    assert not (inner_val_ids & val_ids), 'inner_val overlaps xsub_val!'
    assert inner_val_ids <= train_ids, 'inner_val not subset of xsub_train!'

    print(f'[3/5] inner_val: ratio={ratio:.2f}, {len(sampled)} subjects '
          f'{sampled}, {len(inner_val_ids)} samples; '
          f'inner_train: {len(inner_train_ids)} samples')
    print(f'      per-class coverage: min={min(label_counts.values())}, '
          f'max={max(label_counts.values())}, mean={sum(label_counts.values())/len(label_counts):.1f}')

    out_dir = osp.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # PoseDataset.load_pkl_annotations expects both 'split' and 'annotations'
    # top-level keys; keep the original annotations intact so downstream
    # configs can use this file directly as ann_file.
    split_pkl = {
        'split': {
            'inner_train': sorted(inner_train_ids),
            'inner_val': sorted(inner_val_ids),
        },
        'annotations': annotations,
    }
    mmcv.dump(split_pkl, args.out)
    print(f'[4/5] split pkl written: {args.out} '
          f'({osp.getsize(args.out) / 1024 / 1024:.0f} MB)')

    report = {
        'ann_file': args.ann_file,
        'seed': args.seed,
        'ratio': ratio,
        'sampled_subjects': sampled,
        'inner_train_size': len(inner_train_ids),
        'inner_val_size': len(inner_val_ids),
        'num_classes': num_classes,
        'min_per_class': min(label_counts.values()),
        'max_per_class': max(label_counts.values()),
        'mean_per_class': sum(label_counts.values()) / len(label_counts),
        'per_class_counts': {str(c): label_counts[c] for c in all_labels},
        'disjoint_with_xsub_val': not bool(inner_val_ids & val_ids),
        'subset_of_xsub_train': inner_val_ids <= train_ids,
    }
    if osp.dirname(args.report):
        os.makedirs(osp.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'[5/5] coverage report written: {args.report}')
    print(f'Done in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
