#!/usr/bin/env python
"""Aggregate stage-4 work-1 augmentation results into the 7x16 main table.

Reads the 112 ``result.pkl`` files (7 A-groups x 16 conditions) produced by
``test_aug_baseline.py`` and emits:

  * ``stage4_summary.csv`` -- the 7 x 16 main table (one row per
    group x condition, Top-1 %).
  * ``stage4_mra.csv``     -- per-group mRA (mean Top-1 over 16 conditions)
    plus the clean accuracy and the gap vs Clean.
  * ``stage4_overfit.csv`` -- single-degradation overfitting evidence:
    for A1-A4, the Top-1 gain vs A0 on the *matched* degradation, on the
    *other* three degradations, and on Clean (three-tier averages).
  * ``stage4_report.json`` -- full machine-readable report.

Work 1 uses a single seed (255), so there is no mean/std here; the 3-seed
retrain of the selected B_aug happens in work 5.

Only numpy is required (no mmcv / torch), so this runs on the remote machine
where the result pkls live (and on the dev host once they are synced back).
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np

GROUPS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6')
# Which single degradation each A1-A4 config trains on (for overfit analysis).
GROUP_DEGRADE = {
    'A0': None,
    'A1': 'joint_missing',
    'A2': 'limb_occlusion',
    'A3': 'coord_noise',
    'A4': 'frame_missing',
    'A5': None,   # random_single
    'A6': None,   # mixed
}
DEGRADE_TYPES = ('joint_missing', 'limb_occlusion', 'coord_noise',
                 'frame_missing', 'mixed')
SEVERITIES = ('mild', 'moderate', 'severe')
NUM_CLASSES = 60


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/nturgbd/ntu60_inner_split.pkl')
    parser.add_argument('--split', default='inner_val')
    parser.add_argument(
        '--results-root', default='work_dirs/aug_baseline/stgcnpp_j/results')
    parser.add_argument(
        '--out-dir', default='research_notes/results/stage_04')
    return parser.parse_args()


def condition_list():
    conds = ['clean']
    for sev in SEVERITIES:
        for dt in DEGRADE_TYPES:
            conds.append('{}_{}'.format(dt, sev))
    return conds


def load_labels(ann_file, split):
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    split_ids = set(data['split'][split])
    annotations = [
        item for item in data['annotations']
        if item['frame_dir'] in split_ids]
    if not annotations:
        raise ValueError('No annotations for split {}'.format(split))
    labels = np.asarray([item['label'] for item in annotations], dtype=np.int64)
    return labels


def load_predictions(result_path, num_samples):
    with open(result_path, 'rb') as f:
        scores = pickle.load(f)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (num_samples, NUM_CLASSES):
        raise ValueError(
            '{} shape {} != ({}, {})'.format(
                result_path, scores.shape, num_samples, NUM_CLASSES))
    return scores.argmax(axis=1).astype(np.int64)


def top1_accuracy(predictions, labels):
    return float(np.mean(predictions == labels)) * 100.0


def mean_class_accuracy(predictions, labels):
    correct = np.bincount(
        labels[predictions == labels], minlength=NUM_CLASSES)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    acc = correct / np.maximum(counts, 1)
    return float(np.mean(acc)) * 100.0


def collect(args, labels):
    num_samples = labels.shape[0]
    detail = []
    results = {g: {} for g in GROUPS}
    missing = []

    for group in GROUPS:
        for condition in condition_list():
            path = os.path.join(
                args.results_root, group, condition, 'result.pkl')
            if not os.path.isfile(path):
                missing.append(path)
                results[group][condition] = None
                continue
            try:
                preds = load_predictions(path, num_samples)
            except Exception as exc:
                missing.append('{} ({})'.format(path, exc))
                results[group][condition] = None
                continue
            top1 = top1_accuracy(preds, labels)
            mca = mean_class_accuracy(preds, labels)
            results[group][condition] = dict(top1=top1, mca=mca)
            detail.append(dict(
                group=group, condition=condition,
                top1=round(top1, 2), mean_class_accuracy=round(mca, 2)))

    if missing:
        print('WARNING: {} missing/invalid result file(s):'.format(len(missing)),
              file=sys.stderr)
        for m in missing[:20]:
            print('  ' + m, file=sys.stderr)
        if len(missing) > 20:
            print('  ... and {} more'.format(len(missing) - 20), file=sys.stderr)
    return detail, results


def tier_mean(results, group, degrade_type):
    """Mean Top-1 over the three severities of one degradation type."""
    accs = []
    for sev in SEVERITIES:
        cell = results[group].get('{}_{}'.format(degrade_type, sev))
        if cell is not None:
            accs.append(cell['top1'])
    return float(np.mean(accs)) if accs else None


def build_summary(results):
    rows = []
    for group in GROUPS:
        for condition in condition_list():
            cell = results[group][condition]
            rows.append(dict(
                group=group, condition=condition,
                top1=round(cell['top1'], 2) if cell else '',
                mean_class_accuracy=round(cell['mca'], 2) if cell else ''))
    return rows


def build_mra(results):
    rows = []
    for group in GROUPS:
        accs = []
        for condition in condition_list():
            cell = results[group][condition]
            if cell is not None:
                accs.append(cell['top1'])
        mra = float(np.mean(accs)) if accs else None
        clean = results[group].get('clean')
        clean_top1 = clean['top1'] if clean else None
        gap = None
        if clean_top1 is not None and mra is not None:
            gap = round(clean_top1 - mra, 2)
        rows.append(dict(
            group=group,
            mra=round(mra, 2) if mra is not None else '',
            clean=round(clean_top1, 2) if clean_top1 is not None else '',
            gap_vs_clean=gap,
            n_conditions=len(condition_list())))
    return rows


def build_overfit(results):
    """For A1-A4, gain vs A0 on matched vs other degradations vs clean."""
    rows = []
    for group in ('A1', 'A2', 'A3', 'A4'):
        matched = GROUP_DEGRADE[group]
        others = [t for t in DEGRADE_TYPES if t != matched]

        matched_gain = tier_mean(results, group, matched) - tier_mean(
            results, 'A0', matched)
        other_gains = [
            tier_mean(results, group, t) - tier_mean(results, 'A0', t)
            for t in others]
        other_gain = float(np.mean(other_gains))

        clean_gain = (results[group]['clean']['top1']
                      - results['A0']['clean']['top1'])
        rows.append(dict(
            group=group,
            matched_degradation=matched,
            gain_on_matched=round(matched_gain, 2),
            gain_on_others_mean=round(other_gain, 2),
            gain_on_clean=round(clean_gain, 2),
            overfit_margin=round(matched_gain - other_gain, 2)))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    labels = load_labels(args.ann_file, args.split)
    print('Loaded {} labels from {} ({})'.format(
        labels.shape[0], args.ann_file, args.split))

    detail, results = collect(args, labels)
    summary = build_summary(results)
    mra = build_mra(results)
    overfit = build_overfit(results)

    write_csv(os.path.join(args.out_dir, 'stage4_summary.csv'), summary)
    write_csv(os.path.join(args.out_dir, 'stage4_mra.csv'), mra)
    write_csv(os.path.join(args.out_dir, 'stage4_overfit.csv'), overfit)

    report = dict(
        ann_file=os.path.abspath(args.ann_file),
        split=args.split,
        num_samples=int(labels.shape[0]),
        summary=summary,
        mra=mra,
        overfit=overfit)
    with open(os.path.join(args.out_dir, 'stage4_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('\n=== Stage 4 work-1 mRA (mean over {} conditions) ==='.format(
        len(condition_list())))
    for row in mra:
        print('  {:4s} mRA = {}  clean = {}  gap_vs_clean = {}'.format(
            row['group'], row['mra'], row['clean'], row['gap_vs_clean']))

    print('\n=== Single-degradation overfitting (gain vs A0, pp) ===')
    for row in overfit:
        print('  {:4s} matched={:14s} gain_matched={:6} gain_others={:6} '
              'gain_clean={:6} margin={:6}'.format(
                  row['group'], row['matched_degradation'],
                  row['gain_on_matched'], row['gain_on_others_mean'],
                  row['gain_on_clean'], row['overfit_margin']))

    print('\nOutputs:')
    for name in ('stage4_summary.csv', 'stage4_mra.csv',
                 'stage4_overfit.csv', 'stage4_report.json'):
        print('  ' + os.path.join(args.out_dir, name))


if __name__ == '__main__':
    main()
