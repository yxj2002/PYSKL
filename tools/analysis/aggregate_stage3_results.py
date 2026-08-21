#!/usr/bin/env python
"""Aggregate stage-3 robustness benchmark results into the main table.

Reads the 144 ``result.pkl`` files (3 models x 16 conditions x 3 seeds)
produced by ``run_stage3_robustness.py`` and emits:

  * ``stage3_summary.csv``      -- one row per (model, condition) with
    mean/std Top-1 over seeds.
  * ``stage3_detail.csv``       -- one row per (model, condition, seed).
  * ``stage3_mra.csv``          -- per-model mRA (mean accuracy over the 16
    conditions) with mean/std over seeds.
  * ``stage3_report.json``      -- full machine-readable report.

Only numpy is required (no mmcv / torch), so this can run on the dev host
once the result pkls are synced back from the remote training machine.
"""

import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np

MODELS = ('stgcn', 'stgcnpp', 'ctrgcn')
SEEDS = (255, 2026, 3407)
DEGRADE_TYPES = ('joint_missing', 'limb_occlusion', 'coord_noise',
                 'frame_missing', 'mixed')
SEVERITIES = ('mild', 'moderate', 'severe')
NUM_CLASSES = 60


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/nturgbd/ntu60_3danno.pkl')
    parser.add_argument('--split', default='xsub_val')
    parser.add_argument(
        '--results-root', default='work_dirs/robustness_benchmark')
    parser.add_argument(
        '--out-dir', default='research_notes/results/stage_03')
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=list(SEEDS))
    return parser.parse_args()


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
    # The test dataloader iterates annotations in the same order as the pkl,
    # so label order matches result order.  We assert size consistency later.
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


def condition_list(skip_clean=False):
    conds = [] if skip_clean else ['clean']
    for sev in SEVERITIES:
        for dt in DEGRADE_TYPES:
            conds.append('{}_{}'.format(dt, sev))
    return conds


def collect(args, labels):
    """Return detail rows and a nested dict: results[model][condition][seed]."""
    num_samples = labels.shape[0]
    detail = []
    results = {m: {} for m in MODELS}
    missing = []

    for model in MODELS:
        for condition in condition_list():
            results[model][condition] = {}
            for seed in args.seeds:
                path = os.path.join(
                    args.results_root, model, condition,
                    'seed{}'.format(seed), 'result.pkl')
                if not os.path.isfile(path):
                    missing.append(path)
                    continue
                try:
                    preds = load_predictions(path, num_samples)
                except Exception as exc:
                    missing.append('{} ({})'.format(path, exc))
                    continue
                top1 = top1_accuracy(preds, labels)
                mca = mean_class_accuracy(preds, labels)
                results[model][condition][seed] = dict(
                    top1=top1, mean_class_accuracy=mca)
                detail.append(dict(
                    model=model, condition=condition, seed=seed,
                    top1=round(top1, 2),
                    mean_class_accuracy=round(mca, 2)))

    if missing:
        print('WARNING: {} missing/invalid result file(s):'.format(len(missing)),
              file=sys.stderr)
        for m in missing[:20]:
            print('  ' + m, file=sys.stderr)
        if len(missing) > 20:
            print('  ... and {} more'.format(len(missing) - 20), file=sys.stderr)
    return detail, results


def summarize(results, seeds):
    """Build summary rows (model x condition, mean +/- std over seeds)."""
    rows = []
    for model in MODELS:
        for condition in condition_list():
            seed_data = results[model][condition]
            top1s = [seed_data[s]['top1'] for s in seeds if s in seed_data]
            mcas = [seed_data[s]['mean_class_accuracy']
                    for s in seeds if s in seed_data]
            if not top1s:
                rows.append(dict(
                    model=model, condition=condition,
                    top1_mean='', top1_std='', top1_min='', top1_max='',
                    mean_class_accuracy_mean='', n_seeds=0))
                continue
            rows.append(dict(
                model=model, condition=condition,
                top1_mean=round(float(np.mean(top1s)), 2),
                top1_std=round(float(np.std(top1s, ddof=1)), 2),
                top1_min=round(float(np.min(top1s)), 2),
                top1_max=round(float(np.max(top1s)), 2),
                mean_class_accuracy_mean=round(float(np.mean(mcas)), 2),
                n_seeds=len(top1s)))
    return rows


def compute_mra(results, seeds):
    """mRA = mean Top-1 over the 16 conditions, per seed; then mean/std."""
    rows = []
    for model in MODELS:
        per_seed_mra = {}
        for seed in seeds:
            accs = []
            for condition in condition_list():
                if seed in results[model][condition]:
                    accs.append(results[model][condition][seed]['top1'])
            if accs:
                per_seed_mra[seed] = float(np.mean(accs))
        mra_values = list(per_seed_mra.values())
        rows.append(dict(
            model=model,
            mra_mean=round(float(np.mean(mra_values)), 2) if mra_values else '',
            mra_std=round(float(np.std(mra_values, ddof=1)), 2) if mra_values else '',
            n_conditions=len(condition_list()),
            n_seeds=len(mra_values),
            per_seed={str(s): round(v, 2) for s, v in per_seed_mra.items()}))
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
    summary = summarize(results, args.seeds)
    mra = compute_mra(results, args.seeds)

    write_csv(os.path.join(args.out_dir, 'stage3_detail.csv'), detail)
    write_csv(os.path.join(args.out_dir, 'stage3_summary.csv'), summary)
    write_csv(os.path.join(args.out_dir, 'stage3_mra.csv'), mra)

    report = dict(
        ann_file=os.path.abspath(args.ann_file),
        split=args.split,
        num_samples=int(labels.shape[0]),
        seeds=list(args.seeds),
        summary=summary,
        mra=mra,
        detail=detail)
    with open(os.path.join(args.out_dir, 'stage3_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Console overview
    print('\n=== Stage 3 mRA (mean over {} conditions, {} seeds) ==='.format(
        len(condition_list()), len(args.seeds)))
    for row in mra:
        print('  {:8s} mRA = {} +/- {}  (per-seed: {})'.format(
            row['model'], row['mra_mean'], row['mra_std'],
            ', '.join('{}={}'.format(k, v) for k, v in row['per_seed'].items())))

    print('\nOutputs:')
    for name in ('stage3_detail.csv', 'stage3_summary.csv',
                 'stage3_mra.csv', 'stage3_report.json'):
        print('  ' + os.path.join(args.out_dir, name))


if __name__ == '__main__':
    main()
