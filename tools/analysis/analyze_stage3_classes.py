#!/usr/bin/env python
"""Class-level analysis for the stage-3 multi-seed robustness benchmark.

Extends ``analyze_robustness_results.py`` with a seed dimension.  For every
(model, condition, seed) it computes per-class accuracy, then aggregates over
seeds: per-class mean accuracy, and the top classes that drop the most under
degradation (averaged over seeds).  Also reports top confusion pairs.

Outputs (under --out-dir):
  * per_class_accuracy.csv   -- model, condition, seed, class_id, action, ...
  * per_class_summary.csv    -- model, condition, class_id, mean_acc over seeds
  * top_class_drops.csv      -- model, condition, top-K dropped classes
  * top_confusions.csv       -- model, condition, kind, confusion pairs
  * class_analysis.json      -- full report

Only numpy is required.
"""

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

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
        '--label-map', default='tools/data/label_map/nturgbd_120.txt')
    parser.add_argument(
        '--out-dir', default='research_notes/results/stage_03/class_analysis')
    parser.add_argument('--top-k', type=int, default=10)
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
    return np.asarray([item['label'] for item in annotations], dtype=np.int64)


def load_class_names(label_map):
    with open(label_map, encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) < NUM_CLASSES:
        raise ValueError('Label map has < {} classes'.format(NUM_CLASSES))
    return names[:NUM_CLASSES]


def load_predictions(path, num_samples):
    with open(path, 'rb') as f:
        scores = pickle.load(f)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (num_samples, NUM_CLASSES):
        raise ValueError(
            '{} shape {} != ({}, {})'.format(
                path, scores.shape, num_samples, NUM_CLASSES))
    return scores.argmax(axis=1).astype(np.int64)


def class_accuracy(predictions, labels, counts):
    correct = np.bincount(
        labels[predictions == labels], minlength=NUM_CLASSES).astype(np.int64)
    return correct / np.maximum(counts, 1)


def top_pairs(labels, predictions, mask, top_k):
    pair_counts = np.bincount(
        labels[mask] * NUM_CLASSES + predictions[mask],
        minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    pairs = []
    for true_label, predicted_label in zip(*np.nonzero(pair_counts)):
        if true_label != predicted_label:
            pairs.append((int(pair_counts[true_label, predicted_label]),
                          int(true_label), int(predicted_label)))
    return sorted(pairs, reverse=True)[:top_k]


def condition_list():
    conds = ['clean']
    for sev in SEVERITIES:
        for dt in DEGRADE_TYPES:
            conds.append('{}_{}'.format(dt, sev))
    return conds


def main():
    args = parse_args()
    labels = load_labels(args.ann_file, args.split)
    names = load_class_names(args.label_map)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.int64)
    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_rows = []
    summary_rows = []
    drop_rows = []
    confusion_rows = []
    report = dict(
        ann_file=str(Path(args.ann_file).resolve()),
        split=args.split,
        num_samples=int(labels.size),
        models={})

    for model in MODELS:
        model_report = {'conditions': {}}

        # Clean reference: per-class accuracy averaged over all seeds, plus
        # per-seed clean predictions for the paired new-error analysis.
        clean_preds_by_seed = {}
        clean_per_class_accs = []
        for seed in args.seeds:
            clean_pkl = root / model / 'clean' / 'seed{}'.format(seed) / 'result.pkl'
            if not clean_pkl.is_file():
                continue
            clean_p = load_predictions(str(clean_pkl), labels.size)
            clean_preds_by_seed[seed] = clean_p
            clean_per_class_accs.append(class_accuracy(clean_p, labels, counts))
        if clean_per_class_accs:
            clean_acc = np.mean(clean_per_class_accs, axis=0)
        else:
            clean_acc = np.zeros(NUM_CLASSES)

        for condition in condition_list():
            # Collect per-seed predictions
            seed_preds = {}
            for seed in args.seeds:
                pkl = root / model / condition / 'seed{}'.format(seed) / 'result.pkl'
                if not pkl.is_file():
                    continue
                preds = load_predictions(str(pkl), labels.size)
                seed_preds[seed] = preds

            if not seed_preds:
                continue

            cond_report = dict(
                seeds=list(seed_preds.keys()),
                top1_per_seed={},
                top_drops=[],
                new_error_confusions=[],
                all_error_confusions=[])

            # Per-class accuracy per seed + aggregate
            per_class_accs = []
            for seed, preds in seed_preds.items():
                acc = class_accuracy(preds, labels, counts)
                per_class_accs.append(acc)
                cond_report['top1_per_seed'][str(seed)] = round(
                    float(np.mean(preds == labels)) * 100, 2)
                for cid in range(NUM_CLASSES):
                    class_rows.append(dict(
                        model=model, condition=condition, seed=seed,
                        class_id=cid, action=names[cid],
                        support=int(counts[cid]),
                        clean_accuracy=round(float(clean_acc[cid]), 4),
                        accuracy=round(float(acc[cid]), 4)))

            mean_acc = np.mean(per_class_accs, axis=0)
            std_acc = np.std(per_class_accs, axis=0, ddof=1)
            drops = clean_acc - mean_acc

            for cid in range(NUM_CLASSES):
                summary_rows.append(dict(
                    model=model, condition=condition,
                    class_id=cid, action=names[cid],
                    support=int(counts[cid]),
                    clean_accuracy=round(float(clean_acc[cid]), 4),
                    mean_accuracy=round(float(mean_acc[cid]), 4),
                    std_accuracy=round(float(std_acc[cid]), 4),
                    absolute_drop=round(float(drops[cid]), 4)))

            ranked = sorted(range(NUM_CLASSES), key=lambda i: drops[i], reverse=True)
            for cid in ranked[:args.top_k]:
                record = dict(
                    class_id=int(cid), action=names[cid],
                    support=int(counts[cid]),
                    clean_accuracy=round(float(clean_acc[cid]), 4),
                    mean_accuracy=round(float(mean_acc[cid]), 4),
                    absolute_drop=round(float(drops[cid]), 4))
                cond_report['top_drops'].append(record)
                drop_rows.append(dict(model=model, condition=condition, **record))

            # Confusion pairs: aggregate new-error and all-error across seeds.
            # new_error pairs each seed's degraded predictions against that
            # SAME seed's clean predictions (paired), so a sample counts only
            # when clean-correct and deg-wrong come from the same seed.
            for kind in ('new_error', 'all_error'):
                agg_pairs = {}
                for seed, preds in seed_preds.items():
                    if kind == 'new_error':
                        if seed not in clean_preds_by_seed:
                            continue
                        clean_p = clean_preds_by_seed[seed]
                        mask = (clean_p == labels) & (preds != labels)
                    else:
                        mask = preds != labels
                    for cnt, tid, pid in top_pairs(labels, preds, mask, args.top_k):
                        agg_pairs[(tid, pid)] = agg_pairs.get((tid, pid), 0) + cnt
                ranked_pairs = sorted(agg_pairs.items(), key=lambda x: -x[1])[:args.top_k]
                for (tid, pid), cnt in ranked_pairs:
                    record = dict(
                        count=int(cnt), true_class_id=int(tid),
                        true_action=names[tid],
                        predicted_class_id=int(pid),
                        predicted_action=names[pid])
                    cond_report['{}_confusions'.format(kind)].append(record)
                    confusion_rows.append(
                        dict(model=model, condition=condition, kind=kind, **record))

            model_report['conditions'][condition] = cond_report
        report['models'][model] = model_report

    write_csv(out_dir / 'per_class_accuracy.csv', class_rows)
    write_csv(out_dir / 'per_class_summary.csv', summary_rows)
    write_csv(out_dir / 'top_class_drops.csv', drop_rows)
    write_csv(out_dir / 'top_confusions.csv', confusion_rows)
    with open(out_dir / 'class_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('PASS: analyzed {} samples, {} models; outputs: {}'.format(
        labels.size, len(MODELS), out_dir))


def write_csv(path, rows):
    if not rows:
        return
    with open(str(path), 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
