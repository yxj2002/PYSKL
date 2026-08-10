#!/usr/bin/env python
"""Create class-level analyses from NTU robustness inference results."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


MODELS = ('stgcn', 'stgcnpp', 'ctrgcn')
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
        '--out-dir', default='work_dirs/robustness_benchmark/class_analysis')
    parser.add_argument('--top-k', type=int, default=10)
    return parser.parse_args()


def load_labels(ann_file, split):
    with open(ann_file, 'rb') as file:
        data = pickle.load(file)
    split_ids = set(data['split'][split])
    annotations = [
        item for item in data['annotations'] if item['frame_dir'] in split_ids
    ]
    if not annotations:
        raise ValueError('No annotations found for split: {}'.format(split))
    return np.asarray([item['label'] for item in annotations], dtype=np.int64)


def load_class_names(label_map):
    with open(label_map, encoding='utf-8') as file:
        names = [line.strip() for line in file if line.strip()]
    if len(names) < NUM_CLASSES:
        raise ValueError('Label map has fewer than {} classes'.format(NUM_CLASSES))
    return names[:NUM_CLASSES]


def load_scores(path, num_samples):
    with open(path, 'rb') as file:
        scores = pickle.load(file)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (num_samples, NUM_CLASSES):
        raise ValueError(
            '{} has shape {}, expected ({}, {})'.format(
                path, scores.shape, num_samples, NUM_CLASSES))
    return scores.argmax(axis=1).astype(np.int64)


def class_accuracy(predictions, labels, counts):
    correct = np.bincount(
        labels[predictions == labels], minlength=NUM_CLASSES).astype(np.int64)
    return correct / counts


def top_pairs(labels, predictions, mask, top_k):
    pair_counts = np.bincount(
        labels[mask] * NUM_CLASSES + predictions[mask],
        minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    pairs = []
    for true_label, predicted_label in zip(*np.nonzero(pair_counts)):
        if true_label != predicted_label:
            pairs.append((
                int(pair_counts[true_label, predicted_label]),
                int(true_label), int(predicted_label)))
    return sorted(pairs, reverse=True)[:top_k]


def result_paths(root, model):
    model_root = root / model
    if not model_root.is_dir():
        raise FileNotFoundError('Missing model results: {}'.format(model_root))
    paths = {
        item.name: item / 'result.pkl'
        for item in model_root.iterdir()
        if (item / 'result.pkl').is_file()
    }
    if 'clean' not in paths:
        raise FileNotFoundError('Missing clean result for {}'.format(model))
    return paths


def main():
    args = parse_args()
    labels = load_labels(args.ann_file, args.split)
    names = load_class_names(args.label_map)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.int64)
    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_rows = []
    drop_rows = []
    confusion_rows = []
    report = {
        'ann_file': str(Path(args.ann_file).resolve()),
        'split': args.split,
        'num_samples': int(labels.size),
        'models': {},
    }

    for model in MODELS:
        paths = result_paths(root, model)
        clean_predictions = load_scores(paths['clean'], labels.size)
        clean_accuracy = class_accuracy(clean_predictions, labels, counts)
        model_report = {'conditions': {}}

        for condition, path in sorted(paths.items()):
            predictions = load_scores(path, labels.size)
            accuracy = class_accuracy(predictions, labels, counts)
            drops = clean_accuracy - accuracy
            new_error_mask = (clean_predictions == labels) & (predictions != labels)
            all_error_mask = predictions != labels
            ranked_drops = sorted(
                range(NUM_CLASSES), key=lambda index: drops[index], reverse=True)
            top_drops = ranked_drops[:args.top_k]
            condition_report = {
                'top1_accuracy': float(np.mean(predictions == labels)),
                'mean_class_accuracy': float(np.mean(accuracy)),
                'top_drops': [],
                'new_error_confusions': [],
                'all_error_confusions': [],
            }

            for class_id in range(NUM_CLASSES):
                class_rows.append({
                    'model': model,
                    'condition': condition,
                    'class_id': class_id,
                    'action': names[class_id],
                    'support': int(counts[class_id]),
                    'clean_accuracy': float(clean_accuracy[class_id]),
                    'accuracy': float(accuracy[class_id]),
                    'absolute_drop': float(drops[class_id]),
                })
            for class_id in top_drops:
                record = {
                    'class_id': int(class_id),
                    'action': names[class_id],
                    'support': int(counts[class_id]),
                    'clean_accuracy': float(clean_accuracy[class_id]),
                    'accuracy': float(accuracy[class_id]),
                    'absolute_drop': float(drops[class_id]),
                }
                condition_report['top_drops'].append(record)
                drop_rows.append(dict(model=model, condition=condition, **record))
            for kind, mask, report_key in (
                    ('new_error', new_error_mask, 'new_error_confusions'),
                    ('all_error', all_error_mask, 'all_error_confusions')):
                for count, true_id, pred_id in top_pairs(
                        labels, predictions, mask, args.top_k):
                    record = {
                        'count': count,
                        'true_class_id': true_id,
                        'true_action': names[true_id],
                        'predicted_class_id': pred_id,
                        'predicted_action': names[pred_id],
                    }
                    condition_report[report_key].append(record)
                    confusion_rows.append(
                        dict(model=model, condition=condition, kind=kind, **record))
            model_report['conditions'][condition] = condition_report
        report['models'][model] = model_report

    write_csv(out_dir / 'per_class_accuracy.csv', class_rows)
    write_csv(out_dir / 'top_class_drops.csv', drop_rows)
    write_csv(out_dir / 'top_confusions.csv', confusion_rows)
    with open(out_dir / 'class_analysis.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print('PASS: analyzed {} samples, {} models; outputs: {}'.format(
        labels.size, len(MODELS), out_dir))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
