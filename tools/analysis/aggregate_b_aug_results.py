#!/usr/bin/env python
"""Aggregate B_aug formal xsub test results (work-5) into the main table.

Reads the per-condition ``test.log`` files produced by
``run_b_aug_xsub_test.py`` under
``work_dirs/aug_baseline/b_aug/<model>_j/results/final_xsub/seed<N>/<condition>/``
and emits per-seed detail, per-condition mean±std, and per-model mRA
(mean accuracy over the 16 conditions) into
``research_notes/results/stage_04/``.

Pure stdlib (parses accuracy lines from the test logs), so it runs on the
dev host after result files are synced back.  Compares against the stage-3
clean-baseline numbers when ``research_notes/results/stage_03/stage3_summary.csv``
is available.
"""

import argparse
import csv
import json
import os
import re
from statistics import mean, stdev

SEEDS = (255, 2026, 3407)
MODELS = ('stgcnpp', 'ctrgcn')
DEGRADE_TYPES = ('joint_missing', 'limb_occlusion', 'coord_noise',
                 'frame_missing', 'mixed')
SEVERITIES = ('mild', 'moderate', 'severe')
_B_AUG_ROOT = os.path.join('work_dirs', 'aug_baseline', 'b_aug')
_STAGE3_SUMMARY = os.path.join('research_notes', 'results', 'stage_03',
                               'stage3_summary.csv')

_ACC_RE = {
    'top1': re.compile(r'^top1_acc:\s*([0-9.]+)', re.M),
    'top5': re.compile(r'^top5_acc:\s*([0-9.]+)', re.M),
    'mean_class': re.compile(r'^mean_class_accuracy:\s*([0-9.]+)', re.M),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS,
                        default=['stgcnpp'])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--out-dir', default=os.path.join(
        'research_notes', 'results', 'stage_04'))
    return parser.parse_args()


def condition_list():
    values = ['clean']
    for severity in SEVERITIES:
        for degrade_type in DEGRADE_TYPES:
            values.append('{}_{}'.format(degrade_type, severity))
    return values


def results_root(model):
    return os.path.join(_B_AUG_ROOT, '{}_j'.format(model),
                        'results', 'final_xsub')


def parse_log(log_path):
    with open(log_path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    values = {}
    for key, pattern in _ACC_RE.items():
        matches = pattern.findall(text)
        if matches:
            values[key] = float(matches[-1]) * 100.0
    return values


def collect(model, seeds):
    """Return list of rows: (model, seed, condition, top1, top5, mean_class)."""
    rows = []
    for seed in seeds:
        for condition in condition_list():
            log_path = os.path.join(results_root(model),
                                    'seed{}'.format(seed), condition,
                                    'test.log')
            if not os.path.isfile(log_path):
                raise FileNotFoundError(
                    'Missing test log: {} (test not finished?)'.format(
                        os.path.abspath(log_path)))
            acc = parse_log(log_path)
            if 'top1' not in acc:
                raise ValueError(
                    'No top1_acc line in {} (test incomplete?)'.format(
                        os.path.abspath(log_path)))
            rows.append(dict(
                model=model, seed=seed, condition=condition,
                top1=acc['top1'], top5=acc.get('top5'),
                mean_class=acc.get('mean_class')))
    return rows


def fmt(value):
    return '' if value is None else '{:.2f}'.format(value)


def summarize(rows, seeds):
    """Per-condition mean±std across seeds + per-model mRA."""
    summary = []
    for condition in condition_list():
        tops = [r['top1'] for r in rows
                if r['condition'] == condition and r['seed'] in seeds]
        summary.append(dict(
            condition=condition, top1_mean=mean(tops),
            top1_std=stdev(tops) if len(tops) > 1 else 0.0,
            top1_min=min(tops), top1_max=max(tops), n_seeds=len(tops)))
    mra_per_seed = []
    for seed in seeds:
        tops = [r['top1'] for r in rows if r['seed'] == seed]
        clean = next(r['top1'] for r in rows
                     if r['seed'] == seed and r['condition'] == 'clean')
        mra_per_seed.append(dict(seed=seed, mra=mean(tops), clean=clean))
    mra_values = [entry['mra'] for entry in mra_per_seed]
    return summary, mra_per_seed, dict(
        mra_mean=mean(mra_values),
        mra_std=stdev(mra_values) if len(mra_values) > 1 else 0.0)


def load_stage3_mra():
    """Stage-3 clean-baseline mRA per model (for the A0-vs-B_aug uplift)."""
    if not os.path.isfile(_STAGE3_SUMMARY):
        return {}
    with open(_STAGE3_SUMMARY, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    result = {}
    for model in MODELS:
        tops = [float(r['top1_mean']) for r in rows
                if r.get('model') == model and r.get('condition') != 'clean']
        # mRA in stage 3 = mean over all 16 conditions (clean included)
        all_tops = [float(r['top1_mean']) for r in rows
                    if r.get('model') == model]
        if all_tops:
            result[model] = dict(mra=mean(all_tops),
                                 clean=[float(r['top1_mean']) for r in rows
                                        if r.get('model') == model
                                        and r.get('condition') == 'clean'][0])
    return result


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stage3 = load_stage3_mra()

    all_detail = []
    report = {}
    for model in args.models:
        rows = collect(model, args.seeds)
        all_detail.extend(rows)
        summary, mra_per_seed, mra_stats = summarize(rows, args.seeds)
        clean_mean = next(s['top1_mean'] for s in summary
                          if s['condition'] == 'clean')
        clean_std = next(s['top1_std'] for s in summary
                         if s['condition'] == 'clean')
        report[model] = dict(
            conditions=summary, mra_per_seed=mra_per_seed,
            mra_mean=mra_stats['mra_mean'], mra_std=mra_stats['mra_std'],
            clean_mean=clean_mean, clean_std=clean_std,
            gap_pp=clean_mean - mra_stats['mra_mean'])

        stem = 'b_aug_{}'.format(model)
        with open(os.path.join(args.out_dir, stem + '_summary.csv'), 'w',
                  newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'condition', 'top1_mean', 'top1_std',
                             'top1_min', 'top1_max'])
            for entry in summary:
                writer.writerow([model, entry['condition'],
                                 fmt(entry['top1_mean']), fmt(entry['top1_std']),
                                 fmt(entry['top1_min']), fmt(entry['top1_max'])])
        with open(os.path.join(args.out_dir, stem + '_mra.csv'), 'w',
                  newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'metric', 'seed255', 'seed2026',
                             'seed3407', 'mean', 'std'])
            per_seed = {e['seed']: e for e in mra_per_seed}
            writer.writerow([model, 'mRA'] + [fmt(per_seed[s]['mra'])
                            for s in args.seeds] +
                            [fmt(mra_stats['mra_mean']), fmt(mra_stats['mra_std'])])
            writer.writerow([model, 'clean'] + [fmt(per_seed[s]['clean'])
                            for s in args.seeds] + [fmt(clean_mean), fmt(clean_std)])
            writer.writerow([model, 'gap_pp'] + [''] * len(args.seeds) +
                            [fmt(clean_mean - mra_stats['mra_mean']), ''])

        print('\n=== B_aug {} xsub 16-condition matrix ({} seeds) ==='.format(
            model, len(args.seeds)))
        for entry in summary:
            print('  {:28s} {:6.2f} +- {:4.2f}'.format(
                entry['condition'], entry['top1_mean'], entry['top1_std']))
        print('  {:28s} {:6.2f} +- {:4.2f}'.format(
            'mRA (16-condition mean)', mra_stats['mra_mean'],
            mra_stats['mra_std']))
        print('  clean                          {:6.2f} +- {:4.2f}'.format(
            clean_mean, clean_std))
        print('  gap (clean - mRA)              {:6.2f} pp'.format(
            clean_mean - mra_stats['mra_mean']))
        if model in stage3:
            base = stage3[model]
            print('  stage-3 A0 clean             {:6.2f} | stage-3 A0 mRA {:6.2f}'
                  .format(base['clean'], base['mra']))
            print('  B_aug uplift vs A0 mRA         {:+6.2f} pp'.format(
                mra_stats['mra_mean'] - base['mra']))

    with open(os.path.join(args.out_dir, 'b_aug_detail.csv'), 'w',
              newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'seed', 'condition', 'top1', 'top5',
                         'mean_class_accuracy'])
        for row in all_detail:
            writer.writerow([row['model'], row['seed'], row['condition'],
                             fmt(row['top1']), fmt(row['top5']),
                             fmt(row['mean_class'])])
    with open(os.path.join(args.out_dir, 'b_aug_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('\nWrote: {} ({}_summary.csv / {}_mra.csv / b_aug_detail.csv / '
          'b_aug_report.json)'.format(os.path.abspath(args.out_dir),
                                      'b_aug_<model>', 'b_aug_<model>'))


if __name__ == '__main__':
    main()
