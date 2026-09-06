#!/usr/bin/env python
"""Aggregate augmentation-baseline results from test.log files (stdlib only).

Reads per-condition ``test.log`` files under
``work_dirs/aug_baseline/stgcnpp_j/results/{group}/{condition}/`` and emits:

  * ``stage4_summary.csv`` -- one row per (group, condition) with Top-1 and
    mean-class-accuracy.
  * ``stage4_mra.csv``     -- per-group mRA (mean Top-1 over 16 conditions),
    clean accuracy and gap vs Clean.
  * ``stage4_overfit.csv`` -- single-degradation overfitting evidence for
    A1-A4 (gain vs A0 on matched vs other degradations vs clean).
  * ``stage4_report.json`` -- full machine-readable report.

Groups default to every sub-directory under ``--results-root`` (so work-2/3
variants like ``A6_mixed_p0.3`` are picked up automatically); ``--groups`` can
restrict to a subset, accepting either A-group shorthand (A0..A6) or a full
directory name.  Only the Python standard library is required, so it also runs
on the dev host (no numpy / mmcv / torch).
"""
import argparse
import csv
import json
import os
import re

DEGRADE_TYPES = ('joint_missing', 'limb_occlusion', 'coord_noise',
                 'frame_missing', 'mixed')
SEVERITIES = ('mild', 'moderate', 'severe')

# A-group shorthand -> training degradation tag (used to expand A0..A6).
GROUP_TAG = {
    'A0': 'clean', 'A1': 'joint_missing', 'A2': 'limb_occlusion',
    'A3': 'coord_noise', 'A4': 'frame_missing', 'A5': 'random_single',
    'A6': 'mixed',
}
# Which single degradation each A1-A4 directory trains on (overfit analysis).
GROUP_DEGRADE = {
    'A1_joint_missing': 'joint_missing',
    'A2_limb_occlusion': 'limb_occlusion',
    'A3_coord_noise': 'coord_noise',
    'A4_frame_missing': 'frame_missing',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results-root', default='work_dirs/aug_baseline/stgcnpp_j/results')
    parser.add_argument(
        '--out-dir', default='research_notes/results/stage_04')
    parser.add_argument(
        '--groups', nargs='*', default=None,
        help='Restrict to these groups (A-group shorthand or full directory '
             'names). Default: scan all sub-directories under --results-root.')
    return parser.parse_args()


def condition_list():
    conds = ['clean']
    for sev in SEVERITIES:
        for dt in DEGRADE_TYPES:
            conds.append('{}_{}'.format(dt, sev))
    return conds


def discover_groups(results_root):
    if not os.path.isdir(results_root):
        return []
    return sorted(
        d for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d)))


def expand_groups(args):
    if args.groups:
        groups = []
        for g in args.groups:
            if g in GROUP_TAG:
                groups.append('{}_{}'.format(g, GROUP_TAG[g]))
            else:
                groups.append(g)
        return groups
    return discover_groups(args.results_root)


def _read_log(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()


def read_metrics(results_root, group, condition):
    log = os.path.join(results_root, group, condition, 'test.log')
    if not os.path.isfile(log):
        return None
    txt = _read_log(log)
    top1 = re.findall(r'top1_acc:\s*([\d.]+)', txt)
    mca = re.findall(r'mean_class_accuracy:\s*([\d.]+)', txt)
    if not top1:
        return None
    return dict(top1=float(top1[-1]) * 100.0,
                mca=float(mca[-1]) * 100.0 if mca else None)


def collect(args, groups):
    detail = []
    results = {g: {} for g in groups}
    missing = []
    for group in groups:
        for cond in condition_list():
            m = read_metrics(args.results_root, group, cond)
            if m is None:
                missing.append(os.path.join(
                    args.results_root, group, cond, 'test.log'))
                results[group][cond] = None
                continue
            results[group][cond] = m
            detail.append(dict(group=group, condition=cond,
                               top1=round(m['top1'], 2),
                               mean_class_accuracy=round(m['mca'], 2)
                               if m['mca'] is not None else ''))
    return detail, results, missing


def tier_mean(results, group, degrade_type):
    accs = [results[group].get('{}_{}'.format(degrade_type, sev))
            for sev in SEVERITIES]
    accs = [a['top1'] for a in accs if a is not None]
    return sum(accs) / len(accs) if accs else None


def build_summary(results):
    rows = []
    for group in results:
        for cond in condition_list():
            m = results[group][cond]
            rows.append(dict(
                group=group, condition=cond,
                top1=round(m['top1'], 2) if m else '',
                mean_class_accuracy=round(m['mca'], 2)
                if m and m['mca'] is not None else ''))
    return rows


def build_mra(results):
    rows = []
    for group in results:
        accs = [results[group][c]['top1'] for c in condition_list()
                if results[group][c] is not None]
        mra = sum(accs) / len(accs) if accs else None
        clean = results[group].get('clean')
        clean_top1 = clean['top1'] if clean else None
        gap = round(clean_top1 - mra, 2) if (clean_top1 is not None
                                             and mra is not None) else None
        rows.append(dict(
            group=group, mra=round(mra, 2) if mra is not None else '',
            clean=round(clean_top1, 2) if clean_top1 is not None else '',
            gap_vs_clean=gap, n_conditions=len(condition_list())))
    return rows


def build_overfit(results):
    rows = []
    base = 'A0_clean'
    if base not in results:
        return rows
    for group in GROUP_DEGRADE:
        if group not in results:
            continue
        matched = GROUP_DEGRADE[group]
        others = [t for t in DEGRADE_TYPES if t != matched]
        matched_gain = (tier_mean(results, group, matched)
                        - tier_mean(results, base, matched))
        other_gain = sum(tier_mean(results, group, t)
                         - tier_mean(results, base, t) for t in others) / len(others)
        clean_gain = (results[group]['clean']['top1']
                      - results[base]['clean']['top1'])
        rows.append(dict(
            group=group, matched_degradation=matched,
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
    groups = expand_groups(args)
    if not groups:
        print('No group directories found under {}'.format(args.results_root))
        return

    detail, results, missing = collect(args, groups)

    summary = build_summary(results)
    mra = build_mra(results)
    overfit = build_overfit(results)

    write_csv(os.path.join(args.out_dir, 'stage4_summary.csv'), summary)
    write_csv(os.path.join(args.out_dir, 'stage4_mra.csv'), mra)
    write_csv(os.path.join(args.out_dir, 'stage4_overfit.csv'), overfit)

    report = dict(
        results_root=os.path.abspath(args.results_root),
        groups=groups,
        num_conditions=len(condition_list()),
        summary=summary, mra=mra, overfit=overfit,
        missing=missing)
    with open(os.path.join(args.out_dir, 'stage4_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('=== mRA (mean over {} conditions) ==='.format(len(condition_list())))
    for row in sorted(mra, key=lambda r: -(r['mra'] or 0)):
        print('  {:22s} mRA={:6} clean={:6} gap={}'.format(
            row['group'], row['mra'], row['clean'], row['gap_vs_clean']))

    if missing:
        print('\nWARNING: {} missing test.log'.format(len(missing)))
        for m in missing[:10]:
            print('  ' + m)

    print('\nOutputs:')
    for name in ('stage4_summary.csv', 'stage4_mra.csv',
                 'stage4_overfit.csv', 'stage4_report.json'):
        print('  ' + os.path.join(args.out_dir, name))


if __name__ == '__main__':
    main()
