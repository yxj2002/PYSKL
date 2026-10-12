#!/usr/bin/env python
"""Run B_aug final checkpoints through the formal 16-condition xsub_val matrix.

B_aug (frozen): mixed + prob=0.3 + S3 + 16 epoch, trained on FULL xsub_train.
Supported models (work-5 protocol):
    - ST-GCN++ J  x seeds {255, 2026, 3407}   (main 3-seed report)
    - CTR-GCN  J  x seed(s)                   (transfer to method carrier)

Each run evaluates the best checkpoint of one seed on the frozen stage-3
matrix (1 clean + 5 degradations x 3 severities) with xsub_val, writing
results to ``work_dirs/aug_baseline/b_aug/<model>_j/results/final_xsub/``.

Resumable via a per-model manifest; supports --dry-run / --models / --seeds /
--max-runs.  Uses ``torch.distributed.run --standalone`` (single GPU),
matching stage 3.  Run in the pyskl training environment (mmcv required for
config validation).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import run_stage3_robustness as stage3


SEEDS = (255, 2026, 3407)
MODELS = ('stgcnpp', 'ctrgcn')
B_AUG_ROOT = os.path.join('work_dirs', 'aug_baseline', 'b_aug')
CONFIG_ROOT = os.path.join('configs', 'robust_skeleton', 'ntu60_xsub_3dkp')
_CKPT_RE = re.compile(r'^best_top1_acc_epoch_(\d+)\.pth$')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--models', nargs='+', choices=MODELS, default=['stgcnpp'],
        help='Models to test. Default: stgcnpp (3-seed main report). Add '
             'ctrgcn for the CTR-GCN transfer evaluation.')
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS,
                        default=list(SEEDS))
    parser.add_argument('--max-runs', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def work_root(model):
    return os.path.join(B_AUG_ROOT, '{}_j'.format(model))


def results_root(model):
    return os.path.join(work_root(model), 'results', 'final_xsub')


def conditions():
    values = ['clean']
    for severity in stage3.SEVERITIES:
        for degrade_type in stage3.DEGRADE_TYPES:
            values.append('{}_{}'.format(degrade_type, severity))
    return values


def _find_checkpoint(model, seed):
    """Locate the best-accuracy checkpoint in a B_aug seed directory.

    Picks the highest-epoch ``best_top1_acc_epoch_<N>.pth`` so that leftover
    checkpoints from an interrupted earlier run (e.g. seed2026/...epoch_1.pth)
    never shadow the final one.
    """
    ckpt_dir = os.path.join(work_root(model), 'seed{}'.format(seed))
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            'B_aug seed directory missing: {} (train or sync it first)'.format(
                stage3._normalized(ckpt_dir)))
    candidates = []
    for fname in os.listdir(ckpt_dir):
        match = _CKPT_RE.match(fname)
        if match:
            candidates.append((int(match.group(1)), fname))
    if not candidates:
        raise FileNotFoundError(
            'No best_top1_acc checkpoint in {}'.format(
                stage3._normalized(ckpt_dir)))
    return os.path.join(ckpt_dir, max(candidates)[1])


def build_runs(args):
    runs = []
    for model in args.models:
        for seed in args.seeds:
            checkpoint = _find_checkpoint(model, seed)
            for condition in conditions():
                config = stage3._config_path(CONFIG_ROOT, model, condition)
                output_dir = os.path.join(results_root(model),
                                          'seed{}'.format(seed), condition)
                output = os.path.join(output_dir, 'result.pkl')
                log = os.path.join(output_dir, 'test.log')
                command = [sys.executable, '-m', 'torch.distributed.run',
                           '--standalone', '--nproc_per_node=1',
                           os.path.join('tools', 'test.py'), config,
                           '-C', checkpoint, '--out', output,
                           '--eval', 'top_k_accuracy', 'mean_class_accuracy']
                runs.append(dict(model=model, seed=seed, condition=condition,
                                 config=config, checkpoint=checkpoint,
                                 output=output, log=log, command=command))
    return runs


def _manifest_path(model):
    return os.path.join(results_root(model), 'b_aug_xsub_test_manifest.json')


def write_manifest(path, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(manifest, file_obj, indent=2)


def command_text(command):
    return ' '.join('"{}"'.format(part) if ' ' in part else part
                    for part in command)


def run_model(model, model_runs, args):
    """Execute (or resume) all runs of one model; resumable via manifest."""
    manifest_path = _manifest_path(model)
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding='utf-8') as file_obj:
            manifest = json.load(file_obj)
    if manifest is None:
        manifest = dict(status='pending',
                        ann_file=stage3._normalized(stage3.ANN_FILE),
                        seeds=list(args.seeds), conditions=conditions(),
                        runs=[])
    existing = {run['output']: run for run in manifest.get('runs', [])}
    for run in model_runs:
        old_run = existing.get(run['output'])
        if old_run and old_run.get('returncode') == 0:
            run.update(old_run)
    manifest['runs'] = model_runs
    write_manifest(manifest_path, manifest)

    print('\n=== B_aug formal xsub test: {} ({} runs) ==='.format(
        model, len(model_runs)))
    if args.dry_run:
        print('[dry-run] No outputs or manifest will be written.')
        for index, run in enumerate(model_runs, 1):
            print('[{}/{}] seed{} {}: {}'.format(
                index, len(model_runs), run['seed'], run['condition'],
                command_text(run['command'])))
        return True

    launched = 0
    for index, run in enumerate(model_runs, 1):
        if run.get('returncode') == 0 and os.path.isfile(run['output']):
            print('[{}/{}] SKIP complete: seed{} {}'.format(
                index, len(model_runs), run['seed'], run['condition']))
            continue
        if args.max_runs is not None and launched >= args.max_runs:
            manifest['status'] = 'partial'
            write_manifest(manifest_path, manifest)
            print('STOP: reached --max-runs {}'.format(args.max_runs))
            return False
        os.makedirs(os.path.dirname(run['log']), exist_ok=True)
        print('[{}/{}] RUN seed{} {}'.format(
            index, len(model_runs), run['seed'], run['condition']))
        start = time.time()
        with open(run['log'], 'w', encoding='utf-8') as log_file:
            completed = subprocess.run(run['command'], stdout=log_file,
                                       stderr=subprocess.STDOUT)
        run['returncode'] = completed.returncode
        run['elapsed_seconds'] = time.time() - start
        launched += 1
        write_manifest(manifest_path, manifest)
        if completed.returncode != 0:
            manifest['status'] = 'failed'
            write_manifest(manifest_path, manifest)
            raise RuntimeError(
                'seed{} {} failed (return code {}); inspect {}'.format(
                    run['seed'], run['condition'], completed.returncode,
                    stage3._normalized(run['log'])))

    manifest['status'] = 'completed'
    write_manifest(manifest_path, manifest)
    print('PASS: {} -> all {} runs completed. Manifest: {}'.format(
        model, len(model_runs), stage3._normalized(manifest_path)))
    return True


def main():
    args = parse_args()
    runs = build_runs(args)
    stage3._preflight(runs)
    stage3._validate_configs(runs)
    assert len(conditions()) == 16
    total = len(args.models) * len(args.seeds) * 16
    assert len(runs) == total
    print('Models: {} | Seeds: {} | Conditions: 16 | Total runs: {}'.format(
        ', '.join(args.models),
        ', '.join(str(seed) for seed in args.seeds), len(runs)))
    for model in args.models:
        run_model(model, [r for r in runs if r['model'] == model], args)


if __name__ == '__main__':
    main()
