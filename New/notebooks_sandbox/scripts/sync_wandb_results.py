#!/usr/bin/env python3
"""
sync_wandb_results.py
---------------------
Pull completed runs from the WandB project 'fcl-advanced' (entity: aymentlili)
and write them to the local experiments/ directory structure expected by
per_overlap_ranking.py.

Also supports --local-only mode which reads directly from the local WandB run
directories (notebooks_sandbox/wandb/wandb/run-*/) without any API calls —
useful for offline use or when runs were never fully synced.

Output per run:
  experiments/{model_size}_overlap{N}_{loss}/training_history.json
  experiments/{model_size}_overlap{N}_{loss}/metrics/test_metrics_full_and_layerwise.csv
  experiments/{model_size}_overlap{N}_{loss}/wandb_summary.json

Usage:
  # Sync from cloud (all finished runs):
  conda run -n FCL python3 scripts/sync_wandb_results.py

  # Show status of all runs without writing files:
  conda run -n FCL python3 scripts/sync_wandb_results.py --status

  # Offline parse from local WandB cache:
  conda run -n FCL python3 scripts/sync_wandb_results.py --local-only

  # Sync crashed runs too:
  conda run -n FCL python3 scripts/sync_wandb_results.py --state all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

NOTEBOOKS_SANDBOX = Path(__file__).parent.parent
# WandB saves to WANDB_DIR/wandb/ (adds its own subdirectory)
LOCAL_WANDB_DIR = NOTEBOOKS_SANDBOX / 'wandb' / 'wandb'

sys.path.insert(0, str(NOTEBOOKS_SANDBOX / 'core_modules'))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_run_name(run_name: str) -> Optional[Dict]:
    """Extract model_size, overlap, loss from run name like 'tiny_overlap0_MSE'."""
    parts = run_name.split('_')
    if len(parts) < 3:
        return None
    model_size = parts[0]
    overlap_part = next((p for p in parts if p.startswith('overlap')), None)
    if overlap_part is None:
        return None
    try:
        overlap = int(overlap_part.replace('overlap', ''))
    except ValueError:
        return None
    loss_start = parts.index(overlap_part) + 1
    loss_name = '_'.join(parts[loss_start:])
    return {'model_size': model_size, 'overlap': overlap, 'loss': loss_name}


def write_experiment_files(exp_dir: Path, summary: dict, history: dict):
    """Write wandb_summary.json, training_history.json, test_metrics CSV."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = exp_dir / 'metrics'
    metrics_dir.mkdir(exist_ok=True)

    with open(exp_dir / 'wandb_summary.json', 'w') as f:
        json.dump({k: v for k, v in summary.items()
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool, type(None)))},
                  f, indent=2)

    if history:
        with open(exp_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    metric_keys = {
        'mse':              summary.get('test/mse',             summary.get('val_loss', float('inf'))),
        'mae':              summary.get('test/mae',             float('inf')),
        'cosine_similarity': summary.get('test/cosine_sim',    0.0),
        'wasserstein':      summary.get('test/wasserstein',     float('inf')),
        'gw_distance':      summary.get('topology/gw_distance', float('inf')),
        'cnn_accuracy':     summary.get('cnn/avg_final_acc_id', 0.0),
        'cnn_improvement':  summary.get('cnn/avg_improvement',  0.0),
        'final_epoch':      summary.get('epoch',                0),
    }
    pd.DataFrame([metric_keys]).to_csv(
        metrics_dir / 'test_metrics_full_and_layerwise.csv', index=False
    )


# ──────────────────────────────────────────────────────────────────────────────
# Local (offline) sync
# ──────────────────────────────────────────────────────────────────────────────

def _extract_name_from_local_run(run_dir: Path) -> Optional[str]:
    """Read config.yaml to reconstruct experiment name from CLI args."""
    cfg_file = run_dir / 'files' / 'config.yaml'
    if not cfg_file.exists() or not YAML_AVAILABLE:
        return None
    try:
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        if not cfg:
            return None
        wdata = cfg.get('_wandb', {}).get('value', {})
        for eid, edata in wdata.get('e', {}).items():
            args = edata.get('args', [])
            if '--loss' in args and '--overlap' in args and '--model-size' in args:
                model_size = args[args.index('--model-size') + 1]
                overlap    = args[args.index('--overlap') + 1]
                loss       = args[args.index('--loss') + 1]
                return f"{model_size}_overlap{overlap}_{loss}"
        # Fallback: look for exp_name key
        if 'exp_name' in cfg:
            return cfg['exp_name'].get('value')
    except Exception:
        pass
    return None


def sync_local(experiments_dir: Path, wandb_dir: Path, dry_run: bool = False):
    """Parse local WandB run directories without API calls."""
    print(f"Local WandB dir: {wandb_dir}")
    if not wandb_dir.exists():
        print("ERROR: local wandb dir not found.")
        return

    run_dirs = sorted(wandb_dir.glob('run-*'))
    print(f"Found {len(run_dirs)} local run directories\n")

    ok, skip, crash = 0, 0, 0
    for run_dir in run_dirs:
        name = _extract_name_from_local_run(run_dir)
        summary_file = run_dir / 'files' / 'wandb-summary.json'
        history_file = run_dir / 'files' / 'wandb-history.jsonl'

        summary = {}
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
            except Exception:
                pass

        val_loss = summary.get('val_loss')
        epoch    = summary.get('epoch')
        state    = 'finished' if val_loss is not None else 'crashed'

        if name is None:
            print(f"  SKIP  {run_dir.name} (name unknown, state={state})")
            skip += 1
            continue

        info = parse_run_name(name)
        if info is None:
            print(f"  SKIP  {name} (cannot parse)")
            skip += 1
            continue

        exp_key = f"{info['model_size']}_overlap{info['overlap']}_{info['loss']}"
        exp_dir = experiments_dir / exp_key

        if state == 'crashed':
            print(f"  CRASH {name:45s} | no results")
            crash += 1
            continue

        # Parse history from jsonl
        history = {'train_loss': [], 'val_loss': [], 'epochs': []}
        if history_file.exists():
            try:
                with open(history_file) as f:
                    for line in f:
                        row = json.loads(line.strip())
                        if 'train_loss' in row:
                            history['train_loss'].append(row['train_loss'])
                        if 'val_loss' in row:
                            history['val_loss'].append(row['val_loss'])
                        if 'epoch' in row:
                            history['epochs'].append(int(row['epoch']))
            except Exception:
                pass

        vl_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
        print(f"  OK    {name:45s} | ep={epoch} | val={vl_str}")

        if not dry_run:
            write_experiment_files(exp_dir, summary, history)
        ok += 1

    print(f"\nLocal sync: {ok} written, {crash} crashed, {skip} skipped")


# ──────────────────────────────────────────────────────────────────────────────
# Cloud (API) sync
# ──────────────────────────────────────────────────────────────────────────────

def sync_cloud(experiments_dir: Path, project: str, entity: str,
               state_filter: str, dry_run: bool = False):
    """Sync from WandB cloud API."""
    if not WANDB_AVAILABLE:
        print("ERROR: wandb not installed. Run: pip install wandb")
        return

    api = wandb.Api(timeout=60)
    project_path = f"{entity}/{project}"
    filters = {} if state_filter == 'all' else {'state': state_filter}

    print(f"Querying WandB: {project_path}  (filter state={state_filter})")
    runs = list(api.runs(project_path, filters=filters))
    print(f"Found {len(runs)} runs matching filter\n")

    ok, skip = 0, 0
    for run in runs:
        info = parse_run_name(run.name)
        if info is None:
            print(f"  SKIP  {run.name:45s} (cannot parse)")
            skip += 1
            continue

        exp_key = f"{info['model_size']}_overlap{info['overlap']}_{info['loss']}"
        exp_dir = experiments_dir / exp_key
        summary = dict(run.summary)
        val_loss = summary.get('val_loss')
        vl_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else '?'

        print(f"  {'DRY' if dry_run else 'OK':5s} {run.name:45s} | state={run.state:8s} | val={vl_str}")

        if dry_run:
            ok += 1
            continue

        history = {'train_loss': [], 'val_loss': [], 'epochs': []}
        try:
            hist_df = run.history(samples=5000, keys=['train_loss', 'val_loss', 'epoch'])
            if not hist_df.empty:
                history['train_loss'] = hist_df['train_loss'].dropna().tolist()
                history['val_loss']   = hist_df['val_loss'].dropna().tolist()
                history['epochs']     = hist_df['epoch'].dropna().astype(int).tolist()
        except Exception as e:
            print(f"    history warn: {e}")

        write_experiment_files(exp_dir, summary, history)
        ok += 1

    print(f"\nCloud sync: {ok} {'would sync' if dry_run else 'synced'}, {skip} skipped")


# ──────────────────────────────────────────────────────────────────────────────
# Status report
# ──────────────────────────────────────────────────────────────────────────────

def print_status(project: str, entity: str):
    """Print a quick status table of all runs in the project."""
    if not WANDB_AVAILABLE:
        print("wandb not installed — cannot query cloud status")
        return

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}"))
    print(f"\n{'='*80}")
    print(f"PROJECT: {entity}/{project}   ({len(runs)} total runs)")
    print(f"{'='*80}")
    print(f"{'NAME':<45} {'STATE':<10} {'EPOCH':>6} {'VAL_LOSS':>10} {'CNN_ACC':>8}")
    print('-' * 80)

    by_overlap = {}
    for run in sorted(runs, key=lambda r: r.name):
        s = dict(run.summary)
        epoch    = s.get('epoch', '')
        val_loss = s.get('val_loss', '')
        cnn_acc  = s.get('cnn/avg_final_acc_id', '')
        vl_str   = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
        ca_str   = f"{cnn_acc:.2f}"  if isinstance(cnn_acc, float)  else str(cnn_acc)
        info = parse_run_name(run.name)
        overlap  = info['overlap'] if info else '?'

        print(f"{run.name:<45} {run.state:<10} {str(epoch):>6} {vl_str:>10} {ca_str:>8}")
        key = str(overlap)
        by_overlap.setdefault(key, {'finished': 0, 'crashed': 0, 'other': 0})
        if run.state == 'finished':
            by_overlap[key]['finished'] += 1
        elif run.state in ('crashed', 'failed'):
            by_overlap[key]['crashed'] += 1
        else:
            by_overlap[key]['other'] += 1

    print('\n=== Summary by overlap ===')
    for ov in sorted(by_overlap):
        d = by_overlap[ov]
        print(f"  overlap {ov}: {d['finished']} finished, {d['crashed']} crashed, {d['other']} other")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sync WandB fcl-advanced results to experiments/")
    parser.add_argument('--project',  default='fcl-advanced',  help='WandB project name')
    parser.add_argument('--entity',   default='aymentlili',    help='WandB entity/username')
    parser.add_argument('--experiments-dir', type=Path,
                        default=NOTEBOOKS_SANDBOX / 'experiments')
    parser.add_argument('--local-wandb-dir', type=Path,
                        default=LOCAL_WANDB_DIR,
                        help='Path to local wandb/wandb/ run directories')
    parser.add_argument('--state',    default='finished',
                        choices=['finished', 'crashed', 'failed', 'running', 'all'])
    parser.add_argument('--local-only', action='store_true',
                        help='Read from local WandB cache — no API calls required')
    parser.add_argument('--status',   action='store_true',
                        help='Print run status table from cloud and exit')
    parser.add_argument('--dry-run',  action='store_true')
    args = parser.parse_args()

    if args.status:
        print_status(args.project, args.entity)
        return

    if args.local_only:
        sync_local(args.experiments_dir, args.local_wandb_dir, dry_run=args.dry_run)
    else:
        sync_cloud(args.experiments_dir, args.project, args.entity,
                   args.state, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
