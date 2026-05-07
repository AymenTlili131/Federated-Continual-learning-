#!/usr/bin/env python3
"""
post_experiment_analysis.py
----------------------------
Comprehensive post-training analysis pipeline.  Run this AFTER experiments
are complete (either via run_tournament.sh or direct parallel_training.py).

Steps:
  1. Load all experiment results from local experiments/ (or WandB via sync)
  2. Per-overlap RMT spectral analysis (effective rank, cond. number, KL-MP)
  3. Multipers divergence analysis (per-layer W₁ across loss types)
  4. Correlation analysis (weight correlation matrices by loss + layer)
  5. Summary CSV + figures → paper_results/

Usage:
  conda run -n FCL python3 scripts/post_experiment_analysis.py \
      [--model-size tiny] [--overlaps 0 1 2] [--output-dir paper_results/analysis/]
"""

import sys
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

NOTEBOOKS_SANDBOX = Path(__file__).parent.parent
sys.path.insert(0, str(NOTEBOOKS_SANDBOX))
sys.path.insert(0, str(NOTEBOOKS_SANDBOX / 'core_modules'))

from rmt_analysis import RandomMatrixAnalyzer
from multipers_analysis import (
    compute_multipers_features, compare_multipers_features, MultipersDivergence
)

LAYER_DELIMITERS = [208, 1414, 1514, 2254, 2464]
LAYER_NAMES      = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_experiment_results(experiments_dir: Path, model_size: str,
                            overlap: int) -> List[Dict]:
    """Load all results for a given model size and overlap."""
    pattern = f"{model_size}_overlap{overlap}_*"
    results = []

    for exp_dir in sorted(experiments_dir.glob(pattern)):
        loss_name = exp_dir.name[len(f"{model_size}_overlap{overlap}_"):]

        # Load WandB summary
        summary_file = exp_dir / 'wandb_summary.json'
        summary = {}
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)

        # Load training history
        history_file = exp_dir / 'training_history.json'
        history = {}
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)

        # Load test metrics CSV
        metrics_file = exp_dir / 'metrics' / 'test_metrics_full_and_layerwise.csv'
        metrics = {}
        if metrics_file.exists():
            mdf = pd.read_csv(metrics_file)
            metrics = mdf.iloc[0].to_dict() if len(mdf) > 0 else {}

        # Load tracking CSV (predicted/GT/FN weights for analysis)
        weights = None
        for csv_path in exp_dir.glob('tracking*.csv'):
            try:
                df = pd.read_csv(csv_path, nrows=50)
                w_cols = [c for c in df.columns if c not in ('label', 'epoch') and
                          c.replace('.', '', 1).lstrip('-').isdigit()]
                if len(w_cols) >= 2464:
                    weights = df[w_cols[:2464]].values.astype(np.float32)
            except Exception:
                pass
            break

        results.append({
            'loss_name': loss_name,
            'exp_dir': exp_dir,
            'summary': summary,
            'history': history,
            'metrics': metrics,
            'weights': weights,   # shape (N, 2464) or None
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# RMT analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_rmt_analysis(results: List[Dict], output_dir: Path, overlap: int):
    """RMT spectral analysis across all loss types for one overlap."""
    print(f"\n[RMT] Overlap {overlap}: {len(results)} experiments")
    analyzer = RandomMatrixAnalyzer()
    rmt_rows = []

    fig, axes = plt.subplots(len(LAYER_NAMES), 1,
                             figsize=(10, 3 * len(LAYER_NAMES)), squeeze=False)

    for res in results:
        if res['weights'] is None:
            continue
        w = res['weights'][0]   # take first sample

        rmt_out = analyzer.analyze_all_layers(w.astype(np.float64))
        for layer_name, layer_res in rmt_out.items():
            if layer_res.get('type') != 'weight':
                continue
            rmt_rows.append({
                'overlap': overlap,
                'loss': res['loss_name'],
                'layer': layer_name,
                'effective_rank': layer_res.get('effective_rank', np.nan),
                'condition_number': layer_res.get('condition_number', np.nan),
                'kl_mp': layer_res.get('kl_divergence_mp', np.nan),
                'spectral_radius': layer_res.get('max_eigenvalue', np.nan),
            })

    if not rmt_rows:
        print("  No weight data available for RMT analysis")
        plt.close(fig)
        return

    df_rmt = pd.DataFrame(rmt_rows)
    df_rmt.to_csv(output_dir / f'rmt_overlap{overlap}.csv', index=False)

    # Plot effective rank per layer
    for ax_row, layer in zip(axes, LAYER_NAMES):
        ax = ax_row[0]
        sub = df_rmt[df_rmt['layer'].str.contains(layer, na=False)]
        if sub.empty:
            ax.set_title(layer)
            continue
        ax.barh(sub['loss'], sub['effective_rank'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Effective Rank')
        ax.set_title(f'{layer} — effective rank by loss')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'rmt_effective_rank_overlap{overlap}.png', dpi=150)
    plt.close(fig)
    print(f"  → {output_dir}/rmt_overlap{overlap}.csv + rmt_effective_rank_overlap{overlap}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Multipers divergence
# ──────────────────────────────────────────────────────────────────────────────

def run_multipers_analysis(results: List[Dict], output_dir: Path, overlap: int,
                           n_lines: int = 20, subsample: int = 60):
    """Multipers divergence between loss types."""
    print(f"\n[MULTIPERS] Overlap {overlap}")
    ref_name = 'MSE'
    ref = next((r for r in results if r['loss_name'] == ref_name and r['weights'] is not None), None)

    rows = []
    fig, axes = plt.subplots(1, len(LAYER_NAMES), figsize=(4 * len(LAYER_NAMES), 4))

    for res in results:
        if res['weights'] is None or res is ref:
            continue
        w_test = res['weights'][0].astype('float32')
        w_ref  = (ref['weights'][0].astype('float32')
                  if ref is not None else np.zeros(2464, dtype='float32'))

        f_test = compute_multipers_features(w_test, subsample_per_layer=subsample, n_lines=n_lines)
        f_ref  = compute_multipers_features(w_ref,  subsample_per_layer=subsample, n_lines=n_lines)
        dists  = compare_multipers_features(f_test, f_ref)

        for layer, dist in dists.items():
            rows.append({
                'overlap': overlap,
                'loss': res['loss_name'],
                'ref': ref_name,
                'layer': layer,
                'wasserstein_1': dist,
            })

    if not rows:
        print("  No multipers data available")
        plt.close(fig)
        return

    df_mp = pd.DataFrame(rows)
    df_mp.to_csv(output_dir / f'multipers_overlap{overlap}.csv', index=False)

    for ax, layer in zip(axes, LAYER_NAMES):
        sub = df_mp[df_mp['layer'] == layer]
        if sub.empty:
            ax.set_title(layer)
            continue
        sub_sorted = sub.sort_values('wasserstein_1', ascending=False)
        ax.barh(sub_sorted['loss'], sub_sorted['wasserstein_1'], color='darkorange', alpha=0.7)
        ax.set_xlabel('W₁ vs MSE')
        ax.set_title(layer)
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Multipers W₁ divergence from MSE — overlap {overlap}', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f'multipers_divergence_overlap{overlap}.png', dpi=150)
    plt.close(fig)
    print(f"  → multipers_overlap{overlap}.csv + multipers_divergence_overlap{overlap}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Correlation analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_correlation_analysis(results: List[Dict], output_dir: Path, overlap: int):
    """Per-layer weight correlation matrices across loss types."""
    print(f"\n[CORRELATION] Overlap {overlap}")
    all_rows = {}   # layer -> weight matrix (n_experiments × layer_dim)

    for res in results:
        if res['weights'] is None:
            continue
        w = res['weights'][0].astype(np.float64)
        prev = 0
        for i, (end, layer) in enumerate(zip(LAYER_DELIMITERS, LAYER_NAMES)):
            seg = w[prev:end]
            if layer not in all_rows:
                all_rows[layer] = {'names': [], 'segs': []}
            all_rows[layer]['names'].append(res['loss_name'])
            all_rows[layer]['segs'].append(seg)
            prev = end

    corr_rows = []
    fig, axes = plt.subplots(1, len(LAYER_NAMES),
                             figsize=(4 * len(LAYER_NAMES), 4))

    for ax, layer in zip(axes, LAYER_NAMES):
        if layer not in all_rows or len(all_rows[layer]['segs']) < 2:
            ax.set_title(layer)
            continue
        mat = np.array(all_rows[layer]['segs'])        # (n_losses, layer_dim)
        names = all_rows[layer]['names']
        corr_mat = np.corrcoef(mat)                    # (n_losses, n_losses)

        # Save CSV
        pd.DataFrame(corr_mat, index=names, columns=names).to_csv(
            output_dir / f'corr_{layer}_overlap{overlap}.csv'
        )

        # Heatmap
        sns.heatmap(corr_mat, ax=ax, annot=(len(names) <= 15),
                    fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                    xticklabels=names, yticklabels=names, linewidths=0.3)
        ax.set_title(layer)
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', rotation=0, labelsize=6)

        # Stats
        upper = corr_mat[np.triu_indices(len(names), k=1)]
        corr_rows.append({
            'overlap': overlap,
            'layer': layer,
            'mean_abs_corr': np.mean(np.abs(upper)),
            'max_corr': np.max(upper),
            'min_corr': np.min(upper),
            'pct_high_corr': np.mean(np.abs(upper) > 0.7) * 100,
        })

    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(
            output_dir / f'corr_summary_overlap{overlap}.csv', index=False
        )

    plt.suptitle(f'Weight correlation by loss type — overlap {overlap}', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f'correlation_heatmaps_overlap{overlap}.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  → corr_*_overlap{overlap}.csv + correlation_heatmaps_overlap{overlap}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Post-experiment analysis pipeline')
    parser.add_argument('--experiments-dir', type=Path,
                        default=NOTEBOOKS_SANDBOX / 'experiments')
    parser.add_argument('--output-dir', type=Path,
                        default=NOTEBOOKS_SANDBOX / 'paper_results' / 'analysis')
    parser.add_argument('--model-size', default='tiny')
    parser.add_argument('--overlaps', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--n-lines', type=int, default=20,
                        help='Number of projection lines for multipers')
    parser.add_argument('--subsample', type=int, default=60,
                        help='Subsample per layer for multipers')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {args.output_dir}")
    print(f"Experiments → {args.experiments_dir}")

    all_summary = []
    for overlap in args.overlaps:
        print(f"\n{'='*60}")
        print(f"OVERLAP {overlap}")
        print(f"{'='*60}")

        results = load_experiment_results(args.experiments_dir, args.model_size, overlap)
        print(f"  Found {len(results)} experiments")
        if not results:
            print("  (no results — run sync_wandb_results.py first)")
            continue

        overlap_dir = args.output_dir / f'overlap{overlap}'
        overlap_dir.mkdir(exist_ok=True)

        run_rmt_analysis(results, overlap_dir, overlap)
        run_multipers_analysis(results, overlap_dir, overlap,
                               n_lines=args.n_lines, subsample=args.subsample)
        run_correlation_analysis(results, overlap_dir, overlap)

        # Collect summary metrics
        for res in results:
            m = res['metrics']
            if m:
                all_summary.append({
                    'model_size': args.model_size,
                    'overlap': overlap,
                    'loss': res['loss_name'],
                    'val_loss': res['summary'].get('val_loss', np.nan),
                    'cnn_accuracy': m.get('cnn_accuracy', np.nan),
                    'cnn_improvement': m.get('cnn_improvement', np.nan),
                    'mse': m.get('mse', np.nan),
                    'cosine_similarity': m.get('cosine_similarity', np.nan),
                })

    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(args.output_dir / 'full_results_summary.csv', index=False)
        print(f"\nFull results summary → {args.output_dir}/full_results_summary.csv")

        # Print rankings
        print("\n=== Per-overlap ranking (by cnn_improvement) ===")
        for overlap in args.overlaps:
            sub = summary_df[summary_df['overlap'] == overlap].dropna(subset=['cnn_improvement'])
            if sub.empty:
                continue
            sub_sorted = sub.sort_values('cnn_improvement', ascending=False)
            print(f"\nOverlap {overlap}:")
            print(sub_sorted[['loss', 'cnn_improvement', 'val_loss']].to_string(index=False))


if __name__ == '__main__':
    main()
