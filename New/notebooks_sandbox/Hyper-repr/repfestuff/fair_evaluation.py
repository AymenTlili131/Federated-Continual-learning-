#!/usr/bin/env python3
"""
fair_evaluation.py
------------------
Evaluate ALL trained models on a SINGLE fixed test set with a SINGLE fixed
100-sample finetune split, so every loss function is ranked fairly.

For each experiment with best_model.pth:
  1. Load the saved model (best_model.pth)
  2. Re-run inference on the unified test pairs
  3. Reconstruct and finetune 100 CNNs (same samples, same seed for each experiment)
  4. Record: cnn_accuracy, cnn_improvement, weight_mse, weight_cosine,
             weight_wasserstein, topology features (if --topo)

Results → paper_results/fair_eval/
  - fair_eval_all.csv              (one row per experiment)
  - fair_eval_overlap{N}.csv       (per-overlap ranking)
  - fair_eval_summary.png          (bar chart per overlap)

Usage:
  # Full run
  conda run -n FCL python3 scripts/fair_evaluation.py

  # Quick dry-run (no CNN finetune, just weight-space metrics)
  conda run -n FCL python3 scripts/fair_evaluation.py --no-cnn

  # Only specific overlaps
  conda run -n FCL python3 scripts/fair_evaluation.py --overlaps 0 1 2

  # Resume (skip already-evaluated experiments)
  conda run -n FCL python3 scripts/fair_evaluation.py --resume
"""

import sys
import os
import json
import argparse
import warnings
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import wasserstein_distance

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Path setup (mirrors run_advanced_experiments.py)
# ──────────────────────────────────────────────────────────────────────────────

SCRIPTS_DIR       = Path(__file__).parent
NOTEBOOKS_SANDBOX = SCRIPTS_DIR.parent
PROJECT_ROOT      = NOTEBOOKS_SANDBOX.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(NOTEBOOKS_SANDBOX))
sys.path.insert(0, str(NOTEBOOKS_SANDBOX / 'core_modules'))

from Double_input_transformer import TransformerAE
from config import MODEL_CONFIGS
try:
    from weight_normalization import LayerWiseNormalizer as WeightNormalizer
except ImportError:
    WeightNormalizer = None

MNIST_ROOT  = str(PROJECT_ROOT / 'data' / 'SplitMnist')
ZOO_CSV     = PROJECT_ROOT / 'data' / 'Merged zoo.csv'
EXPERIMENTS = NOTEBOOKS_SANDBOX / 'experiments'

# Fixed random seed for reproducible fair evaluation
FAIR_EVAL_SEED      = 42
N_FINETUNE_SAMPLES  = 100     # same 100 samples for all experiments
N_FINETUNE_EPOCHS   = 5
LAYER_DELIMITERS    = [208, 1414, 1514, 2254, 2464]


# ──────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[TransformerAE, dict]:
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = TransformerAE(
        max_seq_len=cfg.max_seq_len,
        N=cfg.N,
        heads=cfg.heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        neck=cfg.neck,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ck['model_state_dict'])
    model = model.to(device)
    model.eval()
    meta = {
        'epoch':     ck.get('epoch', '?'),
        'val_loss':  ck.get('val_loss', float('inf')),
        'train_loss': ck.get('train_loss', float('inf')),
        'config':    cfg,
    }
    return model, meta


# ──────────────────────────────────────────────────────────────────────────────
# Load unified test data (same for every experiment)
# ──────────────────────────────────────────────────────────────────────────────

_cached_test_data: Dict = {}

def _build_weight_index(df: pd.DataFrame, weight_cols: list):
    """Pre-build O(1) weight lookup index — mirrors run_advanced_experiments.py."""
    weight_matrix = df[weight_cols].values.astype(np.float32)
    labels_arr    = df['label'].values
    epochs_arr    = df['epoch'].values

    label_epoch_idx: Dict = {}
    for act_col in ['leakyrelu', 'relu', 'tanh', 'sigmoid']:
        if act_col not in df.columns:
            continue
        act_mask = (df[act_col].values == 1.0)
        idx_map: Dict = {}
        for row_idx in np.where(act_mask)[0]:
            lbl = labels_arr[row_idx]
            ep  = int(epochs_arr[row_idx])
            if lbl not in idx_map:
                idx_map[lbl] = {}
            if ep not in idx_map[lbl]:
                idx_map[lbl][ep] = row_idx
        label_epoch_idx[act_col] = idx_map

    def lookup(label_str: str, activation: str = 'leakyrelu', epoch: int = 21):
        idx_map = label_epoch_idx.get(activation, {}).get(label_str, {})
        for target in range(epoch, 10, -5):
            if target in idx_map:
                return weight_matrix[idx_map[target]]
        return None

    return lookup


def load_unified_test_data(
    model_size: str,
    overlap: int,
    n_samples: int = N_FINETUNE_SAMPLES,
    seed: int = FAIR_EVAL_SEED,
) -> Dict:
    """Load & cache the fixed test set for a given (model_size, overlap) pair."""
    cache_key = f"{model_size}_{overlap}"
    if cache_key in _cached_test_data:
        return _cached_test_data[cache_key]

    # Load scenario test pairs (pre-generated .npy files)
    scenario_dir = PROJECT_ROOT / 'data' / 'Scenario' / f'overlapping_m{overlap}'
    test_pairs_file = scenario_dir / 'test_pairs.npy'
    if not test_pairs_file.exists():
        raise FileNotFoundError(f"Scenario not found: {test_pairs_file}")
    test_pairs = np.load(test_pairs_file, allow_pickle=True)

    print(f"  Scenario: {scenario_dir} — {len(test_pairs)} test pairs")
    print(f"  Loading zoo CSV (one-time, cached)…")
    df = pd.read_csv(ZOO_CSV)
    weight_cols = list(df.columns[17:-2])  # metadata cols 0-16, last 2 = Accuracy+epoch

    lookup = _build_weight_index(df, weight_cols)

    x1_list, x2_list, y_list, meta_list = [], [], [], []
    for pair in test_pairs:
        task1, task2 = pair
        task_combined = sorted(set(task1) | set(task2))
        w1 = lookup(str(task1))
        w2 = lookup(str(task2))
        wy = lookup(str(task_combined))
        if w1 is not None and w2 is not None and wy is not None:
            x1_list.append(w1)
            x2_list.append(w2)
            y_list.append(wy)
            meta_list.append({'task1': task1, 'task2': task2,
                              'task_combined': task_combined})

    if not x1_list:
        raise ValueError(f"No test pairs resolved from zoo for overlap {overlap}")

    x1 = np.array(x1_list, dtype=np.float32)
    x2 = np.array(x2_list, dtype=np.float32)
    y  = np.array(y_list,  dtype=np.float32)

    # Load saved normalizer (from any finished experiment for this overlap)
    import pickle
    norm_path = next(
        EXPERIMENTS.glob(f"{model_size}_overlap{overlap}_*/weight_normalizer.pkl"),
        None
    )
    normalizer = None
    if norm_path:
        try:
            with open(norm_path, 'rb') as f:
                normalizer = pickle.load(f)
            x1_norm = normalizer.transform(x1)
            x2_norm = normalizer.transform(x2)
            y_norm  = normalizer.transform(y)
        except Exception:
            normalizer = None

    if normalizer is None:
        mu  = x1.mean(0);  sig = x1.std(0) + 1e-8
        x1_norm = (x1 - mu) / sig
        x2_norm = (x2 - mu) / sig
        y_norm  = (y  - mu) / sig

    # Deterministic 100-sample subset (same for every experiment)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x1), min(n_samples, len(x1)), replace=False)

    data = {
        'x1':      x1[idx],       'x2':      x2[idx],      'y':      y[idx],
        'x1_norm': x1_norm[idx],  'x2_norm': x2_norm[idx], 'y_norm': y_norm[idx],
        'meta':    [meta_list[i] for i in idx],
        'normalizer': normalizer,
        'n_total_pairs': len(x1_list),
    }
    _cached_test_data[cache_key] = data
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Weight-space metrics
# ──────────────────────────────────────────────────────────────────────────────

def weight_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """MSE, cosine similarity, Wasserstein-1, layerwise MSEs."""
    mse  = float(np.mean((pred - gt) ** 2))
    cos  = float(np.dot(pred.ravel(), gt.ravel()) /
                 (np.linalg.norm(pred) * np.linalg.norm(gt) + 1e-12))
    w1   = float(wasserstein_distance(pred.ravel(), gt.ravel()))

    layer_mse = {}
    prev = 0
    layer_names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
    for name, end in zip(layer_names, LAYER_DELIMITERS):
        layer_mse[f'mse_{name}'] = float(np.mean((pred[prev:end] - gt[prev:end]) ** 2))
        prev = end
    return {'mse': mse, 'cosine': cos, 'wasserstein': w1, **layer_mse}


# ──────────────────────────────────────────────────────────────────────────────
# CNN finetune evaluation
# ──────────────────────────────────────────────────────────────────────────────

def cnn_eval_single(pred_w: np.ndarray, meta: dict, run_cnn: bool = True) -> Dict:
    """CNN reconstruct + finetune for one predicted weight vector."""
    if not run_cnn:
        return {'cnn_accuracy': float('nan'), 'cnn_baseline': float('nan'),
                'cnn_improvement': float('nan')}
    try:
        from cnn_reconstruction import finetune_reconstructed_cnn
        task_classes = meta.get('task_combined', list(range(10)))
        if isinstance(task_classes, str):
            import ast
            task_classes = ast.literal_eval(task_classes)

        result = finetune_reconstructed_cnn(
            predicted_weights=pred_w,
            task_classes=task_classes,
            activation='leakyrelu',
            mnist_root=MNIST_ROOT,
            n_finetune_epochs=N_FINETUNE_EPOCHS,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        final_acc  = result.get('final_id_accuracy', float('nan'))
        init_acc   = result.get('initial_id_accuracy', float('nan'))
        baseline   = result.get('baseline_accuracy', float('nan'))
        improvement = final_acc - init_acc if not (np.isnan(final_acc) or np.isnan(init_acc)) else float('nan')
        return {
            'cnn_accuracy': final_acc,
            'cnn_baseline': baseline,
            'cnn_improvement': improvement,
            'cnn_initial_acc': init_acc,
        }
    except Exception as e:
        return {'cnn_accuracy': float('nan'), 'cnn_baseline': float('nan'),
                'cnn_improvement': float('nan'), 'cnn_error': str(e)[:80]}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate one experiment
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_experiment(
    exp_dir: Path,
    device: torch.device,
    run_cnn: bool = True,
    n_cnn_samples: int = 20,
) -> Optional[Dict]:
    """Evaluate one experiment directory. Returns metrics dict or None on error."""
    ckpt_path = exp_dir / 'checkpoints' / 'best_model.pth'
    if not ckpt_path.exists():
        return None

    parts = exp_dir.name.split('_')
    model_size   = parts[0]
    overlap_part = next((p for p in parts if p.startswith('overlap')), None)
    if overlap_part is None:
        return None
    overlap   = int(overlap_part.replace('overlap', ''))
    loss_name = '_'.join(parts[parts.index(overlap_part) + 1:])

    try:
        model, model_meta = load_model_from_checkpoint(ckpt_path, device)
    except Exception as e:
        print(f"    ERROR loading {exp_dir.name}: {e}")
        return None

    try:
        data = load_unified_test_data(model_size, overlap)
    except Exception as e:
        print(f"    ERROR loading test data for {exp_dir.name}: {e}")
        return None

    x1_t = torch.from_numpy(data['x1_norm']).float().to(device)
    x2_t = torch.from_numpy(data['x2_norm']).float().to(device)

    with torch.no_grad():
        out = model(x1_t, x2_t)
        pred_norm = out[0].cpu().numpy() if isinstance(out, tuple) else out.cpu().numpy()

    # Denormalize
    normalizer = data['normalizer']
    if normalizer is not None:
        pred_w = normalizer.inverse_transform(pred_norm)
    else:
        pred_w = pred_norm   # already in weight space if no normalizer

    gt_w = data['y']

    # Weight-space metrics (all samples)
    wm = weight_metrics(pred_w.mean(0), gt_w.mean(0))

    # CNN finetune on subset
    cnn_accs, cnn_baselines, cnn_improvements, cnn_initials = [], [], [], []
    n_eval = min(n_cnn_samples, len(pred_w))
    for i in range(n_eval):
        cm = cnn_eval_single(pred_w[i], data['meta'][i], run_cnn=run_cnn)
        cnn_accs.append(cm.get('cnn_accuracy', float('nan')))
        cnn_baselines.append(cm.get('cnn_baseline', float('nan')))
        cnn_improvements.append(cm.get('cnn_improvement', float('nan')))
        cnn_initials.append(cm.get('cnn_initial_acc', float('nan')))

    def nanmean(lst):
        a = [x for x in lst if not np.isnan(x)]
        return float(np.mean(a)) if a else float('nan')

    result = {
        'exp_name':       exp_dir.name,
        'model_size':     model_size,
        'overlap':        overlap,
        'loss_name':      loss_name,
        'epoch':          model_meta['epoch'],
        'saved_val_loss': model_meta['val_loss'],
        'cnn_accuracy':       nanmean(cnn_accs),
        'cnn_baseline':       nanmean(cnn_baselines),
        'cnn_improvement':    nanmean(cnn_improvements),
        'cnn_initial_acc':    nanmean(cnn_initials),
        'n_cnn_evaluated':    n_eval,
        **wm,
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_ranking(df: pd.DataFrame, output_dir: Path):
    for overlap in sorted(df['overlap'].unique()):
        sub = df[df['overlap'] == overlap].copy()
        sub = sub.dropna(subset=['cnn_improvement'])
        if sub.empty:
            sub = df[df['overlap'] == overlap].dropna(subset=['cosine']).copy()

        _rank_cols  = [c for c in ['cnn_accuracy', 'cnn_baseline', 'mse', 'cnn_improvement'] if c in sub.columns]
        _rank_asc   = [c == 'mse' for c in _rank_cols]
        if _rank_cols:
            sub = sub.sort_values(_rank_cols, ascending=_rank_asc)

        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(sub) * 0.4)))

        for ax, col, label, cmap in [
            (axes[0], 'cnn_improvement', 'CNN Improvement (finetuned - initial)', 'RdYlGn'),
            (axes[1], 'cosine',          'Cosine Similarity to GT',               'Blues'),
            (axes[2], 'mse',             'Weight MSE (lower=better)',              'Reds_r'),
        ]:
            vals = sub[col].fillna(0)
            colors = plt.cm.get_cmap(cmap)(
                (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
            )
            axes_sub = ax.barh(sub['loss_name'], vals, color=colors, alpha=0.8)
            ax.set_xlabel(label, fontsize=9)
            ax.set_title(f'Overlap {overlap} — {label}', fontsize=10)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.tick_params(labelsize=7)

        plt.suptitle(f'Fair Evaluation — Overlap {overlap}  (seed={FAIR_EVAL_SEED}, n={N_FINETUNE_SAMPLES})',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'fair_eval_overlap{overlap}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → fair_eval_overlap{overlap}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fair unified evaluation of all trained models')
    parser.add_argument('--experiments-dir', type=Path, default=EXPERIMENTS)
    parser.add_argument('--output-dir', type=Path,
                        default=NOTEBOOKS_SANDBOX / 'paper_results' / 'fair_eval')
    parser.add_argument('--model-size', default='tiny')
    parser.add_argument('--overlaps', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--n-cnn-samples', type=int, default=N_FINETUNE_SAMPLES,
                        help='CNN finetune samples per experiment (default: all 100)')
    parser.add_argument('--no-cnn', action='store_true',
                        help='Skip CNN finetune — weight metrics only (fast)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip experiments already in output CSV')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiments dir: {args.experiments_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Fair eval seed: {FAIR_EVAL_SEED}   n_finetune_samples: {N_FINETUNE_SAMPLES}")
    print(f"CNN eval: {'disabled' if args.no_cnn else f'enabled ({args.n_cnn_samples} samples/exp)'}\n")

    # Load existing results if resuming
    out_csv = args.output_dir / 'fair_eval_all.csv'
    already_done = set()
    if args.resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        already_done = set(prev['exp_name'].tolist())
        print(f"Resuming — {len(already_done)} experiments already evaluated\n")

    # Collect experiment dirs
    pattern = f"{args.model_size}_overlap*_*"
    exp_dirs = []
    for ov in args.overlaps:
        exp_dirs += sorted(args.experiments_dir.glob(f"{args.model_size}_overlap{ov}_*"))
    exp_dirs = [d for d in exp_dirs if (d / 'checkpoints' / 'best_model.pth').exists()]

    print(f"Found {len(exp_dirs)} experiments with best_model.pth\n")

    results = []
    for exp_dir in tqdm(exp_dirs, desc='Evaluating'):
        if exp_dir.name in already_done:
            print(f"  SKIP (already done): {exp_dir.name}")
            continue

        print(f"\n→ {exp_dir.name}")
        t0 = time.time()
        res = evaluate_experiment(
            exp_dir, device,
            run_cnn=not args.no_cnn,
            n_cnn_samples=args.n_cnn_samples,
        )
        if res is not None:
            res['eval_time_s'] = round(time.time() - t0, 1)
            results.append(res)
            print(f"  cnn_improvement={res.get('cnn_improvement', '?'):.3f}  "
                  f"mse={res.get('mse', '?'):.4f}  "
                  f"cosine={res.get('cosine', '?'):.4f}  "
                  f"[{res['eval_time_s']}s]")
        else:
            print(f"  FAILED — skipped")

    if not results:
        print("No results to write.")
        return

    # Merge with previously saved (if resuming)
    new_df = pd.DataFrame(results)
    if args.resume and out_csv.exists():
        old_df = pd.read_csv(out_csv)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(out_csv, index=False)
    print(f"\nAll results → {out_csv}")

    # Per-overlap CSVs  (sort: peak CNN acc → baseline acc → val MSE ↑ → improvement)
    _SORT_COLS = ['cnn_accuracy', 'cnn_baseline', 'mse', 'cnn_improvement']
    for ov in args.overlaps:
        sub = df[df['overlap'] == ov].copy()
        sc  = [c for c in _SORT_COLS if c in sub.columns]
        asc = [c == 'mse' for c in sc]
        if sc:
            sub = sub.sort_values(sc, ascending=asc)
        sub.to_csv(args.output_dir / f'fair_eval_overlap{ov}.csv', index=False)
        print(f"\n=== Overlap {ov} ranking (peak CNN acc → baseline → MSE → improvement) ===")
        cols = ['loss_name', 'cnn_accuracy', 'cnn_baseline', 'mse', 'cnn_improvement', 'cosine', 'epoch']
        print(sub[[c for c in cols if c in sub.columns]].head(10).to_string(index=False))

    plot_ranking(df, args.output_dir)
    print(f"\nDone. Results in {args.output_dir}")


if __name__ == '__main__':
    main()
