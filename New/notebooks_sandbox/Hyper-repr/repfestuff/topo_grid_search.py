#!/usr/bin/env python3
"""
topo_grid_search.py
-------------------
Grid search for optimal topological feature granularity.

Searches over:
  - Filtration types  : value×layer | magnitude×position | value×magnitude
  - subsample_per_layer: [30, 60, 120, 240]
  - n_lines           : [10, 20, 40]

Metric: discriminability = mean silhouette score when grouping weight vectors
by their loss function label (can the features separate different loss types?).

Usage:
  conda run -n FCL python3 scripts/topo_grid_search.py \
      --experiments-dir experiments/ \
      --output-dir paper_results/topo_grid_search/
"""

import sys
import argparse
import json
import warnings
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

NOTEBOOKS_SANDBOX = Path(__file__).parent.parent
sys.path.insert(0, str(NOTEBOOKS_SANDBOX))
sys.path.insert(0, str(NOTEBOOKS_SANDBOX / 'core_modules'))

warnings.filterwarnings('ignore')

try:
    import gudhi
    import multipers
    MULTIPERS_OK = True
except ImportError:
    MULTIPERS_OK = False
    print("Warning: gudhi/multipers not available — only single-param features computed")

try:
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from rmt_analysis import RandomMatrixAnalyzer

# ──────────────────────────────────────────────────────────────────────────────
# Filtration definitions
# ──────────────────────────────────────────────────────────────────────────────

LAYER_DELIMITERS = [208, 1414, 1514, 2254, 2464]
LAYER_NAMES      = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']

def _layer_index_array(n: int) -> np.ndarray:
    """Return normalised layer index for each weight position."""
    out = np.zeros(n, dtype=np.float32)
    prev = 0
    for k, end in enumerate(LAYER_DELIMITERS):
        out[prev:end] = k / len(LAYER_DELIMITERS)
        prev = end
    return out[:n]

def filtration_value_x_layer(w: np.ndarray, seg: slice) -> Tuple[np.ndarray, np.ndarray]:
    """ξ₁ = weight value, ξ₂ = normalised layer index."""
    w_seg = w[seg].astype(np.float32)
    n = len(w_seg)
    f1 = w_seg
    f2 = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return f1, f2

def filtration_magnitude_x_position(w: np.ndarray, seg: slice) -> Tuple[np.ndarray, np.ndarray]:
    """ξ₁ = |weight| magnitude, ξ₂ = normalised position within segment."""
    w_seg = w[seg].astype(np.float32)
    n = len(w_seg)
    f1 = np.abs(w_seg)
    f2 = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return f1, f2

def filtration_value_x_magnitude(w: np.ndarray, seg: slice) -> Tuple[np.ndarray, np.ndarray]:
    """ξ₁ = weight value, ξ₂ = |weight| magnitude."""
    w_seg = w[seg].astype(np.float32)
    f1 = w_seg
    f2 = np.abs(w_seg)
    return f1, f2

FILTRATION_REGISTRY = {
    'value_x_layer':     filtration_value_x_layer,
    'magnitude_x_pos':   filtration_magnitude_x_position,
    'value_x_magnitude': filtration_value_x_magnitude,
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _build_stm(f1: np.ndarray, f2: np.ndarray, subsample: int):
    """Build a 2-parameter path-graph SimplexTreeMulti."""
    n_full = len(f1)
    idx = np.round(np.linspace(0, n_full - 1, min(subsample, n_full))).astype(int)
    s1, s2 = f1[idx], f2[idx]
    n = len(idx)

    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=float(s1[i]))
    for i in range(n - 1):
        st.insert([i, i + 1], filtration=float(max(s1[i], s1[i + 1])))

    stm = multipers.SimplexTreeMulti(st, num_parameters=2)
    for i in range(n):
        stm.assign_filtration([i], filtration=[float(s1[i]), float(s2[i])])
    for i in range(n - 1):
        stm.assign_filtration([i, i + 1],
                              filtration=[float(max(s1[i], s1[i + 1])),
                                         float(max(s2[i], s2[i + 1]))])
    stm.make_filtration_non_decreasing()
    return stm

def _stm_signature(stm, n_lines: int, k_features: int = 10) -> np.ndarray:
    """Project stm onto n_lines and concatenate sorted lifetime signatures."""
    angles = np.linspace(0, np.pi / 2, n_lines, endpoint=False)
    parts = []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        try:
            proj = stm.project_on_line(direction=direction)
            proj.compute_persistence()
            pairs = np.array(proj.persistence_intervals_in_dimension(0))
            if len(pairs) == 0:
                lifetimes = np.zeros(k_features, dtype=np.float32)
            else:
                finite = pairs[pairs[:, 1] != np.inf]
                lt = np.sort((finite[:, 1] - finite[:, 0]))[::-1]
                lifetimes = np.pad(lt[:k_features],
                                   (0, max(0, k_features - len(lt)))).astype(np.float32)
        except Exception:
            lifetimes = np.zeros(k_features, dtype=np.float32)
        parts.append(lifetimes)
    return np.concatenate(parts)

def _fallback_signature(f1: np.ndarray, subsample: int) -> np.ndarray:
    """Single-parameter sorted-gap fallback (no multipers)."""
    s = np.sort(f1)
    gaps = np.sort(s[1:] - s[:-1])[::-1]
    sig = gaps[:subsample]
    return np.pad(sig, (0, max(0, subsample - len(sig)))).astype(np.float32)

def extract_features(
    w: np.ndarray,
    filtration_name: str,
    subsample_per_layer: int,
    n_lines: int,
) -> np.ndarray:
    """Full per-layer feature extraction for one weight vector."""
    filtration_fn = FILTRATION_REGISTRY[filtration_name]
    parts = []
    prev = 0
    for end in LAYER_DELIMITERS:
        seg = slice(prev, end)
        f1, f2 = filtration_fn(w, seg)
        if MULTIPERS_OK:
            try:
                stm = _build_stm(f1, f2, subsample_per_layer)
                sig = _stm_signature(stm, n_lines)
            except Exception:
                sig = _fallback_signature(f1, subsample_per_layer)
        else:
            sig = _fallback_signature(f1, subsample_per_layer)
        parts.append(sig)
        prev = end
    return np.concatenate(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Grid search
# ──────────────────────────────────────────────────────────────────────────────

def load_weight_samples(experiments_dir: Path, model_size: str = 'tiny',
                        max_per_loss: int = 5) -> Tuple[List[np.ndarray], List[str]]:
    """Load weight samples from local experiments + tracking CSVs."""
    weights, labels = [], []
    pattern = f"{model_size}_overlap*_*"
    exp_dirs = sorted(experiments_dir.glob(pattern))

    for exp_dir in exp_dirs:
        loss_name = '_'.join(exp_dir.name.split('_')[3:])  # after "tiny_overlapN_"

        # Look for tracking CSV (contains predicted weights)
        csv_files = list(exp_dir.glob('*.csv')) + list(exp_dir.glob('metrics/*.csv'))
        for csv_path in csv_files[:1]:
            try:
                df = pd.read_csv(csv_path, nrows=max_per_loss)
                # Weight columns are 2464 float columns starting from col 2
                w_cols = [c for c in df.columns if c.isdigit() or
                          (c.replace('.', '', 1).lstrip('-').isdigit())]
                if len(w_cols) >= 2464:
                    for _, row in df.iterrows():
                        w = row[w_cols[:2464]].values.astype(np.float32)
                        weights.append(w)
                        labels.append(loss_name)
                    break
            except Exception:
                continue

    print(f"Loaded {len(weights)} weight samples from {len(set(labels))} loss types")
    return weights, labels


def run_grid_search(
    weights: List[np.ndarray],
    labels: List[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Evaluate all filtration×subsample×n_lines combinations."""
    if not SKLEARN_OK:
        print("sklearn not available — cannot compute silhouette scores")
        return pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    filtration_types   = list(FILTRATION_REGISTRY.keys())
    subsample_options  = [30, 60, 120, 240]
    n_lines_options    = [10, 20, 40]

    grid = list(itertools.product(filtration_types, subsample_options, n_lines_options))
    print(f"\nGrid search: {len(grid)} combinations × {len(weights)} samples\n")

    results = []
    label_arr = np.array(labels)
    unique_labels = list(set(labels))
    label_int = np.array([unique_labels.index(l) for l in labels])

    for i, (filt_name, subsample, n_lines) in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] filt={filt_name} sub={subsample} lines={n_lines}")
        try:
            feats = np.array([
                extract_features(w, filt_name, subsample, n_lines)
                for w in weights
            ])

            if feats.shape[0] < 2 or len(unique_labels) < 2:
                score = float('nan')
            else:
                feats_scaled = StandardScaler().fit_transform(feats)
                score = silhouette_score(feats_scaled, label_int)
        except Exception as e:
            print(f"  ERROR: {e}")
            score = float('nan')

        results.append({
            'filtration': filt_name,
            'subsample_per_layer': subsample,
            'n_lines': n_lines,
            'silhouette_score': score,
            'feature_dim': feats.shape[1] if 'feats' in dir() else -1,
        })
        print(f"  silhouette = {score:.4f}")

    df = pd.DataFrame(results).sort_values('silhouette_score', ascending=False)
    out_csv = output_dir / 'topo_grid_search_results.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")

    # Best config
    best = df.iloc[0]
    print(f"\nBest configuration:")
    print(f"  filtration       = {best['filtration']}")
    print(f"  subsample/layer  = {best['subsample_per_layer']}")
    print(f"  n_lines          = {best['n_lines']}")
    print(f"  silhouette score = {best['silhouette_score']:.4f}")

    best_cfg = {
        'best_filtration': best['filtration'],
        'best_subsample_per_layer': int(best['subsample_per_layer']),
        'best_n_lines': int(best['n_lines']),
        'best_silhouette': float(best['silhouette_score']),
        'all_results': df.to_dict(orient='records'),
    }
    with open(output_dir / 'topo_grid_search_best.json', 'w') as f:
        json.dump(best_cfg, f, indent=2)

    return df


def main():
    parser = argparse.ArgumentParser(description='Grid search for topological feature granularity')
    parser.add_argument('--experiments-dir', type=Path,
                        default=NOTEBOOKS_SANDBOX / 'experiments')
    parser.add_argument('--output-dir', type=Path,
                        default=NOTEBOOKS_SANDBOX / 'paper_results' / 'topo_grid_search')
    parser.add_argument('--model-size', default='tiny')
    parser.add_argument('--max-per-loss', type=int, default=5,
                        help='Max weight samples per loss type')
    parser.add_argument('--use-dummy', action='store_true',
                        help='Use random synthetic weights for quick testing')
    args = parser.parse_args()

    if args.use_dummy or not any(args.experiments_dir.glob(f'{args.model_size}_*')):
        print("Using synthetic dummy weights for grid search validation...")
        np.random.seed(42)
        loss_types = ['MSE', 'MAE', 'Sinkhorn', 'DiffPers', 'RTD']
        weights, labels = [], []
        for loss in loss_types:
            offset = hash(loss) % 100 * 0.01
            for _ in range(8):
                w = np.random.randn(2464).astype(np.float32) * 0.1 + offset
                weights.append(w)
                labels.append(loss)
    else:
        weights, labels = load_weight_samples(
            args.experiments_dir, args.model_size, args.max_per_loss
        )

    if len(weights) < 2:
        print("Not enough samples for grid search. Run with --use-dummy to test.")
        return

    run_grid_search(weights, labels, args.output_dir)


if __name__ == '__main__':
    main()
