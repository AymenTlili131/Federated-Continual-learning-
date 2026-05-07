#!/usr/bin/env python3
"""
stage2_hmmr_topology.py
------------------------
Stage 2 of the tournament: rigorous HMMR timeseries segmentation +
topological / spectral analysis of the top-ranked loss functions from Stage 1.

Inputs:
  - paper_results/fair_eval/fair_eval_all.csv   (from fair_evaluation.py)
  - experiments/*/checkpoints/best_model.pth
  - experiments/*/cnn_validation/               (per-epoch CNN states)

Pipeline:
  1. Load Stage-1 ranking; select top-N per overlap
  2. HMMR segmentation: fit Hidden Markov Models on the training-loss time
     series for each run → detect regime changes, convergence plateaus, and
     instability
  3. Multipers topological analysis: per-layer W1 divergence on predicted
     weight trajectories across training epochs
  4. RMT spectral analysis: effective rank, condition number, KL-MP per
     layer for each top model
  5. Cross-model correlation analysis: cosine/Frobenius similarity between
     the weight manifolds of different loss functions
  6. Output: paper_results/stage2/ — CSVs, figures, ranking JSON for Stage 3

Usage:
  # Full Stage 2 (requires Stage 1 fair_eval_all.csv)
  conda run -n FCL python3 scripts/stage2_hmmr_topology.py --top-n 10

  # HMMR only (fast)
  conda run -n FCL python3 scripts/stage2_hmmr_topology.py --hmmr-only

  # Topology only
  conda run -n FCL python3 scripts/stage2_hmmr_topology.py --topo-only
"""

import sys
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

SCRIPTS_DIR       = Path(__file__).parent
NOTEBOOKS_SANDBOX = SCRIPTS_DIR.parent
PROJECT_ROOT      = NOTEBOOKS_SANDBOX.parent
EXPERIMENTS       = NOTEBOOKS_SANDBOX / 'experiments'

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(NOTEBOOKS_SANDBOX))
sys.path.insert(0, str(NOTEBOOKS_SANDBOX / 'core_modules'))

try:
    from rmt_analysis import RandomMatrixAnalyzer
    RMT_OK = True
except ImportError:
    RMT_OK = False

try:
    from multipers_analysis import compute_multipers_features, compare_multipers_features
    MULTIPERS_OK = True
except ImportError:
    MULTIPERS_OK = False

LAYER_DELIMITERS = [208, 1414, 1514, 2254, 2464]
LAYER_NAMES      = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']


# ──────────────────────────────────────────────────────────────────────────────
# HMMR: Hidden Markov Model + Regime Detection on loss curves
# ──────────────────────────────────────────────────────────────────────────────

class HMMRSegmenter:
    """
    HMMR (Hidden Markov Model Regime) segmentation on 1D time series.

    We use a simple change-point / Gaussian mixture approach since the
    full HMM library (hmmlearn) may not be installed. Falls back to
    a sliding-window regime detection if hmmlearn is unavailable.
    """

    def __init__(self, n_states: int = 3, window: int = 10):
        self.n_states = n_states
        self.window   = window
        self._use_hmmlearn = False
        try:
            from hmmlearn.hmm import GaussianHMM
            self._HMM = GaussianHMM
            self._use_hmmlearn = True
        except ImportError:
            pass

    def fit_predict(self, series: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Segment a loss curve into regimes.

        Returns:
            states: integer array of regime labels (same length as series)
            info:   dict with convergence_epoch, n_plateaus, instability_score, etc.
        """
        series = np.asarray(series, dtype=np.float32)
        if len(series) < 3:
            return np.zeros(len(series), dtype=int), {}

        # Normalise to [0,1]
        s_min, s_max = series.min(), series.max()
        norm = (series - s_min) / (s_max - s_min + 1e-10)

        if self._use_hmmlearn and len(series) >= self.n_states * 5:
            states = self._hmm_segment(norm)
        else:
            states = self._sliding_segment(norm)

        info = self._compute_info(series, states)
        return states, info

    def _hmm_segment(self, norm: np.ndarray) -> np.ndarray:
        try:
            model = self._HMM(
                n_components=self.n_states,
                covariance_type='full',
                n_iter=100,
                random_state=42,
            )
            model.fit(norm.reshape(-1, 1))
            states = model.predict(norm.reshape(-1, 1))
            # Remap state IDs to be ordered by mean (0=lowest loss regime)
            means = [norm[states == s].mean() if (states == s).any() else 1.0
                     for s in range(self.n_states)]
            order = np.argsort(means)
            remap = {old: new for new, old in enumerate(order)}
            return np.array([remap[s] for s in states])
        except Exception:
            return self._sliding_segment(norm)

    def _sliding_segment(self, norm: np.ndarray) -> np.ndarray:
        """Threshold-based regime assignment: low/medium/high."""
        states = np.zeros(len(norm), dtype=int)
        thresholds = np.percentile(norm, [33, 67])
        states[norm > thresholds[1]] = 2
        states[(norm > thresholds[0]) & (norm <= thresholds[1])] = 1
        return states

    def _compute_info(self, series: np.ndarray, states: np.ndarray) -> Dict:
        n = len(series)
        diffs = np.abs(np.diff(series))

        # Convergence: first epoch where loss stops decreasing by >1%
        rolling_min = np.minimum.accumulate(series)
        improvement_rate = (rolling_min[:-1] - rolling_min[1:]) / (rolling_min[:-1] + 1e-10)
        converged_at = next(
            (i for i in range(len(improvement_rate) - 5)
             if improvement_rate[i:i+5].max() < 0.01),
            n
        )

        # Plateaus: extended runs in same regime
        transitions = np.diff(states).nonzero()[0]
        plateau_lengths = np.diff(np.concatenate([[0], transitions, [n]]))
        n_plateaus = len(plateau_lengths[plateau_lengths >= 5])

        # Instability: high variance in diffs
        instability = float(np.std(diffs))

        # Spikes: peaks in loss curve
        peaks, _ = find_peaks(series, prominence=series.std())
        n_spikes = len(peaks)

        # Final convergence value
        final_val = float(series[-min(10, n):].mean())

        return {
            'convergence_epoch':    int(converged_at),
            'convergence_pct':      round(converged_at / n * 100, 1),
            'n_plateaus':           int(n_plateaus),
            'instability_score':    round(instability, 6),
            'n_loss_spikes':        int(n_spikes),
            'final_val_loss':       round(final_val, 6),
            'n_regime_transitions': int(len(transitions)),
        }


def run_hmmr_analysis(
    exp_dirs: List[Path],
    output_dir: Path,
    n_states: int = 3,
) -> pd.DataFrame:
    """Run HMMR segmentation on all experiments."""
    print(f"\n[HMMR] Analysing {len(exp_dirs)} experiments ({n_states} states)")
    segmenter = HMMRSegmenter(n_states=n_states)
    rows = []

    fig_rows = max(1, len(exp_dirs) // 4 + 1)
    fig, axes = plt.subplots(
        fig_rows, min(4, len(exp_dirs)),
        figsize=(16, 4 * fig_rows), squeeze=False
    )
    ax_flat = axes.ravel()
    ax_idx  = 0

    for exp_dir in exp_dirs:
        hist_file = exp_dir / 'training_history.json'
        if not hist_file.exists():
            continue
        try:
            import json as _json
            h = _json.load(open(hist_file))
            val_loss = h.get('val_loss', [])
            if not val_loss:
                continue
        except Exception:
            continue

        series = np.array(val_loss, dtype=np.float32)
        parts  = exp_dir.name.split('_')
        model_size = parts[0]
        ov_part    = next(p for p in parts if p.startswith('overlap'))
        overlap    = int(ov_part.replace('overlap', ''))
        loss_name  = '_'.join(parts[parts.index(ov_part) + 1:])

        states, info = segmenter.fit_predict(series)

        rows.append({
            'exp_name':   exp_dir.name,
            'model_size': model_size,
            'overlap':    overlap,
            'loss_name':  loss_name,
            **info,
        })

        # Plot
        if ax_idx < len(ax_flat):
            ax = ax_flat[ax_idx]
            ax.plot(series, lw=1, alpha=0.8, label='val_loss')
            cmap = plt.cm.Set1
            for s in range(n_states):
                mask = states == s
                xs   = np.where(mask)[0]
                if xs.any():
                    ax.scatter(xs, series[mask], s=2, c=[cmap(s / n_states)], alpha=0.5)
            ax.axvline(info.get('convergence_epoch', len(series)),
                       color='red', lw=1, linestyle='--', alpha=0.7, label='converge')
            ax.set_title(f"{loss_name}\nov{overlap} conv@{info.get('convergence_epoch','?')}ep",
                         fontsize=7)
            ax.set_yscale('log')
            ax.tick_params(labelsize=6)
            ax_idx += 1

    # Hide unused axes
    for ax in ax_flat[ax_idx:]:
        ax.set_visible(False)

    plt.suptitle('HMMR Regime Segmentation — Training Loss Curves', fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / 'hmmr_regimes.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'hmmr_analysis.csv', index=False)
    print(f"  → hmmr_analysis.csv + hmmr_regimes.png  ({len(rows)} experiments)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Topological analysis of weight trajectories
# ──────────────────────────────────────────────────────────────────────────────

def _load_weight_snapshots(exp_dir: Path, max_snapshots: int = 10) -> List[np.ndarray]:
    """Load predicted weight files at regular intervals from cnn_validation/."""
    snapshots = []
    val_dir = exp_dir / 'cnn_validation'
    if not val_dir.exists():
        return snapshots

    epoch_dirs = sorted(val_dir.glob('epoch_*'))
    if not epoch_dirs:
        return snapshots

    step = max(1, len(epoch_dirs) // max_snapshots)
    selected = epoch_dirs[::step][:max_snapshots]

    for ep_dir in selected:
        for f in ep_dir.glob('*.npy'):
            try:
                w = np.load(f).ravel()
                if len(w) >= 2464:
                    snapshots.append(w[:2464].astype(np.float32))
                    break
            except Exception:
                pass
    return snapshots


def run_topology_analysis(
    exp_dirs: List[Path],
    output_dir: Path,
    subsample: int = 60,
    n_lines:   int = 20,
) -> pd.DataFrame:
    """Multipers + RMT topology analysis for each top experiment."""
    print(f"\n[TOPOLOGY] Analysing {len(exp_dirs)} experiments")

    rows = []
    rmt_analyzer = RandomMatrixAnalyzer() if RMT_OK else None

    for exp_dir in exp_dirs:
        parts      = exp_dir.name.split('_')
        model_size = parts[0]
        ov_part    = next(p for p in parts if p.startswith('overlap'))
        overlap    = int(ov_part.replace('overlap', ''))
        loss_name  = '_'.join(parts[parts.index(ov_part) + 1:])

        row = {
            'exp_name':   exp_dir.name,
            'model_size': model_size,
            'overlap':    overlap,
            'loss_name':  loss_name,
        }

        # Load best-epoch predicted weights
        pred_dir = exp_dir / 'predicted_weights'
        best_w   = None
        if pred_dir.exists():
            wfiles = sorted(pred_dir.glob('*.npy'))
            if wfiles:
                try:
                    best_w = np.load(wfiles[-1]).ravel()[:2464].astype(np.float32)
                except Exception:
                    pass

        # Fallback: load from last cnn_validation epoch
        if best_w is None:
            snaps = _load_weight_snapshots(exp_dir, max_snapshots=1)
            if snaps:
                best_w = snaps[0]

        if best_w is None:
            print(f"  SKIP {exp_dir.name} (no weight data)")
            rows.append(row)
            continue

        # RMT spectral analysis
        if rmt_analyzer is not None:
            try:
                rmt_out = rmt_analyzer.analyze_all_layers(best_w.astype(np.float64))
                for layer_name, lr in rmt_out.items():
                    if lr.get('type') == 'weight':
                        row[f'rmt_eff_rank_{layer_name}']    = lr.get('effective_rank', np.nan)
                        row[f'rmt_cond_{layer_name}']        = lr.get('condition_number', np.nan)
                        row[f'rmt_kl_mp_{layer_name}']       = lr.get('kl_divergence_mp', np.nan)
            except Exception as e:
                print(f"  RMT error for {exp_dir.name}: {e}")

        # Multipers topological features
        if MULTIPERS_OK:
            try:
                feats = compute_multipers_features(
                    best_w, subsample_per_layer=subsample, n_lines=n_lines
                )
                for layer, f in feats.items():
                    row[f'topo_mean_{layer}'] = float(np.mean(f))
                    row[f'topo_std_{layer}']  = float(np.std(f))
                    row[f'topo_max_{layer}']  = float(np.max(f))
            except Exception as e:
                print(f"  Multipers error for {exp_dir.name}: {e}")

        # Trajectory analysis: W1 divergence over training epochs
        snaps = _load_weight_snapshots(exp_dir, max_snapshots=8)
        if len(snaps) >= 2 and MULTIPERS_OK:
            try:
                f_first = compute_multipers_features(snaps[0], subsample_per_layer=subsample, n_lines=n_lines)
                f_last  = compute_multipers_features(snaps[-1], subsample_per_layer=subsample, n_lines=n_lines)
                traj_dists = compare_multipers_features(f_last, f_first)
                row['topo_trajectory_w1_total'] = float(np.mean(list(traj_dists.values())))
            except Exception:
                pass

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'topology_analysis.csv', index=False)

    # ── Cross-model correlation heatmap ──
    topo_cols = [c for c in df.columns if c.startswith('topo_mean_')]
    if topo_cols and len(df) >= 2:
        feat_mat = df[topo_cols].fillna(0).values
        if feat_mat.shape[0] >= 2:
            corr = np.corrcoef(feat_mat)
            fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.5), max(8, len(df) * 0.5)))
            sns.heatmap(corr, ax=ax, annot=(len(df) <= 20), fmt='.2f',
                        cmap='RdBu_r', vmin=-1, vmax=1,
                        xticklabels=df['loss_name'].tolist(),
                        yticklabels=df['loss_name'].tolist(),
                        linewidths=0.3)
            ax.set_title('Cross-model topological feature correlation', fontsize=12)
            plt.tight_layout()
            fig.savefig(output_dir / 'cross_model_topo_corr.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"  → topology_analysis.csv  ({len(rows)} experiments)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 ranking
# ──────────────────────────────────────────────────────────────────────────────

def compute_stage2_ranking(
    fair_eval_df: pd.DataFrame,
    hmmr_df:     pd.DataFrame,
    topo_df:     pd.DataFrame,
    output_dir:  Path,
) -> pd.DataFrame:
    """Combine Stage-1 + HMMR + topology into a Stage-2 composite score."""
    # Merge
    df = fair_eval_df.copy()
    if not hmmr_df.empty:
        df = df.merge(hmmr_df[['exp_name', 'convergence_pct', 'instability_score',
                                'n_loss_spikes', 'final_val_loss']],
                      on='exp_name', how='left')
    if not topo_df.empty:
        traj_col = 'topo_trajectory_w1_total'
        if traj_col in topo_df.columns:
            df = df.merge(topo_df[['exp_name', traj_col]], on='exp_name', how='left')

    # Composite score (higher = better):
    # +cnn_improvement   +cosine   -mse   -instability   -n_spikes   -topo_trajectory
    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-12) if mx > mn else s * 0

    score = pd.Series(0.0, index=df.index)
    if 'cnn_improvement' in df.columns:
        score += 0.35 * norm(df['cnn_improvement'].fillna(0))
    if 'cosine' in df.columns:
        score += 0.25 * norm(df['cosine'].fillna(0))
    if 'mse' in df.columns:
        score += 0.20 * (1 - norm(df['mse'].fillna(df['mse'].max())))
    if 'instability_score' in df.columns:
        score += 0.10 * (1 - norm(df['instability_score'].fillna(0)))
    if 'n_loss_spikes' in df.columns:
        score += 0.05 * (1 - norm(df['n_loss_spikes'].fillna(0)))
    if 'topo_trajectory_w1_total' in df.columns:
        score += 0.05 * (1 - norm(df['topo_trajectory_w1_total'].fillna(0)))

    df['stage2_score'] = score

    # Per-overlap ranking
    out = {}
    for overlap in sorted(df['overlap'].unique()):
        sub = df[df['overlap'] == overlap].sort_values('stage2_score', ascending=False)
        out[str(overlap)] = sub[['loss_name', 'stage2_score', 'cnn_improvement',
                                   'cosine', 'mse']].head(20).to_dict(orient='records')
        print(f"\n=== Stage 2 Ranking — Overlap {overlap} (top 10) ===")
        print(sub[['loss_name', 'stage2_score', 'cnn_improvement', 'cosine']].head(10).to_string(index=False))

    df.to_csv(output_dir / 'stage2_ranking_full.csv', index=False)

    ranking_json = {'stage': 2, 'rankings_per_overlap': out}
    with open(output_dir / 'rankings_stage2.json', 'w') as f:
        json.dump(ranking_json, f, indent=2)
    print(f"\n→ stage2_ranking_full.csv + rankings_stage2.json")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_hmmr_summary(hmmr_df: pd.DataFrame, output_dir: Path):
    """Plot HMMR metrics side by side."""
    if hmmr_df.empty:
        return

    for overlap in sorted(hmmr_df['overlap'].unique()):
        sub = hmmr_df[hmmr_df['overlap'] == overlap].sort_values(
            'convergence_epoch', ascending=True
        )
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, max(5, len(sub) * 0.35)))

        for ax, col, title, color in [
            (axes[0], 'convergence_epoch', 'Convergence epoch (lower=faster)', 'steelblue'),
            (axes[1], 'instability_score', 'Instability (std of loss deltas)', 'darkorange'),
            (axes[2], 'n_loss_spikes',     'Number of loss spikes',             'firebrick'),
        ]:
            vals = sub[col].fillna(0)
            axes_sub = ax.barh(sub['loss_name'], vals, color=color, alpha=0.7)
            ax.set_xlabel(title, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.tick_params(labelsize=7)

        plt.suptitle(f'HMMR Analysis — Overlap {overlap}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'hmmr_summary_overlap{overlap}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → hmmr_summary_overlap{overlap}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Stage 2: HMMR + Topological analysis')
    parser.add_argument('--fair-eval-csv', type=Path,
                        default=NOTEBOOKS_SANDBOX / 'paper_results' / 'fair_eval' / 'fair_eval_all.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=NOTEBOOKS_SANDBOX / 'paper_results' / 'stage2')
    parser.add_argument('--top-n',  type=int, default=10,
                        help='Top N experiments per overlap to include in topology analysis')
    parser.add_argument('--overlaps', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--hmmr-only',  action='store_true')
    parser.add_argument('--topo-only',  action='store_true')
    parser.add_argument('--n-states',   type=int, default=3,
                        help='Number of HMM hidden states')
    parser.add_argument('--subsample',  type=int, default=60,
                        help='Subsample per layer for multipers')
    parser.add_argument('--n-lines',    type=int, default=20,
                        help='Number of projection lines for multipers')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage-1 results
    fair_eval_df = pd.DataFrame()
    if args.fair_eval_csv.exists():
        fair_eval_df = pd.read_csv(args.fair_eval_csv)
        print(f"Loaded {len(fair_eval_df)} Stage-1 results from {args.fair_eval_csv}")
    else:
        print(f"WARNING: fair_eval_all.csv not found at {args.fair_eval_csv}")
        print("Falling back to all experiments with best_model.pth\n")
        # Use all experiments
        records = []
        for d in sorted(EXPERIMENTS.glob('tiny_overlap*')):
            if (d / 'checkpoints' / 'best_model.pth').exists():
                parts = d.name.split('_')
                ov    = int(next(p.replace('overlap','') for p in parts if p.startswith('overlap')))
                loss  = '_'.join(parts[parts.index(next(p for p in parts if p.startswith('overlap')))+1:])
                records.append({'exp_name': d.name, 'model_size': 'tiny',
                                'overlap': ov, 'loss_name': loss})
        fair_eval_df = pd.DataFrame(records)

    # Select top-N per overlap for expensive topology analysis
    def get_top_dirs(n: int) -> List[Path]:
        dirs = []
        for ov in args.overlaps:
            sub = fair_eval_df[fair_eval_df['overlap'] == ov]
            if 'cnn_improvement' in sub.columns:
                sub = sub.dropna(subset=['cnn_improvement']).sort_values(
                    'cnn_improvement', ascending=False
                )
            elif 'cosine' in sub.columns:
                sub = sub.dropna(subset=['cosine']).sort_values('cosine', ascending=False)
            top = sub.head(n)
            for _, row in top.iterrows():
                d = EXPERIMENTS / row['exp_name']
                if d.exists():
                    dirs.append(d)
        return dirs

    all_dirs = [
        EXPERIMENTS / row['exp_name']
        for _, row in fair_eval_df.iterrows()
        if (EXPERIMENTS / row['exp_name']).exists()
    ]
    top_dirs = get_top_dirs(args.top_n)

    print(f"\nAll experiments: {len(all_dirs)}")
    print(f"Top {args.top_n} per overlap (for topology): {len(top_dirs)}")

    hmmr_df = pd.DataFrame()
    topo_df = pd.DataFrame()

    if not args.topo_only:
        # HMMR on ALL experiments
        hmmr_df = run_hmmr_analysis(all_dirs, args.output_dir, n_states=args.n_states)
        plot_hmmr_summary(hmmr_df, args.output_dir)

    if not args.hmmr_only:
        # Topology on top-N only (expensive)
        topo_df = run_topology_analysis(
            top_dirs, args.output_dir,
            subsample=args.subsample,
            n_lines=args.n_lines,
        )

    if not fair_eval_df.empty:
        compute_stage2_ranking(fair_eval_df, hmmr_df, topo_df, args.output_dir)

    print(f"\nStage 2 complete → {args.output_dir}")


if __name__ == '__main__':
    main()
