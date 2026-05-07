#!/usr/bin/env python3
"""topo_grid_v2.py — Comprehensive topology with H1, multiple filtrations, and multipers grid.

Per weight vector (5 CNN layers):
  1. Rips complex → H0 + H1 per layer
  2. Alpha complex (PCA-3) → H0 + H1 per layer
  3. Sublevel path-graph → H0 per layer (baseline)
  4. Multipers with 6 filtration pairs × {best scale from 12-combo grid}

Filtration pairs for multipers:
  (f1, f2) combinations:
    A: (sublevel, position)      — current default
    B: (abs_value, position)     — magnitude × position
    C: (sublevel, rank)          — value × quantile rank
    D: (density, position)       — local density × position
    E: (gradient, position)      — local variation × position
    F: (abs_value, cumvar)       — magnitude × cumulative energy
"""
import sys, os, time, pickle, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from scipy.stats import wasserstein_distance

warnings.filterwarnings('ignore')

NB = Path(__file__).resolve().parent.parent
PROJ = NB.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(NB / 'core_modules'))

LAYER_BOUNDS = [(0,200,8,25,'conv1'), (208,1408,6,200,'conv2'),
                (1414,1510,4,24,'conv3'), (1514,2234,20,36,'fc1'),
                (2254,2454,10,20,'fc2')]
LAYER_NAMES = [x[4] for x in LAYER_BOUNDS]
LAYER_DELIMS = [200, 1408, 1510, 2234, 2454]  # end positions (excl bias gaps)

try:
    import gudhi
    import multipers
    HAS_MP = True
except ImportError:
    HAS_MP = False

# ── PH stats helper ───────────────────────────────────────────────────
def _ph_stats(pairs):
    if len(pairs) == 0:
        return dict(n_features=0, total_pers=0.0, entropy=0.0, max_life=0.0, mean_life=0.0)
    f = np.array(pairs)
    f = f[np.isfinite(f[:, 1])]
    if len(f) == 0:
        return dict(n_features=0, total_pers=0.0, entropy=0.0, max_life=0.0, mean_life=0.0)
    lt = f[:, 1] - f[:, 0]
    pr = lt / (lt.sum() + 1e-12)
    return dict(
        n_features=len(f), total_pers=float(lt.sum()),
        entropy=float(-np.sum(pr * np.log(pr + 1e-12))),
        max_life=float(lt.max()), mean_life=float(lt.mean()))

# ── Rips PH (H0 + H1) ────────────────────────────────────────────────
def compute_rips_ph(wv):
    result = {}
    for s, e, rows, cols, name in LAYER_BOUNDS:
        W = wv[s:e].reshape(rows, cols)
        dists = np.linalg.norm(W[:, None] - W[None, :], axis=-1)
        me = float(np.percentile(dists, 95))
        rips = gudhi.RipsComplex(points=W.tolist(), max_edge_length=me)
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        for dim in [0, 1]:
            pairs = st.persistence_intervals_in_dimension(dim)
            for k, v in _ph_stats(pairs).items():
                result[f'rips_{name}_h{dim}_{k}'] = v
    return result

# ── Alpha PH via PCA-3 (H0 + H1) ────────────────────────────────────
def compute_alpha_ph(wv):
    from sklearn.decomposition import PCA
    result = {}
    for s, e, rows, cols, name in LAYER_BOUNDS:
        W = wv[s:e].reshape(rows, cols)
        nc = min(3, cols, rows - 1)
        if nc < 2:
            for dim in [0, 1]:
                for k in ['n_features','total_pers','entropy','max_life','mean_life']:
                    result[f'alpha_{name}_h{dim}_{k}'] = 0.0
            continue
        pts = PCA(n_components=nc).fit_transform(W)
        alpha = gudhi.AlphaComplex(points=pts.tolist())
        st = alpha.create_simplex_tree()
        st.compute_persistence()
        for dim in [0, 1]:
            pairs = st.persistence_intervals_in_dimension(dim)
            for k, v in _ph_stats(pairs).items():
                result[f'alpha_{name}_h{dim}_{k}'] = v
    return result

# ── Sublevel path-graph PH (H0 only, fast baseline) ──────────────────
def compute_sublevel_ph(wv):
    n = len(wv)
    idx = np.round(np.linspace(0, n-1, 500)).astype(int) if n > 500 else np.arange(n)
    ws = wv[idx].astype(np.float64)
    st = gudhi.SimplexTree()
    for i in range(len(ws)): st.insert([i], filtration=float(ws[i]))
    for i in range(len(ws)-1): st.insert([i, i+1], filtration=float(max(ws[i], ws[i+1])))
    st.compute_persistence()
    ph0 = st.persistence_intervals_in_dimension(0)
    return {f'sublevel_h0_{k}': v for k, v in _ph_stats(np.array(ph0)).items()}

# ── Filtration function builders ──────────────────────────────────────
def _knn_density(seg, k=5):
    from scipy.spatial.distance import cdist
    dists = cdist(seg.reshape(-1,1), seg.reshape(-1,1))
    kk = min(k, len(seg)-1)
    knn = np.sort(dists, axis=1)[:, 1:kk+1].mean(axis=1)
    return -np.log(knn + 1e-8)

def build_filtration_pair(seg, pair_name):
    """Return (f1, f2) arrays for a weight segment."""
    n = len(seg)
    pos = np.linspace(0, 1, n, dtype=np.float32)
    rank = np.argsort(np.argsort(seg)).astype(np.float32) / n
    cumvar = np.cumsum(seg**2) / (np.sum(seg**2) + 1e-12)

    if pair_name == 'A':    # sublevel × position (default)
        return seg.astype(np.float32), pos
    elif pair_name == 'B':  # abs_value × position
        return np.abs(seg).astype(np.float32), pos
    elif pair_name == 'C':  # sublevel × rank
        return seg.astype(np.float32), rank.astype(np.float32)
    elif pair_name == 'D':  # density × position
        return _knn_density(seg).astype(np.float32), pos
    elif pair_name == 'E':  # gradient × position
        grad = np.abs(np.diff(seg, prepend=seg[0])).astype(np.float32)
        return grad, pos
    elif pair_name == 'F':  # abs_value × cumulative energy
        return np.abs(seg).astype(np.float32), cumvar.astype(np.float32)
    else:
        raise ValueError(f'Unknown pair: {pair_name}')

FILT_PAIRS = ['A', 'B', 'C', 'D', 'E', 'F']
FILT_LABELS = {
    'A': 'sublevel_x_pos', 'B': 'absval_x_pos', 'C': 'sublevel_x_rank',
    'D': 'density_x_pos', 'E': 'gradient_x_pos', 'F': 'absval_x_cumvar'
}

# ── Multipers with configurable filtration ────────────────────────────
def _stm_to_line_betti_safe(stm, n_lines=20):
    """Project multipers onto random lines, extract Betti signatures."""
    from multipers_analysis import _stm_to_line_betti
    return _stm_to_line_betti(stm, n_lines=n_lines)

def multipers_one_filt(wv, filt_pair, subsample=60, n_lines=20):
    """Compute multipers features for one filtration pair."""
    if not HAS_MP:
        return None
    features = {}
    prev = 0
    for end, name in zip([208, 1414, 1514, 2254, 2464], LAYER_NAMES):
        seg = wv[prev:end].astype(np.float32)
        n_seg = len(seg)
        idx = np.round(np.linspace(0, n_seg-1, min(subsample, n_seg))).astype(int)
        w_sub = seg[idx]
        n = len(idx)

        f1, f2 = build_filtration_pair(w_sub, filt_pair)

        st_1p = gudhi.SimplexTree()
        for i in range(n): st_1p.insert([i], filtration=float(f1[i]))
        for i in range(n-1): st_1p.insert([i, i+1], filtration=float(max(f1[i], f1[i+1])))

        stm = multipers.SimplexTreeMulti(st_1p, num_parameters=2)
        for i in range(n):
            stm.assign_filtration([i], filtration=[float(f1[i]), float(f2[i])])
        for i in range(n-1):
            stm.assign_filtration([i, i+1],
                filtration=[float(max(f1[i], f1[i+1])), float(max(f2[i], f2[i+1]))])
        stm.make_filtration_non_decreasing()

        sig = _stm_to_line_betti_safe(stm, n_lines=n_lines)
        features[name] = sig
        prev = end
    return np.concatenate([features[n] for n in LAYER_NAMES])

def mpdist(f1, f2):
    if f1 is None or f2 is None: return float('nan')
    return float(wasserstein_distance(f1, f2))

# ── Grid config ───────────────────────────────────────────────────────
GRID_SUBS = [30, 60, 120, 200]
GRID_NLS = [10, 20, 40]

# ── Main processing ───────────────────────────────────────────────────
def process_experiment(exp_dir, output_dir):
    df1 = pd.read_pickle(exp_dir / 'df1_weights.pkl')
    N = len(df1)
    exp_name = exp_dir.name
    overlap = int(exp_name.split('overlap')[1][0])
    loss_name = exp_name.split('_', 2)[-1]
    wt_names = ['i1','i2','pred','gt'] + [f'fn{e}' for e in range(1, 6)]

    # ── Phase 1: Rips + Alpha + Sublevel on ALL samples × ALL wt ─────
    rows = []
    for i in tqdm(range(N), desc=f'{exp_name} PH', ncols=100, leave=False):
        for wt in wt_names:
            wv = np.array(df1.iloc[i][wt])
            row = {'sample': i, 'weight_type': wt, 'loss': loss_name, 'overlap': overlap}
            row.update(compute_rips_ph(wv))
            row.update(compute_alpha_ph(wv))
            row.update(compute_sublevel_ph(wv))
            rows.append(row)

    # ── Phase 2: Multipers grid (12 scale combos × 6 filtrations) on 10 samples ──
    rng = np.random.default_rng(42)
    grid_idx = rng.choice(N, min(10, N), replace=False)

    # For each filtration pair, find the best scale
    best_per_filt = {}
    all_grid_dists = {}

    for fp in tqdm(FILT_PAIRS, desc=f'{exp_name} grid', ncols=100, leave=False):
        dist_map = {f'sub{s}_nl{nl}': [] for s in GRID_SUBS for nl in GRID_NLS}
        for i in grid_idx:
            pred_wv = np.array(df1.iloc[i]['pred'])
            gt_wv = np.array(df1.iloc[i]['gt'])
            for s in GRID_SUBS:
                for nl in GRID_NLS:
                    fp_pred = multipers_one_filt(pred_wv, fp, subsample=s, n_lines=nl)
                    fp_gt = multipers_one_filt(gt_wv, fp, subsample=s, n_lines=nl)
                    dist_map[f'sub{s}_nl{nl}'].append(mpdist(fp_pred, fp_gt))

        mean_dists = {k: np.nanmean(v) for k, v in dist_map.items()}
        sorted_scales = sorted(mean_dists.keys(), key=lambda k: mean_dists[k])
        best_per_filt[fp] = sorted_scales[0]
        all_grid_dists[fp] = {k: round(v, 6) for k, v in mean_dists.items()}

    # ── Phase 3: Apply best scale per filtration to all samples × key wt ──
    mp_wts = {'pred', 'gt', 'fn1', 'fn3', 'fn5'}

    for i in tqdm(range(N), desc=f'{exp_name} mp', ncols=100, leave=False):
        gt_wv = np.array(df1.iloc[i]['gt'])
        for wt_idx, wt in enumerate(wt_names):
            row_idx = i * len(wt_names) + wt_idx
            if wt in mp_wts:
                wv = np.array(df1.iloc[i][wt])
                for fp in FILT_PAIRS:
                    scale_key = best_per_filt[fp]
                    sub = int(scale_key.split('_')[0].replace('sub',''))
                    nl = int(scale_key.split('_')[1].replace('nl',''))
                    fw = multipers_one_filt(wv, fp, subsample=sub, n_lines=nl)
                    fg = multipers_one_filt(gt_wv, fp, subsample=sub, n_lines=nl)
                    col = f'mpdist_{FILT_LABELS[fp]}'
                    rows[row_idx][col] = mpdist(fw, fg)
            else:
                for fp in FILT_PAIRS:
                    rows[row_idx][f'mpdist_{FILT_LABELS[fp]}'] = float('nan')

    df = pd.DataFrame(rows)

    scales_info = {
        'best_per_filtration': {FILT_LABELS[fp]: best_per_filt[fp] for fp in FILT_PAIRS},
        'all_grid_dists': {FILT_LABELS[fp]: all_grid_dists[fp] for fp in FILT_PAIRS},
        'grid_subs': GRID_SUBS, 'grid_nls': GRID_NLS,
        'filtration_pairs': {fp: FILT_LABELS[fp] for fp in FILT_PAIRS},
    }

    out = output_dir / exp_name
    out.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out / 'df3v2_topology.pkl')
    with open(out / 'topo_grid_v2_scales.json', 'w') as f:
        json.dump(scales_info, f, indent=2)
    return df, scales_info


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--deep-dir', type=Path, default=NB / 'paper_results' / 'deep_analysis')
    p.add_argument('--output-dir', type=Path, default=NB / 'paper_results' / 'deep_analysis')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()

    exps = sorted([d for d in args.deep_dir.iterdir()
                   if d.is_dir() and (d / 'df1_weights.pkl').exists()])
    print(f'Found {len(exps)} experiments')
    print(f'Filtration pairs: {FILT_LABELS}')
    print(f'Grid: subs={GRID_SUBS} × nls={GRID_NLS} = {len(GRID_SUBS)*len(GRID_NLS)} combos × {len(FILT_PAIRS)} filtrations')

    done, skip = 0, 0
    t0 = time.time()
    for exp_dir in exps:
        out_path = args.output_dir / exp_dir.name / 'df3v2_topology.pkl'
        if args.resume and out_path.exists():
            skip += 1
            continue
        try:
            process_experiment(exp_dir, args.output_dir)
            done += 1
            elapsed = time.time() - t0
            rate = elapsed / done
            remaining = rate * (len(exps) - skip - done)
            print(f'  [{done+skip}/{len(exps)}] {exp_dir.name} ({elapsed/60:.0f}min, ~{remaining/60:.0f}min left)')
        except Exception as e:
            print(f'  ERROR {exp_dir.name}: {e}')
            import traceback; traceback.print_exc()

    print(f'\nDone: {done}  Skipped: {skip}  Total: {len(exps)}  Time: {(time.time()-t0)/3600:.1f}h')

if __name__ == '__main__':
    main()
