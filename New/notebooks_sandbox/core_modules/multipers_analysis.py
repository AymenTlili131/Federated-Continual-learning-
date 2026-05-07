"""
Multi-Parameter Persistent Homology Analysis for CNN Weight Spaces

Novel Contribution
------------------
This module implements the paper's central novel contribution: applying
*multi-parameter persistent homology* (via the multipers library,
https://davidlapous.github.io/multipers/) to the analysis and comparison of
CNN weight vectors.

1-Parameter TDA baseline
    Given a flattened weight vector w ∈ R^2464, single-parameter sublevel-set
    persistence captures how many connected components appear as the threshold
    increases.  This gives H0 / H1 persistence diagrams.

2-Parameter (multipers) approach
    We construct a 2-filtration (ξ₁, ξ₂) where:
      ξ₁ = weight value  (sublevel set filtration)
      ξ₂ = normalised layer index  (layer-aware positional parameter)

    ξ₂ ∈ {0/5, 1/5, 2/5, 3/5, 4/5} encodes *which CNN layer* a weight belongs
    to (using the known delimiters [208, 1414, 1514, 2254, 2464]).

    This 2-parameter module M(ξ₁, ξ₂) captures the *joint* structure of the
    weight value distribution AND layer membership — structure that is invisible
    to single-parameter TDA.

Key classes / functions
-----------------------
compute_layer_filtration_values   Build the (value, layer_index) filtration
build_bifiltered_path_complex     Build the 2-parameter simplicial complex
compute_multipers_features        Full feature extraction pipeline (per-layer)
compare_multipers_features        Wasserstein-1 distance between feature dicts
MultipersDivergence               Compares two weight sets (monitoring, not training)
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

try:
    import multipers
    import multipers.filtrations as mf
    import multipers.grids as mg
    MULTIPERS_AVAILABLE = True
except ImportError:
    MULTIPERS_AVAILABLE = False
    warnings.warn(
        "multipers not available — MultipersDivergence will fall back to "
        "single-parameter features. Install: pip install multipers"
    )

# CNN layer boundaries in the flattened 2464-dim weight vector
LAYER_DELIMITERS = [208, 1414, 1514, 2254, 2464]
LAYER_NAMES = ["conv1", "conv2", "conv3", "fc1", "fc2"]
N_LAYERS = len(LAYER_DELIMITERS)


# ---------------------------------------------------------------------------
# Filtration helpers
# ---------------------------------------------------------------------------

def get_layer_index(position: int) -> float:
    """Return the normalised layer index ∈ [0, 1) for a weight position."""
    for k, end in enumerate(LAYER_DELIMITERS):
        if position < end:
            return k / N_LAYERS
    return (N_LAYERS - 1) / N_LAYERS


def compute_layer_filtration_values(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build the two filtration arrays for a flattened weight vector.

    Parameters
    ----------
    w : np.ndarray of shape (2464,)
        Flattened CNN weight vector.

    Returns
    -------
    f1 : np.ndarray — sublevel-set values (w[i] = weight value)
    f2 : np.ndarray — layer-index parameter  (k/5 for layer k)
    """
    n = len(w)
    f1 = w.astype(np.float32)
    f2 = np.array([get_layer_index(i) for i in range(n)], dtype=np.float32)
    return f1, f2


def build_bifiltered_path_complex(
    w: np.ndarray,
    subsample: int = 300
) -> "multipers.SimplexTreeMulti":
    """Build a 2-parameter simplicial complex from a CNN weight vector.

    The complex is a path graph on `subsample` evenly-spaced weight positions,
    with a 2-parameter filtration (weight_value, layer_index).

    Parameters
    ----------
    w        : (2464,) weight vector
    subsample: number of weight positions to keep (default 300)

    Returns
    -------
    stm : multipers.SimplexTreeMulti with num_parameters=2
    """
    if not (GUDHI_AVAILABLE and MULTIPERS_AVAILABLE):
        raise RuntimeError(
            "gudhi and multipers are both required. "
            "pip install gudhi multipers"
        )

    n_full = len(w)
    # Subsample uniformly across the full 2464-dim vector
    idx = np.round(np.linspace(0, n_full - 1, subsample)).astype(int)
    w_sub = w[idx]
    f1 = w_sub.astype(np.float32)
    f2 = np.array([get_layer_index(int(i)) for i in idx], dtype=np.float32)

    n = len(idx)

    # Build a 1-D path graph in GUDHI (vertices + edges)
    st_1p = gudhi.SimplexTree()
    for i in range(n):
        st_1p.insert([i], filtration=float(f1[i]))
    for i in range(n - 1):
        st_1p.insert([i, i + 1],
                     filtration=float(max(f1[i], f1[i + 1])))

    # Lift to 2-parameter SimplexTreeMulti
    stm = multipers.SimplexTreeMulti(st_1p, num_parameters=2)

    # Assign second filtration (layer index) to all simplices
    # For vertices: f2[i].  For edges [i, i+1]: max(f2[i], f2[i+1])
    for i in range(n):
        stm.assign_filtration([i], filtration=[float(f1[i]), float(f2[i])])
    for i in range(n - 1):
        stm.assign_filtration(
            [i, i + 1],
            filtration=[
                float(max(f1[i], f1[i + 1])),
                float(max(f2[i], f2[i + 1]))
            ]
        )

    stm.make_filtration_non_decreasing()
    return stm


def _stm_to_line_betti(stm, n_lines: int = 20) -> np.ndarray:
    """Project a 2-parameter module onto `n_lines` random lines and return
    the concatenated H0 Betti-number signatures.

    Each projection gives a 1-parameter filtration; we extract the sorted
    persistence lifetimes (top-k features) from each line.

    Returns
    -------
    signature : np.ndarray of shape (n_lines * k_features,)
    """
    k_features = 10
    angles = np.linspace(0, np.pi / 2, n_lines, endpoint=False)
    signatures = []

    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        try:
            proj = stm.project_on_line(direction=direction)
            proj.compute_persistence()
            pairs = proj.persistence_intervals_in_dimension(0)
            if len(pairs) == 0:
                lifetimes = np.zeros(k_features, dtype=np.float32)
            else:
                pairs_arr = np.array(pairs)
                finite_mask = pairs_arr[:, 1] != np.inf
                finite_pairs = pairs_arr[finite_mask]
                if len(finite_pairs) == 0:
                    lifetimes = np.zeros(k_features, dtype=np.float32)
                else:
                    lt = np.sort(finite_pairs[:, 1] - finite_pairs[:, 0])[::-1]
                    lifetimes = np.pad(lt[:k_features],
                                      (0, max(0, k_features - len(lt))))
        except Exception:
            lifetimes = np.zeros(k_features, dtype=np.float32)
        signatures.append(lifetimes)

    return np.concatenate(signatures).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-layer feature extraction
# ---------------------------------------------------------------------------

def compute_multipers_features(
    w: np.ndarray,
    subsample_per_layer: int = 60,
    n_lines: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute multipers line-projection signatures per CNN layer.

    For each of the 5 CNN layers, a 2-parameter simplicial complex is built
    from the layer's weight segment (subsampled) and projected onto `n_lines`
    random lines.  The concatenated sorted-lifetime vectors form a fixed-size
    signature.

    Parameters
    ----------
    w                   : (2464,) flattened weight vector
    subsample_per_layer : max number of weight positions per layer
    n_lines             : number of 1-D line projections per layer

    Returns
    -------
    features : dict mapping layer_name → np.ndarray signature
    """
    if not (GUDHI_AVAILABLE and MULTIPERS_AVAILABLE):
        # Fall back to single-parameter sorted-gap signature
        return _fallback_features(w, subsample_per_layer)

    features = {}
    prev = 0
    for k, (end, name) in enumerate(zip(LAYER_DELIMITERS, LAYER_NAMES)):
        segment = w[prev:end].astype(np.float32)
        n_seg = len(segment)

        idx = np.round(np.linspace(0, n_seg - 1,
                                   min(subsample_per_layer, n_seg))
                       ).astype(int)
        w_sub = segment[idx]

        # Build stm for this layer segment
        n = len(idx)
        f1 = w_sub
        # Within-layer positional normalisation (position within the layer)
        f2 = np.linspace(0.0, 1.0, n, dtype=np.float32)

        st_1p = gudhi.SimplexTree()
        for i in range(n):
            st_1p.insert([i], filtration=float(f1[i]))
        for i in range(n - 1):
            st_1p.insert([i, i + 1],
                         filtration=float(max(f1[i], f1[i + 1])))

        stm = multipers.SimplexTreeMulti(st_1p, num_parameters=2)
        for i in range(n):
            stm.assign_filtration([i], filtration=[float(f1[i]), float(f2[i])])
        for i in range(n - 1):
            stm.assign_filtration(
                [i, i + 1],
                filtration=[
                    float(max(f1[i], f1[i + 1])),
                    float(max(f2[i], f2[i + 1]))
                ]
            )
        stm.make_filtration_non_decreasing()

        sig = _stm_to_line_betti(stm, n_lines=n_lines)
        features[name] = sig
        prev = end

    return features


def _fallback_features(w: np.ndarray, subsample: int) -> Dict[str, np.ndarray]:
    """Single-parameter fallback when multipers is unavailable."""
    features = {}
    prev = 0
    for end, name in zip(LAYER_DELIMITERS, LAYER_NAMES):
        seg = w[prev:end]
        s = np.sort(seg)
        gaps = np.sort(s[1:] - s[:-1])[::-1]
        sig = gaps[:subsample]
        if len(sig) < subsample:
            sig = np.pad(sig, (0, subsample - len(sig)))
        features[name] = sig.astype(np.float32)
        prev = end
    return features


# ---------------------------------------------------------------------------
# Comparison / divergence
# ---------------------------------------------------------------------------

def compare_multipers_features(
    feat1: Dict[str, np.ndarray],
    feat2: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Wasserstein-1 distance between per-layer multipers signatures.

    Returns
    -------
    distances : dict mapping layer_name → float distance
    """
    distances = {}
    for name in LAYER_NAMES:
        if name in feat1 and name in feat2:
            s1 = np.sort(feat1[name])
            s2 = np.sort(feat2[name])
            # Pad to same length
            n = max(len(s1), len(s2))
            s1 = np.pad(s1, (0, n - len(s1)))
            s2 = np.pad(s2, (0, n - len(s2)))
            distances[name] = float(np.mean(np.abs(s1 - s2)))
    return distances


class MultipersDivergence:
    """Compare two sets of CNN weights using multi-parameter persistence.

    This is a MONITORING-ONLY class (not differentiable).  It is used for:
    - Offline analysis: comparing predicted vs. ground-truth weights
    - Tracking the evolution of weight topology during training
    - Generating paper figures showing multi-parameter persistence structure

    Parameters
    ----------
    subsample_per_layer : int  (default 60)
    n_lines             : int  number of line projections  (default 20)
    """

    def __init__(self, subsample_per_layer: int = 60, n_lines: int = 20):
        self.subsample_per_layer = subsample_per_layer
        self.n_lines = n_lines

    def compare(
        self,
        w_pred: np.ndarray,
        w_gt: np.ndarray,
        w_finetuned: Optional[np.ndarray] = None,
    ) -> Dict:
        """Full divergence analysis between predicted and ground-truth weights.

        Parameters
        ----------
        w_pred     : (2464,) predicted weights
        w_gt       : (2464,) ground-truth weights
        w_finetuned: (2464,) finetuned weights (optional)

        Returns
        -------
        results : dict with keys
            'pred_features'      : per-layer multipers signatures for w_pred
            'gt_features'        : per-layer multipers signatures for w_gt
            'pred_vs_gt'         : per-layer Wasserstein-1 distances
            'finetuned_features' : (if w_finetuned given)
            'finetuned_vs_gt'    : (if w_finetuned given)
            'summary'            : aggregated mean divergence
        """
        feat_pred = compute_multipers_features(
            w_pred, self.subsample_per_layer, self.n_lines
        )
        feat_gt = compute_multipers_features(
            w_gt, self.subsample_per_layer, self.n_lines
        )

        dist_pred_gt = compare_multipers_features(feat_pred, feat_gt)

        results = {
            'pred_features': feat_pred,
            'gt_features': feat_gt,
            'pred_vs_gt': dist_pred_gt,
            'summary': {
                'mean_divergence_pred_gt': float(np.mean(list(dist_pred_gt.values()))),
            }
        }

        if w_finetuned is not None:
            feat_fn = compute_multipers_features(
                w_finetuned, self.subsample_per_layer, self.n_lines
            )
            dist_fn_gt = compare_multipers_features(feat_fn, feat_gt)
            results['finetuned_features'] = feat_fn
            results['finetuned_vs_gt'] = dist_fn_gt
            results['summary']['mean_divergence_finetuned_gt'] = float(
                np.mean(list(dist_fn_gt.values()))
            )

        return results

    def timeline(
        self,
        weight_snapshots: List[np.ndarray],
        epochs: Optional[List[int]] = None,
    ) -> Dict:
        """Track multipers divergence across training epochs.

        Parameters
        ----------
        weight_snapshots : list of (2464,) arrays, one per epoch
        epochs           : optional epoch labels

        Returns
        -------
        timeline : dict  with 'epochs', 'per_layer', 'mean_divergence'
        """
        n = len(weight_snapshots)
        if epochs is None:
            epochs = list(range(n))

        # Compute features for all snapshots
        all_features = [
            compute_multipers_features(w, self.subsample_per_layer, self.n_lines)
            for w in weight_snapshots
        ]

        # Divergence from final snapshot (or baseline = epoch 0)
        baseline = all_features[0]
        per_layer_divergence = {name: [] for name in LAYER_NAMES}
        mean_divergence = []

        for feat in all_features:
            d = compare_multipers_features(feat, baseline)
            for name in LAYER_NAMES:
                per_layer_divergence[name].append(d.get(name, 0.0))
            mean_divergence.append(float(np.mean(list(d.values()))))

        return {
            'epochs': epochs,
            'per_layer': per_layer_divergence,
            'mean_divergence': mean_divergence,
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("Testing multipers_analysis.py")
    print(f"  gudhi:     {'OK' if GUDHI_AVAILABLE else 'MISSING'}")
    print(f"  multipers: {'OK (v' + multipers.__version__ + ')' if MULTIPERS_AVAILABLE else 'MISSING'}")

    np.random.seed(0)
    w_pred = np.random.randn(2464).astype(np.float32) * 0.1
    w_gt   = np.random.randn(2464).astype(np.float32) * 0.1

    div = MultipersDivergence(subsample_per_layer=30, n_lines=5)
    res = div.compare(w_pred, w_gt)

    print("\nPer-layer multipers divergence (pred vs GT):")
    for layer, d in res['pred_vs_gt'].items():
        print(f"  {layer}: {d:.6f}")
    print(f"Mean divergence: {res['summary']['mean_divergence_pred_gt']:.6f}")
