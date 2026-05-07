"""
Differentiable Persistence-Inspired Loss Functions

All nn.Module losses here are fully differentiable via PyTorch autograd and
run entirely on GPU — no GUDHI, no NumPy, no CPU offloading.

Mathematical basis
------------------
For a 1-D weight sequence f: {0,...,n-1} → R on a path graph,
the H0 sublevel-set persistence diagram D(f) has birth-times equal to the
local minima of f.  The stability theorem gives:

    d_W1(D(f), D(g))  ≤  || f_sorted – g_sorted ||_1

so the Wasserstein-1 distance between sorted weight vectors is a valid
upper bound on the H0 persistence diagram distance.

The persistence LANDSCAPE (λ_k) of D(f) can be approximated by the
k-th largest gap between consecutive sorted values (= lifetime of the
k-th most persistent H0 feature).  L1 between these sorted gap sequences
gives a differentiable proxy for the Wasserstein distance on landscapes.

The persistence IMAGE is approximated via a differentiable 2-D Gaussian
KDE over the (birth, lifetime) scatter of the sorted sequence.

For offline MONITORING (not training), GUDHI-based helpers are provided
as plain functions prefixed with compute_gudhi_*.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List

# GUDHI is only used for offline monitoring helpers, not for training losses
try:
    import gudhi
    from gudhi.representations import Landscape, PersistenceImage as GUDHIPersistenceImage
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

try:
    import multipers
    MULTIPERS_AVAILABLE = True
except ImportError:
    MULTIPERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# CNN layer boundaries (matches AutoregressiveLoss / LAYER_DELIMITERS)
# ---------------------------------------------------------------------------
LAYER_DELIMITERS = [208, 1414, 1514, 2254, 2464]


# ===========================================================================
# DIFFERENTIABLE TRAINING LOSSES  (true autograd, 100% GPU)
# ===========================================================================

class PersistenceLandscapeLoss(nn.Module):
    """Differentiable persistence landscape loss.

    The k-th persistence landscape λ_k is approximated by the k-th largest
    gap between consecutive sorted weight values (= lifetime of the k-th most
    persistent H0 feature in the sublevel filtration on a path graph).

    Loss = Wasserstein-1 distance between the sorted gap (landscape) vectors.
    Fully differentiable via torch.sort.  No GUDHI, no NumPy, no CPU ops.
    """
    def __init__(self, num_landscapes: int = 5, resolution: int = 100):
        super().__init__()
        self.num_landscapes = num_landscapes
        self.resolution = resolution

    @staticmethod
    def _sorted_gaps(w: torch.Tensor) -> torch.Tensor:
        """Sorted weight vector → sorted gaps (persistence lifetimes), descending."""
        s = torch.sort(w, dim=-1).values           # (N,)
        gaps = s[1:] - s[:-1]                      # (N-1,) all ≥ 0
        return torch.sort(gaps, descending=True).values

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_gaps = self._sorted_gaps(pred.reshape(-1))   # flatten batch → 1 sequence
        tgt_gaps = self._sorted_gaps(target.reshape(-1))
        return F.l1_loss(pred_gaps, tgt_gaps)


class PersistenceImageLoss(nn.Module):
    """Differentiable persistence image loss via Gaussian soft-histogram KDE.

    A 2-D soft image is built from the (birth, lifetime) scatter of sorted
    weight values.  Pixel intensities are differentiable w.r.t. the weights
    because the Gaussian kernel is applied directly to the sorted tensors.
    Loss = MSE between the two normalised persistence images.

    No GUDHI, no NumPy, no CPU offloading.
    """
    def __init__(self, bandwidth: float = 1.0, resolution: int = 20):
        super().__init__()
        self.bandwidth = bandwidth
        self.resolution = resolution

    def _build_image(self, w: torch.Tensor) -> torch.Tensor:
        """Build a (resolution × resolution) differentiable persistence image."""
        s = torch.sort(w.reshape(-1)).values       # (N,)
        births = s[:-1]                            # (N-1,)
        lifetimes = (s[1:] - s[:-1]).clamp(min=0) # (N-1,)

        R = self.resolution
        device, dtype = w.device, w.dtype

        b_min, b_max = births.detach().min(), births.detach().max()
        l_max = lifetimes.detach().max()

        b_range = (b_max - b_min).clamp(min=1e-6)
        l_range = l_max.clamp(min=1e-6)

        b_grid = torch.linspace(b_min.item(), b_max.item(), R, device=device, dtype=dtype)  # (R,)
        l_grid = torch.linspace(0.0, l_max.item(), R, device=device, dtype=dtype)           # (R,)

        sigma_b = self.bandwidth * b_range / R
        sigma_l = self.bandwidth * l_range / R

        # Gaussian kernels: (N-1, R)
        kb = torch.exp(-0.5 * ((births.unsqueeze(-1) - b_grid) / sigma_b) ** 2)
        kl = torch.exp(-0.5 * ((lifetimes.unsqueeze(-1) - l_grid) / sigma_l) ** 2)

        # Weight each point by its lifetime (more persistent = more important)
        w_pt = lifetimes / (lifetimes.sum() + 1e-8)  # (N-1,)

        # img[i,j] = sum_k  w_pt[k] * kb[k,i] * kl[k,j]
        img = torch.einsum('k,ki,kj->ij', w_pt, kb, kl)  # (R, R)
        return img / (img.sum() + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_img = self._build_image(pred.reshape(-1))
        tgt_img = self._build_image(target.reshape(-1))
        return F.mse_loss(pred_img, tgt_img)


class LayerwisePersistenceLandscapeLoss(nn.Module):
    """Layerwise version of PersistenceLandscapeLoss — applies per CNN layer.

    All layers are processed (no skipping).  The layer boundaries defined by
    LAYER_DELIMITERS match the AutoregressiveLoss / CNN injection delimiters.
    """
    def __init__(self, delimiters: Optional[List[int]] = None,
                 num_landscapes: int = 5, resolution: int = 100):
        super().__init__()
        self.delimiters = delimiters or LAYER_DELIMITERS
        self.base_loss = PersistenceLandscapeLoss(num_landscapes, resolution)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses, start = [], 0
        for end in self.delimiters:
            losses.append(self.base_loss(pred[:, start:end], target[:, start:end]))
            start = end
        return torch.stack(losses).mean()


class LayerwisePersistenceImageLoss(nn.Module):
    """Layerwise version of PersistenceImageLoss — applies per CNN layer."""
    def __init__(self, delimiters: Optional[List[int]] = None,
                 bandwidth: float = 1.0, resolution: int = 20):
        super().__init__()
        self.delimiters = delimiters or LAYER_DELIMITERS
        self.base_loss = PersistenceImageLoss(bandwidth, resolution)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses, start = [], 0
        for end in self.delimiters:
            losses.append(self.base_loss(pred[:, start:end], target[:, start:end]))
            start = end
        return torch.stack(losses).mean()


# ===========================================================================
# OFFLINE MONITORING HELPERS  (GUDHI-based, NOT for training / backprop)
# ===========================================================================

def compute_gudhi_persistence_diagram(weights_numpy: np.ndarray):
    """[MONITORING ONLY] Compute exact H0 sublevel-set persistence via GUDHI.

    Returns list of (birth, death) pairs.  Do NOT call inside a training
    forward pass — this uses GUDHI C++ and breaks the gradient graph.
    """
    if not GUDHI_AVAILABLE:
        raise RuntimeError("GUDHI not installed. pip install gudhi")
    w = weights_numpy.flatten()
    st = gudhi.SimplexTree()
    for i, v in enumerate(w):
        st.insert([i], filtration=float(v))
    for i in range(len(w) - 1):
        st.insert([i, i + 1], filtration=float(max(w[i], w[i + 1])))
    st.compute_persistence()
    return st.persistence_intervals_in_dimension(0)


def compute_gudhi_landscape(weights_numpy: np.ndarray,
                            num_landscapes: int = 5,
                            resolution: int = 100) -> np.ndarray:
    """[MONITORING ONLY] Compute exact GUDHI persistence landscape vector."""
    diagram = compute_gudhi_persistence_diagram(weights_numpy)
    if len(diagram) == 0:
        return np.zeros(num_landscapes * resolution)
    lsc = Landscape(num_landscapes=num_landscapes, resolution=resolution)
    return lsc.fit_transform([diagram])[0]


# ---------------------------------------------------------------------------
# Convenience lookup
# ---------------------------------------------------------------------------

def get_persistence_loss(name: str) -> nn.Module:
    """Return a differentiable persistence loss module by name."""
    catalogue = {
        'PersLandscape':    PersistenceLandscapeLoss(),
        'PersImage':        PersistenceImageLoss(),
        'LW_PersLandscape': LayerwisePersistenceLandscapeLoss(),
        'LW_PersImage':     LayerwisePersistenceImageLoss(),
    }
    if name not in catalogue:
        raise KeyError(f"Unknown persistence loss '{name}'. "
                       f"Available: {list(catalogue.keys())}")
    return catalogue[name]
