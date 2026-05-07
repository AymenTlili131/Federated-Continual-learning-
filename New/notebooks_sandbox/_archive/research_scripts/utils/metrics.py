"""Metrics computation utilities."""

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from scipy.linalg import eigvalsh
from typing import Dict, List, Tuple, Optional

def compute_eigenvalues(weights: np.ndarray, per_layer: bool = False) -> Dict:
    """Compute eigenvalues of weight matrix."""
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # Compute covariance matrix
    cov = np.cov(weights.T)
    eigenvalues = eigvalsh(cov)
    
    return {
        'eigenvalues': eigenvalues,
        'max_eigenvalue': eigenvalues.max(),
        'min_eigenvalue': eigenvalues.min(),
        'spectral_norm': np.abs(eigenvalues).max()
    }

def compute_persistent_homology(weights: np.ndarray, max_dim: int = 2) -> Dict:
    """Compute persistent homology features."""
    # Placeholder - requires giotto-tda
    return {
        'betti_0': 0,
        'betti_1': 0,
        'persistence_entropy': 0.0
    }

def compute_rmt_metrics(weights: np.ndarray) -> Dict:
    """Compute Random Matrix Theory metrics."""
    eigenvalues = compute_eigenvalues(weights)['eigenvalues']
    
    # Spectral density
    hist, bins = np.histogram(eigenvalues, bins=50, density=True)
    
    return {
        'eigenvalues': eigenvalues,
        'spectral_density': hist,
        'bins': bins
    }

def compute_wasserstein_distance(weights1: np.ndarray, weights2: np.ndarray) -> float:
    """Compute Wasserstein distance between weight distributions."""
    return wasserstein_distance(weights1.flatten(), weights2.flatten())
