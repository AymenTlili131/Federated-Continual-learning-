"""
Persistent Homology and Topological Data Analysis for Weight Comparison
Computes persistence diagrams, Betti curves, and vectorizable features
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude, BettiCurve
    from gtda.plotting import plot_diagram, plot_betti_curves
    GIOTTO_AVAILABLE = True
except ImportError:
    GIOTTO_AVAILABLE = False
    warnings.warn("giotto-tda not available. Install with: pip install giotto-tda")

try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("ripser not available. Install with: pip install ripser persim")

from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import matplotlib.pyplot as plt


class PersistentHomologyAnalyzer:
    """
    Compute persistent homology features for weight vectors
    """
    def __init__(self, max_dimension: int = 2, max_edge_length: float = np.inf):
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        
        if GIOTTO_AVAILABLE:
            self.vr_persistence = VietorisRipsPersistence(
                homology_dimensions=list(range(max_dimension + 1)),
                max_edge_length=max_edge_length,
                n_jobs=-1
            )
            self.betti_curve = BettiCurve(n_bins=100)
            self.persistence_entropy = PersistenceEntropy()
            self.amplitude = Amplitude(metric='landscape')
    
    def compute_persistence_diagram(self, weights: np.ndarray) -> Dict:
        """
        Compute persistence diagram for weight vector
        
        Args:
            weights: (n_samples, n_features) or (n_features,)
        
        Returns:
            Dictionary with persistence diagrams and features
        """
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        
        results = {}
        
        if RIPSER_AVAILABLE:
            # Use Ripser for computation
            diagrams = ripser(weights.T, maxdim=self.max_dimension)
            results['ripser_diagrams'] = diagrams['dgms']
            
            # Extract features from diagrams
            results['features'] = self._extract_diagram_features(diagrams['dgms'])
        
        elif GIOTTO_AVAILABLE:
            # Use giotto-tda
            # Reshape for giotto-tda: (n_samples, n_points, n_dimensions)
            point_cloud = weights.T.reshape(1, -1, 1)
            
            diagrams = self.vr_persistence.fit_transform(point_cloud)
            results['giotto_diagrams'] = diagrams
            
            # Compute Betti curves
            betti_curves = self.betti_curve.fit_transform(diagrams)
            results['betti_curves'] = betti_curves
            
            # Compute persistence entropy
            entropies = self.persistence_entropy.fit_transform(diagrams)
            results['persistence_entropy'] = entropies
            
            # Compute amplitude (persistence landscape)
            amplitudes = self.amplitude.fit_transform(diagrams)
            results['amplitudes'] = amplitudes
        
        else:
            # Fallback: simple distance-based features
            results['features'] = self._compute_distance_features(weights)
        
        return results
    
    def _extract_diagram_features(self, diagrams: List[np.ndarray]) -> Dict[str, float]:
        """Extract numerical features from persistence diagrams"""
        features = {}
        
        for dim, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                continue
            
            # Remove infinite death times
            dgm_finite = dgm[dgm[:, 1] != np.inf]
            
            if len(dgm_finite) == 0:
                continue
            
            # Birth and death times
            births = dgm_finite[:, 0]
            deaths = dgm_finite[:, 1]
            lifetimes = deaths - births
            
            # Features for this dimension
            features[f'h{dim}_num_features'] = len(dgm_finite)
            features[f'h{dim}_max_lifetime'] = np.max(lifetimes)
            features[f'h{dim}_mean_lifetime'] = np.mean(lifetimes)
            features[f'h{dim}_std_lifetime'] = np.std(lifetimes)
            features[f'h{dim}_total_persistence'] = np.sum(lifetimes)
            features[f'h{dim}_entropy'] = entropy(lifetimes / np.sum(lifetimes) + 1e-10)
            
            # Percentiles
            for p in [25, 50, 75, 90]:
                features[f'h{dim}_lifetime_p{p}'] = np.percentile(lifetimes, p)
        
        return features
    
    def _compute_distance_features(self, weights: np.ndarray) -> Dict[str, float]:
        """Compute simple distance-based features as fallback"""
        # Pairwise distances
        if weights.shape[0] > 1:
            distances = pdist(weights)
        else:
            # For single sample, use autocorrelation-like features
            distances = pdist(weights.reshape(-1, 1))
        
        features = {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances),
            'min_distance': np.min(distances),
        }
        
        return features
    
    def compare_diagrams(self, weights1: np.ndarray, weights2: np.ndarray,
                        metric: str = 'wasserstein') -> float:
        """
        Compare two persistence diagrams
        
        Args:
            weights1, weights2: Weight vectors to compare
            metric: Distance metric ('wasserstein', 'bottleneck')
        
        Returns:
            Distance between diagrams
        """
        if not RIPSER_AVAILABLE:
            # Fallback to simple comparison
            return np.linalg.norm(weights1 - weights2)
        
        dgm1 = ripser(weights1.reshape(-1, 1), maxdim=self.max_dimension)['dgms']
        dgm2 = ripser(weights2.reshape(-1, 1), maxdim=self.max_dimension)['dgms']
        
        # Compute distance for each dimension
        total_distance = 0.0
        
        try:
            from persim import wasserstein, bottleneck
            
            for dim in range(len(dgm1)):
                d1 = dgm1[dim][dgm1[dim][:, 1] != np.inf]
                d2 = dgm2[dim][dgm2[dim][:, 1] != np.inf]
                
                if len(d1) == 0 or len(d2) == 0:
                    continue
                
                if metric == 'wasserstein':
                    dist = wasserstein(d1, d2)
                else:  # bottleneck
                    dist = bottleneck(d1, d2)
                
                total_distance += dist
        except ImportError:
            # Simple L2 distance as fallback
            total_distance = np.linalg.norm(weights1 - weights2)
        
        return total_distance
    
    def plot_persistence_diagram(self, weights: np.ndarray, 
                                 title: str = "Persistence Diagram",
                                 save_path: Optional[str] = None):
        """Plot persistence diagram"""
        if not RIPSER_AVAILABLE:
            print("Ripser not available for plotting")
            return None
        
        diagrams = ripser(weights.reshape(-1, 1), maxdim=self.max_dimension)['dgms']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_diagrams(diagrams, ax=ax)
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class BettiCurveAnalyzer:
    """
    Compute and analyze Betti curves
    """
    def __init__(self, n_bins: int = 100, max_dimension: int = 2):
        self.n_bins = n_bins
        self.max_dimension = max_dimension
    
    def compute_betti_curves(self, weights: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute Betti curves for each homology dimension
        
        Returns:
            Dictionary mapping dimension to Betti curve
        """
        if not RIPSER_AVAILABLE:
            return {}
        
        diagrams = ripser(weights.reshape(-1, 1), maxdim=self.max_dimension)['dgms']
        
        betti_curves = {}
        
        for dim, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                betti_curves[dim] = np.zeros(self.n_bins)
                continue
            
            # Remove infinite points
            dgm_finite = dgm[dgm[:, 1] != np.inf]
            
            if len(dgm_finite) == 0:
                betti_curves[dim] = np.zeros(self.n_bins)
                continue
            
            # Create filtration values
            min_val = np.min(dgm_finite)
            max_val = np.max(dgm_finite)
            filtration = np.linspace(min_val, max_val, self.n_bins)
            
            # Compute Betti numbers at each filtration value
            betti = np.zeros(self.n_bins)
            for i, f in enumerate(filtration):
                # Count features alive at filtration value f
                alive = (dgm_finite[:, 0] <= f) & (dgm_finite[:, 1] > f)
                betti[i] = np.sum(alive)
            
            betti_curves[dim] = betti
        
        return betti_curves
    
    def compare_betti_curves(self, curve1: np.ndarray, curve2: np.ndarray,
                            metric: str = 'l2') -> float:
        """Compare two Betti curves"""
        if metric == 'l2':
            return np.linalg.norm(curve1 - curve2)
        elif metric == 'l1':
            return np.sum(np.abs(curve1 - curve2))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def plot_betti_curves(self, betti_curves: Dict[int, np.ndarray],
                         title: str = "Betti Curves",
                         save_path: Optional[str] = None):
        """Plot Betti curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for dim, curve in betti_curves.items():
            ax.plot(curve, label=f'$\\beta_{dim}$', linewidth=2)
        
        ax.set_xlabel('Filtration')
        ax.set_ylabel('Betti Number')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class TopologicalFeatureExtractor:
    """
    Extract vectorizable topological features for ML
    """
    def __init__(self, max_dimension: int = 2):
        self.max_dimension = max_dimension
        self.ph_analyzer = PersistentHomologyAnalyzer(max_dimension=max_dimension)
        self.betti_analyzer = BettiCurveAnalyzer(max_dimension=max_dimension)
    
    def extract_features(self, weights: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive topological features
        
        Returns:
            Feature vector combining all topological descriptors
        """
        features = []
        
        # Persistence diagram features
        ph_results = self.ph_analyzer.compute_persistence_diagram(weights)
        if 'features' in ph_results:
            # Convert to ordered array
            feature_dict = ph_results['features']
            features.extend([v for v in feature_dict.values()])
        
        # Betti curve features
        betti_curves = self.betti_analyzer.compute_betti_curves(weights)
        for dim in range(self.max_dimension + 1):
            if dim in betti_curves:
                curve = betti_curves[dim]
                # Summary statistics of Betti curve
                features.extend([
                    np.max(curve),
                    np.mean(curve),
                    np.std(curve),
                    np.sum(curve > 0) / len(curve),  # Fraction non-zero
                ])
        
        return np.array(features)
    
    def batch_extract_features(self, weights_batch: np.ndarray) -> np.ndarray:
        """Extract features for batch of weight vectors"""
        feature_list = []
        
        for i in range(weights_batch.shape[0]):
            features = self.extract_features(weights_batch[i])
            feature_list.append(features)
        
        return np.array(feature_list)


def compare_weight_topology(predicted: np.ndarray, 
                            finetuned: np.ndarray,
                            ground_truth: np.ndarray,
                            max_dimension: int = 2) -> Dict[str, Dict]:
    """
    Compare topological features of predicted, finetuned, and ground truth weights
    
    Returns:
        Dictionary with comparison metrics
    """
    ph_analyzer = PersistentHomologyAnalyzer(max_dimension=max_dimension)
    betti_analyzer = BettiCurveAnalyzer(max_dimension=max_dimension)
    
    results = {
        'predicted': {},
        'finetuned': {},
        'ground_truth': {},
        'comparisons': {}
    }
    
    # Compute features for each
    for name, weights in [('predicted', predicted), 
                          ('finetuned', finetuned),
                          ('ground_truth', ground_truth)]:
        ph_results = ph_analyzer.compute_persistence_diagram(weights)
        betti_curves = betti_analyzer.compute_betti_curves(weights)
        
        results[name]['persistence'] = ph_results
        results[name]['betti_curves'] = betti_curves
    
    # Compare predicted vs ground truth
    results['comparisons']['pred_vs_gt_diagram'] = ph_analyzer.compare_diagrams(
        predicted, ground_truth
    )
    
    # Compare finetuned vs ground truth
    results['comparisons']['finetuned_vs_gt_diagram'] = ph_analyzer.compare_diagrams(
        finetuned, ground_truth
    )
    
    # Compare Betti curves
    for dim in range(max_dimension + 1):
        if dim in results['predicted']['betti_curves'] and dim in results['ground_truth']['betti_curves']:
            pred_curve = results['predicted']['betti_curves'][dim]
            gt_curve = results['ground_truth']['betti_curves'][dim]
            ft_curve = results['finetuned']['betti_curves'][dim]
            
            results['comparisons'][f'pred_vs_gt_betti_{dim}'] = betti_analyzer.compare_betti_curves(
                pred_curve, gt_curve
            )
            results['comparisons'][f'finetuned_vs_gt_betti_{dim}'] = betti_analyzer.compare_betti_curves(
                ft_curve, gt_curve
            )
    
    return results


if __name__ == "__main__":
    print("Testing Persistent Homology Analysis")
    print("=" * 60)
    
    # Create test weight vectors
    np.random.seed(42)
    weights = np.random.randn(2464)
    
    # Test persistence diagram
    ph_analyzer = PersistentHomologyAnalyzer(max_dimension=2)
    results = ph_analyzer.compute_persistence_diagram(weights)
    
    print("\nPersistence Diagram Features:")
    if 'features' in results:
        for key, value in results['features'].items():
            print(f"  {key}: {value:.4f}")
    
    # Test Betti curves
    betti_analyzer = BettiCurveAnalyzer(n_bins=50, max_dimension=2)
    betti_curves = betti_analyzer.compute_betti_curves(weights)
    
    print("\nBetti Curves:")
    for dim, curve in betti_curves.items():
        print(f"  H{dim}: max={np.max(curve):.2f}, mean={np.mean(curve):.2f}")
    
    # Test feature extraction
    feature_extractor = TopologicalFeatureExtractor(max_dimension=2)
    features = feature_extractor.extract_features(weights)
    
    print(f"\nExtracted {len(features)} topological features")
    
    print("\nLibrary Status:")
    print(f"  giotto-tda: {'✓' if GIOTTO_AVAILABLE else '✗'}")
    print(f"  ripser: {'✓' if RIPSER_AVAILABLE else '✗'}")
