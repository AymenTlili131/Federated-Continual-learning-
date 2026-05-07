"""
Random Matrix Theory (RMT) Analysis for Neural Network Weights
Implements eigenvalue analysis, spectral density, and Marchenko-Pastur law comparison
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import warnings


class RandomMatrixAnalyzer:
    """
    Analyze weight matrices using Random Matrix Theory
    """
    def __init__(self, layer_shapes: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            layer_shapes: List of (rows, cols) for each layer's weight matrix
        """
        # Default CNN layer shapes from Silu.py
        if layer_shapes is None:
            self.layer_shapes = [
                (8, 25),      # Conv1: 8 filters, 5x5x1 = 25
                (6, 200),     # Conv2: 6 filters, 5x5x8 = 200
                (4, 24),      # Conv3: 4 filters, 2x2x6 = 24
                (20, 36),     # FC1: 20 outputs, 36 inputs
                (10, 20),     # FC2: 10 outputs, 20 inputs
            ]
        else:
            self.layer_shapes = layer_shapes
        
        # Define layer boundaries in flattened vector
        self.layer_ranges = self._compute_layer_ranges()
    
    def _compute_layer_ranges(self) -> List[Tuple[int, int, str]]:
        """Compute start/end indices for each layer in flattened vector"""
        ranges = []
        current_idx = 0
        
        layer_names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
        
        for i, (name, shape) in enumerate(zip(layer_names, self.layer_shapes)):
            n_weights = shape[0] * shape[1]
            n_bias = shape[0]
            
            # Weight range
            ranges.append((current_idx, current_idx + n_weights, f'{name}_weight'))
            current_idx += n_weights
            
            # Bias range
            ranges.append((current_idx, current_idx + n_bias, f'{name}_bias'))
            current_idx += n_bias
        
        return ranges
    
    def extract_weight_matrices(self, flattened_weights: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract individual weight matrices from flattened vector
        
        Args:
            flattened_weights: (2464,) flattened weight vector
        
        Returns:
            Dictionary mapping layer names to weight matrices
        """
        matrices = {}
        
        for start, end, name in self.layer_ranges:
            if 'bias' in name:
                # Bias is 1D
                matrices[name] = flattened_weights[start:end]
            else:
                # Reshape to matrix using clean layer-name→index lookup
                base = name.split('_')[0]  # e.g. 'conv1', 'fc2'
                if base.startswith('conv'):
                    layer_idx = ['conv1', 'conv2', 'conv3'].index(base)
                else:
                    layer_idx = ['fc1', 'fc2'].index(base) + 3

                shape = self.layer_shapes[layer_idx]
                weights = flattened_weights[start:end]
                matrices[name] = weights.reshape(shape)
        
        return matrices
    
    def compute_eigenvalues(self, weight_matrix: np.ndarray) -> np.ndarray:
        """
        Compute eigenvalues of weight matrix
        For non-square matrices, use singular values
        """
        if weight_matrix.ndim == 1:
            # For bias vectors, return the values themselves
            return weight_matrix
        
        m, n = weight_matrix.shape
        
        if m == n:
            # Square matrix: use eigenvalues
            eigenvalues = np.linalg.eigvalsh(weight_matrix @ weight_matrix.T)
        else:
            # Non-square: use singular values
            singular_values = np.linalg.svd(weight_matrix, compute_uv=False)
            eigenvalues = singular_values ** 2
        
        return eigenvalues
    
    def marchenko_pastur_density(self, x: np.ndarray, gamma: float, sigma: float = 1.0) -> np.ndarray:
        """
        Marchenko-Pastur distribution density
        
        Args:
            x: Points to evaluate density
            gamma: Aspect ratio (n/p) where n < p
            sigma: Variance of original matrix entries
        
        Returns:
            Density values
        """
        lambda_minus = sigma**2 * (1 - np.sqrt(gamma))**2
        lambda_plus = sigma**2 * (1 + np.sqrt(gamma))**2
        
        density = np.zeros_like(x)
        mask = (x >= lambda_minus) & (x <= lambda_plus)
        
        density[mask] = (1 / (2 * np.pi * sigma**2 * gamma * x[mask])) * \
                        np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus))
        
        return density
    
    def compute_spectral_density(self, eigenvalues: np.ndarray, 
                                 n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute empirical spectral density
        
        Returns:
            bins, density
        """
        hist, bin_edges = np.histogram(eigenvalues, bins=n_bins, density=True)
        bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bins, hist
    
    def analyze_layer(self, weight_matrix: np.ndarray, layer_name: str) -> Dict:
        """
        Comprehensive RMT analysis of a single layer
        
        Returns:
            Dictionary with analysis results
        """
        results = {'layer_name': layer_name}
        
        if weight_matrix.ndim == 1:
            # Bias vector
            results['type'] = 'bias'
            results['mean'] = np.mean(weight_matrix)
            results['std'] = np.std(weight_matrix)
            results['max'] = np.max(weight_matrix)
            results['min'] = np.min(weight_matrix)
            return results
        
        results['type'] = 'weight'
        results['shape'] = weight_matrix.shape
        
        # Compute eigenvalues
        eigenvalues = self.compute_eigenvalues(weight_matrix)
        results['eigenvalues'] = eigenvalues
        
        # Eigenvalue statistics
        results['max_eigenvalue'] = np.max(eigenvalues)
        results['min_eigenvalue'] = np.min(eigenvalues)
        results['mean_eigenvalue'] = np.mean(eigenvalues)
        results['std_eigenvalue'] = np.std(eigenvalues)
        results['trace'] = np.sum(eigenvalues)
        
        # Spectral radius / max eigenvalue magnitude
        max_eig = float(np.max(np.abs(eigenvalues)))
        results['spectral_radius'] = max_eig

        # Rank (relative threshold: 1e-4 × max eigenvalue magnitude)
        results['rank'] = int(np.sum(eigenvalues > 1e-4 * max_eig))

        # Condition number (ratio of largest to smallest non-negligible eigenvalue)
        significant = eigenvalues[eigenvalues > 1e-4 * max_eig]
        if len(significant) >= 2:
            results['condition_number'] = float(significant.max() / significant.min())
        elif len(significant) == 1:
            results['condition_number'] = 1.0
        else:
            results['condition_number'] = float('nan')
        # Effective rank: exp(Shannon entropy of normalised non-negative spectrum)
        eig_pos = np.clip(eigenvalues, 0, None)  # eigenvalues can be slightly negative due to fp errors
        eig_norm = eig_pos / (eig_pos.sum() + 1e-30)
        results['effective_rank'] = float(np.exp(stats.entropy(eig_norm + 1e-30)))
        
        # Spectral density
        bins, density = self.compute_spectral_density(eigenvalues)
        results['spectral_bins'] = bins
        results['spectral_density'] = density
        
        # Marchenko-Pastur comparison
        m, n = weight_matrix.shape
        gamma = min(m, n) / max(m, n)
        sigma = np.std(weight_matrix)
        
        mp_density = self.marchenko_pastur_density(bins, gamma, sigma)
        results['mp_density'] = mp_density
        results['mp_gamma'] = gamma
        
        # KL divergence from Marchenko-Pastur
        empirical_density = density + 1e-10
        theoretical_density = mp_density + 1e-10
        
        # Normalize
        empirical_density = empirical_density / np.sum(empirical_density)
        theoretical_density = theoretical_density / np.sum(theoretical_density)
        
        results['kl_divergence_from_mp'] = stats.entropy(empirical_density, theoretical_density)
        
        return results
    
    def analyze_all_layers(self, flattened_weights: np.ndarray) -> Dict[str, Dict]:
        """
        Analyze all layers in the network
        
        Returns:
            Dictionary mapping layer names to analysis results
        """
        matrices = self.extract_weight_matrices(flattened_weights)
        
        results = {}
        for layer_name, matrix in matrices.items():
            if 'weight' in layer_name:  # Only analyze weight matrices, skip biases
                results[layer_name] = self.analyze_layer(matrix, layer_name)
        
        return results
    
    def compare_weights(self, weights1: np.ndarray, weights2: np.ndarray,
                       layer_name: Optional[str] = None) -> Dict:
        """
        Compare eigenvalue distributions of two weight sets
        
        Returns:
            Comparison metrics
        """
        if weights1.ndim == 1 and len(weights1) == 2464:
            # Flattened weights - extract specific layer or analyze all
            matrices1 = self.extract_weight_matrices(weights1)
            matrices2 = self.extract_weight_matrices(weights2)
            
            if layer_name:
                w1 = matrices1[layer_name]
                w2 = matrices2[layer_name]
            else:
                # Compare all layers
                comparisons = {}
                for name in matrices1.keys():
                    if 'weight' in name:
                        comparisons[name] = self.compare_weights(
                            matrices1[name], matrices2[name], name
                        )
                return comparisons
        else:
            w1 = weights1
            w2 = weights2
        
        # Compute eigenvalues
        eig1 = self.compute_eigenvalues(w1)
        eig2 = self.compute_eigenvalues(w2)
        
        # Wasserstein distance between eigenvalue distributions
        eig1_sorted = np.sort(eig1)
        eig2_sorted = np.sort(eig2)
        
        # Pad to same length
        max_len = max(len(eig1_sorted), len(eig2_sorted))
        if len(eig1_sorted) < max_len:
            eig1_sorted = np.pad(eig1_sorted, (0, max_len - len(eig1_sorted)), 'edge')
        if len(eig2_sorted) < max_len:
            eig2_sorted = np.pad(eig2_sorted, (0, max_len - len(eig2_sorted)), 'edge')
        
        wasserstein_dist = np.mean(np.abs(eig1_sorted - eig2_sorted))
        
        # KL divergence between spectral densities
        bins1, density1 = self.compute_spectral_density(eig1)
        bins2, density2 = self.compute_spectral_density(eig2)
        
        # Use common bins
        min_bin = min(bins1.min(), bins2.min())
        max_bin = max(bins1.max(), bins2.max())
        common_bins = np.linspace(min_bin, max_bin, 100)
        
        hist1, _ = np.histogram(eig1, bins=common_bins, density=True)
        hist2, _ = np.histogram(eig2, bins=common_bins, density=True)
        
        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        kl_div = stats.entropy(hist1, hist2)
        
        return {
            'wasserstein_distance': wasserstein_dist,
            'kl_divergence': kl_div,
            'spectral_radius_diff': abs(np.max(np.abs(eig1)) - np.max(np.abs(eig2))),
            'mean_eigenvalue_diff': abs(np.mean(eig1) - np.mean(eig2)),
        }
    
    def plot_spectral_density(self, weight_matrix: np.ndarray, 
                             layer_name: str = "",
                             compare_mp: bool = True,
                             save_path: Optional[str] = None):
        """Plot spectral density with optional Marchenko-Pastur overlay"""
        results = self.analyze_layer(weight_matrix, layer_name)
        
        if results['type'] == 'bias':
            print(f"Cannot plot spectral density for bias vector")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot empirical density
        ax.plot(results['spectral_bins'], results['spectral_density'], 
               'b-', linewidth=2, label='Empirical')
        
        if compare_mp:
            # Plot Marchenko-Pastur
            ax.plot(results['spectral_bins'], results['mp_density'],
                   'r--', linewidth=2, label='Marchenko-Pastur')
        
        ax.set_xlabel('Eigenvalue', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Spectral Density: {layer_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def compare_weight_stages_rmt(predicted: np.ndarray,
                               finetuned: np.ndarray,
                               ground_truth: np.ndarray) -> Dict:
    """
    Compare predicted, finetuned, and ground truth weights using RMT
    
    Returns:
        Comprehensive comparison metrics
    """
    analyzer = RandomMatrixAnalyzer()
    
    results = {
        'predicted': analyzer.analyze_all_layers(predicted),
        'finetuned': analyzer.analyze_all_layers(finetuned),
        'ground_truth': analyzer.analyze_all_layers(ground_truth),
        'comparisons': {}
    }
    
    # Compare predicted vs ground truth
    results['comparisons']['pred_vs_gt'] = analyzer.compare_weights(
        predicted, ground_truth
    )
    
    # Compare finetuned vs ground truth
    results['comparisons']['finetuned_vs_gt'] = analyzer.compare_weights(
        finetuned, ground_truth
    )
    
    # Compare predicted vs finetuned
    results['comparisons']['pred_vs_finetuned'] = analyzer.compare_weights(
        predicted, finetuned
    )
    
    return results


if __name__ == "__main__":
    print("Testing Random Matrix Theory Analysis")
    print("=" * 60)
    
    # Create test weight vector
    np.random.seed(42)
    weights = np.random.randn(2464) * 0.1
    
    # Analyze
    analyzer = RandomMatrixAnalyzer()
    results = analyzer.analyze_all_layers(weights)
    
    print("\nLayer Analysis:")
    for layer_name, layer_results in results.items():
        if layer_results['type'] == 'weight':
            print(f"\n{layer_name}:")
            print(f"  Shape: {layer_results['shape']}")
            print(f"  Spectral radius: {layer_results['spectral_radius']:.4f}")
            print(f"  Condition number: {layer_results['condition_number']:.4f}")
            print(f"  Effective rank: {layer_results['effective_rank']:.2f}")
            print(f"  KL div from MP: {layer_results['kl_divergence_from_mp']:.4f}")
    
    # Test comparison
    weights2 = np.random.randn(2464) * 0.1
    comparison = analyzer.compare_weights(weights, weights2)
    
    print("\n\nWeight Comparison:")
    for layer_name, metrics in comparison.items():
        print(f"\n{layer_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
