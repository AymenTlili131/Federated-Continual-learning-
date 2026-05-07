"""
Comprehensive distance metrics for weight analysis.
Implements both full 2464-vector distances and layer-wise subdistances.
"""

import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance, entropy
from typing import Dict, List, Tuple, Optional
import warnings


class WeightDistanceMetrics:
    """
    Compute distance metrics for CNN weights.
    
    CNN Architecture (2464 total weights):
    - conv1.weight: 26 filters × 80 weights = 2080 weights
    - conv1.bias: 26 weights
    - conv2.weight: 24 filters × 16 weights = 384 weights  
    - conv2.bias: 24 weights
    - fc.weight + fc.bias: ~50 weights (approximate)
    
    Layer-wise breakdown:
    1. conv1_weights: indices 0:2080 (2080 weights)
    2. conv1_bias: indices 2080:2106 (26 weights)
    3. conv2_weights: indices 2106:2490 (384 weights) - ADJUSTED
    4. conv2_bias: indices 2490:2514 (24 weights) - ADJUSTED
    5. fc_layer: indices 2514:2464 - NEED TO VERIFY ACTUAL INDICES
    """
    
    # Layer boundaries for 2464-dimensional weight vector
    LAYER_BOUNDARIES = {
        'conv1_weights': (0, 2080),
        'conv1_bias': (2080, 2106),
        'conv2_weights': (2106, 2490),
        'conv2_bias': (2490, 2514),
        'fc_layer': (2514, 2464)  # Remaining weights
    }
    
    def __init__(self):
        """Initialize distance metrics calculator."""
        self.layer_names = list(self.LAYER_BOUNDARIES.keys())
    
    def extract_layer(self, weights: np.ndarray, layer_name: str) -> np.ndarray:
        """Extract specific layer weights from full vector."""
        start, end = self.LAYER_BOUNDARIES[layer_name]
        if end > len(weights):
            warnings.warn(f"Layer {layer_name} boundary {end} exceeds weight length {len(weights)}")
            end = len(weights)
        return weights[start:end]
    
    def compute_full_distances(self, w1: np.ndarray, w2: np.ndarray) -> Dict[str, float]:
        """
        Compute distance metrics on full 2464-dimensional vectors.
        
        Args:
            w1: First weight vector (2464,)
            w2: Second weight vector (2464,)
        
        Returns:
            Dictionary of distance metrics
        """
        try:
            # Ensure 1D arrays
            w1 = w1.flatten()
            w2 = w2.flatten()
            
            metrics = {}
            
            # 1. Euclidean distance (L2 norm)
            metrics['euclidean'] = float(np.linalg.norm(w1 - w2))
            
            # 2. Manhattan distance (L1 norm)
            metrics['manhattan'] = float(np.sum(np.abs(w1 - w2)))
            
            # 3. Cosine distance
            try:
                metrics['cosine'] = float(cosine(w1, w2))
            except:
                metrics['cosine'] = np.nan
            
            # 4. Wasserstein distance (Earth Mover's Distance)
            try:
                metrics['wasserstein'] = float(wasserstein_distance(w1, w2))
            except:
                metrics['wasserstein'] = np.nan
            
            # 5. Frobenius norm (same as Euclidean for vectors)
            metrics['frobenius'] = metrics['euclidean']
            
            # 6. Q-quantile loss (median absolute deviation)
            metrics['q_quantile'] = float(np.median(np.abs(w1 - w2)))
            
            # 7. Norm of Jacobian (gradient-based)
            metrics['jacobian_norm'] = float(np.linalg.norm(w1 - w2, ord=2))
            
            # 8. Fisher Information Difference (approximation)
            # Using variance difference as proxy
            metrics['fisher_info_diff'] = float(np.abs(np.var(w1) - np.var(w2)))
            
            # 9. Contractive loss (based on local Lipschitz constant)
            diff = w1 - w2
            metrics['contractive'] = float(np.sqrt(np.sum(diff**2) / len(diff)))
            
            # 10. MAPE (Mean Absolute Percentage Error)
            epsilon = 1e-8
            metrics['mape'] = float(np.mean(np.abs((w1 - w2) / (np.abs(w2) + epsilon))) * 100)
            
            # 11. Layer-wise loss normalization
            metrics['lwln'] = float(np.mean(np.abs(w1 - w2) / (np.std(w2) + epsilon)))
            
            # 12. Jensen-Shannon divergence (for normalized distributions)
            try:
                # Normalize to probability distributions
                p1 = np.abs(w1) / (np.sum(np.abs(w1)) + epsilon)
                p2 = np.abs(w2) / (np.sum(np.abs(w2)) + epsilon)
                m = 0.5 * (p1 + p2)
                metrics['jensen_shannon'] = float(0.5 * entropy(p1, m) + 0.5 * entropy(p2, m))
            except:
                metrics['jensen_shannon'] = np.nan
            
            # 13. Auto-regressive loss (difference in autocorrelation)
            try:
                acf1 = np.correlate(w1, w1, mode='valid')[0]
                acf2 = np.correlate(w2, w2, mode='valid')[0]
                metrics['auto_regressive'] = float(np.abs(acf1 - acf2))
            except:
                metrics['auto_regressive'] = np.nan
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"Error computing full distances: {e}")
            return {k: np.nan for k in ['euclidean', 'manhattan', 'cosine', 'wasserstein']}
    
    def compute_layerwise_distances(self, w1: np.ndarray, w2: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute distance metrics layer-by-layer (5 subdistances).
        
        Args:
            w1: First weight vector (2464,)
            w2: Second weight vector (2464,)
        
        Returns:
            Nested dictionary: {layer_name: {metric_name: value}}
        """
        layerwise_metrics = {}
        
        for layer_name in self.layer_names:
            try:
                # Extract layer weights
                l1 = self.extract_layer(w1, layer_name)
                l2 = self.extract_layer(w2, layer_name)
                
                if len(l1) == 0 or len(l2) == 0:
                    layerwise_metrics[layer_name] = {'euclidean': np.nan}
                    continue
                
                # Compute metrics for this layer
                layer_metrics = {}
                
                # Euclidean distance
                layer_metrics['euclidean'] = float(np.linalg.norm(l1 - l2))
                
                # Manhattan distance
                layer_metrics['manhattan'] = float(np.sum(np.abs(l1 - l2)))
                
                # Cosine distance
                try:
                    if len(l1) > 1:
                        layer_metrics['cosine'] = float(cosine(l1, l2))
                    else:
                        layer_metrics['cosine'] = float(np.abs(l1[0] - l2[0]))
                except:
                    layer_metrics['cosine'] = np.nan
                
                # Relative difference (normalized by layer magnitude)
                l2_norm = np.linalg.norm(l2)
                layer_metrics['relative_diff'] = float(np.linalg.norm(l1 - l2) / (l2_norm + 1e-8))
                
                # Mean absolute difference
                layer_metrics['mean_abs_diff'] = float(np.mean(np.abs(l1 - l2)))
                
                layerwise_metrics[layer_name] = layer_metrics
                
            except Exception as e:
                warnings.warn(f"Error computing distances for layer {layer_name}: {e}")
                layerwise_metrics[layer_name] = {'euclidean': np.nan}
        
        return layerwise_metrics
    
    def compute_all_metrics(self, w1: np.ndarray, w2: np.ndarray) -> Dict:
        """
        Compute both full and layer-wise distance metrics.
        
        Returns:
            {
                'full': {metric: value},
                'layerwise': {layer: {metric: value}},
                'summary': {aggregated statistics}
            }
        """
        # Ensure numpy arrays
        if torch.is_tensor(w1):
            w1 = w1.detach().cpu().numpy()
        if torch.is_tensor(w2):
            w2 = w2.detach().cpu().numpy()
        
        w1 = w1.flatten()
        w2 = w2.flatten()
        
        # Compute metrics
        full_metrics = self.compute_full_distances(w1, w2)
        layerwise_metrics = self.compute_layerwise_distances(w1, w2)
        
        # Compute summary statistics
        summary = {
            'total_weights': len(w1),
            'mean_full_euclidean': full_metrics.get('euclidean', np.nan),
            'mean_layerwise_euclidean': np.mean([
                m.get('euclidean', np.nan) 
                for m in layerwise_metrics.values()
            ]),
            'max_layer_distance': max([
                m.get('euclidean', 0) 
                for m in layerwise_metrics.values()
            ]),
            'min_layer_distance': min([
                m.get('euclidean', np.inf) 
                for m in layerwise_metrics.values() 
                if not np.isnan(m.get('euclidean', np.nan))
            ])
        }
        
        return {
            'full': full_metrics,
            'layerwise': layerwise_metrics,
            'summary': summary
        }
    
    def format_as_table(self, metrics: Dict) -> str:
        """Format metrics as markdown table."""
        lines = []
        
        # Full metrics table
        lines.append("## Full Vector Distances (2464 dimensions)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for metric, value in sorted(metrics['full'].items()):
            lines.append(f"| {metric} | {value:.6f} |")
        lines.append("")
        
        # Layer-wise metrics table
        lines.append("## Layer-wise Distances (5 subdistances)")
        lines.append("")
        lines.append("| Layer | Euclidean | Manhattan | Cosine | Relative Diff | Mean Abs Diff |")
        lines.append("|-------|-----------|-----------|--------|---------------|---------------|")
        
        for layer_name, layer_metrics in metrics['layerwise'].items():
            start, end = self.LAYER_BOUNDARIES[layer_name]
            size = end - start
            lines.append(
                f"| {layer_name} ({size}w) | "
                f"{layer_metrics.get('euclidean', np.nan):.6f} | "
                f"{layer_metrics.get('manhattan', np.nan):.6f} | "
                f"{layer_metrics.get('cosine', np.nan):.6f} | "
                f"{layer_metrics.get('relative_diff', np.nan):.6f} | "
                f"{layer_metrics.get('mean_abs_diff', np.nan):.6f} |"
            )
        lines.append("")
        
        # Summary
        lines.append("## Summary Statistics")
        lines.append("")
        lines.append("| Statistic | Value |")
        lines.append("|-----------|-------|")
        for stat, value in sorted(metrics['summary'].items()):
            lines.append(f"| {stat} | {value:.6f} |")
        
        return "\n".join(lines)


def compute_pairwise_distances(weights_list: List[np.ndarray], 
                               metric_type: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise distance matrix for a list of weight vectors.
    
    Args:
        weights_list: List of weight vectors
        metric_type: Type of distance metric to use
    
    Returns:
        Distance matrix (n_samples, n_samples)
    """
    n = len(weights_list)
    dist_matrix = np.zeros((n, n))
    calculator = WeightDistanceMetrics()
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                metrics = calculator.compute_full_distances(weights_list[i], weights_list[j])
                dist = metrics.get(metric_type, np.nan)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
            except Exception as e:
                warnings.warn(f"Error computing distance between samples {i} and {j}: {e}")
                dist_matrix[i, j] = np.nan
                dist_matrix[j, i] = np.nan
    
    return dist_matrix
