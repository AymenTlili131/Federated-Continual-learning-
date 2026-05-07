"""
Layer-wise weight normalization for CNN weight vectors

Handles the fact that CNN weights (small, centered near 0) and biases (large)
have very different distributions. Normalizes per layer to preserve information.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Layer boundaries for 2464-dimensional CNN weight vector
LAYER_BOUNDARIES = {
    'conv1_weight': (0, 200),      # [8, 1, 5, 5] = 200
    'conv1_bias': (200, 208),      # [8] = 8
    'conv2_weight': (208, 1408),   # [6, 8, 5, 5] = 1200
    'conv2_bias': (1408, 1414),    # [6] = 6
    'conv3_weight': (1414, 1510),  # [4, 6, 2, 2] = 96
    'conv3_bias': (1510, 1514),    # [4] = 4
    'fc1_weight': (1514, 2234),    # [20, 36] = 720
    'fc1_bias': (2234, 2254),      # [20] = 20
    'fc2_weight': (2254, 2454),    # [10, 20] = 200
    'fc2_bias': (2454, 2464),      # [10] = 10
}


class LayerWiseNormalizer:
    """
    Normalize CNN weights layer-by-layer
    
    Preserves layer-specific distributions while enabling transformer training
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        self.method = method
        self.scalers: Dict[str, object] = {}
        self.fitted = False
        
    def fit(self, weight_vectors: np.ndarray):
        """
        Fit scalers on weight vectors
        
        Args:
            weight_vectors: (N, 2464) array of weight vectors
        """
        print(f"Fitting layer-wise {self.method} scalers...")
        
        for layer_name, (start, end) in LAYER_BOUNDARIES.items():
            layer_data = weight_vectors[:, start:end]
            
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            scaler.fit(layer_data)
            self.scalers[layer_name] = scaler
            
            # Print statistics
            mean = layer_data.mean()
            std = layer_data.std()
            min_val = layer_data.min()
            max_val = layer_data.max()
            
            print(f"  {layer_name:15s} [{start:4d}:{end:4d}] "
                  f"mean={mean:8.4f} std={std:7.4f} "
                  f"min={min_val:8.4f} max={max_val:8.4f}")
        
        self.fitted = True
        print("✓ Scalers fitted")
        
    def transform(self, weight_vectors: np.ndarray) -> np.ndarray:
        """
        Transform weight vectors using fitted scalers
        
        Args:
            weight_vectors: (N, 2464) array
            
        Returns:
            Normalized weight vectors (N, 2464)
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        normalized = np.zeros_like(weight_vectors)
        
        for layer_name, (start, end) in LAYER_BOUNDARIES.items():
            layer_data = weight_vectors[:, start:end]
            normalized[:, start:end] = self.scalers[layer_name].transform(layer_data)
        
        return normalized
    
    def inverse_transform(self, normalized_vectors: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized vectors back to original scale
        
        Args:
            normalized_vectors: (N, 2464) normalized array
            
        Returns:
            Original scale weight vectors (N, 2464)
        """
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        original = np.zeros_like(normalized_vectors)
        
        for layer_name, (start, end) in LAYER_BOUNDARIES.items():
            layer_data = normalized_vectors[:, start:end]
            original[:, start:end] = self.scalers[layer_name].inverse_transform(layer_data)
        
        return original
    
    def fit_transform(self, weight_vectors: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(weight_vectors)
        return self.transform(weight_vectors)
    
    def save(self, filepath: str):
        """Save fitted scalers to disk"""
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted normalizer")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'method': self.method,
            'scalers': self.scalers,
            'fitted': self.fitted,
            'layer_boundaries': LAYER_BOUNDARIES
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✓ Normalizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LayerWiseNormalizer':
        """Load fitted scalers from disk"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        normalizer = cls(method=save_data['method'])
        normalizer.scalers = save_data['scalers']
        normalizer.fitted = save_data['fitted']
        
        print(f"✓ Normalizer loaded from {filepath}")
        return normalizer
    
    def get_statistics(self) -> Dict:
        """Get normalization statistics for each layer"""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted")
        
        stats = {}
        for layer_name, scaler in self.scalers.items():
            if self.method == 'standard':
                stats[layer_name] = {
                    'mean': scaler.mean_.tolist(),
                    'std': scaler.scale_.tolist(),
                    'var': scaler.var_.tolist()
                }
            elif self.method == 'minmax':
                stats[layer_name] = {
                    'min': scaler.data_min_.tolist(),
                    'max': scaler.data_max_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
        
        return stats


def analyze_weight_distributions(weight_vectors: np.ndarray, title: str = "Weight Distribution Analysis"):
    """
    Analyze and print weight distribution statistics per layer
    
    Useful for understanding if normalization is needed
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Total vectors: {len(weight_vectors)}")
    print(f"\n{'Layer':<15} {'Range':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    
    for layer_name, (start, end) in LAYER_BOUNDARIES.items():
        layer_data = weight_vectors[:, start:end]
        
        mean = layer_data.mean()
        std = layer_data.std()
        min_val = layer_data.min()
        max_val = layer_data.max()
        range_str = f"[{start}:{end}]"
        
        print(f"{layer_name:<15} {range_str:<12} {mean:>9.4f} {std:>9.4f} "
              f"{min_val:>9.4f} {max_val:>9.4f}")
    
    # Overall statistics
    print("-" * 70)
    print(f"{'OVERALL':<15} {'[0:2464]':<12} {weight_vectors.mean():>9.4f} "
          f"{weight_vectors.std():>9.4f} {weight_vectors.min():>9.4f} "
          f"{weight_vectors.max():>9.4f}")
    print("=" * 70)


def compare_normalization_methods(weight_vectors: np.ndarray, sample_size: int = 1000):
    """
    Compare different normalization strategies
    
    Args:
        weight_vectors: (N, 2464) weight array
        sample_size: Number of samples to use for comparison
    """
    if len(weight_vectors) > sample_size:
        indices = np.random.choice(len(weight_vectors), sample_size, replace=False)
        sample = weight_vectors[indices]
    else:
        sample = weight_vectors
    
    print("\n" + "="*70)
    print("NORMALIZATION METHOD COMPARISON")
    print("="*70)
    
    # Original
    print("\n1. ORIGINAL (No normalization)")
    analyze_weight_distributions(sample, "Original Distribution")
    
    # Global StandardScaler
    print("\n2. GLOBAL STANDARD SCALER")
    global_scaler = StandardScaler()
    global_normalized = global_scaler.fit_transform(sample)
    analyze_weight_distributions(global_normalized, "Global Standard Normalization")
    
    # Layer-wise StandardScaler
    print("\n3. LAYER-WISE STANDARD SCALER")
    layerwise_std = LayerWiseNormalizer(method='standard')
    layerwise_std_normalized = layerwise_std.fit_transform(sample)
    analyze_weight_distributions(layerwise_std_normalized, "Layer-wise Standard Normalization")
    
    # Layer-wise MinMaxScaler
    print("\n4. LAYER-WISE MINMAX SCALER")
    layerwise_mm = LayerWiseNormalizer(method='minmax')
    layerwise_mm_normalized = layerwise_mm.fit_transform(sample)
    analyze_weight_distributions(layerwise_mm_normalized, "Layer-wise MinMax Normalization")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("Layer-wise StandardScaler is recommended because:")
    print("  1. Preserves layer-specific distributions")
    print("  2. Handles different scales of weights vs biases")
    print("  3. Zero-centered output works well with transformers")
    print("  4. Can be inverted for CNN reconstruction")
    print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Layer-wise Weight Normalization Module")
    print("="*70)
    print("\nLayer boundaries:")
    for layer_name, (start, end) in LAYER_BOUNDARIES.items():
        size = end - start
        print(f"  {layer_name:15s} [{start:4d}:{end:4d}] = {size:4d} parameters")
    print(f"\nTotal: 2464 parameters")
