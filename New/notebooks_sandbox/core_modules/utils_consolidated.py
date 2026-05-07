"""
Consolidated utilities for weight-space analysis.
Contains all distance metrics, topology helpers, and common functions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance, entropy
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path
import json


# ============================================================================
# DISTANCE METRICS
# ============================================================================

class WeightDistanceMetrics:
    """
    Comprehensive distance metrics for CNN weight analysis.
    
    CNN Architecture (2464 total weights):
    - conv1.weight: 26 filters × 80 weights = 2080
    - conv1.bias: 26 weights
    - conv2.weight: 24 filters × 16 weights = 384
    - conv2.bias: 24 weights
    - fc layers: remaining ~50 weights
    """
    
    LAYER_BOUNDARIES = {
        'conv1_weights': (0, 2080),
        'conv1_bias': (2080, 2106),
        'conv2_weights': (2106, 2490),
        'conv2_bias': (2490, 2514),
        'fc_layer': (2514, 2464)
    }
    
    @staticmethod
    def frobenius_norm(w1: np.ndarray, w2: np.ndarray) -> float:
        """Frobenius norm of difference (L2 norm for vectors)."""
        return float(np.linalg.norm(w1 - w2, ord=2))
    
    @staticmethod
    def q_quantile_loss(w1: np.ndarray, w2: np.ndarray, q: float = 0.5) -> float:
        """Q-quantile loss (median absolute deviation when q=0.5)."""
        errors = w1 - w2
        return float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))
    
    @staticmethod
    def norm_of_jacobian(w1: np.ndarray, w2: np.ndarray) -> float:
        """Norm of Jacobian (gradient-based distance)."""
        return float(np.linalg.norm(w1 - w2, ord=2))
    
    @staticmethod
    def fisher_information_diff(w1: np.ndarray, w2: np.ndarray) -> float:
        """Fisher Information difference (variance-based)."""
        return float(np.abs(np.var(w1) - np.var(w2)))
    
    @staticmethod
    def contractive_loss(w1: np.ndarray, w2: np.ndarray) -> float:
        """Contractive loss based on local Lipschitz constant."""
        diff = w1 - w2
        return float(np.sqrt(np.sum(diff**2) / len(diff)))
    
    @staticmethod
    def wasserstein_distance_metric(w1: np.ndarray, w2: np.ndarray) -> float:
        """Wasserstein distance (Earth Mover's Distance)."""
        try:
            return float(wasserstein_distance(w1.flatten(), w2.flatten()))
        except:
            return np.nan
    
    @staticmethod
    def mape_loss(w1: np.ndarray, w2: np.ndarray, epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error."""
        return float(np.mean(np.abs((w1 - w2) / (np.abs(w2) + epsilon))) * 100)
    
    @staticmethod
    def lwln_loss(w1: np.ndarray, w2: np.ndarray, epsilon: float = 1e-8) -> float:
        """Layer-wise Loss Normalization."""
        std = np.std(w2)
        return float(np.mean(np.abs(w1 - w2) / (std + epsilon)))
    
    @staticmethod
    def jensen_shannon_divergence(w1: np.ndarray, w2: np.ndarray, epsilon: float = 1e-8) -> float:
        """Jensen-Shannon divergence for normalized distributions."""
        try:
            # Normalize to probability distributions
            p1 = np.abs(w1) / (np.sum(np.abs(w1)) + epsilon)
            p2 = np.abs(w2) / (np.sum(np.abs(w2)) + epsilon)
            m = 0.5 * (p1 + p2)
            return float(0.5 * entropy(p1, m) + 0.5 * entropy(p2, m))
        except:
            return np.nan
    
    @staticmethod
    def auto_regressive_loss(w1: np.ndarray, w2: np.ndarray) -> float:
        """Auto-regressive loss (difference in autocorrelation)."""
        try:
            acf1 = np.correlate(w1, w1, mode='valid')[0]
            acf2 = np.correlate(w2, w2, mode='valid')[0]
            return float(np.abs(acf1 - acf2))
        except:
            return np.nan
    
    @staticmethod
    def compute_all_full_distances(w1: np.ndarray, w2: np.ndarray) -> dict:
        """Compute all full-vector distance metrics with NaN/Inf handling."""
        cls = WeightDistanceMetrics
        
        # Replace NaN/Inf with zeros for distance computation
        w1_clean = np.nan_to_num(w1, nan=0.0, posinf=0.0, neginf=0.0)
        w2_clean = np.nan_to_num(w2, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            'euclidean': float(euclidean(w1_clean, w2_clean)),
            'manhattan': float(np.sum(np.abs(w1_clean - w2_clean))),
            'cosine': float(cosine(w1_clean, w2_clean)) if np.any(w1_clean) and np.any(w2_clean) else 0.0,
            'frobenius': cls.frobenius_norm(w1_clean, w2_clean),
            'q_quantile': cls.q_quantile_loss(w1_clean, w2_clean),
            'norm_jacobian': cls.norm_of_jacobian(w1_clean, w2_clean),
            'fisher_info': cls.fisher_information_diff(w1_clean, w2_clean),
            'contractive': cls.contractive_loss(w1_clean, w2_clean),
            'wasserstein': cls.wasserstein_distance_metric(w1_clean, w2_clean),
            'mape': cls.mape_loss(w1_clean, w2_clean),
            'lwln': cls.lwln_loss(w1_clean, w2_clean),
            'js_divergence': cls.jensen_shannon_divergence(w1_clean, w2_clean),
            'autoregressive': cls.auto_regressive_loss(w1_clean, w2_clean)
        }
    
    @staticmethod
    def compute_all_layerwise_distances(w1: np.ndarray, w2: np.ndarray) -> dict:
        """Compute all metrics layer-by-layer."""
        cls = WeightDistanceMetrics
        layerwise_metrics = {}
        
        for layer_name, (start, end) in cls.LAYER_BOUNDARIES.items():
            w1_layer = w1[start:end]
            w2_layer = w2[start:end]
            
            layerwise_metrics[f'{layer_name}_euclidean'] = float(euclidean(w1_layer, w2_layer))
            layerwise_metrics[f'{layer_name}_manhattan'] = float(np.sum(np.abs(w1_layer - w2_layer)))
            layerwise_metrics[f'{layer_name}_cosine'] = float(cosine(w1_layer, w2_layer))
            layerwise_metrics[f'{layer_name}_frobenius'] = cls.frobenius_norm(w1_layer, w2_layer)
            layerwise_metrics[f'{layer_name}_mape'] = cls.mape_loss(w1_layer, w2_layer)
        
        return layerwise_metrics
    
    @classmethod
    def compute_layerwise_distances(cls, w1: np.ndarray, w2: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute distance metrics layer-by-layer."""
        w1 = w1.flatten()
        w2 = w2.flatten()
        
        layerwise = {}
        for layer_name, (start, end) in cls.LAYER_BOUNDARIES.items():
            if end > len(w1):
                end = len(w1)
            if start >= end:
                continue
                
            l1 = w1[start:end]
            l2 = w2[start:end]
            
            layerwise[layer_name] = {
                'euclidean': float(np.linalg.norm(l1 - l2)),
                'manhattan': float(np.sum(np.abs(l1 - l2))),
                'cosine': float(cosine(l1, l2)) if len(l1) > 1 else np.nan,
                'relative_diff': float(np.linalg.norm(l1 - l2) / (np.linalg.norm(l2) + 1e-8)),
                'mean_abs_diff': float(np.mean(np.abs(l1 - l2)))
            }
        
        return layerwise
    
    @classmethod
    def compute_all_metrics_to_csv(cls, w1: np.ndarray, w2: np.ndarray, 
                                   output_path: Path, metadata: Optional[Dict] = None):
        """Compute all metrics and save to CSV."""
        full_metrics = cls.compute_all_full_distances(w1, w2)
        layerwise_metrics = cls.compute_layerwise_distances(w1, w2)
        
        # Create DataFrame for full metrics
        full_df = pd.DataFrame([full_metrics])
        if metadata:
            for key, value in metadata.items():
                full_df[key] = value
        
        # Create DataFrame for layerwise metrics
        layerwise_rows = []
        for layer_name, metrics in layerwise_metrics.items():
            row = {'layer': layer_name, **metrics}
            if metadata:
                row.update(metadata)
            layerwise_rows.append(row)
        layerwise_df = pd.DataFrame(layerwise_rows)
        
        # Save both
        output_path = Path(output_path)
        full_df.to_csv(output_path.parent / f"{output_path.stem}_full.csv", index=False)
        layerwise_df.to_csv(output_path.parent / f"{output_path.stem}_layerwise.csv", index=False)
        
        return full_df, layerwise_df


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def detect_model_dimensions(state_dict: Dict) -> Dict[str, int]:
    """
    Auto-detect model dimensions from checkpoint OrderedDict.
    
    Returns:
        Dictionary with d_model, N (layers), heads, d_ff, neck
    """
    config = {}
    
    # Detect d_model from embedding layer
    for key in state_dict.keys():
        if 'embed' in key.lower() and 'weight' in key:
            config['d_model'] = state_dict[key].shape[-1]
            break
    
    # Count encoder layers
    encoder_layers = set()
    for key in state_dict.keys():
        if 'enc' in key.lower() and 'layers' in key:
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        encoder_layers.add(int(parts[i + 1]))
                    except:
                        pass
    config['N'] = len(encoder_layers) if encoder_layers else 2
    
    # Detect attention heads
    for key in state_dict.keys():
        if 'attn' in key and 'q_linear' in key and 'weight' in key:
            d_model = state_dict[key].shape[0]
            # Assume heads divide d_model evenly
            for h in [1, 2, 4, 8, 12, 16]:
                if d_model % h == 0:
                    config['heads'] = h
                    break
            break
    
    # Detect d_ff from feed-forward layer
    for key in state_dict.keys():
        if 'ff' in key and 'linear_1' in key and 'weight' in key:
            config['d_ff'] = state_dict[key].shape[0]
            break
    
    # Detect neck dimension
    for key in state_dict.keys():
        if 'vec2neck' in key and 'weight' in key:
            config['neck'] = state_dict[key].shape[0]
            break
    
    # Set defaults if not found
    config.setdefault('d_model', 128)
    config.setdefault('N', 3)
    config.setdefault('heads', 4)
    config.setdefault('d_ff', 512)
    config.setdefault('neck', 64)
    config['max_seq_len'] = 50  # Fixed for this architecture
    config['dropout'] = 0.1
    
    return config


def load_checkpoint_auto(checkpoint_path: Path, device: str = 'cpu'):
    """
    Load checkpoint with auto-detected dimensions.
    
    Returns:
        model, config, metadata
    """
    from Double_input_transformer import TransformerAE
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Auto-detect config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if hasattr(config, '__dict__'):
            config = vars(config)
    else:
        config = detect_model_dimensions(checkpoint['model_state_dict'])
    
    # Create model
    model = TransformerAE(
        max_seq_len=config.get('max_seq_len', 50),
        N=config.get('N', 3),
        heads=config.get('heads', 4),
        d_model=config.get('d_model', 128),
        d_ff=config.get('d_ff', 512),
        neck=config.get('neck', 64),
        dropout=config.get('dropout', 0.1)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', -1),
        'train_loss': checkpoint.get('train_loss', np.nan),
        'val_loss': checkpoint.get('val_loss', np.nan),
        **config
    }
    
    return model, config, metadata


# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_merged_zoo(csv_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """Load Merged zoo.csv with optional row limit."""
    df = pd.read_csv(csv_path)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    return df


def extract_weights_from_zoo(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extract weight columns from zoo DataFrame.
    
    Returns:
        weights array (n_samples, 2464), weight column names
    """
    weight_cols = [col for col in df.columns if col.startswith('weight ')]
    weights = df[weight_cols].values.astype(np.float32)
    
    # Ensure exactly 2464 features
    if weights.shape[1] > 2464:
        weights = weights[:, :2464]
        weight_cols = weight_cols[:2464]
    elif weights.shape[1] < 2464:
        padding = np.zeros((weights.shape[0], 2464 - weights.shape[1]), dtype=np.float32)
        weights = np.concatenate([weights, padding], axis=1)
    
    return weights, weight_cols


def create_weight_pairs(weights: np.ndarray, overlap: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create pairs of weights for training.
    
    Args:
        weights: Weight matrix (n_samples, 2464)
        overlap: Overlap parameter (0, 1, or 2)
    
    Returns:
        x1, x2, y (target is average)
    """
    n_samples = len(weights)
    n_pairs = n_samples // 2
    
    indices = np.random.permutation(n_samples)
    idx1 = indices[:n_pairs]
    idx2 = indices[n_pairs:2*n_pairs]
    
    x1 = weights[idx1]
    x2 = weights[idx2]
    y = (x1 + x2) / 2  # Average as target
    
    return x1, x2, y


# ============================================================================
# TOPOLOGY UTILITIES
# ============================================================================

def safe_mapper_analysis(X: np.ndarray, n_intervals: int = 10, overlap: float = 0.3):
    """
    Safe Mapper algorithm with error handling.
    
    Returns:
        Dictionary with nodes, edges, or error message
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        
        if len(X) < 10:
            return {'error': 'insufficient_samples', 'n_samples': len(X)}
        
        # Compute lens (PCA projection)
        pca = PCA(n_components=1)
        lens = pca.fit_transform(X).flatten()
        
        # Create cover
        lens_min, lens_max = lens.min(), lens.max()
        if lens_min == lens_max:
            return {'error': 'constant_lens'}
        
        interval_length = (lens_max - lens_min) / (n_intervals * (1 - overlap))
        overlap_length = interval_length * overlap
        
        nodes = []
        node_id = 0
        
        for i in range(n_intervals):
            interval_start = lens_min + i * (interval_length - overlap_length)
            interval_end = interval_start + interval_length
            
            mask = (lens >= interval_start) & (lens <= interval_end)
            interval_points = np.where(mask)[0]
            
            if len(interval_points) < 3:
                continue
            
            # Cluster
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(X[interval_points])
            
            for cluster_id in np.unique(labels):
                if cluster_id == -1:
                    continue
                cluster_points = interval_points[labels == cluster_id]
                if len(cluster_points) >= 2:
                    nodes.append({
                        'id': node_id,
                        'size': len(cluster_points),
                        'interval': i,
                        'cluster': int(cluster_id)
                    })
                    node_id += 1
        
        return {'nodes': nodes, 'n_nodes': len(nodes), 'n_intervals': n_intervals}
        
    except Exception as e:
        return {'error': str(e)}


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def save_metrics_csv(metrics_dict: Dict, output_path: Path, metadata: Optional[Dict] = None):
    """Save metrics dictionary to CSV."""
    df = pd.DataFrame([metrics_dict])
    if metadata:
        for key, value in metadata.items():
            df[key] = value
    df.to_csv(output_path, index=False)
    return df


def append_metrics_csv(metrics_dict: Dict, output_path: Path, metadata: Optional[Dict] = None):
    """Append metrics to existing CSV or create new one."""
    row = {**metrics_dict}
    if metadata:
        row.update(metadata)
    
    df_new = pd.DataFrame([row])
    
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv(output_path, index=False)
    return df
