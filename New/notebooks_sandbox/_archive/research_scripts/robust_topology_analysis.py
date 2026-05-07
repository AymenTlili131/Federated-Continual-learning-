"""
Robust topological analysis with comprehensive error handling.
Includes Mapper algorithm, persistent homology, and clustering with fallbacks.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Optional imports with fallbacks
try:
    from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Clustering features disabled.")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    DIMRED_AVAILABLE = True
except ImportError:
    DIMRED_AVAILABLE = False
    warnings.warn("Dimensionality reduction libraries not available.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Graph features disabled.")


class RobustMapper:
    """
    Robust implementation of Mapper algorithm with comprehensive error handling.
    """
    
    def __init__(self, 
                 n_intervals: int = 10,
                 overlap: float = 0.3,
                 min_cluster_size: int = 2,
                 clustering_method: str = 'dbscan'):
        """
        Initialize Mapper algorithm.
        
        Args:
            n_intervals: Number of intervals for cover
            overlap: Overlap percentage between intervals
            min_cluster_size: Minimum points required for a cluster
            clustering_method: 'dbscan', 'kmeans', or 'hierarchical'
        """
        self.n_intervals = n_intervals
        self.overlap = overlap
        self.min_cluster_size = min_cluster_size
        self.clustering_method = clustering_method
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for Mapper algorithm")
    
    def _safe_clustering(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform clustering with error handling and fallbacks.
        
        Returns:
            Cluster labels, or array of -1 if clustering fails
        """
        try:
            n_samples = len(X)
            
            # Check if we have enough distinct points
            if n_samples < self.min_cluster_size:
                warnings.warn(f"Only {n_samples} samples, less than min_cluster_size={self.min_cluster_size}")
                return np.zeros(n_samples, dtype=int)
            
            # Check for duplicate points
            unique_points = np.unique(X, axis=0)
            if len(unique_points) < self.min_cluster_size:
                warnings.warn(f"Only {len(unique_points)} unique points, insufficient for clustering")
                return np.zeros(n_samples, dtype=int)
            
            # Try primary clustering method
            if self.clustering_method == 'dbscan':
                clusterer = DBSCAN(eps=kwargs.get('eps', 0.5), 
                                  min_samples=self.min_cluster_size)
                labels = clusterer.fit_predict(X)
                
                # If DBSCAN finds no clusters, fall back to KMeans
                if len(np.unique(labels)) <= 1 or np.all(labels == -1):
                    warnings.warn("DBSCAN found no clusters, falling back to KMeans")
                    n_clusters = min(3, n_samples // self.min_cluster_size)
                    if n_clusters < 2:
                        return np.zeros(n_samples, dtype=int)
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(X)
                
            elif self.clustering_method == 'kmeans':
                n_clusters = min(kwargs.get('n_clusters', 3), n_samples // self.min_cluster_size)
                if n_clusters < 2:
                    return np.zeros(n_samples, dtype=int)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(X)
                
            elif self.clustering_method == 'hierarchical':
                n_clusters = min(kwargs.get('n_clusters', 3), n_samples // self.min_cluster_size)
                if n_clusters < 2:
                    return np.zeros(n_samples, dtype=int)
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clusterer.fit_predict(X)
                
            else:
                warnings.warn(f"Unknown clustering method: {self.clustering_method}, using KMeans")
                n_clusters = min(3, n_samples // self.min_cluster_size)
                if n_clusters < 2:
                    return np.zeros(n_samples, dtype=int)
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(X)
            
            return labels
            
        except Exception as e:
            warnings.warn(f"Clustering failed: {e}. Returning single cluster.")
            return np.zeros(len(X), dtype=int)
    
    def fit(self, X: np.ndarray, lens: Optional[np.ndarray] = None) -> Dict:
        """
        Fit Mapper algorithm with robust error handling.
        
        Args:
            X: Data matrix (n_samples, n_features)
            lens: Lens function values (n_samples,). If None, uses PCA projection.
        
        Returns:
            Dictionary with graph structure and metadata
        """
        try:
            n_samples = X.shape[0]
            
            # Validate input
            if n_samples < self.min_cluster_size:
                warnings.warn(f"Dataset too small ({n_samples} samples) for Mapper")
                return {'nodes': [], 'edges': [], 'error': 'insufficient_samples'}
            
            # Compute lens function if not provided
            if lens is None:
                try:
                    if DIMRED_AVAILABLE:
                        pca = PCA(n_components=1)
                        lens = pca.fit_transform(X).flatten()
                    else:
                        # Fallback: use first principal component manually
                        X_centered = X - X.mean(axis=0)
                        cov = np.cov(X_centered.T)
                        eigenvalues, eigenvectors = np.linalg.eig(cov)
                        lens = X_centered @ eigenvectors[:, 0].real
                except Exception as e:
                    warnings.warn(f"Lens computation failed: {e}. Using mean as lens.")
                    lens = X.mean(axis=1)
            
            # Create cover intervals
            lens_min, lens_max = lens.min(), lens.max()
            if lens_min == lens_max:
                warnings.warn("Lens function is constant")
                return {'nodes': [{'id': 0, 'size': n_samples}], 'edges': [], 'error': 'constant_lens'}
            
            interval_length = (lens_max - lens_min) / (self.n_intervals * (1 - self.overlap))
            overlap_length = interval_length * self.overlap
            
            nodes = []
            node_id = 0
            point_to_nodes = {i: [] for i in range(n_samples)}
            
            # Process each interval
            for i in range(self.n_intervals):
                interval_start = lens_min + i * (interval_length - overlap_length)
                interval_end = interval_start + interval_length
                
                # Get points in this interval
                mask = (lens >= interval_start) & (lens <= interval_end)
                interval_points = np.where(mask)[0]
                
                if len(interval_points) < self.min_cluster_size:
                    continue
                
                # Cluster points in interval
                X_interval = X[interval_points]
                labels = self._safe_clustering(X_interval)
                
                # Create nodes for each cluster
                for cluster_id in np.unique(labels):
                    if cluster_id == -1:  # Noise points in DBSCAN
                        continue
                    
                    cluster_mask = labels == cluster_id
                    cluster_points = interval_points[cluster_mask]
                    
                    if len(cluster_points) >= self.min_cluster_size:
                        nodes.append({
                            'id': node_id,
                            'size': len(cluster_points),
                            'points': cluster_points.tolist(),
                            'interval': i,
                            'cluster': int(cluster_id)
                        })
                        
                        for point in cluster_points:
                            point_to_nodes[point].append(node_id)
                        
                        node_id += 1
            
            # Create edges between overlapping nodes
            edges = []
            for point, node_list in point_to_nodes.items():
                if len(node_list) > 1:
                    for i in range(len(node_list)):
                        for j in range(i+1, len(node_list)):
                            edge = tuple(sorted([node_list[i], node_list[j]]))
                            if edge not in edges:
                                edges.append(edge)
            
            return {
                'nodes': nodes,
                'edges': edges,
                'n_nodes': len(nodes),
                'n_edges': len(edges),
                'n_intervals': self.n_intervals,
                'overlap': self.overlap
            }
            
        except Exception as e:
            warnings.warn(f"Mapper algorithm failed: {e}")
            return {'nodes': [], 'edges': [], 'error': str(e)}


class RobustPersistentHomology:
    """
    Robust persistent homology computation with error handling.
    """
    
    def __init__(self, max_dimension: int = 2):
        """
        Initialize persistent homology analyzer.
        
        Args:
            max_dimension: Maximum homology dimension to compute
        """
        self.max_dimension = max_dimension
        
        # Try to import ripser
        try:
            import ripser
            from persim import plot_diagrams
            self.ripser = ripser
            self.plot_diagrams = plot_diagrams
            self.available = True
        except ImportError:
            warnings.warn("ripser/persim not available. Persistent homology disabled.")
            self.available = False
    
    def compute(self, X: np.ndarray, metric: str = 'euclidean') -> Dict:
        """
        Compute persistent homology with error handling.
        
        Args:
            X: Data matrix (n_samples, n_features)
            metric: Distance metric to use
        
        Returns:
            Dictionary with persistence diagrams and statistics
        """
        if not self.available:
            return {'error': 'ripser_not_available', 'diagrams': None}
        
        try:
            # Subsample if too many points
            n_samples = X.shape[0]
            if n_samples > 1000:
                warnings.warn(f"Subsampling {n_samples} points to 1000 for efficiency")
                indices = np.random.choice(n_samples, 1000, replace=False)
                X = X[indices]
            
            # Compute persistence diagrams
            result = self.ripser.ripser(X, maxdim=self.max_dimension, metric=metric)
            diagrams = result['dgms']
            
            # Compute statistics
            stats = {}
            for dim in range(min(self.max_dimension + 1, len(diagrams))):
                dgm = diagrams[dim]
                if len(dgm) > 0:
                    # Filter out infinite bars
                    finite_bars = dgm[np.isfinite(dgm).all(axis=1)]
                    if len(finite_bars) > 0:
                        lifetimes = finite_bars[:, 1] - finite_bars[:, 0]
                        stats[f'betti_{dim}'] = len(finite_bars)
                        stats[f'max_lifetime_{dim}'] = float(lifetimes.max())
                        stats[f'mean_lifetime_{dim}'] = float(lifetimes.mean())
                        stats[f'total_persistence_{dim}'] = float(lifetimes.sum())
                    else:
                        stats[f'betti_{dim}'] = 0
                else:
                    stats[f'betti_{dim}'] = 0
            
            return {
                'diagrams': diagrams,
                'stats': stats,
                'n_samples': X.shape[0],
                'max_dimension': self.max_dimension
            }
            
        except Exception as e:
            warnings.warn(f"Persistent homology computation failed: {e}")
            return {'error': str(e), 'diagrams': None}


def safe_compute_topology_metrics(X: np.ndarray) -> Dict:
    """
    Safely compute all topology metrics with comprehensive error handling.
    
    Args:
        X: Data matrix (n_samples, n_features)
    
    Returns:
        Dictionary with all computed metrics
    """
    results = {
        'mapper': None,
        'persistence': None,
        'errors': []
    }
    
    # Try Mapper
    try:
        mapper = RobustMapper(n_intervals=10, overlap=0.3, min_cluster_size=3)
        results['mapper'] = mapper.fit(X)
        if 'error' in results['mapper']:
            results['errors'].append(f"Mapper: {results['mapper']['error']}")
    except Exception as e:
        results['errors'].append(f"Mapper failed: {e}")
        results['mapper'] = {'error': str(e)}
    
    # Try Persistent Homology
    try:
        ph = RobustPersistentHomology(max_dimension=2)
        results['persistence'] = ph.compute(X)
        if 'error' in results['persistence']:
            results['errors'].append(f"Persistence: {results['persistence']['error']}")
    except Exception as e:
        results['errors'].append(f"Persistent homology failed: {e}")
        results['persistence'] = {'error': str(e)}
    
    return results
