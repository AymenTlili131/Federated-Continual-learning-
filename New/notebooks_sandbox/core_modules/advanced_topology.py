"""
Advanced Topological Analysis

Features:
- Persistent homology with persistence images and landscapes
- Mapper algorithm
- Gromov-Wasserstein distance
- Betti numbers and persistence diagrams
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional

# Topology imports
try:
    from ripser import ripser
    from persim import plot_diagrams, PersistenceImager, persistence_images
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("ripser/persim not available")

try:
    import kmapper as km
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    KMAPPER_AVAILABLE = True
except ImportError:
    KMAPPER_AVAILABLE = False
    warnings.warn("kmapper not available")

try:
    import ot  # Python Optimal Transport
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    warnings.warn("POT (Python Optimal Transport) not available for Gromov-Wasserstein")


# ============================================================================
# PERSISTENT HOMOLOGY WITH IMAGES AND LANDSCAPES
# ============================================================================

def compute_persistent_homology_full(representations, max_dim=2, subsample=1000):
    """
    Compute comprehensive persistent homology analysis
    
    Returns:
        - Persistence diagrams
        - Betti numbers
        - Persistence images
        - Persistence landscapes
    """
    if not RIPSER_AVAILABLE:
        return None
    
    try:
        # Subsample if needed
        if len(representations) > subsample:
            indices = np.random.choice(len(representations), subsample, replace=False)
            representations = representations[indices]
        
        # Compute persistence diagrams
        result = ripser(representations, maxdim=max_dim)
        diagrams = result['dgms']
        
        # Betti numbers
        betti_numbers = {}
        for dim in range(max_dim + 1):
            if dim < len(diagrams):
                dgm = diagrams[dim]
                finite_intervals = dgm[dgm[:, 1] < np.inf]
                betti_numbers[f'betti_{dim}'] = len(finite_intervals)
        
        # Persistence images
        pers_images = {}
        try:
            pimgr = PersistenceImager(pixel_size=0.1)
            for dim in range(max_dim + 1):
                if dim < len(diagrams):
                    dgm = diagrams[dim]
                    finite_dgm = dgm[dgm[:, 1] < np.inf]
                    if len(finite_dgm) > 0:
                        pimg = pimgr.transform(finite_dgm)
                        pers_images[f'dim_{dim}'] = pimg
        except Exception as e:
            warnings.warn(f"Persistence images failed: {e}")
        
        # Persistence landscapes (simplified)
        landscapes = {}
        for dim in range(max_dim + 1):
            if dim < len(diagrams):
                dgm = diagrams[dim]
                finite_dgm = dgm[dgm[:, 1] < np.inf]
                if len(finite_dgm) > 0:
                    # Compute landscape function (simplified version)
                    births = finite_dgm[:, 0]
                    deaths = finite_dgm[:, 1]
                    lifetimes = deaths - births
                    landscapes[f'dim_{dim}'] = {
                        'mean_lifetime': np.mean(lifetimes),
                        'max_lifetime': np.max(lifetimes),
                        'total_persistence': np.sum(lifetimes)
                    }
        
        return {
            'diagrams': diagrams,
            'betti_numbers': betti_numbers,
            'persistence_images': pers_images,
            'landscapes': landscapes
        }
    
    except Exception as e:
        warnings.warn(f"Persistent homology computation failed: {e}")
        return None


# ============================================================================
# MAPPER ALGORITHM
# ============================================================================

def compute_mapper(representations, n_cubes=10, overlap=0.3, subsample=1000, n_jobs=1):
    """
    Compute Mapper algorithm for topological data analysis
    
    Args:
        representations: Data (n_samples, n_features)
        n_cubes: Number of hypercubes per dimension
        overlap: Overlap percentage between cubes
        subsample: Maximum samples to use
        n_jobs: Number of CPU workers for DBSCAN (1=sequential, -1=all cores)
    
    Returns:
        Dictionary with Mapper graph and statistics
    """
    if not KMAPPER_AVAILABLE:
        return None
    
    try:
        # Subsample if needed
        if len(representations) > subsample:
            indices = np.random.choice(len(representations), subsample, replace=False)
            representations = representations[indices]
        
        # Initialize Mapper
        mapper = km.KeplerMapper(verbose=0)
        
        # Project to 2D using PCA
        pca = PCA(n_components=2)
        lens = pca.fit_transform(representations)
        
        # Create Mapper graph with n_jobs for DBSCAN
        graph = mapper.map(
            lens,
            representations,
            clusterer=DBSCAN(eps=0.5, min_samples=3, n_jobs=n_jobs),
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
        )
        
        # Extract statistics
        stats = {
            'n_nodes': len(graph['nodes']),
            'n_edges': len(graph['links']),
            'node_sizes': [len(members) for members in graph['nodes'].values()],
        }
        
        stats['mean_node_size'] = np.mean(stats['node_sizes'])
        stats['max_node_size'] = np.max(stats['node_sizes'])
        stats['graph_density'] = 2 * stats['n_edges'] / (stats['n_nodes'] * (stats['n_nodes'] - 1)) if stats['n_nodes'] > 1 else 0
        
        return {
            'graph': graph,
            'stats': stats,
            'lens': lens
        }
    
    except Exception as e:
        warnings.warn(f"Mapper computation failed: {e}")
        return None


# ============================================================================
# GROMOV-WASSERSTEIN DISTANCE
# ============================================================================

def compute_gromov_wasserstein(X1, X2, subsample=500):
    """
    Compute Gromov-Wasserstein distance between two point clouds
    
    Args:
        X1, X2: Point clouds (n_samples, n_features)
        subsample: Maximum samples to use
    
    Returns:
        Gromov-Wasserstein distance and transport plan
    """
    if not OT_AVAILABLE:
        return None
    
    try:
        # Subsample if needed
        if len(X1) > subsample:
            idx1 = np.random.choice(len(X1), subsample, replace=False)
            X1 = X1[idx1]
        if len(X2) > subsample:
            idx2 = np.random.choice(len(X2), subsample, replace=False)
            X2 = X2[idx2]
        
        # Compute cost matrices (pairwise distances)
        C1 = ot.dist(X1, X1, metric='euclidean')
        C2 = ot.dist(X2, X2, metric='euclidean')
        
        # Uniform distributions
        p = ot.unif(len(X1))
        q = ot.unif(len(X2))
        
        # Compute Gromov-Wasserstein distance
        gw_dist, log = ot.gromov.gromov_wasserstein2(
            C1, C2, p, q, 'square_loss', verbose=False, log=True
        )
        
        return {
            'distance': float(gw_dist),
            'transport_plan': log.get('T', None),
            'n_samples_1': len(X1),
            'n_samples_2': len(X2)
        }
    
    except Exception as e:
        warnings.warn(f"Gromov-Wasserstein computation failed: {e}")
        return None


# ============================================================================
# COMPREHENSIVE TOPOLOGY ANALYSIS
# ============================================================================

def compute_comprehensive_topology(representations, epoch=0, n_jobs=1):
    """
    Compute all topological analyses
    
    Args:
        representations: Data representations (n_samples, n_features)
        epoch: Current epoch number
        n_jobs: Number of CPU workers (1=sequential, -1=all cores)
                Note: Currently only affects DBSCAN in Mapper
    
    Returns:
        Dictionary with all topology results
    """
    results = {'epoch': epoch}
    
    # Persistent homology
    print(f"    Computing persistent homology...")
    ph_results = compute_persistent_homology_full(representations, max_dim=2)
    if ph_results:
        results['persistent_homology'] = ph_results
        results.update(ph_results['betti_numbers'])
        
        # Add landscape statistics
        for dim, landscape in ph_results['landscapes'].items():
            for key, val in landscape.items():
                results[f'landscape_{dim}_{key}'] = val
    
    # Mapper algorithm (with n_jobs for DBSCAN)
    print(f"    Computing Mapper (n_jobs={n_jobs})...")
    mapper_results = compute_mapper(representations, n_cubes=10, overlap=0.3, n_jobs=n_jobs)
    if mapper_results:
        results['mapper'] = mapper_results
        results.update({f'mapper_{k}': v for k, v in mapper_results['stats'].items()})
    
    # Gromov-Wasserstein (compare to random baseline)
    print(f"    Computing Gromov-Wasserstein...")
    random_baseline = np.random.randn(*representations.shape)
    gw_results = compute_gromov_wasserstein(representations, random_baseline)
    if gw_results:
        results['gromov_wasserstein'] = gw_results
        results['gw_distance'] = gw_results['distance']
    
    return results


# ============================================================================
# SAVE TOPOLOGY RESULTS
# ============================================================================

def save_topology_results(results, save_dir, epoch):
    """Save topology results to files"""
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numerical results as JSON
    json_results = {
        'epoch': epoch,
        'betti_numbers': {},
        'landscapes': {},
        'mapper_stats': {},
        'gw_distance': None
    }
    
    for key, val in results.items():
        if key.startswith('betti_'):
            json_results['betti_numbers'][key] = int(val)
        elif key.startswith('landscape_'):
            json_results['landscapes'][key] = float(val)
        elif key.startswith('mapper_'):
            json_results['mapper_stats'][key] = float(val) if isinstance(val, (int, float, np.number)) else val
        elif key == 'gw_distance':
            json_results['gw_distance'] = float(val)
    
    with open(save_dir / f"topology_epoch_{epoch:04d}.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save persistence images if available
    if 'persistent_homology' in results and 'persistence_images' in results['persistent_homology']:
        pers_images = results['persistent_homology']['persistence_images']
        for dim, img in pers_images.items():
            if img is not None:
                plt.figure(figsize=(6, 6))
                plt.imshow(img, cmap='viridis', origin='lower')
                plt.colorbar(label='Persistence')
                plt.title(f'Persistence Image - Dimension {dim} - Epoch {epoch}')
                plt.xlabel('Birth')
                plt.ylabel('Persistence')
                plt.savefig(save_dir / f"pers_image_{dim}_epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
                plt.close()
    
    return json_results
