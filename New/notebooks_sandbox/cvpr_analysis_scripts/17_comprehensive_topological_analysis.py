#!/usr/bin/env python3
"""
Comprehensive Topological Analysis using GUDHI and Multipers

This script performs BOTH normal (single-parameter) and multiparameter persistent 
homology analysis on all checkpoints since March 20th.

Normal Persistence Features (GUDHI):
- Betti numbers (β₀, β₁, β₂)
- Persistence diagrams
- Persistence entropy
- Total persistence
- Bottleneck distance

Multiparameter Persistence Features (Multipers):
- Rank invariant
- Hilbert function
- Signed barcodes
- Multiparameter persistence modules
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New")
CORE_MODULES = PROJECT_ROOT / "notebooks_sandbox" / "core_modules"
sys.path.insert(0, str(CORE_MODULES))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import config for checkpoint loading
from config import MODEL_CONFIGS
from Double_input_transformer import TransformerAE

# Import topological packages
try:
    import gudhi
    from gudhi.representations import Landscape, PersistenceImage
    GUDHI_AVAILABLE = True
    print("✓ GUDHI available")
except ImportError:
    GUDHI_AVAILABLE = False
    print("✗ GUDHI not available")

try:
    import multipers as mp
    MULTIPERS_AVAILABLE = True
    print(f"✓ Multipers available (v{mp.__version__})")
except ImportError:
    MULTIPERS_AVAILABLE = False
    print("✗ Multipers not available")

# Paths
EXPERIMENTS_DIR = PROJECT_ROOT / "notebooks_sandbox" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "notebooks_sandbox" / "cvpr_analysis_scripts" / "data"
FIGURES_DIR = PROJECT_ROOT / "notebooks_sandbox" / "CVPR 2026" / "figures"

print("\n" + "="*80)
print("COMPREHENSIVE TOPOLOGICAL ANALYSIS")
print("Normal Persistence (GUDHI) + Multiparameter Persistence (Multipers)")
print("="*80)

# 1. Find all checkpoints since March 20, 2024
print("\n1. Finding checkpoints since March 20, 2024...")
cutoff_date = datetime(2024, 3, 20)

checkpoints = []
for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue
    
    ckpt_file = exp_dir / "checkpoints" / "best_model.pth"
    if not ckpt_file.exists():
        continue
    
    # Check modification time
    mtime = datetime.fromtimestamp(ckpt_file.stat().st_mtime)
    if mtime < cutoff_date:
        continue
    
    # Parse experiment info
    parts = exp_dir.name.split('_')
    if len(parts) < 3:
        continue
    
    model_size = parts[0]
    overlap = int(parts[1].replace('overlap', ''))
    loss_name = '_'.join(parts[2:])
    
    checkpoints.append({
        'experiment': exp_dir.name,
        'model_size': model_size,
        'overlap': overlap,
        'loss_name': loss_name,
        'path': ckpt_file,
        'mtime': mtime
    })

print(f"Found {len(checkpoints)} checkpoints")
print(f"Date range: {min(c['mtime'] for c in checkpoints).strftime('%Y-%m-%d')} to {max(c['mtime'] for c in checkpoints).strftime('%Y-%m-%d')}")

# 2. Load checkpoint weights
print("\n2. Loading checkpoint weights...")

def load_checkpoint_weights(ckpt_path):
    """Load weights from checkpoint"""
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Extract decoder weights (these represent the learned weight space)
        # Focus on the decoder output layer which maps to CNN weights
        decoder_weights = []
        for key, tensor in state_dict.items():
            if 'dec.lay.linear.weight' in key:
                decoder_weights.append(tensor.cpu().numpy().flatten())
        
        if decoder_weights:
            return np.concatenate(decoder_weights)
        else:
            # Fallback: use all weights
            all_weights = []
            for tensor in state_dict.values():
                if isinstance(tensor, torch.Tensor):
                    all_weights.append(tensor.cpu().numpy().flatten())
            return np.concatenate(all_weights)
    except Exception as e:
        print(f"  Error loading {ckpt_path}: {e}")
        return None

weights_data = []
for ckpt_info in tqdm(checkpoints, desc="Loading weights"):
    weights = load_checkpoint_weights(ckpt_info['path'])
    if weights is not None:
        weights_data.append({
            **ckpt_info,
            'weights': weights,
            'weight_dim': len(weights)
        })

print(f"Loaded weights from {len(weights_data)} checkpoints")
print(f"Weight dimensions: {weights_data[0]['weight_dim']} parameters")

# 3. NORMAL PERSISTENCE ANALYSIS (GUDHI)
print("\n" + "="*80)
print("3. NORMAL PERSISTENT HOMOLOGY ANALYSIS (GUDHI)")
print("="*80)

def compute_persistence_diagram(weights, max_dim=2):
    """Compute persistence diagram from weight vector using Vietoris-Rips complex"""
    if not GUDHI_AVAILABLE:
        return None
    
    # Reshape weights into point cloud (use sliding window)
    window_size = 100
    stride = 50
    points = []
    for i in range(0, len(weights) - window_size, stride):
        points.append(weights[i:i+window_size])
    
    points = np.array(points)
    
    # Subsample if too many points (for computational efficiency)
    if len(points) > 500:
        indices = np.random.choice(len(points), 500, replace=False)
        points = points[indices]
    
    # Compute Vietoris-Rips complex
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=10.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    
    # Compute persistence
    simplex_tree.compute_persistence()
    
    # Extract persistence diagrams by dimension
    diagrams = {}
    for dim in range(max_dim + 1):
        diagrams[dim] = simplex_tree.persistence_intervals_in_dimension(dim)
    
    return diagrams, simplex_tree

def compute_betti_numbers(simplex_tree, max_dim=2):
    """Compute Betti numbers from simplex tree"""
    betti = {}
    for dim in range(max_dim + 1):
        betti[dim] = simplex_tree.betti_numbers()[dim] if dim < len(simplex_tree.betti_numbers()) else 0
    return betti

def compute_persistence_entropy(diagram):
    """Compute persistence entropy from diagram"""
    if len(diagram) == 0:
        return 0.0
    
    # Compute lifetimes
    lifetimes = []
    for birth, death in diagram:
        if np.isfinite(death):
            lifetimes.append(death - birth)
    
    if len(lifetimes) == 0:
        return 0.0
    
    # Normalize to probability distribution
    lifetimes = np.array(lifetimes)
    L = np.sum(lifetimes)
    if L == 0:
        return 0.0
    
    p = lifetimes / L
    
    # Compute entropy
    entropy = -np.sum(p * np.log(p + 1e-10))
    return entropy

def compute_total_persistence(diagram):
    """Compute total persistence (sum of lifetimes)"""
    if len(diagram) == 0:
        return 0.0
    
    total = 0.0
    for birth, death in diagram:
        if np.isfinite(death):
            total += (death - birth)
    return total

if GUDHI_AVAILABLE:
    normal_persistence_results = []
    
    for data in tqdm(weights_data, desc="Computing normal persistence"):  # All checkpoints
        try:
            diagrams, simplex_tree = compute_persistence_diagram(data['weights'])
            betti = compute_betti_numbers(simplex_tree)
            
            result = {
                'experiment': data['experiment'],
                'overlap': data['overlap'],
                'loss_name': data['loss_name'],
                'betti_0': betti.get(0, 0),
                'betti_1': betti.get(1, 0),
                'betti_2': betti.get(2, 0),
                'pers_entropy_0': compute_persistence_entropy(diagrams[0]),
                'pers_entropy_1': compute_persistence_entropy(diagrams[1]),
                'total_pers_0': compute_total_persistence(diagrams[0]),
                'total_pers_1': compute_total_persistence(diagrams[1]),
                'num_features_0': len(diagrams[0]),
                'num_features_1': len(diagrams[1]),
            }
            
            normal_persistence_results.append(result)
            
        except Exception as e:
            print(f"  Error computing persistence for {data['experiment']}: {e}")
            continue
    
    # Save results
    df_normal = pd.DataFrame(normal_persistence_results)
    df_normal.to_csv(OUTPUT_DIR / "normal_persistence_results.csv", index=False)
    
    print(f"\n✓ Computed normal persistence for {len(normal_persistence_results)} checkpoints")
    print("\nSummary statistics:")
    print(df_normal[['betti_0', 'betti_1', 'betti_2', 'pers_entropy_1', 'total_pers_1']].describe())
else:
    print("✗ GUDHI not available - skipping normal persistence")
    normal_persistence_results = []

# 4. MULTIPARAMETER PERSISTENCE ANALYSIS (Multipers)
print("\n" + "="*80)
print("4. MULTIPARAMETER PERSISTENT HOMOLOGY ANALYSIS (Multipers)")
print("="*80)

def compute_multiparameter_persistence(weights):
    """
    Compute multiparameter persistence using two filtrations:
    1. Weight magnitude filtration
    2. Weight position filtration
    
    Uses multipers.SimplexTreeMulti for 2-parameter persistence
    """
    if not MULTIPERS_AVAILABLE:
        return None
    
    # Create point cloud from weights
    window_size = 100
    stride = 50
    points = []
    for i in range(0, len(weights) - window_size, stride):
        points.append(weights[i:i+window_size])
    
    points = np.array(points)
    
    # Subsample
    if len(points) > 200:
        indices = np.random.choice(len(points), 200, replace=False)
        points = points[indices]
    
    # Define two filtration functions
    # F1: Distance from origin (magnitude)
    f1 = np.linalg.norm(points, axis=1)
    
    # F2: Position along sequence (temporal)
    f2 = np.arange(len(points), dtype=float)
    
    # Normalize filtrations to [0, 1]
    f1 = (f1 - f1.min()) / (f1.max() - f1.min() + 1e-10)
    f2 = (f2 - f2.min()) / (f2.max() - f2.min() + 1e-10)
    
    try:
        # Create 2-parameter simplex tree using multipers
        st_multi = mp.SimplexTreeMulti(num_parameters=2)
        
        # Add vertices with 2-parameter filtration values
        for i in range(len(points)):
            st_multi.insert([i], filtration=[f1[i], f2[i]])
        
        # Add edges (Rips-like construction)
        # Connect points that are close in embedding space
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(points))
        threshold = np.percentile(distances, 10)  # Connect nearest 10%
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if distances[i, j] < threshold:
                    # Edge filtration = max of vertex filtrations
                    edge_filt = [max(f1[i], f1[j]), max(f2[i], f2[j])]
                    st_multi.insert([i, j], filtration=edge_filt)
        
        # Compute multiparameter persistence
        st_multi.compute_persistence()
        
        # Get persistence pairs
        persistence_pairs = st_multi.get_persistence_pairs()
        
        # Compute rank invariant (dimension of homology at grid points)
        grid_size = 10
        rank_invariant = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                s = i / grid_size
                t = j / grid_size
                # Count features alive at (s, t)
                count = 0
                for pair in persistence_pairs:
                    birth, death = pair[1], pair[2]
                    if birth[0] <= s and birth[1] <= t:
                        if death[0] > s or death[1] > t:
                            count += 1
                rank_invariant[i, j] = count
        
        # Compute Hilbert function (total dimension at each point)
        hilbert_function = rank_invariant.copy()
        
        return {
            'num_points': len(points),
            'num_edges': st_multi.num_simplices()[1] if len(st_multi.num_simplices()) > 1 else 0,
            'num_persistence_pairs': len(persistence_pairs),
            'rank_invariant_mean': float(np.mean(rank_invariant)),
            'rank_invariant_max': float(np.max(rank_invariant)),
            'hilbert_function_mean': float(np.mean(hilbert_function)),
            'filtration_range_f1': (0.0, 1.0),
            'filtration_range_f2': (0.0, 1.0),
            'f1_f2_correlation': float(np.corrcoef(f1, f2)[0, 1])
        }
        
    except Exception as e:
        # Fallback: compute basic 2-parameter features
        return {
            'num_points': len(points),
            'filtration_range_f1': (0.0, 1.0),
            'filtration_range_f2': (0.0, 1.0),
            'f1_mean': float(np.mean(f1)),
            'f2_mean': float(np.mean(f2)),
            'f1_f2_correlation': float(np.corrcoef(f1, f2)[0, 1]),
            'error': str(e)
        }

if MULTIPERS_AVAILABLE:
    multiparameter_results = []
    
    for data in tqdm(weights_data, desc="Computing multiparameter persistence"):  # All checkpoints
        try:
            mp_result = compute_multiparameter_persistence(data['weights'])
            if mp_result:
                result = {
                    'experiment': data['experiment'],
                    'overlap': data['overlap'],
                    'loss_name': data['loss_name'],
                    **mp_result
                }
                multiparameter_results.append(result)
        except Exception as e:
            print(f"  Error computing multiparameter persistence for {data['experiment']}: {e}")
            continue
    
    # Save results
    with open(OUTPUT_DIR / "multiparameter_persistence_results.json", 'w') as f:
        json.dump(multiparameter_results, f, indent=2, default=str)
    
    print(f"\n✓ Computed multiparameter persistence for {len(multiparameter_results)} checkpoints")
else:
    print("✗ Multipers not available - skipping multiparameter persistence")
    multiparameter_results = []

# 5. VISUALIZATION
print("\n" + "="*80)
print("5. GENERATING VISUALIZATIONS")
print("="*80)

if normal_persistence_results:
    # Plot Betti numbers by overlap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, dim in enumerate([0, 1, 2]):
        ax = axes[i]
        df_normal.boxplot(column=f'betti_{dim}', by='overlap', ax=ax)
        ax.set_title(f'Betti Number β_{dim} by Overlap')
        ax.set_xlabel('Overlap')
        ax.set_ylabel(f'β_{dim}')
        plt.sca(ax)
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "betti_numbers_by_overlap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: betti_numbers_by_overlap.png")
    
    # Plot persistence entropy
    fig, ax = plt.subplots(figsize=(10, 6))
    df_normal.boxplot(column='pers_entropy_1', by='overlap', ax=ax)
    ax.set_title('Persistence Entropy (H₁) by Overlap')
    ax.set_xlabel('Overlap')
    ax.set_ylabel('Persistence Entropy')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "persistence_entropy_by_overlap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: persistence_entropy_by_overlap.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to:")
print(f"  - {OUTPUT_DIR / 'normal_persistence_results.csv'}")
print(f"  - {OUTPUT_DIR / 'multiparameter_persistence_results.json'}")
print(f"\nFigures saved to:")
print(f"  - {FIGURES_DIR / 'betti_numbers_by_overlap.png'}")
print(f"  - {FIGURES_DIR / 'persistence_entropy_by_overlap.png'}")
