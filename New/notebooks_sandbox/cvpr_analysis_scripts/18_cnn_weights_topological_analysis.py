#!/usr/bin/env python3
"""
Topological Analysis on CNN Weights (Ground Truth, Predicted, Finetuned)

This script performs comprehensive topological analysis using GUDHI and Multipers
on the actual CNN weights (2464 dimensions), not TransformerAE weights.

Analyzes three weight types:
1. Ground Truth (GT) - Input CNN weights from test set
2. Predicted (PD) - CNN weights predicted by TransformerAE
3. Finetuned (FN) - CNN weights after finetuning on task
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
import torch.nn as nn
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import config and model
from config import MODEL_CONFIGS
from Double_input_transformer import TransformerAE

# Import topological packages
try:
    import gudhi
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
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "notebooks_sandbox" / "cvpr_analysis_scripts" / "data"
FIGURES_DIR = PROJECT_ROOT / "notebooks_sandbox" / "CVPR 2026" / "figures"

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*80)
print("CNN WEIGHTS TOPOLOGICAL ANALYSIS")
print("Ground Truth, Predicted, and Finetuned CNN Weights")
print("="*80)

# 1. Find tracking data files
print("\n1. Finding tracking data files...")
experiments_root = PROJECT_ROOT / "Experiments"
tracking_files = []

for exp_dir in sorted(experiments_root.iterdir()):
    if not exp_dir.is_dir():
        continue
    
    # Look for Tracking subdirectory
    tracking_dir = exp_dir / "Tracking"
    if tracking_dir.exists():
        for tracking_file in tracking_dir.glob("*.csv"):
            tracking_files.append({
                'experiment': exp_dir.name,
                'tracking_path': tracking_file
            })

print(f"Found {len(tracking_files)} tracking files")

# 2. Load tracking data to get GT, PD, FN CNN weights
print("\n2. Loading CNN weights from tracking data...")
print("   Format: columns are 'weight 0', 'weight 1', ..., 'weight 2463' for each type")

all_weights = {
    'GT': [],
    'PD': [],
    'FN': [],
    'experiment': [],
    'file': []
}

for track_info in tqdm(tracking_files[:20], desc="Loading tracking data"):  # Sample 20 files
    try:
        df_track = pd.read_csv(track_info['tracking_path'])
        
        # Find weight columns - they're named "weight 0", "weight 1", etc.
        all_cols = df_track.columns.tolist()
        
        # Find indices of weight columns
        weight_col_indices = [i for i, col in enumerate(all_cols) if col.startswith('weight ')]
        
        if len(weight_col_indices) < 2464 * 3:
            print(f"  Skipping {track_info['experiment']}: insufficient columns ({len(weight_col_indices)})")
            continue
        
        # Weight columns structure (based on previous analysis):
        # Columns 2-2465: GT weights (2464 dims)
        # Columns 2466-4929: PD weights (2464 dims)  
        # Columns 4930-7393: FN weights (2464 dims)
        
        # But columns are named "weight 0", "weight 1", etc.
        # So we need to extract by position after label columns
        
        # Find where weight columns start
        first_weight_idx = all_cols.index('weight 0')
        
        # Extract weight column names
        gt_cols = all_cols[first_weight_idx : first_weight_idx + 2464]
        pd_cols = all_cols[first_weight_idx + 2464 : first_weight_idx + 2464*2]
        fn_cols = all_cols[first_weight_idx + 2464*2 : first_weight_idx + 2464*3]
        
        # Sample up to 5 weight vectors from this file
        n_samples = min(5, len(df_track))
        sample_indices = np.random.choice(len(df_track), n_samples, replace=False)
        
        for idx in sample_indices:
            gt_vec = df_track.iloc[idx][gt_cols].values.astype(float)
            pd_vec = df_track.iloc[idx][pd_cols].values.astype(float)
            fn_vec = df_track.iloc[idx][fn_cols].values.astype(float)
            
            # Verify dimensions
            if len(gt_vec) == 2464 and len(pd_vec) == 2464 and len(fn_vec) == 2464:
                all_weights['GT'].append(gt_vec)
                all_weights['PD'].append(pd_vec)
                all_weights['FN'].append(fn_vec)
                all_weights['experiment'].append(track_info['experiment'])
                all_weights['file'].append(track_info['tracking_path'].name)
            
    except Exception as e:
        print(f"  Error loading {track_info['experiment']}: {e}")
        continue

# Convert to arrays
gt_weights_samples = np.array(all_weights['GT'])
pd_weights_samples = np.array(all_weights['PD'])
fn_weights_samples = np.array(all_weights['FN'])

print(f"\nLoaded CNN weights:")
print(f"  Ground Truth: {gt_weights_samples.shape}")
print(f"  Predicted: {pd_weights_samples.shape}")
print(f"  Finetuned: {fn_weights_samples.shape}")

# 4. NORMAL PERSISTENCE ANALYSIS (GUDHI)
print("\n" + "="*80)
print("4. NORMAL PERSISTENT HOMOLOGY ANALYSIS (GUDHI)")
print("="*80)

def compute_persistence_cnn_weights(weights_matrix, max_dim=2, max_samples=50):
    """
    Compute persistence diagram from CNN weight matrix
    Each row is a 2464-dimensional CNN weight vector
    """
    if not GUDHI_AVAILABLE:
        return None
    
    # Subsample if too many
    if len(weights_matrix) > max_samples:
        indices = np.random.choice(len(weights_matrix), max_samples, replace=False)
        weights_matrix = weights_matrix[indices]
    
    # Compute Vietoris-Rips complex
    rips_complex = gudhi.RipsComplex(points=weights_matrix, max_edge_length=50.0)
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
    betti_list = simplex_tree.betti_numbers()
    for dim in range(max_dim + 1):
        betti[dim] = betti_list[dim] if dim < len(betti_list) else 0
    return betti

def compute_persistence_entropy(diagram):
    """Compute persistence entropy from diagram"""
    if len(diagram) == 0:
        return 0.0
    
    lifetimes = []
    for birth, death in diagram:
        if np.isfinite(death):
            lifetimes.append(death - birth)
    
    if len(lifetimes) == 0:
        return 0.0
    
    lifetimes = np.array(lifetimes)
    L = np.sum(lifetimes)
    if L == 0:
        return 0.0
    
    p = lifetimes / L
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
    normal_results = []
    
    for weight_type, weights in [('GT', gt_weights_samples), 
                                  ('PD', pd_weights_samples), 
                                  ('FN', fn_weights_samples)]:
        print(f"\nComputing normal persistence for {weight_type} weights...")
        try:
            diagrams, simplex_tree = compute_persistence_cnn_weights(weights)
            betti = compute_betti_numbers(simplex_tree)
            
            result = {
                'weight_type': weight_type,
                'num_samples': len(weights),
                'betti_0': betti.get(0, 0),
                'betti_1': betti.get(1, 0),
                'betti_2': betti.get(2, 0),
                'pers_entropy_0': compute_persistence_entropy(diagrams[0]),
                'pers_entropy_1': compute_persistence_entropy(diagrams[1]),
                'pers_entropy_2': compute_persistence_entropy(diagrams[2]),
                'total_pers_0': compute_total_persistence(diagrams[0]),
                'total_pers_1': compute_total_persistence(diagrams[1]),
                'total_pers_2': compute_total_persistence(diagrams[2]),
                'num_features_0': len(diagrams[0]),
                'num_features_1': len(diagrams[1]),
                'num_features_2': len(diagrams[2]),
            }
            
            normal_results.append(result)
            print(f"  β₀={result['betti_0']}, β₁={result['betti_1']}, β₂={result['betti_2']}")
            print(f"  Entropy(H₁)={result['pers_entropy_1']:.3f}, Total Pers(H₁)={result['total_pers_1']:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    df_normal = pd.DataFrame(normal_results)
    df_normal.to_csv(OUTPUT_DIR / "cnn_weights_normal_persistence.csv", index=False)
    print(f"\n✓ Saved normal persistence results")
else:
    print("✗ GUDHI not available - skipping normal persistence")
    normal_results = []

# 5. MULTIPARAMETER PERSISTENCE ANALYSIS (Multipers)
print("\n" + "="*80)
print("5. MULTIPARAMETER PERSISTENT HOMOLOGY ANALYSIS (Multipers)")
print("="*80)

def compute_multiparameter_persistence_cnn(weights_matrix, max_samples=30):
    """
    Compute 2-parameter persistence on CNN weights
    F1: Weight magnitude (L2 norm)
    F2: Weight position (index in sequence)
    """
    if not MULTIPERS_AVAILABLE:
        return None
    
    # Subsample
    if len(weights_matrix) > max_samples:
        indices = np.random.choice(len(weights_matrix), max_samples, replace=False)
        weights_matrix = weights_matrix[indices]
    
    # Define two filtration functions
    # F1: L2 norm of each weight vector
    f1 = np.linalg.norm(weights_matrix, axis=1)
    
    # F2: Index (position in sequence)
    f2 = np.arange(len(weights_matrix), dtype=float)
    
    # Normalize to [0, 1]
    f1 = (f1 - f1.min()) / (f1.max() - f1.min() + 1e-10)
    f2 = (f2 - f2.min()) / (f2.max() - f2.min() + 1e-10)
    
    try:
        # Create 2-parameter simplex tree
        st_multi = mp.SimplexTreeMulti(num_parameters=2)
        
        # Add vertices with 2-parameter filtration
        for i in range(len(weights_matrix)):
            st_multi.insert([i], filtration=[f1[i], f2[i]])
        
        # Add edges (Rips-like construction)
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(weights_matrix))
        threshold = np.percentile(distances, 10)  # Connect nearest 10%
        
        for i in range(len(weights_matrix)):
            for j in range(i+1, len(weights_matrix)):
                if distances[i, j] < threshold:
                    edge_filt = [max(f1[i], f1[j]), max(f2[i], f2[j])]
                    st_multi.insert([i, j], filtration=edge_filt)
        
        # Compute persistence
        st_multi.compute_persistence()
        
        # Get persistence pairs
        persistence_pairs = st_multi.get_persistence_pairs()
        
        # Compute rank invariant on grid
        grid_size = 10
        rank_invariant = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                s = i / grid_size
                t = j / grid_size
                count = 0
                for pair in persistence_pairs:
                    birth, death = pair[1], pair[2]
                    if birth[0] <= s and birth[1] <= t:
                        if death[0] > s or death[1] > t:
                            count += 1
                rank_invariant[i, j] = count
        
        return {
            'num_points': len(weights_matrix),
            'num_edges': st_multi.num_simplices()[1] if len(st_multi.num_simplices()) > 1 else 0,
            'num_persistence_pairs': len(persistence_pairs),
            'rank_invariant_mean': float(np.mean(rank_invariant)),
            'rank_invariant_max': float(np.max(rank_invariant)),
            'rank_invariant_std': float(np.std(rank_invariant)),
            'f1_f2_correlation': float(np.corrcoef(f1, f2)[0, 1])
        }
        
    except Exception as e:
        return {
            'num_points': len(weights_matrix),
            'f1_f2_correlation': float(np.corrcoef(f1, f2)[0, 1]),
            'error': str(e)
        }

if MULTIPERS_AVAILABLE:
    multiparameter_results = []
    
    for weight_type, weights in [('GT', gt_weights_samples), 
                                  ('PD', pd_weights_samples), 
                                  ('FN', fn_weights_samples)]:
        print(f"\nComputing multiparameter persistence for {weight_type} weights...")
        try:
            mp_result = compute_multiparameter_persistence_cnn(weights)
            if mp_result:
                mp_result['weight_type'] = weight_type
                mp_result['num_samples'] = len(weights)
                multiparameter_results.append(mp_result)
                print(f"  Correlation(F1,F2)={mp_result['f1_f2_correlation']:.3f}")
                if 'rank_invariant_mean' in mp_result:
                    print(f"  Rank invariant: mean={mp_result['rank_invariant_mean']:.3f}, max={mp_result['rank_invariant_max']:.1f}")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    with open(OUTPUT_DIR / "cnn_weights_multiparameter_persistence.json", 'w') as f:
        json.dump(multiparameter_results, f, indent=2, default=str)
    print(f"\n✓ Saved multiparameter persistence results")
else:
    print("✗ Multipers not available - skipping multiparameter persistence")
    multiparameter_results = []

# 6. VISUALIZATION
print("\n" + "="*80)
print("6. GENERATING VISUALIZATIONS")
print("="*80)

if normal_results:
    # Plot Betti numbers comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, dim in enumerate([0, 1, 2]):
        ax = axes[i]
        betti_values = [r[f'betti_{dim}'] for r in normal_results]
        weight_types = [r['weight_type'] for r in normal_results]
        
        ax.bar(weight_types, betti_values, color=['blue', 'orange', 'green'])
        ax.set_title(f'Betti Number β_{dim}')
        ax.set_ylabel(f'β_{dim}')
        ax.set_xlabel('Weight Type')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cnn_weights_betti_numbers_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: cnn_weights_betti_numbers_comparison.png")
    
    # Plot persistence entropy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    weight_types = [r['weight_type'] for r in normal_results]
    entropy_h1 = [r['pers_entropy_1'] for r in normal_results]
    total_pers_h1 = [r['total_pers_1'] for r in normal_results]
    
    x = np.arange(len(weight_types))
    width = 0.35
    
    ax.bar(x - width/2, entropy_h1, width, label='Persistence Entropy (H₁)', color='steelblue')
    ax.bar(x + width/2, total_pers_h1, width, label='Total Persistence (H₁)', color='coral')
    
    ax.set_xlabel('Weight Type')
    ax.set_ylabel('Value')
    ax.set_title('Persistence Features Comparison (H₁)')
    ax.set_xticks(x)
    ax.set_xticklabels(weight_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cnn_weights_persistence_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: cnn_weights_persistence_comparison.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to:")
print(f"  - {OUTPUT_DIR / 'cnn_weights_normal_persistence.csv'}")
print(f"  - {OUTPUT_DIR / 'cnn_weights_multiparameter_persistence.json'}")
print(f"\nFigures saved to:")
print(f"  - {FIGURES_DIR / 'cnn_weights_betti_numbers_comparison.png'}")
print(f"  - {FIGURES_DIR / 'cnn_weights_persistence_comparison.png'}")
