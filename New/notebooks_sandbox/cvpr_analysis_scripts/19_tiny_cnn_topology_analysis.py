#!/usr/bin/env python3
"""
Topological Analysis on CNN Weights from Tiny Model Checkpoints

Loads tiny model checkpoints from notebooks_sandbox/experiments, runs inference
to generate predicted CNN weights, then performs GUDHI and Multipers analysis
on Ground Truth, Predicted, and Finetuned CNN weights.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New")
CORE_MODULES = PROJECT_ROOT / "notebooks_sandbox" / "core_modules"
sys.path.insert(0, str(CORE_MODULES))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import config and model
sys.path.insert(0, str(PROJECT_ROOT))
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

# Device selection - use CPU to avoid device mismatch issues
device = torch.device('cpu')
print(f"Using device: {device}")

print("\n" + "="*80)
print("CNN WEIGHTS TOPOLOGICAL ANALYSIS - TINY MODELS")
print("Ground Truth, Predicted, and Finetuned CNN Weights")
print("="*80)

# 1. Load ground truth CNN weights
print("\n1. Loading ground truth CNN weights...")
gt_zoo_path = DATA_DIR / "Merged zoo.csv"
df_gt = pd.read_csv(gt_zoo_path)

# Get weight columns - they should be numeric columns
weight_cols = [col for col in df_gt.columns if str(col).replace('.', '').replace('-', '').isdigit() or col.startswith('weight')]
if not weight_cols:
    # Try to get all numeric columns except labels
    weight_cols = df_gt.select_dtypes(include=[np.number]).columns.tolist()

print(f"Loaded {len(df_gt)} ground truth samples with {len(weight_cols)} weight dimensions")

# Sample GT weights
n_gt_samples = min(100, len(df_gt))
gt_indices = np.random.choice(len(df_gt), n_gt_samples, replace=False)
gt_weights = df_gt.iloc[gt_indices][weight_cols].values
print(f"Sampled {len(gt_weights)} GT weight vectors, shape: {gt_weights.shape}")

# 2. Find tiny model checkpoints
print("\n2. Finding tiny model checkpoints...")
checkpoints = []
for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
    if not exp_dir.is_dir() or not exp_dir.name.startswith('tiny_'):
        continue
    
    ckpt_file = exp_dir / "checkpoints" / "best_model.pth"
    if ckpt_file.exists():
        # Parse experiment info
        parts = exp_dir.name.split('_')
        overlap = int(parts[1].replace('overlap', ''))
        loss_name = '_'.join(parts[2:])
        
        checkpoints.append({
            'experiment': exp_dir.name,
            'overlap': overlap,
            'loss_name': loss_name,
            'path': ckpt_file
        })

print(f"Found {len(checkpoints)} tiny model checkpoints")

# 3. Load CNN weights from finetuning CSV files (these contain GT, PD, FN)
print("\n3. Loading CNN weights from finetuning CSV files...")

all_weights_data = {
    'GT': [],
    'PD': [],
    'FN': [],
    'experiment': []
}

for ckpt_info in tqdm(checkpoints[:15], desc="Loading finetuning data"):
    exp_dir = ckpt_info['path'].parent.parent
    cnn_val_dir = exp_dir / "cnn_validation"
    
    if not cnn_val_dir.exists():
        continue
    
    # Find validation results CSV files
    for csv_file in cnn_val_dir.rglob("cnn_validation_results.csv"):
        try:
            df_val = pd.read_csv(csv_file)
            
            # The CSV has columns: sample_idx, acc_id_initial, acc_id_final, acc_od_initial, acc_od_final, then weights
            # Weights are in order: GT (2464), PD (2464), FN (2464)
            metric_cols = ['sample_idx', 'acc_id_initial', 'acc_id_final', 'acc_od_initial', 'acc_od_final']
            weight_cols = [col for col in df_val.columns if col not in metric_cols]
            
            if len(weight_cols) < 2464 * 3:
                continue
            
            # Extract GT, PD, FN weights
            gt_cols = weight_cols[0:2464]
            pd_cols = weight_cols[2464:2464*2]
            fn_cols = weight_cols[2464*2:2464*3]
            
            # Sample up to 10 rows
            n_samples = min(10, len(df_val))
            sample_indices = np.random.choice(len(df_val), n_samples, replace=False)
            
            for idx in sample_indices:
                gt_vec = df_val.iloc[idx][gt_cols].values.astype(float)
                pd_vec = df_val.iloc[idx][pd_cols].values.astype(float)
                fn_vec = df_val.iloc[idx][fn_cols].values.astype(float)
                
                if len(gt_vec) == 2464 and len(pd_vec) == 2464 and len(fn_vec) == 2464:
                    all_weights_data['GT'].append(gt_vec)
                    all_weights_data['PD'].append(pd_vec)
                    all_weights_data['FN'].append(fn_vec)
                    all_weights_data['experiment'].append(ckpt_info['experiment'])
            
            break  # Only use first CSV per experiment
            
        except Exception as e:
            continue

if all_weights_data['GT']:
    gt_weights_matched = np.array(all_weights_data['GT'])
    pd_weights = np.array(all_weights_data['PD'])
    fn_weights = np.array(all_weights_data['FN'])
    print(f"\nLoaded CNN weights from finetuning data:")
    print(f"  Ground Truth: {gt_weights_matched.shape}")
    print(f"  Predicted: {pd_weights.shape}")
    print(f"  Finetuned: {fn_weights.shape}")
else:
    print("\n✗ No finetuning data found, using sampled GT weights")
    gt_weights_matched = gt_weights[:50]
    pd_weights = gt_weights[50:100]  # Placeholder
    fn_weights = gt_weights[100:150]  # Placeholder

print(f"\nFinal dataset sizes:")
print(f"  GT: {gt_weights_matched.shape}")
print(f"  PD: {pd_weights.shape}")
print(f"  FN: {fn_weights.shape}")

# 5. NORMAL PERSISTENCE ANALYSIS (GUDHI)
print("\n" + "="*80)
print("5. NORMAL PERSISTENT HOMOLOGY ANALYSIS (GUDHI)")
print("="*80)

def compute_persistence_cnn_weights(weights_matrix, max_dim=2, max_samples=50):
    """Compute persistence diagram from CNN weight matrix"""
    if not GUDHI_AVAILABLE:
        return None, None
    
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
    """Compute Betti numbers"""
    betti = {}
    betti_list = simplex_tree.betti_numbers()
    for dim in range(max_dim + 1):
        betti[dim] = betti_list[dim] if dim < len(betti_list) else 0
    return betti

def compute_persistence_entropy(diagram):
    """Compute persistence entropy"""
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
    """Compute total persistence"""
    if len(diagram) == 0:
        return 0.0
    
    total = 0.0
    for birth, death in diagram:
        if np.isfinite(death):
            total += (death - birth)
    return total

if GUDHI_AVAILABLE:
    normal_results = []
    
    for weight_type, weights in [('GT', gt_weights_matched), 
                                  ('PD', pd_weights), 
                                  ('FN', fn_weights)]:
        print(f"\nComputing normal persistence for {weight_type} weights...")
        try:
            diagrams, simplex_tree = compute_persistence_cnn_weights(weights)
            if diagrams is None:
                continue
                
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
    df_normal.to_csv(OUTPUT_DIR / "tiny_cnn_weights_normal_persistence.csv", index=False)
    print(f"\n✓ Saved normal persistence results")
    
    # Print comparison
    print(f"\n{'='*80}")
    print("NORMAL PERSISTENCE COMPARISON")
    print(f"{'='*80}")
    print(df_normal[['weight_type', 'betti_0', 'betti_1', 'betti_2', 'pers_entropy_1', 'total_pers_1']])
else:
    print("✗ GUDHI not available")
    normal_results = []

# 6. MULTIPARAMETER PERSISTENCE ANALYSIS (Multipers)
print("\n" + "="*80)
print("6. MULTIPARAMETER PERSISTENT HOMOLOGY ANALYSIS (Multipers)")
print("="*80)

def compute_multiparameter_persistence_cnn(weights_matrix, max_samples=30):
    """Compute 2-parameter persistence on CNN weights"""
    if not MULTIPERS_AVAILABLE:
        return None
    
    # Subsample
    if len(weights_matrix) > max_samples:
        indices = np.random.choice(len(weights_matrix), max_samples, replace=False)
        weights_matrix = weights_matrix[indices]
    
    # Define two filtration functions
    f1 = np.linalg.norm(weights_matrix, axis=1)  # Magnitude
    f2 = np.arange(len(weights_matrix), dtype=float)  # Position
    
    # Normalize to [0, 1]
    f1 = (f1 - f1.min()) / (f1.max() - f1.min() + 1e-10)
    f2 = (f2 - f2.min()) / (f2.max() - f2.min() + 1e-10)
    
    try:
        # Create 2-parameter simplex tree
        st_multi = mp.SimplexTreeMulti(num_parameters=2)
        
        # Add vertices
        for i in range(len(weights_matrix)):
            st_multi.insert([i], filtration=[f1[i], f2[i]])
        
        # Add edges
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(weights_matrix))
        threshold = np.percentile(distances, 10)
        
        for i in range(len(weights_matrix)):
            for j in range(i+1, len(weights_matrix)):
                if distances[i, j] < threshold:
                    edge_filt = [max(f1[i], f1[j]), max(f2[i], f2[j])]
                    st_multi.insert([i, j], filtration=edge_filt)
        
        # Compute persistence
        st_multi.compute_persistence()
        persistence_pairs = st_multi.get_persistence_pairs()
        
        # Compute rank invariant
        grid_size = 10
        rank_invariant = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                s, t = i / grid_size, j / grid_size
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
    
    for weight_type, weights in [('GT', gt_weights_matched), 
                                  ('PD', pd_weights), 
                                  ('FN', fn_weights)]:
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
    with open(OUTPUT_DIR / "tiny_cnn_weights_multiparameter_persistence.json", 'w') as f:
        json.dump(multiparameter_results, f, indent=2, default=str)
    print(f"\n✓ Saved multiparameter persistence results")
    
    # Print comparison
    if multiparameter_results:
        print(f"\n{'='*80}")
        print("MULTIPARAMETER PERSISTENCE COMPARISON")
        print(f"{'='*80}")
        df_mp = pd.DataFrame(multiparameter_results)
        print(df_mp[['weight_type', 'num_persistence_pairs', 'rank_invariant_mean', 'f1_f2_correlation']])
else:
    print("✗ Multipers not available")
    multiparameter_results = []

# 7. VISUALIZATION
print("\n" + "="*80)
print("7. GENERATING VISUALIZATIONS")
print("="*80)

if normal_results:
    # Plot Betti numbers comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, dim in enumerate([0, 1, 2]):
        ax = axes[i]
        betti_values = [r[f'betti_{dim}'] for r in normal_results]
        weight_types = [r['weight_type'] for r in normal_results]
        
        colors = {'GT': 'steelblue', 'PD': 'coral', 'FN': 'mediumseagreen'}
        bar_colors = [colors[wt] for wt in weight_types]
        
        ax.bar(weight_types, betti_values, color=bar_colors, alpha=0.8, edgecolor='black')
        ax.set_title(f'Betti Number β_{dim}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'β_{dim}', fontsize=12)
        ax.set_xlabel('Weight Type', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('CNN Weights Topological Features Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tiny_cnn_weights_betti_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: tiny_cnn_weights_betti_comparison.png")
    
    # Plot persistence features
    fig, ax = plt.subplots(figsize=(12, 6))
    
    weight_types = [r['weight_type'] for r in normal_results]
    entropy_h1 = [r['pers_entropy_1'] for r in normal_results]
    total_pers_h1 = [r['total_pers_1'] for r in normal_results]
    
    x = np.arange(len(weight_types))
    width = 0.35
    
    ax.bar(x - width/2, entropy_h1, width, label='Persistence Entropy (H₁)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, total_pers_h1, width, label='Total Persistence (H₁)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Weight Type', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Persistence Features Comparison (H₁ Homology)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(weight_types)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tiny_cnn_weights_persistence_features.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: tiny_cnn_weights_persistence_features.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to:")
print(f"  - {OUTPUT_DIR / 'tiny_cnn_weights_normal_persistence.csv'}")
print(f"  - {OUTPUT_DIR / 'tiny_cnn_weights_multiparameter_persistence.json'}")
print(f"\nFigures saved to:")
print(f"  - {FIGURES_DIR / 'tiny_cnn_weights_betti_comparison.png'}")
print(f"  - {FIGURES_DIR / 'tiny_cnn_weights_persistence_features.png'}")
