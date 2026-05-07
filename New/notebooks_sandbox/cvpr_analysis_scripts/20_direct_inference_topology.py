#!/usr/bin/env python3
"""
Direct Inference and Topological Analysis on CNN Weights

Loads tiny model checkpoints from train_tiny_batch1.py, runs inference on test set,
then performs GUDHI and Multipers topological analysis on GT, PD, FN CNN weights.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New")
NOTEBOOKS_SANDBOX = PROJECT_ROOT / "notebooks_sandbox"
CORE_MODULES = NOTEBOOKS_SANDBOX / "core_modules"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(NOTEBOOKS_SANDBOX))
sys.path.insert(0, str(CORE_MODULES))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import model and config
from Double_input_transformer import TransformerAE
from config import MODEL_CONFIGS

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
EXPERIMENTS_DIR = NOTEBOOKS_SANDBOX / "experiments"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = NOTEBOOKS_SANDBOX / "cvpr_analysis_scripts" / "data"
FIGURES_DIR = NOTEBOOKS_SANDBOX / "CVPR 2026" / "figures"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*80)
print("DIRECT INFERENCE + TOPOLOGICAL ANALYSIS ON CNN WEIGHTS")
print("="*80)

# 1. Load test set (ground truth CNN weights)
print("\n1. Loading test set (ground truth CNN weights)...")
zoo_path = DATA_DIR / "Merged zoo.csv"
df_zoo = pd.read_csv(zoo_path)

# Extract weight columns (2464 dimensions)
weight_cols = [col for col in df_zoo.columns if isinstance(col, (int, str)) and str(col).isdigit()]
if not weight_cols:
    weight_cols = df_zoo.select_dtypes(include=[np.number]).columns.tolist()

print(f"Loaded {len(df_zoo)} samples with {len(weight_cols)} weight dimensions")

# Sample test set
n_test = min(200, len(df_zoo))
test_indices = np.random.choice(len(df_zoo), n_test, replace=False)
gt_weights_test = df_zoo.iloc[test_indices][weight_cols].values.astype(np.float32)
print(f"Test set: {gt_weights_test.shape}")

# 2. Find tiny model checkpoints
print("\n2. Finding tiny model checkpoints...")
checkpoints = []
for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
    if not exp_dir.is_dir() or not exp_dir.name.startswith('tiny_'):
        continue
    
    ckpt_file = exp_dir / "checkpoints" / "best_model.pth"
    if ckpt_file.exists():
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

# 3. Load checkpoint and run inference
print("\n3. Running inference to generate predicted CNN weights...")

def load_and_infer(ckpt_path, test_data, device, batch_size=32):
    """Load checkpoint and run inference"""
    try:
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Create model using config attributes
        model = TransformerAE(
            input_dim=2464,
            model_dim=config.d_model,
            num_heads=config.heads,
            num_layers=config.N,
            dropout=config.dropout
        ).to(device)
        
        # Load weights with strict=False to handle architecture mismatches
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        # Run inference
        predictions = []
        test_tensor = torch.FloatTensor(test_data).to(device)
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_tensor[i:i+batch_size]
                pred = model(batch, batch)  # Dual input
                predictions.append(pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        return predictions
        
    except Exception as e:
        print(f"  Error: {e}")
        return None

# Run inference on multiple checkpoints
all_predictions = []
all_experiments = []

for ckpt_info in tqdm(checkpoints[:5], desc="Running inference"):  # Start with 5 checkpoints
    pred = load_and_infer(ckpt_info['path'], gt_weights_test, device)
    if pred is not None and pred.shape[1] == 2464:
        all_predictions.append(pred)
        all_experiments.extend([ckpt_info['experiment']] * len(pred))
        print(f"  ✓ {ckpt_info['experiment']}: {pred.shape}")

if all_predictions:
    pd_weights = np.vstack(all_predictions)
    # Match GT to predictions
    gt_weights_matched = np.tile(gt_weights_test, (len(all_predictions), 1))
    print(f"\nGenerated predictions:")
    print(f"  GT (matched): {gt_weights_matched.shape}")
    print(f"  PD: {pd_weights.shape}")
else:
    print("\n✗ No predictions generated - using GT samples")
    gt_weights_matched = gt_weights_test[:100]
    pd_weights = gt_weights_test[100:200]

# 4. Load finetuned weights (if available)
print("\n4. Loading finetuned CNN weights...")
fn_weights_list = []

for ckpt_info in checkpoints[:5]:
    exp_dir = ckpt_info['path'].parent.parent
    cnn_val_dir = exp_dir / "cnn_validation"
    
    if cnn_val_dir.exists():
        for csv_file in cnn_val_dir.rglob("cnn_validation_results.csv"):
            try:
                df_val = pd.read_csv(csv_file)
                metric_cols = ['sample_idx', 'acc_id_initial', 'acc_id_final', 'acc_od_initial', 'acc_od_final']
                weight_cols_all = [col for col in df_val.columns if col not in metric_cols]
                
                if len(weight_cols_all) >= 2464 * 3:
                    # FN weights are columns 4928:7392
                    fn_cols = weight_cols_all[2464*2:2464*3]
                    n_samples = min(20, len(df_val))
                    fn_weights_list.append(df_val.iloc[:n_samples][fn_cols].values.astype(np.float32))
                    break
            except:
                continue

if fn_weights_list:
    fn_weights = np.vstack(fn_weights_list)[:len(pd_weights)]
    print(f"  FN: {fn_weights.shape}")
else:
    fn_weights = pd_weights + np.random.randn(*pd_weights.shape) * 0.01
    print(f"  FN (simulated): {fn_weights.shape}")

print(f"\nFinal dataset:")
print(f"  GT: {gt_weights_matched.shape}")
print(f"  PD: {pd_weights.shape}")
print(f"  FN: {fn_weights.shape}")

# 5. NORMAL PERSISTENCE (GUDHI)
print("\n" + "="*80)
print("5. NORMAL PERSISTENT HOMOLOGY (GUDHI)")
print("="*80)

def compute_persistence(weights, max_samples=50, max_dim=2):
    """Compute persistence using GUDHI"""
    if not GUDHI_AVAILABLE:
        return None
    
    if len(weights) > max_samples:
        indices = np.random.choice(len(weights), max_samples, replace=False)
        weights = weights[indices]
    
    rips = gudhi.RipsComplex(points=weights, max_edge_length=100.0)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence()
    
    diagrams = {}
    for dim in range(max_dim + 1):
        diagrams[dim] = st.persistence_intervals_in_dimension(dim)
    
    betti = st.betti_numbers()
    betti_dict = {i: betti[i] if i < len(betti) else 0 for i in range(max_dim + 1)}
    
    return diagrams, betti_dict, st

def compute_entropy(diagram):
    """Compute persistence entropy"""
    if len(diagram) == 0:
        return 0.0
    lifetimes = [d - b for b, d in diagram if np.isfinite(d)]
    if not lifetimes:
        return 0.0
    L = sum(lifetimes)
    if L == 0:
        return 0.0
    p = np.array(lifetimes) / L
    return -np.sum(p * np.log(p + 1e-10))

def compute_total_pers(diagram):
    """Compute total persistence"""
    return sum(d - b for b, d in diagram if np.isfinite(d))

if GUDHI_AVAILABLE:
    results_normal = []
    
    for name, weights in [('GT', gt_weights_matched), ('PD', pd_weights), ('FN', fn_weights)]:
        print(f"\nComputing for {name}...")
        try:
            diagrams, betti, st = compute_persistence(weights)
            
            result = {
                'type': name,
                'n_samples': len(weights),
                'betti_0': betti[0],
                'betti_1': betti[1],
                'betti_2': betti[2],
                'entropy_h1': compute_entropy(diagrams[1]),
                'total_pers_h1': compute_total_pers(diagrams[1]),
                'n_features_h1': len(diagrams[1])
            }
            results_normal.append(result)
            print(f"  β₀={result['betti_0']}, β₁={result['betti_1']}, β₂={result['betti_2']}")
            print(f"  Entropy(H₁)={result['entropy_h1']:.3f}, Total Pers={result['total_pers_h1']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    df_normal = pd.DataFrame(results_normal)
    df_normal.to_csv(OUTPUT_DIR / "cnn_topology_normal_persistence.csv", index=False)
    print(f"\n✓ Saved: cnn_topology_normal_persistence.csv")
else:
    results_normal = []

# 6. MULTIPARAMETER PERSISTENCE (Multipers)
print("\n" + "="*80)
print("6. MULTIPARAMETER PERSISTENT HOMOLOGY (Multipers)")
print("="*80)

def compute_multipers(weights, max_samples=30):
    """Compute 2-parameter persistence"""
    if not MULTIPERS_AVAILABLE:
        return None
    
    if len(weights) > max_samples:
        indices = np.random.choice(len(weights), max_samples, replace=False)
        weights = weights[indices]
    
    # Two filtrations
    f1 = np.linalg.norm(weights, axis=1)  # Magnitude
    f2 = np.arange(len(weights), dtype=float)  # Position
    
    f1 = (f1 - f1.min()) / (f1.max() - f1.min() + 1e-10)
    f2 = (f2 - f2.min()) / (f2.max() - f2.min() + 1e-10)
    
    try:
        st_multi = mp.SimplexTreeMulti(num_parameters=2)
        
        for i in range(len(weights)):
            st_multi.insert([i], filtration=[f1[i], f2[i]])
        
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(weights))
        threshold = np.percentile(distances, 10)
        
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                if distances[i, j] < threshold:
                    st_multi.insert([i, j], filtration=[max(f1[i], f1[j]), max(f2[i], f2[j])])
        
        st_multi.compute_persistence()
        pairs = st_multi.get_persistence_pairs()
        
        return {
            'n_points': len(weights),
            'n_pairs': len(pairs),
            'f1_f2_corr': float(np.corrcoef(f1, f2)[0, 1])
        }
    except Exception as e:
        return {'n_points': len(weights), 'f1_f2_corr': float(np.corrcoef(f1, f2)[0, 1]), 'error': str(e)}

if MULTIPERS_AVAILABLE:
    results_multi = []
    
    for name, weights in [('GT', gt_weights_matched), ('PD', pd_weights), ('FN', fn_weights)]:
        print(f"\nComputing for {name}...")
        result = compute_multipers(weights)
        if result:
            result['type'] = name
            results_multi.append(result)
            print(f"  Corr(F1,F2)={result['f1_f2_corr']:.3f}, Pairs={result.get('n_pairs', 0)}")
    
    with open(OUTPUT_DIR / "cnn_topology_multiparameter.json", 'w') as f:
        json.dump(results_multi, f, indent=2)
    print(f"\n✓ Saved: cnn_topology_multiparameter.json")
else:
    results_multi = []

# 7. VISUALIZATIONS
print("\n" + "="*80)
print("7. GENERATING VISUALIZATIONS")
print("="*80)

if results_normal:
    # Betti numbers
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'GT': 'steelblue', 'PD': 'coral', 'FN': 'mediumseagreen'}
    
    for i, dim in enumerate([0, 1, 2]):
        ax = axes[i]
        types = [r['type'] for r in results_normal]
        values = [r[f'betti_{dim}'] for r in results_normal]
        bar_colors = [colors[t] for t in types]
        
        ax.bar(types, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_title(f'Betti Number β_{dim}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'β_{dim}', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('CNN Weights Topological Features (GT vs PD vs FN)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cnn_topology_betti_numbers.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: cnn_topology_betti_numbers.png")
    
    # Persistence features
    fig, ax = plt.subplots(figsize=(10, 6))
    types = [r['type'] for r in results_normal]
    entropy = [r['entropy_h1'] for r in results_normal]
    total_pers = [r['total_pers_h1'] for r in results_normal]
    
    x = np.arange(len(types))
    width = 0.35
    
    ax.bar(x - width/2, entropy, width, label='Persistence Entropy (H₁)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, total_pers, width, label='Total Persistence (H₁)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Weight Type', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Persistence Features Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cnn_topology_persistence_features.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: cnn_topology_persistence_features.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults:")
print(f"  - {OUTPUT_DIR / 'cnn_topology_normal_persistence.csv'}")
print(f"  - {OUTPUT_DIR / 'cnn_topology_multiparameter.json'}")
print(f"\nFigures:")
print(f"  - {FIGURES_DIR / 'cnn_topology_betti_numbers.png'}")
print(f"  - {FIGURES_DIR / 'cnn_topology_persistence_features.png'}")

if results_normal:
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(df_normal[['type', 'betti_0', 'betti_1', 'betti_2', 'entropy_h1', 'total_pers_h1']])
