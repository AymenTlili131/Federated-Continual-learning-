#!/usr/bin/env python3
"""
Complete CNN Weight Analysis Pipeline - Hardcoded Example
Follows the exact pattern from run_advanced_experiments.py

HARDCODED CONFIGURATION - Modify these paths for your checkpoint:
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HARDCODED PATHS - MODIFY THESE FOR YOUR CHECKPOINT
# ============================================================================

# Example: tiny_overlap0_LW_Sinkhorn
EXPERIMENT_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/experiments/tiny_overlap0_LW_Sinkhorn")
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "best_model.pth"
DATA_PATH = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/Merged zoo.csv")
SCENARIO_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/Scenario/overlapping_m0")

# Training config (match your checkpoint settings)
ACTIVATION = 'leakyrelu'
EPOCH = 21
OVERLAP = 0  # Extracted from experiment name

# ============================================================================
# Setup Paths
# ============================================================================

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
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
import json
from sklearn.decomposition import PCA

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# Step 1: Load Model from Checkpoint
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading Model from Checkpoint")
print("="*80)

from Double_input_transformer import TransformerAE
from config import MODEL_CONFIGS

def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint with proper device handling"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Val loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"Model config: N={config.N}, heads={config.heads}, d_model={config.d_model}, neck={config.neck}")
    
    # Create model
    model = TransformerAE(
        max_seq_len=config.max_seq_len,
        N=config.N,
        heads=config.heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        neck=config.neck,
        dropout=config.dropout
    )
    model = model.to(device)
    
    # Load state dict with device alignment
    state_dict = checkpoint['model_state_dict']
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config

model, config = load_checkpoint(CHECKPOINT_PATH, device)

# ============================================================================
# Step 2: Load Scenario Test Set (EXACT same as training)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Loading Scenario Test Set")
print("="*80)

def load_scenario_testset(scenario_dir, data_path, activation='leakyrelu', epoch=21):
    """
    Load the exact testset used during training.
    Follows the exact pattern from run_advanced_experiments.py
    """
    # Load scenario test pairs
    test_pairs_file = scenario_dir / "test_pairs.npy"
    
    if not test_pairs_file.exists():
        raise FileNotFoundError(f"Scenario test pairs not found: {test_pairs_file}")
    
    test_pairs = np.load(test_pairs_file, allow_pickle=True)
    print(f"Loaded {len(test_pairs)} test pairs from {test_pairs_file}")
    
    # Load merged zoo
    df = pd.read_csv(data_path)
    weight_cols = list(df.columns[17:-2])  # Same as training
    print(f"Zoo: {len(df)} samples, {len(weight_cols)} weight dimensions")
    
    # Load weights for test pairs (EXACT same logic as training)
    x1_list, x2_list, y_list, metadata_list = [], [], [], []
    missing_count = 0
    
    print(f"Loading weights for test pairs (activation={activation}, epoch={epoch})...")
    
    for pair in tqdm(test_pairs, desc="Processing pairs"):
        task1, task2 = pair
        task_combined = sorted(set(task1) | set(task2))
        
        task1_str = str(task1)
        task2_str = str(task2)
        task_combined_str = str(task_combined)
        
        # Find weights with fallback to nearby epochs (EXACT same as training)
        found1, found2, found3 = False, False, False
        
        for target_epoch in range(epoch, 10, -5):
            if not found1:
                mask1 = (df['label'] == task1_str) & (df['epoch'] == target_epoch) & (df[activation] == 1.0)
                if mask1.sum() > 0:
                    w1 = df[mask1].iloc[0][weight_cols].values.astype(np.float32)
                    found1 = True
            
            if not found2:
                mask2 = (df['label'] == task2_str) & (df['epoch'] == target_epoch) & (df[activation] == 1.0)
                if mask2.sum() > 0:
                    w2 = df[mask2].iloc[0][weight_cols].values.astype(np.float32)
                    found2 = True
            
            if not found3:
                mask_combined = (df['label'] == task_combined_str) & (df['epoch'] == target_epoch) & (df[activation] == 1.0)
                if mask_combined.sum() > 0:
                    w_combined = df[mask_combined].iloc[0][weight_cols].values.astype(np.float32)
                    found3 = True
            
            if found1 and found2 and found3:
                break
        
        if found1 and found2 and found3:
            x1_list.append(w1)
            x2_list.append(w2)
            y_list.append(w_combined)
            metadata_list.append({
                'task1': task1,
                'task2': task2,
                'task_combined': task_combined,
                'activation': activation,
                'epoch': epoch
            })
        else:
            missing_count += 1
    
    if missing_count > 0:
        print(f"⚠️  Warning: {missing_count}/{len(test_pairs)} test pairs not found in zoo")
    
    x1_test = np.array(x1_list)
    x2_test = np.array(x2_list)
    y_test = np.array(y_list)
    
    print(f"\nTest set loaded successfully:")
    print(f"  x1 (source tasks): {x1_test.shape}")
    print(f"  x2 (target tasks): {x2_test.shape}")
    print(f"  y (combined GT): {y_test.shape}")
    
    return x1_test, x2_test, y_test, metadata_list

x1_test, x2_test, y_test, test_metadata = load_scenario_testset(
    SCENARIO_DIR, DATA_PATH, activation=ACTIVATION, epoch=EPOCH
)

# ============================================================================
# Step 3: Apply Normalization (EXACT same as training)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Applying Layer-wise Normalization")
print("="*80)

from layer_wise_normalizer import LayerWiseNormalizer

# Try to load saved normalizer, otherwise fit on y_test
normalizer_path = EXPERIMENT_DIR / "weight_normalizer.pkl"
if normalizer_path.exists():
    print(f"Loading saved normalizer from {normalizer_path}")
    normalizer = LayerWiseNormalizer.load(str(normalizer_path))
else:
    print("Fitting normalizer on test set (y_test)...")
    normalizer = LayerWiseNormalizer(method='standard')
    normalizer.fit(y_test)

# Transform test data
x1_test_norm = normalizer.transform(x1_test)
x2_test_norm = normalizer.transform(x2_test)
y_test_norm = normalizer.transform(y_test)

print(f"Normalization complete:")
print(f"  x1_norm: mean={x1_test_norm.mean():.4f}, std={x1_test_norm.std():.4f}")
print(f"  y_norm: mean={y_test_norm.mean():.4f}, std={y_test_norm.std():.4f}")

# ============================================================================
# Step 4: Run Inference (EXACT same forward pass as training)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Running Inference")
print("="*80)

def run_inference(model, x1_norm, x2_norm, device, batch_size=32):
    """
    Run inference following the EXACT pattern from training.
    Model takes TWO inputs: x1 and x2 (source and target tasks)
    """
    predictions = []
    
    # Convert to tensors and move to device
    x1_tensor = torch.from_numpy(x1_norm).float().to(device)
    x2_tensor = torch.from_numpy(x2_norm).float().to(device)
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x1_norm), batch_size):
            batch_x1 = x1_tensor[i:i+batch_size]
            batch_x2 = x2_tensor[i:i+batch_size]
            
            # EXACT forward pass from training:
            # output, _, _, _, _ = model(x1, x2)
            output, _, _, _, _ = model(batch_x1, batch_x2)
            predictions.append(output.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

# Run inference
pd_weights_norm = run_inference(model, x1_test_norm, x2_test_norm, device)
print(f"Predictions shape (normalized): {pd_weights_norm.shape}")

# Denormalize to get actual weight values
pd_weights = normalizer.inverse_transform(pd_weights_norm)
print(f"Predictions shape (denormalized): {pd_weights.shape}")

# Ground truth (already in original space)
gt_weights = y_test.copy()
print(f"Ground truth shape: {gt_weights.shape}")

# ============================================================================
# Step 5: Eigenvalue Analysis Per Layer
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Eigenvalue Analysis Per Layer")
print("="*80)

# Layer mapping for 2464-dim weights
LAYER_MAPPING = {
    'layer_1': {'start': 0, 'end': 400, 'name': 'Early Features'},
    'layer_2': {'start': 400, 'end': 800, 'name': 'Mid Features'},
    'layer_3': {'start': 800, 'end': 1600, 'name': 'Deep Features'},
    'layer_4': {'start': 1600, 'end': 2464, 'name': 'Classifier'}
}

print("Layer mapping:")
for key, val in LAYER_MAPPING.items():
    print(f"  {key}: {val['name']} [{val['start']}:{val['end']}] ({val['end']-val['start']} dims)")

def extract_eigenvalues_per_layer(weights, layer_mapping, sample_idx=0):
    """Extract eigenvalues for each layer"""
    eigenvalues = {}
    weight_sample = weights[sample_idx]
    
    for layer_name, layer_info in layer_mapping.items():
        start, end = layer_info['start'], layer_info['end']
        layer_weights = weight_sample[start:end]
        
        # Reshape to square-ish matrix for SVD
        n = len(layer_weights)
        side = int(np.sqrt(n))
        if side * side == n:
            matrix = layer_weights.reshape(side, side)
        else:
            padded = np.zeros(side * side)
            padded[:n] = layer_weights
            matrix = padded.reshape(side, side)
        
        # Compute eigenvalues
        try:
            eigenvals = np.linalg.eigvalsh(matrix)
            eigenvalues[layer_name] = eigenvals
        except:
            eigenvalues[layer_name] = np.array([0])
    
    return eigenvalues

# Extract eigenvalues for GT and PD
gt_eigen = extract_eigenvalues_per_layer(gt_weights, LAYER_MAPPING, sample_idx=0)
pd_eigen = extract_eigenvalues_per_layer(pd_weights, LAYER_MAPPING, sample_idx=0)

print("Eigenvalues extracted for all layers")

# ============================================================================
# Step 6: Plot Eigenvalue Histograms
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Plotting Eigenvalue Histograms")
print("="*80)

def plot_eigenvalue_histograms(gt_eigen, pd_eigen, layer_mapping, save_path=None):
    """Plot histograms of eigenvalues per layer"""
    n_layers = len(layer_mapping)
    fig, axes = plt.subplots(2, n_layers//2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (layer_name, layer_info) in enumerate(layer_mapping.items()):
        ax = axes[idx]
        
        gt_vals = gt_eigen[layer_name]
        pd_vals = pd_eigen[layer_name]
        
        # Plot histograms
        ax.hist(gt_vals, bins=30, alpha=0.6, label='GT', color='steelblue', density=True)
        ax.hist(pd_vals, bins=30, alpha=0.6, label='PD', color='coral', density=True)
        
        ax.set_title(f"{layer_info['name']}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle("Eigenvalue Distribution per Layer: Ground Truth vs Predicted", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig

# Create output directory
output_dir = EXPERIMENT_DIR / "analysis_results"
output_dir.mkdir(exist_ok=True)

fig_eigen = plot_eigenvalue_histograms(
    gt_eigen, pd_eigen, LAYER_MAPPING, 
    save_path=output_dir / "eigenvalue_histograms.png"
)

# ============================================================================
# Step 7: Wasserstein Distance Calculations
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Computing Wasserstein Distances")
print("="*80)

try:
    import ot  # POT: Python Optimal Transport
    POT_AVAILABLE = True
    print("Using POT (Python Optimal Transport)")
except ImportError:
    POT_AVAILABLE = False
    print("POT not available, using scipy fallback")

def compute_wasserstein_distance(dist1, dist2, p=2):
    """Compute Wasserstein distance between two distributions"""
    if POT_AVAILABLE:
        n1, n2 = len(dist1), len(dist2)
        a = np.ones((n1,)) / n1
        b = np.ones((n2,)) / n2
        M = cdist(dist1.reshape(-1, 1), dist2.reshape(-1, 1), metric='euclidean')
        M = M ** p
        w_dist = ot.emd2(a, b, M)
        w_dist = w_dist ** (1/p) if p > 1 else w_dist
        return float(w_dist)
    else:
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(dist1, dist2)

def compute_layerwise_wasserstein(gt_weights, pd_weights, layer_mapping, n_samples=10):
    """Compute Wasserstein distance per layer between GT and PD"""
    wasserstein_results = {}
    
    for layer_name, layer_info in layer_mapping.items():
        start, end = layer_info['start'], layer_info['end']
        layer_distances = []
        
        for i in range(min(n_samples, len(gt_weights))):
            gt_layer = gt_weights[i, start:end]
            pd_layer = pd_weights[i, start:end]
            w_dist = compute_wasserstein_distance(gt_layer, pd_layer)
            layer_distances.append(w_dist)
        
        wasserstein_results[layer_name] = {
            'mean': np.mean(layer_distances),
            'std': np.std(layer_distances),
            'values': layer_distances
        }
    
    return wasserstein_results

wasserstein_distances = compute_layerwise_wasserstein(gt_weights, pd_weights, LAYER_MAPPING, n_samples=len(gt_weights))

print("\nWasserstein Distances per Layer (GT vs PD):")
print("="*60)
for layer_name, results in wasserstein_distances.items():
    print(f"{layer_name:15s}: {results['mean']:.4f} ± {results['std']:.4f}")

# Plot Wasserstein distances
def plot_wasserstein_comparison(wasserstein_results, save_path=None):
    """Plot Wasserstein distances per layer"""
    layer_names = list(wasserstein_results.keys())
    means = [wasserstein_results[l]['mean'] for l in layer_names]
    stds = [wasserstein_results[l]['std'] for l in layer_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(layer_names))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color='steelblue', edgecolor='black')
    
    ax.set_xlabel("Layer", fontsize=12, fontweight='bold')
    ax.set_ylabel("Wasserstein Distance", fontsize=12, fontweight='bold')
    ax.set_title("Wasserstein Distance per Layer (GT vs PD)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', ' ').title() for n in layer_names], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig

fig_wasserstein = plot_wasserstein_comparison(
    wasserstein_distances,
    save_path=output_dir / "wasserstein_distances.png"
)

# ============================================================================
# Step 8: Finetuning Simulation
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Finetuning Predictions")
print("="*80)

def finetune_prediction(pd_weight, gt_weight, n_steps=50, lr=0.01):
    """Simulate finetuning by gradient descent toward ground truth"""
    current = pd_weight.copy()
    trajectory = [current.copy()]
    
    for step in range(n_steps):
        grad = current - gt_weight
        current = current - lr * grad
        trajectory.append(current.copy())
    
    return current, trajectory

# Finetune predictions
n_finetune = min(10, len(pd_weights))
fn_weights = []
finetune_trajectories = []

print(f"Finetuning {n_finetune} predictions...")
for i in tqdm(range(n_finetune)):
    pd_w = pd_weights[i]
    gt_w = gt_weights[i]
    
    fn_w, trajectory = finetune_prediction(pd_w, gt_w, n_steps=30, lr=0.05)
    fn_weights.append(fn_w)
    finetune_trajectories.append(trajectory)

fn_weights = np.array(fn_weights)
print(f"Finetuned weights shape: {fn_weights.shape}")

# Extract eigenvalues for FN
fn_eigen_list = []
for i in range(len(fn_weights)):
    fn_eigen = extract_eigenvalues_per_layer(fn_weights, LAYER_MAPPING, sample_idx=i)
    fn_eigen_list.append(fn_eigen)

# ============================================================================
# Step 9: Compare GT vs PD vs FN
# ============================================================================

print("\n" + "="*80)
print("STEP 9: Comparing GT vs PD vs FN")
print("="*80)

def compare_eigenvalues_layerwise(gt_eigen, pd_eigen, fn_eigen_list, layer_mapping):
    """Compare eigenvalue statistics across GT, PD, FN"""
    comparison = {}
    
    for layer_name in layer_mapping.keys():
        gt_vals = gt_eigen[layer_name]
        pd_vals = pd_eigen[layer_name]
        fn_vals = np.mean([fn_e[layer_name] for fn_e in fn_eigen_list], axis=0)
        
        comparison[layer_name] = {
            'gt_mean': np.mean(gt_vals),
            'gt_std': np.std(gt_vals),
            'pd_mean': np.mean(pd_vals),
            'pd_std': np.std(pd_vals),
            'fn_mean': np.mean(fn_vals),
            'fn_std': np.std(fn_vals),
            'pd_error': np.abs(np.mean(gt_vals) - np.mean(pd_vals)),
            'fn_error': np.abs(np.mean(gt_vals) - np.mean(fn_vals))
        }
    
    return comparison

eigen_comparison = compare_eigenvalues_layerwise(gt_eigen, pd_eigen, fn_eigen_list, LAYER_MAPPING)

print("\nEigenvalue Comparison per Layer:")
print("="*80)
print(f"{'Layer':<20} {'GT Mean':<12} {'PD Mean':<12} {'FN Mean':<12} {'PD Err':<10} {'FN Err':<10}")
print("-"*80)
for layer_name, stats in eigen_comparison.items():
    print(f"{layer_name:<20} {stats['gt_mean']:<12.4f} {stats['pd_mean']:<12.4f} "
          f"{stats['fn_mean']:<12.4f} {stats['pd_error']:<10.4f} {stats['fn_error']:<10.4f}")

# ============================================================================
# Step 10: Save Results
# ============================================================================

print("\n" + "="*80)
print("STEP 10: Saving Results")
print("="*80)

# Save Wasserstein distances
wasserstein_df = pd.DataFrame([
    {
        'layer': layer,
        'mean_distance': data['mean'],
        'std_distance': data['std'],
        'n_samples': len(data['values'])
    }
    for layer, data in wasserstein_distances.items()
])
wasserstein_df.to_csv(output_dir / "wasserstein_distances.csv", index=False)

# Save eigenvalue comparison
eigen_comparison_df = pd.DataFrame([
    {
        'layer': layer,
        **stats
    }
    for layer, stats in eigen_comparison.items()
])
eigen_comparison_df.to_csv(output_dir / "eigenvalue_comparison.csv", index=False)

# Save summary
summary = {
    'checkpoint': str(CHECKPOINT_PATH),
    'n_test_samples': len(y_test),
    'overlap': OVERLAP,
    'activation': ACTIVATION,
    'epoch': EPOCH,
    'wasserstein_distances': {
        layer: {'mean': data['mean'], 'std': data['std']}
        for layer, data in wasserstein_distances.items()
    },
    'eigenvalue_comparison': eigen_comparison
}

with open(output_dir / "analysis_summary.json", 'w') as f:
    json.dump(summary, f, indent=2, default=float)

print(f"\nAll results saved to: {output_dir}")
print(f"  - eigenvalue_histograms.png")
print(f"  - wasserstein_distances.png")
print(f"  - wasserstein_distances.csv")
print(f"  - eigenvalue_comparison.csv")
print(f"  - analysis_summary.json")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
