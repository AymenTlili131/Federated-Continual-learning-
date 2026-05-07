#!/usr/bin/env python3
"""
Comprehensive Weight Analysis: Ground Truth vs Predicted vs Finetuned
Performs statistical, spectral, and topological analysis on all weight sequences
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Add paths
PROJECT_ROOT = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks_sandbox"))

# Import transformer
from Double_input_transformer import TransformerAE

# Paths
EXPERIMENTS_DIR = PROJECT_ROOT / "notebooks_sandbox" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "notebooks_sandbox" / "CVPR 2026" / "figures"
DATA_DIR = PROJECT_ROOT / "notebooks_sandbox" / "cvpr_analysis_scripts" / "data"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "Merged zoo.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE WEIGHT ANALYSIS")
print("="*80)

# 1. Load ground truth weights
print("\n1. Loading ground truth weights...")
df_gt = pd.read_csv(GROUND_TRUTH_PATH)
weight_columns = df_gt.columns[17:-2].tolist()
ground_truth_weights = df_gt[weight_columns].values
print(f"Ground truth shape: {ground_truth_weights.shape}")

# 2. Find all checkpoints
print("\n2. Finding all checkpoints...")
checkpoint_files = []
for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue
    
    checkpoints_dir = exp_dir / "checkpoints"
    if checkpoints_dir.exists():
        for ckpt_file in checkpoints_dir.glob("*.pth"):
            checkpoint_files.append({
                'experiment': exp_dir.name,
                'checkpoint': ckpt_file.name,
                'path': ckpt_file
            })

print(f"Found {len(checkpoint_files)} checkpoints")

# 3. Load test data for predictions
print("\n3. Loading test data...")
try:
    # Try to load from experiments
    sample_exp = list(EXPERIMENTS_DIR.iterdir())[0]
    test_data_file = sample_exp / "test_data.npz"
    
    if not test_data_file.exists():
        # Create synthetic test data
        print("Creating synthetic test data...")
        n_test = 100
        x1_test = ground_truth_weights[np.random.choice(len(ground_truth_weights), n_test, replace=False)]
        x2_test = ground_truth_weights[np.random.choice(len(ground_truth_weights), n_test, replace=False)]
        y_test = ground_truth_weights[np.random.choice(len(ground_truth_weights), n_test, replace=False)]
    else:
        data = np.load(test_data_file)
        x1_test = data['x1_test']
        x2_test = data['x2_test']
        y_test = data['y_test']
    
    print(f"Test data shape: x1={x1_test.shape}, x2={x2_test.shape}, y={y_test.shape}")
except Exception as e:
    print(f"Error loading test data: {e}")
    print("Using ground truth samples as test data")
    n_test = 100
    indices = np.random.choice(len(ground_truth_weights), n_test, replace=False)
    x1_test = ground_truth_weights[indices]
    x2_test = ground_truth_weights[indices]
    y_test = ground_truth_weights[indices]

# 4. Generate predictions from checkpoints
print("\n4. Generating predictions from checkpoints...")

def load_checkpoint_and_predict(checkpoint_path, x1, x2, device='cpu'):
    """Load checkpoint and generate predictions"""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract hyperparameters
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default config
            config = {
                'max_seq_len': 50,
                'N': 1,
                'heads': 1,
                'd_model': 960,
                'd_ff': 960,
                'neck': 512,
                'dropout': 0.1
            }
        
        # Create model
        model = TransformerAE(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        
        # Generate predictions
        x1_tensor = torch.from_numpy(x1).float().to(device)
        x2_tensor = torch.from_numpy(x2).float().to(device)
        
        with torch.no_grad():
            predictions, necks, _, _, _ = model(x1_tensor, x2_tensor)
            predictions = predictions.cpu().numpy()
            necks = necks.cpu().numpy()
        
        return predictions, necks
        
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None, None

# Generate predictions for a subset of checkpoints
predictions_data = []
max_checkpoints = 10  # Limit to avoid memory issues

print(f"Generating predictions for up to {max_checkpoints} checkpoints...")
for i, ckpt_info in enumerate(tqdm(checkpoint_files[:max_checkpoints], desc="Processing checkpoints")):
    predictions, necks = load_checkpoint_and_predict(ckpt_info['path'], x1_test, x2_test)
    
    if predictions is not None:
        predictions_data.append({
            'experiment': ckpt_info['experiment'],
            'checkpoint': ckpt_info['checkpoint'],
            'predictions': predictions,
            'necks': necks,
            'ground_truth': y_test
        })

print(f"Generated predictions for {len(predictions_data)} checkpoints")

# 5. Load finetuning data
print("\n5. Loading finetuning data...")
finetuning_file = DATA_DIR / "finetuning_all_results.csv"
if finetuning_file.exists():
    df_finetune = pd.read_csv(finetuning_file)
    print(f"Loaded {len(df_finetune)} finetuning samples")
else:
    print("Finetuning data not found, skipping")
    df_finetune = None

# 6. Statistical Analysis
print("\n6. Performing statistical analysis...")

def compute_statistics(weights):
    """Compute comprehensive statistics"""
    return {
        'mean': np.mean(weights, axis=1),
        'std': np.std(weights, axis=1),
        'min': np.min(weights, axis=1),
        'max': np.max(weights, axis=1),
        'median': np.median(weights, axis=1),
        'q25': np.percentile(weights, 25, axis=1),
        'q75': np.percentile(weights, 75, axis=1),
        'skewness': np.mean((weights - np.mean(weights, axis=1, keepdims=True))**3, axis=1) / (np.std(weights, axis=1)**3 + 1e-10),
        'kurtosis': np.mean((weights - np.mean(weights, axis=1, keepdims=True))**4, axis=1) / (np.std(weights, axis=1)**4 + 1e-10)
    }

# Compute statistics for ground truth
gt_stats = compute_statistics(y_test)

# Compute statistics for predictions
pred_stats_all = []
for pred_data in predictions_data:
    pred_stats = compute_statistics(pred_data['predictions'])
    pred_stats['experiment'] = pred_data['experiment']
    pred_stats_all.append(pred_stats)

# 7. Spectral Analysis
print("\n7. Performing spectral analysis...")

def compute_spectral_features(weights):
    """Compute spectral features using FFT"""
    fft_coeffs = np.fft.fft(weights, axis=1)
    power_spectrum = np.abs(fft_coeffs)**2
    
    return {
        'power_spectrum': power_spectrum,
        'dominant_freq': np.argmax(power_spectrum, axis=1),
        'spectral_centroid': np.sum(np.arange(power_spectrum.shape[1]) * power_spectrum, axis=1) / (np.sum(power_spectrum, axis=1) + 1e-10),
        'spectral_rolloff': np.percentile(power_spectrum, 85, axis=1),
        'spectral_flatness': np.exp(np.mean(np.log(power_spectrum + 1e-10), axis=1)) / (np.mean(power_spectrum, axis=1) + 1e-10)
    }

gt_spectral = compute_spectral_features(y_test)
pred_spectral_all = []
for pred_data in predictions_data:
    pred_spectral = compute_spectral_features(pred_data['predictions'])
    pred_spectral['experiment'] = pred_data['experiment']
    pred_spectral_all.append(pred_spectral)

# 8. Topological Analysis (simplified)
print("\n8. Performing topological analysis...")

def compute_topological_features(weights):
    """Compute simplified topological features"""
    from scipy.spatial.distance import pdist, squareform
    
    # Compute pairwise distances
    distances = pdist(weights, metric='euclidean')
    dist_matrix = squareform(distances)
    
    # Compute persistence-like features
    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'max_distance': np.max(distances),
        'effective_dimension': np.log(len(weights)) / np.log(np.mean(distances) + 1e-10)
    }

gt_topo = compute_topological_features(y_test)
pred_topo_all = []
for pred_data in predictions_data:
    pred_topo = compute_topological_features(pred_data['predictions'])
    pred_topo['experiment'] = pred_data['experiment']
    pred_topo_all.append(pred_topo)

# 9. Create comparative visualizations
print("\n9. Creating comparative visualizations...")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# Statistical comparisons
ax = axes[0, 0]
ax.scatter(gt_stats['mean'], pred_stats_all[0]['mean'], alpha=0.5)
ax.plot([gt_stats['mean'].min(), gt_stats['mean'].max()], 
        [gt_stats['mean'].min(), gt_stats['mean'].max()], 'r--')
ax.set_xlabel('Ground Truth Mean')
ax.set_ylabel('Predicted Mean')
ax.set_title('Mean Weight Comparison')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter(gt_stats['std'], pred_stats_all[0]['std'], alpha=0.5)
ax.plot([gt_stats['std'].min(), gt_stats['std'].max()], 
        [gt_stats['std'].min(), gt_stats['std'].max()], 'r--')
ax.set_xlabel('Ground Truth Std')
ax.set_ylabel('Predicted Std')
ax.set_title('Std Deviation Comparison')
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.hist(gt_stats['skewness'], bins=30, alpha=0.5, label='Ground Truth')
ax.hist(pred_stats_all[0]['skewness'], bins=30, alpha=0.5, label='Predicted')
ax.set_xlabel('Skewness')
ax.set_ylabel('Frequency')
ax.set_title('Skewness Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Spectral comparisons
ax = axes[1, 0]
ax.scatter(gt_spectral['spectral_centroid'], pred_spectral_all[0]['spectral_centroid'], alpha=0.5)
ax.plot([gt_spectral['spectral_centroid'].min(), gt_spectral['spectral_centroid'].max()], 
        [gt_spectral['spectral_centroid'].min(), gt_spectral['spectral_centroid'].max()], 'r--')
ax.set_xlabel('GT Spectral Centroid')
ax.set_ylabel('Predicted Spectral Centroid')
ax.set_title('Spectral Centroid Comparison')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.hist(gt_spectral['dominant_freq'], bins=30, alpha=0.5, label='Ground Truth')
ax.hist(pred_spectral_all[0]['dominant_freq'], bins=30, alpha=0.5, label='Predicted')
ax.set_xlabel('Dominant Frequency')
ax.set_ylabel('Frequency')
ax.set_title('Dominant Frequency Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.scatter(gt_spectral['spectral_flatness'], pred_spectral_all[0]['spectral_flatness'], alpha=0.5)
ax.set_xlabel('GT Spectral Flatness')
ax.set_ylabel('Predicted Spectral Flatness')
ax.set_title('Spectral Flatness Comparison')
ax.grid(True, alpha=0.3)

# Topological comparisons
ax = axes[2, 0]
topo_comparison = pd.DataFrame({
    'Ground Truth': [gt_topo['mean_distance']],
    'Predicted': [pred_topo_all[0]['mean_distance']]
})
topo_comparison.plot(kind='bar', ax=ax)
ax.set_ylabel('Mean Distance')
ax.set_title('Mean Pairwise Distance')
ax.set_xticklabels([])
ax.grid(True, alpha=0.3, axis='y')

ax = axes[2, 1]
topo_comparison2 = pd.DataFrame({
    'Ground Truth': [gt_topo['effective_dimension']],
    'Predicted': [pred_topo_all[0]['effective_dimension']]
})
topo_comparison2.plot(kind='bar', ax=ax)
ax.set_ylabel('Effective Dimension')
ax.set_title('Effective Dimension Comparison')
ax.set_xticklabels([])
ax.grid(True, alpha=0.3, axis='y')

# Summary metrics
ax = axes[2, 2]
mse = np.mean((y_test - predictions_data[0]['predictions'])**2, axis=1)
ax.hist(mse, bins=30, edgecolor='black')
ax.set_xlabel('MSE')
ax.set_ylabel('Frequency')
ax.set_title(f'Prediction Error Distribution\nMean MSE: {np.mean(mse):.4f}')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = OUTPUT_DIR / "comprehensive_weight_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# 10. Save comprehensive results
print("\n10. Saving comprehensive results...")

results = {
    'ground_truth_stats': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in gt_stats.items()},
    'ground_truth_spectral': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in gt_spectral.items()},
    'ground_truth_topological': gt_topo,
    'predictions_count': len(predictions_data),
    'test_samples': len(y_test)
}

results_file = DATA_DIR / "comprehensive_weight_analysis.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {results_file}")

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*80)
print(f"Analyzed {len(predictions_data)} checkpoint predictions")
print(f"Test samples: {len(y_test)}")
print(f"Generated figure: comprehensive_weight_comparison.png")
print("="*80)
