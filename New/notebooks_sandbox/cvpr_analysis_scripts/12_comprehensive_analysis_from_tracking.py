#!/usr/bin/env python3
"""
Comprehensive Analysis Using Tracking Data
Uses the tracking CSVs which contain: Ground Truth, Predicted, and Finetuned weights
Performs statistical, spectral, topological analysis and segmentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New")
EXPERIMENTS_DIR = PROJECT_ROOT / "Experiments"
OUTPUT_DIR = PROJECT_ROOT / "notebooks_sandbox" / "CVPR 2026" / "figures"
DATA_DIR = PROJECT_ROOT / "notebooks_sandbox" / "cvpr_analysis_scripts" / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE WEIGHT ANALYSIS FROM TRACKING DATA")
print("="*80)

# 1. Find all tracking files
print("\n1. Finding tracking files...")
tracking_files = []
for tracking_dir in EXPERIMENTS_DIR.rglob("Tracking"):
    if tracking_dir.is_dir():
        for csv_file in tracking_dir.glob("*.csv"):
            tracking_files.append(csv_file)

print(f"Found {len(tracking_files)} tracking files")

# 2. Load tracking data (sample subset to avoid memory issues)
print("\n2. Loading tracking data...")
all_data = []
max_files = 20  # Limit to avoid memory issues

for i, csv_file in enumerate(tqdm(tracking_files[:max_files], desc="Loading")):
    try:
        df = pd.read_csv(csv_file)
        df = df.drop(columns=["Unnamed: 0"], errors='ignore')
        
        cols = df.columns.tolist()
        if len(cols) >= 7394:
            # Extract weight matrices
            GT_cols = cols[2:2466]
            PD_cols = cols[2466:4930]
            FN_cols = cols[4930:7394]
            
            GT_data = df[GT_cols].to_numpy()
            PD_data = df[PD_cols].to_numpy()
            FN_data = df[FN_cols].to_numpy()
            
            # Sample to reduce size
            n_samples = min(100, len(GT_data))
            indices = np.random.choice(len(GT_data), n_samples, replace=False)
            
            all_data.append({
                'file': csv_file.name,
                'experiment': csv_file.parent.parent.name,
                'GT': GT_data[indices],
                'PD': PD_data[indices],
                'FN': FN_data[indices]
            })
    except Exception as e:
        print(f"  Error loading {csv_file.name}: {e}")

print(f"Loaded {len(all_data)} tracking files with weight data")

# 3. Statistical Analysis
print("\n3. Performing statistical analysis...")

def compute_comprehensive_stats(weights):
    """Compute comprehensive statistical features"""
    return {
        'mean': np.mean(weights, axis=1),
        'std': np.std(weights, axis=1),
        'median': np.median(weights, axis=1),
        'min': np.min(weights, axis=1),
        'max': np.max(weights, axis=1),
        'q25': np.percentile(weights, 25, axis=1),
        'q75': np.percentile(weights, 75, axis=1),
        'skewness': stats.skew(weights, axis=1),
        'kurtosis': stats.kurtosis(weights, axis=1),
        'range': np.ptp(weights, axis=1),
        'iqr': np.percentile(weights, 75, axis=1) - np.percentile(weights, 25, axis=1)
    }

statistical_results = []
for data in all_data:
    gt_stats = compute_comprehensive_stats(data['GT'])
    pd_stats = compute_comprehensive_stats(data['PD'])
    fn_stats = compute_comprehensive_stats(data['FN'])
    
    statistical_results.append({
        'experiment': data['experiment'],
        'file': data['file'],
        'gt_stats': gt_stats,
        'pd_stats': pd_stats,
        'fn_stats': fn_stats
    })

# 4. Spectral Analysis
print("\n4. Performing spectral analysis...")

def compute_spectral_features(weights):
    """Compute spectral features using FFT"""
    fft_coeffs = fft(weights, axis=1)
    power_spectrum = np.abs(fft_coeffs)**2
    
    # Compute spectral features
    freqs = np.fft.fftfreq(weights.shape[1])
    
    return {
        'power_spectrum': power_spectrum,
        'dominant_freq_idx': np.argmax(power_spectrum, axis=1),
        'spectral_centroid': np.sum(np.arange(power_spectrum.shape[1]) * power_spectrum, axis=1) / (np.sum(power_spectrum, axis=1) + 1e-10),
        'spectral_spread': np.sqrt(np.sum((np.arange(power_spectrum.shape[1])[:, None].T - np.sum(np.arange(power_spectrum.shape[1]) * power_spectrum, axis=1, keepdims=True) / (np.sum(power_spectrum, axis=1, keepdims=True) + 1e-10))**2 * power_spectrum, axis=1) / (np.sum(power_spectrum, axis=1) + 1e-10)),
        'spectral_rolloff': np.percentile(power_spectrum, 85, axis=1),
        'spectral_flatness': np.exp(np.mean(np.log(power_spectrum + 1e-10), axis=1)) / (np.mean(power_spectrum, axis=1) + 1e-10)
    }

spectral_results = []
for data in all_data:
    gt_spectral = compute_spectral_features(data['GT'])
    pd_spectral = compute_spectral_features(data['PD'])
    fn_spectral = compute_spectral_features(data['FN'])
    
    spectral_results.append({
        'experiment': data['experiment'],
        'file': data['file'],
        'gt_spectral': gt_spectral,
        'pd_spectral': pd_spectral,
        'fn_spectral': fn_spectral
    })

# 5. Topological Analysis
print("\n5. Performing topological analysis...")

def compute_topological_features(weights):
    """Compute topological features"""
    # Sample to reduce computation
    n_sample = min(50, len(weights))
    sample_idx = np.random.choice(len(weights), n_sample, replace=False)
    weights_sample = weights[sample_idx]
    
    # Compute pairwise distances
    distances = pdist(weights_sample, metric='euclidean')
    dist_matrix = squareform(distances)
    
    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'median_distance': np.median(distances),
        'effective_dimension': np.log(len(weights_sample)) / np.log(np.mean(distances) + 1e-10),
        'intrinsic_dimension': 2 * np.mean(distances)**2 / np.var(distances)
    }

topological_results = []
for data in all_data:
    gt_topo = compute_topological_features(data['GT'])
    pd_topo = compute_topological_features(data['PD'])
    fn_topo = compute_topological_features(data['FN'])
    
    topological_results.append({
        'experiment': data['experiment'],
        'file': data['file'],
        'gt_topo': gt_topo,
        'pd_topo': pd_topo,
        'fn_topo': fn_topo
    })

# 6. Segmentation Analysis
print("\n6. Performing segmentation analysis...")

def detect_changepoints(signal, threshold=0.15):
    """Simple changepoint detection"""
    if len(signal) < 3:
        return []
    
    gradients = np.diff(signal)
    gradient_changes = np.abs(np.diff(gradients))
    
    if gradient_changes.max() > 0:
        gradient_changes = gradient_changes / gradient_changes.max()
    
    changepoints = np.where(gradient_changes > threshold)[0] + 1
    return changepoints.tolist()

segmentation_results = []
for data in all_data:
    # Analyze mean trajectory
    gt_mean_traj = np.mean(data['GT'], axis=0)
    pd_mean_traj = np.mean(data['PD'], axis=0)
    fn_mean_traj = np.mean(data['FN'], axis=0)
    
    gt_changepoints = detect_changepoints(gt_mean_traj)
    pd_changepoints = detect_changepoints(pd_mean_traj)
    fn_changepoints = detect_changepoints(fn_mean_traj)
    
    segmentation_results.append({
        'experiment': data['experiment'],
        'file': data['file'],
        'gt_segments': len(gt_changepoints) + 1,
        'pd_segments': len(pd_changepoints) + 1,
        'fn_segments': len(fn_changepoints) + 1,
        'gt_changepoints': gt_changepoints,
        'pd_changepoints': pd_changepoints,
        'fn_changepoints': fn_changepoints
    })

# 7. Create comprehensive visualizations
print("\n7. Creating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# Use first dataset for visualization
sample_data = all_data[0]
sample_stats = statistical_results[0]
sample_spectral = spectral_results[0]
sample_topo = topological_results[0]

# Statistical comparisons
ax = fig.add_subplot(gs[0, 0])
ax.scatter(sample_stats['gt_stats']['mean'], sample_stats['pd_stats']['mean'], alpha=0.5, label='Predicted', s=20)
ax.scatter(sample_stats['gt_stats']['mean'], sample_stats['fn_stats']['mean'], alpha=0.5, label='Finetuned', s=20)
ax.plot([sample_stats['gt_stats']['mean'].min(), sample_stats['gt_stats']['mean'].max()],
        [sample_stats['gt_stats']['mean'].min(), sample_stats['gt_stats']['mean'].max()], 'r--', alpha=0.5)
ax.set_xlabel('Ground Truth Mean')
ax.set_ylabel('Predicted/Finetuned Mean')
ax.set_title('Mean Weight Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.scatter(sample_stats['gt_stats']['std'], sample_stats['pd_stats']['std'], alpha=0.5, label='Predicted', s=20)
ax.scatter(sample_stats['gt_stats']['std'], sample_stats['fn_stats']['std'], alpha=0.5, label='Finetuned', s=20)
ax.plot([sample_stats['gt_stats']['std'].min(), sample_stats['gt_stats']['std'].max()],
        [sample_stats['gt_stats']['std'].min(), sample_stats['gt_stats']['std'].max()], 'r--', alpha=0.5)
ax.set_xlabel('Ground Truth Std')
ax.set_ylabel('Predicted/Finetuned Std')
ax.set_title('Std Deviation Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 2])
ax.hist(sample_stats['gt_stats']['skewness'], bins=30, alpha=0.5, label='GT', edgecolor='black')
ax.hist(sample_stats['pd_stats']['skewness'], bins=30, alpha=0.5, label='PD', edgecolor='black')
ax.hist(sample_stats['fn_stats']['skewness'], bins=30, alpha=0.5, label='FN', edgecolor='black')
ax.set_xlabel('Skewness')
ax.set_ylabel('Frequency')
ax.set_title('Skewness Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 3])
ax.hist(sample_stats['gt_stats']['kurtosis'], bins=30, alpha=0.5, label='GT', edgecolor='black')
ax.hist(sample_stats['pd_stats']['kurtosis'], bins=30, alpha=0.5, label='PD', edgecolor='black')
ax.hist(sample_stats['fn_stats']['kurtosis'], bins=30, alpha=0.5, label='FN', edgecolor='black')
ax.set_xlabel('Kurtosis')
ax.set_ylabel('Frequency')
ax.set_title('Kurtosis Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Spectral comparisons
ax = fig.add_subplot(gs[1, 0])
ax.scatter(sample_spectral['gt_spectral']['spectral_centroid'], 
          sample_spectral['pd_spectral']['spectral_centroid'], alpha=0.5, label='Predicted', s=20)
ax.scatter(sample_spectral['gt_spectral']['spectral_centroid'], 
          sample_spectral['fn_spectral']['spectral_centroid'], alpha=0.5, label='Finetuned', s=20)
ax.plot([sample_spectral['gt_spectral']['spectral_centroid'].min(), sample_spectral['gt_spectral']['spectral_centroid'].max()],
        [sample_spectral['gt_spectral']['spectral_centroid'].min(), sample_spectral['gt_spectral']['spectral_centroid'].max()], 'r--', alpha=0.5)
ax.set_xlabel('GT Spectral Centroid')
ax.set_ylabel('PD/FN Spectral Centroid')
ax.set_title('Spectral Centroid Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.hist(sample_spectral['gt_spectral']['spectral_flatness'], bins=30, alpha=0.5, label='GT', edgecolor='black')
ax.hist(sample_spectral['pd_spectral']['spectral_flatness'], bins=30, alpha=0.5, label='PD', edgecolor='black')
ax.hist(sample_spectral['fn_spectral']['spectral_flatness'], bins=30, alpha=0.5, label='FN', edgecolor='black')
ax.set_xlabel('Spectral Flatness')
ax.set_ylabel('Frequency')
ax.set_title('Spectral Flatness Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
# Power spectrum comparison (average)
gt_ps_mean = np.mean(sample_spectral['gt_spectral']['power_spectrum'], axis=0)
pd_ps_mean = np.mean(sample_spectral['pd_spectral']['power_spectrum'], axis=0)
fn_ps_mean = np.mean(sample_spectral['fn_spectral']['power_spectrum'], axis=0)
freqs = np.arange(len(gt_ps_mean))
ax.semilogy(freqs[:100], gt_ps_mean[:100], label='GT', alpha=0.7)
ax.semilogy(freqs[:100], pd_ps_mean[:100], label='PD', alpha=0.7)
ax.semilogy(freqs[:100], fn_ps_mean[:100], label='FN', alpha=0.7)
ax.set_xlabel('Frequency')
ax.set_ylabel('Power (log scale)')
ax.set_title('Average Power Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 3])
ax.hist(sample_spectral['gt_spectral']['dominant_freq_idx'], bins=30, alpha=0.5, label='GT', edgecolor='black')
ax.hist(sample_spectral['pd_spectral']['dominant_freq_idx'], bins=30, alpha=0.5, label='PD', edgecolor='black')
ax.hist(sample_spectral['fn_spectral']['dominant_freq_idx'], bins=30, alpha=0.5, label='FN', edgecolor='black')
ax.set_xlabel('Dominant Frequency Index')
ax.set_ylabel('Frequency')
ax.set_title('Dominant Frequency Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Topological comparisons
ax = fig.add_subplot(gs[2, 0])
topo_comparison = pd.DataFrame({
    'Ground Truth': [sample_topo['gt_topo']['mean_distance']],
    'Predicted': [sample_topo['pd_topo']['mean_distance']],
    'Finetuned': [sample_topo['fn_topo']['mean_distance']]
})
topo_comparison.T.plot(kind='bar', ax=ax, legend=False)
ax.set_ylabel('Mean Pairwise Distance')
ax.set_title('Mean Distance Comparison')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

ax = fig.add_subplot(gs[2, 1])
topo_comparison2 = pd.DataFrame({
    'Ground Truth': [sample_topo['gt_topo']['effective_dimension']],
    'Predicted': [sample_topo['pd_topo']['effective_dimension']],
    'Finetuned': [sample_topo['fn_topo']['effective_dimension']]
})
topo_comparison2.T.plot(kind='bar', ax=ax, legend=False)
ax.set_ylabel('Effective Dimension')
ax.set_title('Effective Dimension Comparison')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

ax = fig.add_subplot(gs[2, 2])
topo_comparison3 = pd.DataFrame({
    'Ground Truth': [sample_topo['gt_topo']['intrinsic_dimension']],
    'Predicted': [sample_topo['pd_topo']['intrinsic_dimension']],
    'Finetuned': [sample_topo['fn_topo']['intrinsic_dimension']]
})
topo_comparison3.T.plot(kind='bar', ax=ax, legend=False)
ax.set_ylabel('Intrinsic Dimension')
ax.set_title('Intrinsic Dimension Comparison')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# Segmentation comparison
ax = fig.add_subplot(gs[2, 3])
seg_data = pd.DataFrame(segmentation_results)
seg_summary = seg_data[['gt_segments', 'pd_segments', 'fn_segments']].mean()
seg_summary.plot(kind='bar', ax=ax)
ax.set_ylabel('Average Number of Segments')
ax.set_title('Segmentation Comparison')
ax.set_xticklabels(['GT', 'PD', 'FN'], rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# Error distributions
ax = fig.add_subplot(gs[3, 0])
mse_pd = np.mean((sample_data['GT'] - sample_data['PD'])**2, axis=1)
mse_fn = np.mean((sample_data['GT'] - sample_data['FN'])**2, axis=1)
ax.hist(mse_pd, bins=30, alpha=0.5, label=f'PD (μ={np.mean(mse_pd):.4f})', edgecolor='black')
ax.hist(mse_fn, bins=30, alpha=0.5, label=f'FN (μ={np.mean(mse_fn):.4f})', edgecolor='black')
ax.set_xlabel('MSE')
ax.set_ylabel('Frequency')
ax.set_title('Prediction Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[3, 1])
mae_pd = np.mean(np.abs(sample_data['GT'] - sample_data['PD']), axis=1)
mae_fn = np.mean(np.abs(sample_data['GT'] - sample_data['FN']), axis=1)
ax.hist(mae_pd, bins=30, alpha=0.5, label=f'PD (μ={np.mean(mae_pd):.4f})', edgecolor='black')
ax.hist(mae_fn, bins=30, alpha=0.5, label=f'FN (μ={np.mean(mae_fn):.4f})', edgecolor='black')
ax.set_xlabel('MAE')
ax.set_ylabel('Frequency')
ax.set_title('Mean Absolute Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Correlation analysis
ax = fig.add_subplot(gs[3, 2])
corr_pd = np.corrcoef(sample_data['GT'].flatten(), sample_data['PD'].flatten())[0, 1]
corr_fn = np.corrcoef(sample_data['GT'].flatten(), sample_data['FN'].flatten())[0, 1]
correlations = pd.Series({'Predicted': corr_pd, 'Finetuned': corr_fn})
correlations.plot(kind='bar', ax=ax)
ax.set_ylabel('Correlation with Ground Truth')
ax.set_title('Overall Correlation')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# Summary metrics
ax = fig.add_subplot(gs[3, 3])
summary_text = f"""Summary Statistics
Samples: {len(sample_data['GT'])}
Dimensions: {sample_data['GT'].shape[1]}

MSE (Predicted): {np.mean(mse_pd):.4f}
MSE (Finetuned): {np.mean(mse_fn):.4f}

Correlation (PD): {corr_pd:.4f}
Correlation (FN): {corr_fn:.4f}

Segments (GT): {segmentation_results[0]['gt_segments']}
Segments (PD): {segmentation_results[0]['pd_segments']}
Segments (FN): {segmentation_results[0]['fn_segments']}
"""
ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=10, 
        verticalalignment='center', family='monospace')
ax.axis('off')

plt.suptitle(f'Comprehensive Weight Analysis: {sample_data["experiment"]}', fontsize=16, y=0.995)

output_file = OUTPUT_DIR / "comprehensive_weight_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# 8. Save comprehensive results
print("\n8. Saving comprehensive results...")

results_summary = {
    'files_analyzed': len(all_data),
    'statistical_analysis': {
        'experiments': [r['experiment'] for r in statistical_results],
        'files': [r['file'] for r in statistical_results]
    },
    'spectral_analysis': {
        'experiments': [r['experiment'] for r in spectral_results],
        'files': [r['file'] for r in spectral_results]
    },
    'topological_analysis': {
        'experiments': [r['experiment'] for r in topological_results],
        'files': [r['file'] for r in topological_results]
    },
    'segmentation_analysis': segmentation_results
}

results_file = DATA_DIR / "comprehensive_analysis_results.json"
with open(results_file, 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"Saved: {results_file}")

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*80)
print(f"Analyzed {len(all_data)} tracking files")
print(f"Statistical analysis: ✓")
print(f"Spectral analysis: ✓")
print(f"Topological analysis: ✓")
print(f"Segmentation analysis: ✓")
print(f"Generated figure: comprehensive_weight_analysis.png")
print("="*80)
