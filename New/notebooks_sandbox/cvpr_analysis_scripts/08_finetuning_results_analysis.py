#!/usr/bin/env python3
"""
Comprehensive Finetuning Results Analysis
Analyzes the CNN finetuning data that WAS saved during tournament execution
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
EXPERIMENTS_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/experiments")
OUTPUT_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/CVPR 2026/figures")
DATA_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/cvpr_analysis_scripts/data")

print("="*80)
print("COMPREHENSIVE FINETUNING RESULTS ANALYSIS")
print("="*80)

# 1. Collect all finetuning data
print("\n1. Collecting finetuning data from all experiments...")
all_finetuning_data = []

exp_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])

for exp_dir in tqdm(exp_dirs, desc="Processing experiments"):
    # Parse experiment name
    parts = exp_dir.name.split('_')
    if len(parts) < 3:
        continue
    
    model_size = parts[0]
    overlap = int(parts[1].replace('overlap', ''))
    loss_name = '_'.join(parts[2:])
    
    # Find CNN validation directories
    cnn_val_dir = exp_dir / "cnn_validation"
    if not cnn_val_dir.exists():
        continue
    
    # Process each validation epoch
    for epoch_dir in sorted(cnn_val_dir.iterdir()):
        if not epoch_dir.is_dir():
            continue
        
        epoch_num = int(epoch_dir.name.split('_')[-1])
        
        # Read CNN validation results
        csv_file = epoch_dir / "cnn_validation_results.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                
                # Add metadata
                df['experiment'] = exp_dir.name
                df['model_size'] = model_size
                df['overlap'] = overlap
                df['loss_name'] = loss_name
                df['validation_epoch'] = epoch_num
                
                all_finetuning_data.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

# Combine all data
if all_finetuning_data:
    df_all = pd.concat(all_finetuning_data, ignore_index=True)
    print(f"\nCollected {len(df_all)} finetuning samples from {df_all['experiment'].nunique()} experiments")
else:
    print("\nNo finetuning data found!")
    exit(1)

# 2. Analyze finetuning performance
print("\n2. Analyzing finetuning performance...")

# Compute improvement metrics
df_all['acc_improvement'] = df_all['acc_id_final'] - df_all['acc_id_initial']
df_all['ood_improvement'] = df_all['acc_ood_initial']  # OOD at initial state

# Summary statistics
print("\nFinetuning Performance Summary:")
print(f"  Initial ID Accuracy: {df_all['acc_id_initial'].mean():.3f} ± {df_all['acc_id_initial'].std():.3f}")
print(f"  Final ID Accuracy: {df_all['acc_id_final'].mean():.3f} ± {df_all['acc_id_final'].std():.3f}")
print(f"  Average Improvement: {df_all['acc_improvement'].mean():.3f} ± {df_all['acc_improvement'].std():.3f}")
print(f"  Initial OOD Accuracy: {df_all['ood_improvement'].mean():.3f} ± {df_all['ood_improvement'].std():.3f}")

# 3. Analyze by loss function
print("\n3. Analyzing by loss function...")
loss_performance = df_all.groupby('loss_name').agg({
    'acc_id_initial': ['mean', 'std', 'count'],
    'acc_id_final': ['mean', 'std'],
    'acc_improvement': ['mean', 'std'],
    'acc_ood_initial': ['mean', 'std']
}).round(3)

print("\nTop 10 Loss Functions by Final ID Accuracy:")
top_losses = loss_performance.sort_values(('acc_id_final', 'mean'), ascending=False).head(10)
print(top_losses[('acc_id_final', 'mean')])

# 4. Analyze by overlap
print("\n4. Analyzing by overlap...")
overlap_performance = df_all.groupby('overlap').agg({
    'acc_id_initial': ['mean', 'std'],
    'acc_id_final': ['mean', 'std'],
    'acc_improvement': ['mean', 'std'],
    'acc_ood_initial': ['mean', 'std']
}).round(3)

print("\nPerformance by Overlap:")
print(overlap_performance)

# 5. Analyze finetuning trajectory
print("\n5. Analyzing finetuning trajectories...")

# Extract per-epoch finetuning data
finetune_epoch_cols = [c for c in df_all.columns if c.startswith('epoch_') and '_acc_id' in c]
print(f"Found {len(finetune_epoch_cols)} finetuning epoch columns")

# 6. Create visualizations
print("\n6. Creating visualizations...")

# Figure 1: Finetuning improvement distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Initial vs Final accuracy
ax = axes[0, 0]
ax.scatter(df_all['acc_id_initial'], df_all['acc_id_final'], alpha=0.3, s=20)
ax.plot([0, 1], [0, 1], 'r--', label='No improvement')
ax.set_xlabel('Initial ID Accuracy')
ax.set_ylabel('Final ID Accuracy')
ax.set_title('Finetuning: Initial vs Final Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# Improvement distribution
ax = axes[0, 1]
ax.hist(df_all['acc_improvement'], bins=50, alpha=0.7, edgecolor='black')
ax.axvline(df_all['acc_improvement'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df_all["acc_improvement"].mean():.3f}')
ax.set_xlabel('Accuracy Improvement')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Finetuning Improvements')
ax.legend()
ax.grid(True, alpha=0.3)

# Performance by overlap
ax = axes[1, 0]
overlap_data = [df_all[df_all['overlap'] == o]['acc_id_final'].values for o in sorted(df_all['overlap'].unique())]
ax.boxplot(overlap_data, labels=[f'Overlap {o}' for o in sorted(df_all['overlap'].unique())])
ax.set_ylabel('Final ID Accuracy')
ax.set_title('Final Accuracy by Overlap')
ax.grid(True, alpha=0.3, axis='y')

# Top loss functions
ax = axes[1, 1]
top_10_losses = df_all.groupby('loss_name')['acc_id_final'].mean().nlargest(10)
top_10_losses.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Mean Final ID Accuracy')
ax.set_title('Top 10 Loss Functions (by Final Accuracy)')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
output_file = OUTPUT_DIR / "finetuning_performance_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# Figure 2: Finetuning trajectory analysis
if finetune_epoch_cols:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract trajectory data
    trajectories = []
    for idx, row in df_all.iterrows():
        traj = []
        for col in sorted(finetune_epoch_cols):
            if pd.notna(row[col]):
                traj.append(row[col])
        if traj:
            trajectories.append(traj)
    
    # Plot average trajectory
    ax = axes[0]
    if trajectories:
        max_len = max(len(t) for t in trajectories)
        trajectory_array = np.full((len(trajectories), max_len), np.nan)
        for i, traj in enumerate(trajectories):
            trajectory_array[i, :len(traj)] = traj
        
        mean_traj = np.nanmean(trajectory_array, axis=0)
        std_traj = np.nanstd(trajectory_array, axis=0)
        epochs = np.arange(len(mean_traj))
        
        ax.plot(epochs, mean_traj, 'b-', linewidth=2, label='Mean')
        ax.fill_between(epochs, mean_traj - std_traj, mean_traj + std_traj, 
                        alpha=0.3, label='±1 std')
        ax.set_xlabel('Finetuning Epoch')
        ax.set_ylabel('ID Accuracy')
        ax.set_title('Average Finetuning Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot sample trajectories
    ax = axes[1]
    for i, traj in enumerate(trajectories[:50]):  # Plot first 50
        ax.plot(range(len(traj)), traj, alpha=0.2, color='blue')
    ax.set_xlabel('Finetuning Epoch')
    ax.set_ylabel('ID Accuracy')
    ax.set_title('Sample Finetuning Trajectories (50 samples)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "finetuning_trajectories.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

# 7. Save comprehensive results
print("\n7. Saving comprehensive results...")

# Save aggregated data
output_csv = DATA_DIR / "finetuning_all_results.csv"
df_all.to_csv(output_csv, index=False)
print(f"Saved all finetuning data: {output_csv} ({len(df_all)} samples)")

# Save summary statistics
summary_stats = {
    'total_samples': len(df_all),
    'total_experiments': df_all['experiment'].nunique(),
    'total_validation_epochs': df_all['validation_epoch'].nunique(),
    'mean_initial_acc': float(df_all['acc_id_initial'].mean()),
    'mean_final_acc': float(df_all['acc_id_final'].mean()),
    'mean_improvement': float(df_all['acc_improvement'].mean()),
    'mean_ood_acc': float(df_all['acc_ood_initial'].mean()),
    'best_loss_function': top_losses.index[0],
    'best_loss_final_acc': float(top_losses.iloc[0]['acc_id_final']['mean']),
}

summary_file = DATA_DIR / "finetuning_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"Saved summary: {summary_file}")

# Save per-loss performance
loss_perf_file = DATA_DIR / "finetuning_by_loss.csv"
loss_performance.to_csv(loss_perf_file)
print(f"Saved per-loss performance: {loss_perf_file}")

# 8. Analyze eigenvalue data
print("\n8. Analyzing eigenvalue data...")
eigenvalue_files = []

for exp_dir in exp_dirs:
    cnn_val_dir = exp_dir / "cnn_validation"
    if cnn_val_dir.exists():
        eigenvalue_files.extend(list(cnn_val_dir.glob("**/sample_*_eigenvalues.json")))

print(f"Found {len(eigenvalue_files)} eigenvalue analysis files")

if eigenvalue_files:
    # Sample a few eigenvalue files
    sample_eigenvalues = []
    for eig_file in eigenvalue_files[:10]:
        try:
            with open(eig_file, 'r') as f:
                eig_data = json.load(f)
                sample_eigenvalues.append(eig_data)
        except Exception as e:
            pass
    
    if sample_eigenvalues:
        print(f"\nSample eigenvalue analysis structure:")
        print(f"  Keys: {list(sample_eigenvalues[0].keys())}")
        if 'predicted' in sample_eigenvalues[0]:
            print(f"  Predicted layers: {list(sample_eigenvalues[0]['predicted'].keys())}")

print("\n" + "="*80)
print("FINETUNING ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Findings:")
print(f"  Total finetuning samples: {len(df_all)}")
print(f"  Experiments with finetuning: {df_all['experiment'].nunique()}")
print(f"  Mean initial accuracy: {df_all['acc_id_initial'].mean():.3f}")
print(f"  Mean final accuracy: {df_all['acc_id_final'].mean():.3f}")
print(f"  Mean improvement: {df_all['acc_improvement'].mean():.3f}")
print(f"  Best loss function: {summary_stats['best_loss_function']}")
print(f"  Best final accuracy: {summary_stats['best_loss_final_acc']:.3f}")

print("\nGenerated figures:")
print("  - finetuning_performance_analysis.png")
print("  - finetuning_trajectories.png")

print("\nSaved data:")
print("  - finetuning_all_results.csv")
print("  - finetuning_summary.json")
print("  - finetuning_by_loss.csv")
print("="*80)
