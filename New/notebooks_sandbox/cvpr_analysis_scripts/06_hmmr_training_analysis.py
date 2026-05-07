#!/usr/bin/env python3
"""
HMMR Time-Series Segmentation on Training Trajectories
Analyzes training loss curves and metrics evolution using Hidden Markov Model Regression
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
EXPERIMENTS_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/experiments")
OUTPUT_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/CVPR 2026/figures")
DATA_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/cvpr_analysis_scripts/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("HMMR-STYLE TIME SERIES ANALYSIS ON TRAINING TRAJECTORIES")
print("="*80)

def simple_changepoint_detection(signal, threshold=0.1):
    """
    Simple changepoint detection using gradient changes.
    Returns indices where significant changes occur.
    """
    if len(signal) < 3:
        return []
    
    # Compute gradients
    gradients = np.diff(signal)
    
    # Compute second derivative (change in gradient)
    gradient_changes = np.abs(np.diff(gradients))
    
    # Normalize
    if gradient_changes.max() > 0:
        gradient_changes = gradient_changes / gradient_changes.max()
    
    # Find changepoints
    changepoints = np.where(gradient_changes > threshold)[0] + 1
    
    return changepoints.tolist()

def segment_by_changepoints(signal, changepoints):
    """Segment signal by changepoints."""
    if not changepoints:
        return [(0, len(signal), 0)]
    
    segments = []
    changepoints = [0] + list(changepoints) + [len(signal)]
    
    for i in range(len(changepoints) - 1):
        start = changepoints[i]
        end = changepoints[i + 1]
        segments.append((start, end, i))
    
    return segments

def analyze_training_trajectory(exp_dir):
    """Analyze single experiment training trajectory."""
    history_file = exp_dir / "training_history.json"
    
    if not history_file.exists():
        return None
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        train_losses = np.array(history.get('train_loss', []))
        val_losses = np.array(history.get('val_loss', []))
        
        if len(train_losses) == 0 or len(val_losses) == 0:
            return None
        
        # Detect changepoints in validation loss
        changepoints_val = simple_changepoint_detection(val_losses, threshold=0.15)
        changepoints_train = simple_changepoint_detection(train_losses, threshold=0.15)
        
        # Segment the trajectory
        segments_val = segment_by_changepoints(val_losses, changepoints_val)
        segments_train = segment_by_changepoints(train_losses, changepoints_train)
        
        # Compute segment statistics
        segment_stats = []
        for start, end, seg_id in segments_val:
            segment_stats.append({
                'segment_id': seg_id,
                'start_epoch': start,
                'end_epoch': end,
                'duration': end - start,
                'val_loss_mean': val_losses[start:end].mean(),
                'val_loss_std': val_losses[start:end].std(),
                'val_loss_trend': val_losses[end-1] - val_losses[start] if end > start else 0,
                'train_loss_mean': train_losses[start:end].mean() if end <= len(train_losses) else np.nan,
            })
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'changepoints_val': changepoints_val,
            'changepoints_train': changepoints_train,
            'segments_val': segments_val,
            'segments_train': segments_train,
            'segment_stats': segment_stats,
            'num_segments': len(segments_val),
        }
    except Exception as e:
        print(f"Error analyzing {exp_dir.name}: {e}")
        return None

# 1. Collect all training trajectories
print("\n1. Analyzing training trajectories...")
all_trajectories = {}
exp_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])

for exp_dir in tqdm(exp_dirs, desc="Processing experiments"):
    parts = exp_dir.name.split('_')
    if len(parts) < 3:
        continue
    
    overlap = int(parts[1].replace('overlap', ''))
    loss_name = '_'.join(parts[2:])
    
    result = analyze_training_trajectory(exp_dir)
    if result:
        all_trajectories[exp_dir.name] = {
            'overlap': overlap,
            'loss_name': loss_name,
            **result
        }

print(f"Analyzed {len(all_trajectories)} experiments")

# 2. Create comprehensive visualization
print("\n2. Creating training trajectory segmentation plots...")

# Select representative experiments for visualization
representative_exps = []
for overlap in [0, 1, 2]:
    overlap_exps = [k for k, v in all_trajectories.items() if v['overlap'] == overlap]
    if overlap_exps:
        # Pick first one for each overlap
        representative_exps.append(overlap_exps[0])

fig, axes = plt.subplots(len(representative_exps), 2, figsize=(14, 4*len(representative_exps)))
if len(representative_exps) == 1:
    axes = axes.reshape(1, -1)

for idx, exp_name in enumerate(representative_exps):
    data = all_trajectories[exp_name]
    
    # Plot training loss with segments
    ax = axes[idx, 0]
    epochs = np.arange(len(data['train_losses']))
    ax.plot(epochs, data['train_losses'], 'b-', alpha=0.7, label='Train Loss')
    
    # Mark changepoints
    for cp in data['changepoints_train']:
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Color segments
    for start, end, seg_id in data['segments_train']:
        ax.axvspan(start, end, alpha=0.1, color=f'C{seg_id % 10}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title(f'{exp_name}\nTraining Loss Segmentation ({len(data["segments_train"])} segments)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot validation loss with segments
    ax = axes[idx, 1]
    epochs = np.arange(len(data['val_losses']))
    ax.plot(epochs, data['val_losses'], 'g-', alpha=0.7, label='Val Loss')
    
    # Mark changepoints
    for cp in data['changepoints_val']:
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Color segments
    for start, end, seg_id in data['segments_val']:
        ax.axvspan(start, end, alpha=0.1, color=f'C{seg_id % 10}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title(f'Validation Loss Segmentation ({len(data["segments_val"])} segments)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = OUTPUT_DIR / "hmmr_training_segmentation.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# 3. Analyze segment statistics across all experiments
print("\n3. Analyzing segment statistics...")

all_segment_data = []
for exp_name, data in all_trajectories.items():
    for seg_stat in data['segment_stats']:
        all_segment_data.append({
            'experiment': exp_name,
            'overlap': data['overlap'],
            'loss_name': data['loss_name'],
            **seg_stat
        })

df_segments = pd.DataFrame(all_segment_data)

# 4. Create segment analysis plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Segment duration distribution
ax = axes[0, 0]
for overlap in sorted(df_segments['overlap'].unique()):
    data = df_segments[df_segments['overlap'] == overlap]['duration']
    ax.hist(data, alpha=0.5, label=f'Overlap {overlap}', bins=20)
ax.set_xlabel('Segment Duration (epochs)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Segment Durations')
ax.legend()
ax.grid(True, alpha=0.3)

# Number of segments per experiment
ax = axes[0, 1]
num_segments = df_segments.groupby(['experiment', 'overlap'])['segment_id'].max().reset_index()
num_segments['num_segments'] = num_segments['segment_id'] + 1
for overlap in sorted(num_segments['overlap'].unique()):
    data = num_segments[num_segments['overlap'] == overlap]['num_segments']
    ax.hist(data, alpha=0.5, label=f'Overlap {overlap}', bins=10)
ax.set_xlabel('Number of Segments')
ax.set_ylabel('Frequency')
ax.set_title('Number of Segments per Experiment')
ax.legend()
ax.grid(True, alpha=0.3)

# Segment loss trends
ax = axes[1, 0]
df_segments.boxplot(column='val_loss_trend', by='overlap', ax=ax)
ax.set_xlabel('Overlap')
ax.set_ylabel('Validation Loss Trend')
ax.set_title('Loss Trend by Segment')
plt.sca(ax)
plt.xticks(rotation=0)

# Segment stability (std)
ax = axes[1, 1]
df_segments.boxplot(column='val_loss_std', by='overlap', ax=ax)
ax.set_xlabel('Overlap')
ax.set_ylabel('Validation Loss Std Dev')
ax.set_title('Segment Stability')
plt.sca(ax)
plt.xticks(rotation=0)

plt.tight_layout()
output_file = OUTPUT_DIR / "hmmr_segment_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# 5. Save segment statistics
output_csv = DATA_DIR / "training_segments_statistics.csv"
df_segments.to_csv(output_csv, index=False)
print(f"Saved segment statistics: {output_csv}")

# 6. Create summary statistics
print("\n" + "="*80)
print("SEGMENTATION SUMMARY")
print("="*80)

summary_stats = df_segments.groupby('overlap').agg({
    'duration': ['mean', 'std', 'min', 'max'],
    'val_loss_mean': ['mean', 'std'],
    'val_loss_trend': ['mean', 'std'],
    'segment_id': 'max'
}).round(3)

print("\nSegment Statistics by Overlap:")
print(summary_stats)

# Count experiments by number of segments
print("\nExperiments by Number of Segments:")
segment_counts = df_segments.groupby('experiment')['segment_id'].max() + 1
print(segment_counts.value_counts().sort_index())

print("\n" + "="*80)
print("HMMR-STYLE ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated figures:")
print(f"  - hmmr_training_segmentation.png")
print(f"  - hmmr_segment_analysis.png")
print(f"\nSaved data:")
print(f"  - training_segments_statistics.csv ({len(df_segments)} segments)")
print("="*80)
