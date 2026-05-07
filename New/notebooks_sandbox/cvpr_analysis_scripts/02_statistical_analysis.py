#!/usr/bin/env python3
"""
Statistical Analysis for CVPR Paper
Creates correlation matrices and statistical visualizations similar to progress report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/cvpr_analysis_scripts/data")
FIGURES_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/CVPR 2026/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_data():
    """Load collected experiment data"""
    metrics_file = DATA_DIR / "experiment_metrics.csv"
    weights_file = DATA_DIR / "weight_representations.pkl"
    
    df = pd.read_csv(metrics_file)
    
    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)
    
    return df, weights

def create_correlation_matrix_by_overlap(df, overlap_id, save_name):
    """Create correlation matrix for specific overlap (C0, D1 style)"""
    
    # Filter by overlap
    df_overlap = df[df['overlap'] == overlap_id].copy()
    
    if len(df_overlap) == 0:
        print(f"No data for overlap {overlap_id}")
        return
    
    # Select numeric columns for correlation - use available columns
    available_cols = df_overlap.select_dtypes(include=[np.number]).columns.tolist()
    
    # Prioritize key metrics if available
    priority_cols = ['final_val_loss', 'best_val_loss', 'final_train_loss', 
                    'best_train_loss', 'total_epochs', 'convergence_epoch',
                    'euclidean_mean', 'manhattan_mean', 'cosine_mean', 
                    'frobenius_mean', 'wasserstein_mean', 'mape_mean',
                    'final_gw_distance', 'gw_distance_mean', 'cnn_validation_samples']
    
    numeric_cols = [col for col in priority_cols if col in available_cols]
    
    # Add more columns if we have fewer than 10
    if len(numeric_cols) < 10:
        extra_cols = [col for col in available_cols if col not in numeric_cols][:15-len(numeric_cols)]
        numeric_cols.extend(extra_cols)
    
    # Add loss_name as categorical (encoded)
    df_overlap['loss_encoded'] = pd.Categorical(df_overlap['loss_name']).codes
    numeric_cols.append('loss_encoded')
    
    # Compute correlation matrix
    corr_matrix = df_overlap[numeric_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax)
    
    ax.set_title(f'Correlation Matrix - Overlap {overlap_id}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / f"{save_name}_correlation_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    return corr_matrix

def create_weight_statistics_correlation(weights_dict, overlap_id):
    """Create correlation matrix from weight representation statistics"""
    
    # Filter experiments by overlap
    overlap_experiments = {k: v for k, v in weights_dict.items() 
                          if f'overlap{overlap_id}' in k}
    
    if not overlap_experiments:
        print(f"No weight data for overlap {overlap_id}")
        return
    
    # Compute statistics for each experiment
    stats_data = []
    
    for exp_name, weights in overlap_experiments.items():
        if weights.size == 0:
            continue
        
        # Flatten weights if needed
        if weights.ndim > 2:
            weights_flat = weights.reshape(weights.shape[0], -1)
        else:
            weights_flat = weights
        
        # Compute statistics across all samples
        stats_record = {
            'experiment': exp_name,
            'mean': np.mean(weights_flat),
            'std': np.std(weights_flat),
            'min': np.min(weights_flat),
            'max': np.max(weights_flat),
            'median': np.median(weights_flat),
            'skewness': stats.skew(weights_flat.flatten()),
            'kurtosis': stats.kurtosis(weights_flat.flatten()),
            'q25': np.percentile(weights_flat, 25),
            'q75': np.percentile(weights_flat, 75),
        }
        
        stats_data.append(stats_record)
    
    # Create DataFrame
    df_stats = pd.DataFrame(stats_data)
    
    # Correlation matrix
    numeric_cols = ['mean', 'std', 'min', 'max', 'median', 
                    'skewness', 'kurtosis', 'q25', 'q75']
    corr_matrix = df_stats[numeric_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Correlation'},
                ax=ax)
    
    ax.set_title(f'Weight Statistics Correlation - Overlap {overlap_id}',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / f"weight_stats_overlap{overlap_id}_correlation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    return corr_matrix, df_stats

def create_loss_comparison_plot(df):
    """Compare loss functions across overlaps"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Val loss by overlap
    ax = axes[0, 0]
    for overlap in sorted(df['overlap'].unique()):
        df_overlap = df[df['overlap'] == overlap]
        ax.scatter(range(len(df_overlap)), df_overlap['final_val_loss'],
                  label=f'Overlap {overlap}', alpha=0.6, s=50)
    ax.set_xlabel('Experiment Index')
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Validation Loss by Overlap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top right: Convergence epoch by overlap (or validation samples if cnn_accuracy not available)
    ax = axes[0, 1]
    if 'cnn_validation_samples' in df.columns:
        df.boxplot(column='cnn_validation_samples', by='overlap', ax=ax)
        ax.set_ylabel('CNN Validation Samples')
        ax.set_title('Validation Samples by Overlap')
    elif 'convergence_epoch' in df.columns:
        df.boxplot(column='convergence_epoch', by='overlap', ax=ax)
        ax.set_ylabel('Convergence Epoch')
        ax.set_title('Convergence Speed by Overlap')
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Overlap')
    plt.sca(ax)
    plt.xticks(rotation=0)
    
    # 3. Loss function performance
    ax = axes[1, 0]
    # Top 10 loss functions by average CNN accuracy
    top_losses = df.groupby('loss_name')['cnn_accuracy'].mean().nlargest(10)
    top_losses.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Average CNN Accuracy (%)')
    ax.set_title('Top 10 Loss Functions by CNN Accuracy')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Training vs validation loss
    ax = axes[1, 1]
    ax.scatter(df['final_train_loss'], df['final_val_loss'], 
              c=df['overlap'], cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel('Final Training Loss')
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.plot([df['final_train_loss'].min(), df['final_train_loss'].max()],
           [df['final_train_loss'].min(), df['final_train_loss'].max()],
           'r--', alpha=0.5, label='y=x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / "loss_comparison_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def create_statistical_summary_table(df):
    """Create summary statistics table"""
    
    summary = df.groupby('overlap').agg({
        'final_val_loss': ['mean', 'std', 'min', 'max'],
        'cnn_accuracy': ['mean', 'std', 'min', 'max'],
        'total_epochs': ['mean', 'std'],
        'loss_name': 'count'
    }).round(3)
    
    # Save to CSV
    output_file = FIGURES_DIR / "statistical_summary.csv"
    summary.to_csv(output_file)
    print(f"Saved summary table: {output_file}")
    
    return summary

def main():
    print("="*80)
    print("STATISTICAL ANALYSIS FOR CVPR PAPER")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df, weights = load_data()
    
    print(f"Loaded {len(df)} experiments")
    print(f"Loaded {len(weights)} weight representations")
    
    # Create correlation matrices (like C0, D1 from progress report)
    print("\n1. Creating correlation matrices...")
    for overlap_id in sorted(df['overlap'].unique()):
        if overlap_id == 0:
            save_name = "C0"
        elif overlap_id == 1:
            save_name = "D1"
        else:
            save_name = f"E{overlap_id}"
        
        create_correlation_matrix_by_overlap(df, overlap_id, save_name)
    
    # Weight statistics correlation
    print("\n2. Creating weight statistics correlations...")
    for overlap_id in sorted(df['overlap'].unique()):
        create_weight_statistics_correlation(weights, overlap_id)
    
    # Loss comparison plots
    print("\n3. Creating loss comparison plots...")
    create_loss_comparison_plot(df)
    
    # Summary table
    print("\n4. Creating statistical summary table...")
    summary = create_statistical_summary_table(df)
    print("\nSummary Statistics:")
    print(summary)
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
