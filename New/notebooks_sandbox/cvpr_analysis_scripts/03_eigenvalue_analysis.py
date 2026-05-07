#!/usr/bin/env python3
"""
Eigenvalue Analysis for CVPR Paper
Analyzes spectral properties of weight representations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.linalg import svd
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/cvpr_analysis_scripts/data")
FIGURES_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/CVPR 2026/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-paper')

def load_data():
    """Load weight representations"""
    weights_file = DATA_DIR / "weight_representations.pkl"
    
    with open(weights_file, 'rb') as f:
        weights = pickle.load(f)
    
    return weights

def compute_eigenvalue_spectrum(weights_matrix):
    """Compute eigenvalue spectrum of weight covariance matrix"""
    
    # Flatten if needed
    if weights_matrix.ndim > 2:
        weights_flat = weights_matrix.reshape(weights_matrix.shape[0], -1)
    else:
        weights_flat = weights_matrix
    
    # Center the data
    weights_centered = weights_flat - weights_flat.mean(axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(weights_centered.T)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    return eigenvalues

def analyze_eigenvalue_decay(weights_dict):
    """Analyze eigenvalue decay patterns across experiments"""
    
    eigenvalue_data = {}
    
    for exp_name, weights in weights_dict.items():
        if weights.size == 0:
            continue
        
        try:
            eigenvalues = compute_eigenvalue_spectrum(weights)
            eigenvalue_data[exp_name] = eigenvalues
        except Exception as e:
            print(f"Error computing eigenvalues for {exp_name}: {e}")
    
    return eigenvalue_data

def plot_eigenvalue_spectrum_by_overlap(eigenvalue_data):
    """Plot eigenvalue spectra grouped by overlap"""
    
    # Group by overlap
    overlap_groups = {0: [], 1: [], 2: []}
    
    for exp_name, eigenvalues in eigenvalue_data.items():
        if 'overlap0' in exp_name:
            overlap_groups[0].append(eigenvalues)
        elif 'overlap1' in exp_name:
            overlap_groups[1].append(eigenvalues)
        elif 'overlap2' in exp_name:
            overlap_groups[2].append(eigenvalues)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for overlap_id, ax in enumerate(axes):
        eigenvalue_list = overlap_groups[overlap_id]
        
        if not eigenvalue_list:
            continue
        
        # Plot each experiment's spectrum (first 100 eigenvalues)
        for i, eigenvalues in enumerate(eigenvalue_list[:10]):  # Plot first 10 for clarity
            n_plot = min(100, len(eigenvalues))
            ax.plot(range(1, n_plot+1), eigenvalues[:n_plot], 
                   alpha=0.5, linewidth=1)
        
        # Compute and plot mean spectrum
        min_len = min(len(ev) for ev in eigenvalue_list)
        eigenvalues_array = np.array([ev[:min_len] for ev in eigenvalue_list])
        mean_spectrum = eigenvalues_array.mean(axis=0)
        
        n_plot = min(100, len(mean_spectrum))
        ax.plot(range(1, n_plot+1), mean_spectrum[:n_plot],
               'r-', linewidth=3, label='Mean Spectrum', alpha=0.8)
        
        ax.set_xlabel('Eigenvalue Index', fontsize=12)
        ax.set_ylabel('Eigenvalue Magnitude', fontsize=12)
        ax.set_title(f'Overlap {overlap_id} - Eigenvalue Spectrum', 
                    fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / "eigenvalue_spectrum_by_overlap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_eigenvalue_decay_rate(eigenvalue_data):
    """Plot eigenvalue decay rates"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Group by overlap
    overlap_groups = {0: [], 1: [], 2: []}
    
    for exp_name, eigenvalues in eigenvalue_data.items():
        if 'overlap0' in exp_name:
            overlap_groups[0].append(eigenvalues)
        elif 'overlap1' in exp_name:
            overlap_groups[1].append(eigenvalues)
        elif 'overlap2' in exp_name:
            overlap_groups[2].append(eigenvalues)
    
    # 1. Cumulative explained variance
    ax = axes[0, 0]
    for overlap_id, eigenvalue_list in overlap_groups.items():
        if not eigenvalue_list:
            continue
        
        # Average cumulative variance
        cumvar_list = []
        for eigenvalues in eigenvalue_list:
            cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            cumvar_list.append(cumvar[:100])
        
        min_len = min(len(cv) for cv in cumvar_list)
        cumvar_array = np.array([cv[:min_len] for cv in cumvar_list])
        mean_cumvar = cumvar_array.mean(axis=0)
        
        ax.plot(range(1, len(mean_cumvar)+1), mean_cumvar,
               label=f'Overlap {overlap_id}', linewidth=2)
    
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    
    # 2. Eigenvalue ratios (consecutive)
    ax = axes[0, 1]
    for overlap_id, eigenvalue_list in overlap_groups.items():
        if not eigenvalue_list:
            continue
        
        ratio_list = []
        for eigenvalues in eigenvalue_list:
            ratios = eigenvalues[1:50] / eigenvalues[0:49]
            ratio_list.append(ratios)
        
        min_len = min(len(r) for r in ratio_list)
        ratio_array = np.array([r[:min_len] for r in ratio_list])
        mean_ratios = ratio_array.mean(axis=0)
        
        ax.plot(range(1, len(mean_ratios)+1), mean_ratios,
               label=f'Overlap {overlap_id}', linewidth=2)
    
    ax.set_xlabel('Eigenvalue Index', fontsize=12)
    ax.set_ylabel('λ(i+1) / λ(i)', fontsize=12)
    ax.set_title('Eigenvalue Decay Rate', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Effective rank
    ax = axes[1, 0]
    effective_ranks = {0: [], 1: [], 2: []}
    
    for exp_name, eigenvalues in eigenvalue_data.items():
        # Effective rank: exp(entropy of normalized eigenvalues)
        normalized_ev = eigenvalues / eigenvalues.sum()
        entropy = -np.sum(normalized_ev * np.log(normalized_ev + 1e-10))
        eff_rank = np.exp(entropy)
        
        if 'overlap0' in exp_name:
            effective_ranks[0].append(eff_rank)
        elif 'overlap1' in exp_name:
            effective_ranks[1].append(eff_rank)
        elif 'overlap2' in exp_name:
            effective_ranks[2].append(eff_rank)
    
    # Box plot - only include labels for non-empty data
    data_to_plot = [effective_ranks[i] for i in [0, 1, 2] if effective_ranks[i]]
    labels_to_plot = [f'Overlap {i}' for i in [0, 1, 2] if effective_ranks[i]]
    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels_to_plot)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('Effective Rank Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Top eigenvalue magnitude
    ax = axes[1, 1]
    top_eigenvalues = {0: [], 1: [], 2: []}
    
    for exp_name, eigenvalues in eigenvalue_data.items():
        if 'overlap0' in exp_name:
            top_eigenvalues[0].append(eigenvalues[0])
        elif 'overlap1' in exp_name:
            top_eigenvalues[1].append(eigenvalues[0])
        elif 'overlap2' in exp_name:
            top_eigenvalues[2].append(eigenvalues[0])
    
    data_to_plot = [top_eigenvalues[i] for i in [0, 1, 2] if top_eigenvalues[i]]
    labels_to_plot = [f'Overlap {i}' for i in [0, 1, 2] if top_eigenvalues[i]]
    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels_to_plot)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Top Eigenvalue Magnitude', fontsize=12)
    ax.set_title('Leading Eigenvalue Distribution', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / "eigenvalue_decay_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    print("="*80)
    print("EIGENVALUE ANALYSIS FOR CVPR PAPER")
    print("="*80)
    
    # Load data
    print("\nLoading weight representations...")
    weights = load_data()
    print(f"Loaded {len(weights)} weight representations")
    
    # Compute eigenvalue spectra
    print("\nComputing eigenvalue spectra...")
    eigenvalue_data = analyze_eigenvalue_decay(weights)
    print(f"Computed eigenvalues for {len(eigenvalue_data)} experiments")
    
    # Create visualizations
    print("\n1. Plotting eigenvalue spectra by overlap...")
    plot_eigenvalue_spectrum_by_overlap(eigenvalue_data)
    
    print("\n2. Plotting eigenvalue decay analysis...")
    plot_eigenvalue_decay_rate(eigenvalue_data)
    
    # Save eigenvalue data
    output_file = DATA_DIR / "eigenvalue_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(eigenvalue_data, f)
    print(f"\nSaved eigenvalue data: {output_file}")
    
    print("\n" + "="*80)
    print("EIGENVALUE ANALYSIS COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
