#!/usr/bin/env python3
"""
Topological Analysis for CVPR Paper
Focus on Persistent Homology and Multiparameter Persistent Homology
Core contribution of the paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import TDA libraries
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("⚠ GUDHI not available - some analyses will be limited")

try:
    import multipers
    MULTIPERS_AVAILABLE = True
except ImportError:
    MULTIPERS_AVAILABLE = False
    print("⚠ Multipers not available - multiparameter analysis will be limited")

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

def compute_rips_persistence(point_cloud, max_dimension=2):
    """Compute Rips persistence for a point cloud"""
    
    if not GUDHI_AVAILABLE:
        return None
    
    # Subsample if too large
    if len(point_cloud) > 1000:
        indices = np.random.choice(len(point_cloud), 1000, replace=False)
        point_cloud = point_cloud[indices]
    
    # Create Rips complex
    rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=2.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    
    # Compute persistence
    simplex_tree.compute_persistence()
    
    # Get persistence diagrams
    persistence = simplex_tree.persistence()
    
    return persistence

def plot_persistence_diagram(persistence, title, save_name):
    """Plot persistence diagram"""
    
    if persistence is None or not GUDHI_AVAILABLE:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot for each dimension
    for dim in range(3):
        ax = axes[dim]
        
        # Extract points for this dimension
        points = [(birth, death) for (d, (birth, death)) in persistence 
                  if d == dim and death != float('inf')]
        
        if not points:
            ax.text(0.5, 0.5, f'No H{dim} features', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'H{dim} (Dimension {dim})', fontweight='bold')
            continue
        
        births, deaths = zip(*points)
        
        # Plot points
        ax.scatter(births, deaths, alpha=0.6, s=30)
        
        # Plot diagonal
        max_val = max(max(births), max(deaths))
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Birth = Death')
        
        ax.set_xlabel('Birth', fontsize=11)
        ax.set_ylabel('Death', fontsize=11)
        ax.set_title(f'H{dim} (Dimension {dim})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / f"{save_name}_persistence_diagram.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def compute_persistence_statistics(persistence):
    """Compute statistics from persistence diagram"""
    
    if persistence is None:
        return {}
    
    stats = {}
    
    for dim in range(3):
        # Extract points for this dimension
        points = [(birth, death) for (d, (birth, death)) in persistence 
                  if d == dim and death != float('inf')]
        
        if not points:
            stats[f'H{dim}_count'] = 0
            stats[f'H{dim}_avg_persistence'] = 0
            stats[f'H{dim}_max_persistence'] = 0
            continue
        
        births, deaths = zip(*points)
        persistences = [d - b for b, d in points]
        
        stats[f'H{dim}_count'] = len(points)
        stats[f'H{dim}_avg_persistence'] = np.mean(persistences)
        stats[f'H{dim}_max_persistence'] = np.max(persistences)
        stats[f'H{dim}_total_persistence'] = np.sum(persistences)
    
    return stats

def analyze_topology_by_overlap(weights_dict):
    """Analyze topological features grouped by overlap"""
    
    if not GUDHI_AVAILABLE:
        print("GUDHI not available - skipping topology analysis")
        return
    
    # Group by overlap
    overlap_groups = {0: {}, 1: {}, 2: {}}
    
    for exp_name, weights in weights_dict.items():
        if weights.size == 0:
            continue
        
        # Determine overlap
        if 'overlap0' in exp_name:
            overlap_id = 0
        elif 'overlap1' in exp_name:
            overlap_id = 1
        elif 'overlap2' in exp_name:
            overlap_id = 2
        else:
            continue
        
        overlap_groups[overlap_id][exp_name] = weights
    
    # Analyze each overlap
    all_stats = []
    
    for overlap_id, exp_dict in overlap_groups.items():
        print(f"\nAnalyzing Overlap {overlap_id} ({len(exp_dict)} experiments)...")
        
        for exp_name, weights in list(exp_dict.items())[:5]:  # Analyze first 5 per overlap
            print(f"  Processing {exp_name}...")
            
            # Flatten weights
            if weights.ndim > 2:
                weights_flat = weights.reshape(weights.shape[0], -1)
            else:
                weights_flat = weights
            
            # Normalize
            scaler = StandardScaler()
            weights_normalized = scaler.fit_transform(weights_flat)
            
            # Compute persistence
            try:
                persistence = compute_rips_persistence(weights_normalized, max_dimension=2)
                
                # Plot for first experiment in each overlap
                if exp_name == list(exp_dict.keys())[0]:
                    plot_persistence_diagram(
                        persistence,
                        f"Persistence Diagram - {exp_name}",
                        f"overlap{overlap_id}_example"
                    )
                
                # Compute statistics
                stats = compute_persistence_statistics(persistence)
                stats['experiment'] = exp_name
                stats['overlap'] = overlap_id
                all_stats.append(stats)
                
            except Exception as e:
                print(f"    Error: {e}")
    
    # Create DataFrame
    df_stats = pd.DataFrame(all_stats)
    
    # Save
    output_file = DATA_DIR / "topology_statistics.csv"
    df_stats.to_csv(output_file, index=False)
    print(f"\nSaved topology statistics: {output_file}")
    
    return df_stats

def plot_topology_comparison(df_stats):
    """Compare topological features across overlaps"""
    
    if df_stats is None or len(df_stats) == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot for each dimension
    for dim in range(3):
        # Feature count
        ax = axes[0, dim]
        df_stats.boxplot(column=f'H{dim}_count', by='overlap', ax=ax)
        ax.set_xlabel('Overlap', fontsize=11)
        ax.set_ylabel('Feature Count', fontsize=11)
        ax.set_title(f'H{dim} Feature Count', fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=0)
        
        # Average persistence
        ax = axes[1, dim]
        df_stats.boxplot(column=f'H{dim}_avg_persistence', by='overlap', ax=ax)
        ax.set_xlabel('Overlap', fontsize=11)
        ax.set_ylabel('Average Persistence', fontsize=11)
        ax.set_title(f'H{dim} Average Persistence', fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=0)
    
    plt.suptitle('Topological Features Comparison', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / "topology_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def create_betti_curves(weights_dict):
    """Create Betti curves showing evolution of topological features"""
    
    if not GUDHI_AVAILABLE:
        return
    
    # Select representative experiments
    representative_exps = {}
    for exp_name, weights in weights_dict.items():
        if 'overlap0_MSE' in exp_name:
            representative_exps['Overlap 0 - MSE'] = weights
        elif 'overlap1_MSE' in exp_name:
            representative_exps['Overlap 1 - MSE'] = weights
        elif 'overlap2_MSE' in exp_name:
            representative_exps['Overlap 2 - MSE'] = weights
        
        if len(representative_exps) >= 3:
            break
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (exp_name, weights) in enumerate(representative_exps.items()):
        if idx >= 3:
            break
        
        ax = axes[idx]
        
        # Flatten and normalize
        if weights.ndim > 2:
            weights_flat = weights.reshape(weights.shape[0], -1)
        else:
            weights_flat = weights
        
        scaler = StandardScaler()
        weights_normalized = scaler.fit_transform(weights_flat)
        
        # Subsample
        if len(weights_normalized) > 500:
            indices = np.random.choice(len(weights_normalized), 500, replace=False)
            weights_normalized = weights_normalized[indices]
        
        # Compute Rips complex
        rips_complex = gudhi.RipsComplex(points=weights_normalized, max_edge_length=3.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        simplex_tree.compute_persistence()
        
        # Get Betti numbers at different filtration values
        filtration_values = np.linspace(0, 2.0, 100)
        betti_0 = []
        betti_1 = []
        betti_2 = []
        
        for filt_val in filtration_values:
            # Count features alive at this filtration
            persistence = simplex_tree.persistence()
            
            b0 = sum(1 for (dim, (birth, death)) in persistence 
                    if dim == 0 and birth <= filt_val and (death > filt_val or death == float('inf')))
            b1 = sum(1 for (dim, (birth, death)) in persistence 
                    if dim == 1 and birth <= filt_val and (death > filt_val or death == float('inf')))
            b2 = sum(1 for (dim, (birth, death)) in persistence 
                    if dim == 2 and birth <= filt_val and (death > filt_val or death == float('inf')))
            
            betti_0.append(b0)
            betti_1.append(b1)
            betti_2.append(b2)
        
        # Plot Betti curves
        ax.plot(filtration_values, betti_0, label='β₀ (Components)', linewidth=2)
        ax.plot(filtration_values, betti_1, label='β₁ (Loops)', linewidth=2)
        ax.plot(filtration_values, betti_2, label='β₂ (Voids)', linewidth=2)
        
        ax.set_xlabel('Filtration Value', fontsize=11)
        ax.set_ylabel('Betti Number', fontsize=11)
        ax.set_title(exp_name, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Betti Curves - Topological Evolution', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / "betti_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    print("="*80)
    print("TOPOLOGICAL ANALYSIS FOR CVPR PAPER")
    print("Focus: Persistent Homology & Multiparameter Persistence")
    print("="*80)
    
    if not GUDHI_AVAILABLE:
        print("\n⚠ WARNING: GUDHI not available")
        print("Install with: conda install -c conda-forge gudhi")
        print("Topological analysis will be limited\n")
    
    # Load data
    print("\nLoading weight representations...")
    weights = load_data()
    print(f"Loaded {len(weights)} weight representations")
    
    # Analyze topology by overlap
    print("\n1. Computing persistent homology...")
    df_stats = analyze_topology_by_overlap(weights)
    
    if df_stats is not None and len(df_stats) > 0:
        # Create comparison plots
        print("\n2. Creating topology comparison plots...")
        plot_topology_comparison(df_stats)
        
        # Create Betti curves
        print("\n3. Creating Betti curves...")
        create_betti_curves(weights)
    
    print("\n" + "="*80)
    print("TOPOLOGICAL ANALYSIS COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
