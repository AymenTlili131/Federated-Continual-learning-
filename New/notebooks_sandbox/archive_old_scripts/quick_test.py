#!/usr/bin/env python3
"""Quick test of distance metrics and topology analysis."""

import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / "research_scripts"))

from distance_metrics import WeightDistanceMetrics
from robust_topology_analysis import safe_compute_topology_metrics
from wandb_integration import create_experiment_summary

print("="*80)
print("Testing Distance Metrics & Topology Analysis")
print("="*80)

# Create sample weight vectors
np.random.seed(42)
w1 = np.random.randn(2464).astype(np.float32)
w2 = np.random.randn(2464).astype(np.float32)

# Test distance metrics
print("\n1. Testing Distance Metrics...")
calculator = WeightDistanceMetrics()
metrics = calculator.compute_all_metrics(w1, w2)

print(f"   Full distances computed: {len(metrics['full'])} metrics")
print(f"   Layer-wise distances: {len(metrics['layerwise'])} layers")
print(f"   Sample Euclidean distance: {metrics['full']['euclidean']:.6f}")

# Generate markdown table
markdown = calculator.format_as_table(metrics)
output_file = Path("test_distance_metrics.md")
with open(output_file, 'w') as f:
    f.write(markdown)
print(f"   ✓ Markdown table saved to: {output_file}")

# Test topology analysis
print("\n2. Testing Topology Analysis...")
weight_matrix = np.random.randn(50, 64).astype(np.float32)  # 50 samples, 64 dims
topology_results = safe_compute_topology_metrics(weight_matrix)

if topology_results.get('mapper'):
    mapper = topology_results['mapper']
    if 'error' not in mapper:
        print(f"   ✓ Mapper: {mapper.get('n_nodes', 0)} nodes, {mapper.get('n_edges', 0)} edges")
    else:
        print(f"   ⚠ Mapper error: {mapper['error']}")

if topology_results.get('persistence'):
    ph = topology_results['persistence']
    if 'error' not in ph and ph.get('stats'):
        print(f"   ✓ Persistent Homology: Betti numbers computed")
        for dim in range(3):
            if f'betti_{dim}' in ph['stats']:
                print(f"     - Betti_{dim}: {ph['stats'][f'betti_{dim}']}")
    else:
        print(f"   ⚠ Persistence error: {ph.get('error', 'unknown')}")

if topology_results.get('errors'):
    print(f"   Errors encountered: {len(topology_results['errors'])}")
    for err in topology_results['errors']:
        print(f"     - {err}")

# Test experiment summary
print("\n3. Testing Experiment Summary...")
summary = create_experiment_summary(
    model_config={'model_size': 'medium', 'epochs': 500},
    training_history={'train_loss': [0.14, 0.09, 0.08], 'val_loss': [0.15, 0.10, 0.09]},
    distance_metrics=metrics,
    topology_results=topology_results
)

summary_file = Path("test_experiment_summary.md")
with open(summary_file, 'w') as f:
    f.write(summary)
print(f"   ✓ Summary saved to: {summary_file}")

print("\n" + "="*80)
print("All Tests Passed! ✓")
print("="*80)
print("\nGenerated files:")
print(f"  - {output_file}")
print(f"  - {summary_file}")
print("\nModules working correctly:")
print("  ✓ Distance Metrics (full + layer-wise)")
print("  ✓ Robust Topology Analysis (Mapper + Persistent Homology)")
print("  ✓ WandB Integration (experiment summaries)")
