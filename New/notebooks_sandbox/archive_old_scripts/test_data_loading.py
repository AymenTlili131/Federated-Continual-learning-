#!/usr/bin/env python3
"""
Quick diagnostic to test data loading and verify PyTorch differentiable losses
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Test 1: Check CSV structure
print("=" * 80)
print("TEST 1: CSV Structure")
print("=" * 80)

csv_path = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/Merged zoo.csv")
df = pd.read_csv(csv_path)

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns[:20])}")  # First 20 columns
print(f"\nColumn 'label' exists: {'label' in df.columns}")
print(f"Column 'epoch' exists: {'epoch' in df.columns}")
print(f"Column 'leakyrelu' exists: {'leakyrelu' in df.columns}")

# Check a sample row
print(f"\nSample row:")
print(f"  label: {df.iloc[0]['label']}")
print(f"  epoch: {df.iloc[0]['epoch']}")
print(f"  leakyrelu: {df.iloc[0]['leakyrelu']}")

# Test 2: Check scenario pairs
print("\n" + "=" * 80)
print("TEST 2: Scenario Pairs")
print("=" * 80)

scenario_dir = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/Scenario/overlapping_m0")
train_pairs_file = scenario_dir / "train_pairs.npy"

if train_pairs_file.exists():
    train_pairs = np.load(train_pairs_file, allow_pickle=True)
    print(f"Train pairs loaded: {len(train_pairs)}")
    print(f"Sample pair: {train_pairs[0]}")
    
    # Test matching
    task1, task2 = train_pairs[0]
    task_combined = sorted(set(task1) | set(task2))
    
    task1_str = str(task1)
    task2_str = str(task2)
    task_combined_str = str(task_combined)
    
    print(f"\nTask 1: {task1_str}")
    print(f"Task 2: {task2_str}")
    print(f"Combined: {task_combined_str}")
    
    # Try to find in CSV
    mask1 = (df['label'] == task1_str) & (df['epoch'] == 21) & (df['leakyrelu'] == 1.0)
    mask2 = (df['label'] == task2_str) & (df['epoch'] == 21) & (df['leakyrelu'] == 1.0)
    mask_combined = (df['label'] == task_combined_str) & (df['epoch'] == 21) & (df['leakyrelu'] == 1.0)
    
    print(f"\nMatches found:")
    print(f"  Task 1: {mask1.sum()}")
    print(f"  Task 2: {mask2.sum()}")
    print(f"  Combined: {mask_combined.sum()}")
    
    if mask1.sum() == 0:
        # Try different label formats
        print(f"\nTrying alternative label formats for task1...")
        print(f"  Unique labels in CSV (first 10): {df['label'].unique()[:10]}")
else:
    print(f"ERROR: Scenario file not found: {train_pairs_file}")

# Test 3: Check losses are PyTorch differentiable
print("\n" + "=" * 80)
print("TEST 3: Loss Functions (PyTorch Differentiable)")
print("=" * 80)

import torch
import sys
sys.path.insert(0, str(Path(__file__).parent))

from advanced_losses import HierarchicalLossRegistry

registry = HierarchicalLossRegistry()
loss_names = registry.list_losses()

print(f"Total losses: {len(loss_names)}")

# Test each loss
print("\nTesting losses with dummy tensors...")
x = torch.randn(4, 100, requires_grad=True)
y = torch.randn(4, 100)

failed_losses = []
for loss_name in loss_names[:10]:  # Test first 10
    try:
        loss_fn = registry.get_loss(loss_name)
        loss = loss_fn(x, y)
        
        # Check if differentiable
        if loss.requires_grad:
            loss.backward()
            print(f"  ✓ {loss_name}: differentiable")
        else:
            print(f"  ✗ {loss_name}: NOT differentiable")
            failed_losses.append(loss_name)
        
        # Reset gradients
        x.grad = None
        
    except Exception as e:
        print(f"  ✗ {loss_name}: ERROR - {e}")
        failed_losses.append(loss_name)

if failed_losses:
    print(f"\n⚠️  WARNING: {len(failed_losses)} losses failed:")
    for name in failed_losses:
        print(f"    - {name}")
else:
    print(f"\n✓ All tested losses are PyTorch differentiable!")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
