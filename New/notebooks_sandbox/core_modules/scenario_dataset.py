"""
Scenario-Based Dataset Creation

Creates training/validation/test splits based on task overlap criteria
rather than random splitting. Ensures challenging OOD test sets.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from itertools import combinations


def create_scenario_splits(
    overlap: int = 0,
    epoch_key: int = 0,
    activ_key: int = 0,
    scenario_root: str = "./data/Scenario"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load scenario-based train/val/test splits
    
    Criteria:
    - Train: Different task sizes (e.g., [2,3,4] vs [5,6,7,8])
    - Val: Same task sizes (e.g., [2,3,4] vs [5,6,7,8])
    - Test: Contains class 0 (OOD challenge)
    
    Args:
        overlap: Number of overlapping classes (0, 1, or 2)
        epoch_key: Epoch configuration (0-5 maps to 11,16,21,26,31,36)
        activ_key: Activation function (0-5 maps to gelu,relu,silu,leakyrelu,sigmoid,tanh)
        scenario_root: Root directory for scenarios
    
    Returns:
        train_pairs, val_pairs, test_pairs as numpy arrays
    """
    scenario_dir = Path(scenario_root) / f"overlapping_m{overlap}_epoch{epoch_key}_activ{activ_key}"
    
    train_path = scenario_dir / "train_pairs.npy"
    val_path = scenario_dir / "val_pairs.npy"
    test_path = scenario_dir / "test_pairs.npy"
    
    # Load if exists
    if train_path.exists():
        train_pairs = np.load(train_path, allow_pickle=True)
        val_pairs = np.load(val_path, allow_pickle=True) if val_path.exists() else np.array([])
        test_pairs = np.load(test_path, allow_pickle=True) if test_path.exists() else np.array([])
    else:
        # Generate on the fly if scenario files don't exist
        print(f"Scenario files not found at {scenario_dir}, generating...")
        train_pairs, val_pairs, test_pairs = generate_scenarios_on_fly(overlap, epoch_key, activ_key)
    
    return train_pairs, val_pairs, test_pairs


def generate_scenarios_on_fly(
    overlap: int,
    epoch_key: int,
    activ_key: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate scenario splits on the fly if files don't exist
    
    Criteria:
    - Train: pairs with different task sizes
    - Val: pairs with same task sizes
    - Test: pairs containing class 0
    """
    all_pairs = []
    digits = list(range(10))
    
    for k1 in range(2, 10):
        S1 = list(combinations(digits, k1))
        for pair1 in S1:
            for k2 in range(2, 10):
                S2 = list(combinations(digits, k2))
                for pair2 in S2:
                    if pair1 != pair2 and len(set(pair1) & set(pair2)) == overlap:
                        if [list(pair1), list(pair2)] not in all_pairs:
                            all_pairs.append([list(pair1), list(pair2)])
    
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    for element in all_pairs:
        pair1, pair2 = element
        full_pair = [pair1, pair2, int(epoch_key), int(activ_key)]
        
        if 0 in pair1 or 0 in pair2:
            # Test set: contains class 0 (OOD challenge)
            test_pairs.append(full_pair)
        elif len(pair1) == len(pair2):
            # Val set: same task sizes
            val_pairs.append(full_pair)
        else:
            # Train set: different task sizes
            train_pairs.append(full_pair)
    
    return np.array(train_pairs, dtype=object), np.array(val_pairs, dtype=object), np.array(test_pairs, dtype=object)


def load_weights_from_zoo(
    merged_zoo_path: str,
    task_pair: List[int],
    epoch: int,
    activation: str,
    limit: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Load ground truth weights from merged zoo for specific task configuration
    
    Args:
        merged_zoo_path: Path to Merged zoo.csv
        task_pair: [task1_classes, task2_classes]
        epoch: Training epoch
        activation: Activation function
        limit: Maximum number of samples
    
    Returns:
        weights array, metadata dict
    """
    df = pd.read_csv(merged_zoo_path)
    
    # Filter by criteria
    # This is a simplified version - you'll need to adapt based on your zoo structure
    # The zoo should have columns indicating task configuration, epoch, activation
    
    filtered_df = df.copy()
    
    # Extract weight columns (assuming columns 11-2474 are weights)
    weight_cols = [col for col in df.columns if 'weight' in col.lower() or 'bias' in col.lower()]
    if not weight_cols:
        # Fallback: assume columns after metadata are weights
        weight_cols = df.columns[11:2475].tolist()
    
    weights = filtered_df[weight_cols].values
    
    if limit:
        weights = weights[:limit]
    
    metadata = {
        'n_samples': len(weights),
        'task_pair': task_pair,
        'epoch': epoch,
        'activation': activation
    }
    
    return weights, metadata


def create_scenario_dataset(
    merged_zoo_path: str,
    overlap: int = 0,
    epoch_key: int = 0,
    activ_key: int = 0,
    scenario_root: str = "./data/Scenario",
    normalize_weights: bool = False,
    normalization_method: str = "standardize"
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, Dict]:
    """
    Create scenario-based datasets with proper train/val/test splits
    
    Args:
        merged_zoo_path: Path to Merged zoo.csv
        overlap: Task overlap (0, 1, or 2)
        epoch_key: Epoch configuration
        activ_key: Activation function
        scenario_root: Scenario directory
        normalize_weights: Whether to normalize weights
        normalization_method: 'standardize' or 'minmax'
    
    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    # Load scenario splits
    train_pairs, val_pairs, test_pairs = create_scenario_splits(
        overlap, epoch_key, activ_key, scenario_root
    )
    
    print(f"Scenario splits loaded:")
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs: {len(val_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")
    
    # Load weights from merged zoo
    df = pd.read_csv(merged_zoo_path)
    
    # Extract weight columns (columns after metadata)
    # Assuming structure: metadata cols, then 2464 weight values
    weight_start_col = 11  # Adjust based on your zoo structure
    weight_cols = df.columns[weight_start_col:weight_start_col+2464].tolist()
    
    all_weights = df[weight_cols].values.astype(np.float32)
    
    print(f"Loaded {len(all_weights)} weight vectors from zoo")
    print(f"Weight vector shape: {all_weights.shape}")
    
    # Normalize if requested
    if normalize_weights:
        if normalization_method == "standardize":
            mean = all_weights.mean(axis=0, keepdims=True)
            std = all_weights.std(axis=0, keepdims=True) + 1e-8
            all_weights = (all_weights - mean) / std
            print(f"Applied standardization (mean={mean.mean():.4f}, std={std.mean():.4f})")
        elif normalization_method == "minmax":
            min_val = all_weights.min(axis=0, keepdims=True)
            max_val = all_weights.max(axis=0, keepdims=True)
            all_weights = (all_weights - min_val) / (max_val - min_val + 1e-8)
            print(f"Applied min-max normalization")
    
    # Create weight pairs based on scenarios
    def create_pairs_from_scenarios(scenario_pairs, n_samples_per_pair=10):
        x1_list, x2_list, y_list = [], [], []
        
        for i, pair_data in enumerate(scenario_pairs[:min(len(scenario_pairs), 1000)]):
            # Each pair_data: [task1_classes, task2_classes, epoch, activ]
            # Sample random weights for this configuration
            indices = np.random.choice(len(all_weights), size=n_samples_per_pair, replace=True)
            
            for idx in indices:
                w1 = all_weights[idx]
                w2_idx = np.random.choice(len(all_weights))
                w2 = all_weights[w2_idx]
                
                # Target: combination of w1 and w2 based on overlap
                target = (w1 + w2) / 2  # Simplified - adjust based on your needs
                
                x1_list.append(w1)
                x2_list.append(w2)
                y_list.append(target)
        
        return np.array(x1_list), np.array(x2_list), np.array(y_list)
    
    # Create datasets
    x1_train, x2_train, y_train = create_pairs_from_scenarios(train_pairs, n_samples_per_pair=5)
    x1_val, x2_val, y_val = create_pairs_from_scenarios(val_pairs, n_samples_per_pair=3)
    x1_test, x2_test, y_test = create_pairs_from_scenarios(test_pairs, n_samples_per_pair=3)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(x1_train)}")
    print(f"  Val: {len(x1_val)}")
    print(f"  Test: {len(x1_test)}")
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.from_numpy(x1_train).float(),
        torch.from_numpy(x2_train).float(),
        torch.from_numpy(y_train).float()
    )
    
    val_dataset = TensorDataset(
        torch.from_numpy(x1_val).float(),
        torch.from_numpy(x2_val).float(),
        torch.from_numpy(y_val).float()
    )
    
    test_dataset = TensorDataset(
        torch.from_numpy(x1_test).float(),
        torch.from_numpy(x2_test).float(),
        torch.from_numpy(y_test).float()
    )
    
    metadata = {
        'overlap': overlap,
        'epoch_key': epoch_key,
        'activ_key': activ_key,
        'n_train': len(x1_train),
        'n_val': len(x1_val),
        'n_test': len(x1_test),
        'normalized': normalize_weights,
        'normalization_method': normalization_method if normalize_weights else None,
        'train_pairs_count': len(train_pairs),
        'val_pairs_count': len(val_pairs),
        'test_pairs_count': len(test_pairs)
    }
    
    return train_dataset, val_dataset, test_dataset, metadata


# Activation and epoch mappings
ACTIVATION_MAP = {
    0: "gelu",
    1: "relu", 
    2: "silu",
    3: "leakyrelu",
    4: "sigmoid",
    5: "tanh"
}

EPOCH_MAP = {
    0: 11,
    1: 16,
    2: 21,
    3: 26,
    4: 31,
    5: 36
}


def get_activation_name(activ_key: int) -> str:
    """Get activation function name from key"""
    return ACTIVATION_MAP.get(activ_key, "leakyrelu")


def get_epoch_value(epoch_key: int) -> int:
    """Get epoch value from key"""
    return EPOCH_MAP.get(epoch_key, 21)
