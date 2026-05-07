"""
Data loading utilities for the research pipeline.
Integrates with existing FCL data infrastructure.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_merged_zoo(
    data_path: Path,
    limit_samples: Optional[int] = None,
    activation_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Load the merged zoo CSV file.
    
    Args:
        data_path: Path to Merged zoo.csv
        limit_samples: Optional limit on number of samples
        activation_filter: Optional filter by activation function
    
    Returns:
        DataFrame with weight data
    """
    print(f"Loading merged zoo from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Merged zoo not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows")
    
    # Filter by activation if specified
    if activation_filter and 'activation' in df.columns:
        df = df[df['activation'] == activation_filter]
        print(f"  Filtered to {len(df)} rows with activation={activation_filter}")
    
    # Limit samples if specified
    if limit_samples and len(df) > limit_samples:
        df = df.sample(n=limit_samples, random_state=42)
        print(f"  Limited to {limit_samples} samples")
    
    return df


def extract_weight_columns(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract weight columns and metadata from dataframe.
    
    Args:
        df: Merged zoo dataframe
    
    Returns:
        weights: Weight matrix (n_samples, n_features)
        weight_cols: List of weight column names
        metadata_cols: List of metadata column names
    """
    # Identify weight columns (numeric columns that aren't metadata)
    metadata_keywords = [
        'label', 'activation', 'epoch', 'accuracy', 'loss',
        'overlap', 'init', 'scenario', 'timestamp'
    ]
    
    weight_cols = []
    metadata_cols = []
    
    for col in df.columns:
        is_metadata = any(keyword in col.lower() for keyword in metadata_keywords)
        if is_metadata or df[col].dtype == 'object':
            metadata_cols.append(col)
        else:
            weight_cols.append(col)
    
    print(f"  Found {len(weight_cols)} weight columns")
    print(f"  Found {len(metadata_cols)} metadata columns")
    
    weights = df[weight_cols].values.astype(np.float32)
    
    # TransformerAE expects exactly 2464 features (26*80 + 24*16)
    if weights.shape[1] != 2464:
        print(f"  Warning: Expected 2464 features, got {weights.shape[1]}")
        if weights.shape[1] > 2464:
            print(f"  Truncating to first 2464 features")
            weights = weights[:, :2464]
            weight_cols = weight_cols[:2464]
        else:
            print(f"  Padding with zeros to 2464 features")
            padding = np.zeros((weights.shape[0], 2464 - weights.shape[1]), dtype=np.float32)
            weights = np.concatenate([weights, padding], axis=1)
    
    return weights, weight_cols, metadata_cols


def create_weight_pairs(
    weights: np.ndarray,
    labels: np.ndarray,
    overlap: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create pairs of weights for transformer input.
    
    Args:
        weights: Weight matrix (n_samples, n_features)
        labels: Labels for each sample
        overlap: Number of overlapping classes (0, 1, or 2)
        random_state: Random seed
    
    Returns:
        x1: First weight in pair (n_pairs, n_features)
        x2: Second weight in pair (n_pairs, n_features)
        y: Target merged weights (n_pairs, n_features)
    """
    np.random.seed(random_state)
    n_samples = len(weights)
    
    # For simplicity, create random pairs
    # In production, you'd use overlap criteria
    n_pairs = min(n_samples // 2, 10000)
    
    indices = np.random.permutation(n_samples)
    idx1 = indices[:n_pairs]
    idx2 = indices[n_pairs:2*n_pairs]
    
    x1 = weights[idx1]
    x2 = weights[idx2]
    
    # Target is average (simple merging strategy)
    y = (x1 + x2) / 2
    
    return x1, x2, y


def ensure_ood_test_set(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    min_class_distance: int = 2,
    max_overlap: float = 0.3
) -> bool:
    """
    Check if test set is sufficiently out-of-distribution.
    
    Args:
        train_labels: Training set labels
        test_labels: Test set labels
        min_class_distance: Minimum number of different classes
        max_overlap: Maximum allowed overlap ratio
    
    Returns:
        is_ood: True if test set is sufficiently OOD
    """
    # Convert labels to sets for comparison
    train_set = set(train_labels)
    test_set = set(test_labels)
    
    # Calculate overlap
    overlap = len(train_set & test_set)
    overlap_ratio = overlap / len(test_set) if len(test_set) > 0 else 1.0
    
    # Check criteria
    is_ood = overlap_ratio <= max_overlap
    
    return is_ood


def create_dataloaders(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        x1: First weight in pair
        x2: Second weight in pair
        y: Target weights
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
    
    Returns:
        train_loader, val_loader, test_loader
    """
    n_samples = len(x1)
    
    # Calculate split sizes
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_test - n_val
    
    # Create indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Convert to tensors
    x1_tensor = torch.from_numpy(x1).float()
    x2_tensor = torch.from_numpy(x2).float()
    y_tensor = torch.from_numpy(y).float()
    
    # Create datasets
    train_dataset = TensorDataset(
        x1_tensor[train_idx],
        x2_tensor[train_idx],
        y_tensor[train_idx]
    )
    val_dataset = TensorDataset(
        x1_tensor[val_idx],
        x2_tensor[val_idx],
        y_tensor[val_idx]
    )
    test_dataset = TensorDataset(
        x1_tensor[test_idx],
        x2_tensor[test_idx],
        y_tensor[test_idx]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples ({n_train/n_samples*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({n_val/n_samples*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({n_test/n_samples*100:.1f}%)")
    
    return train_loader, val_loader, test_loader


def prepare_data_for_pipeline(
    data_dir: Path,
    batch_size: int = 32,
    overlap: int = 2,
    activation_filter: Optional[str] = None,
    limit_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Complete data preparation pipeline.
    
    Args:
        data_dir: Directory containing Merged zoo.csv
        batch_size: Batch size for dataloaders
        overlap: Overlap configuration
        activation_filter: Optional activation filter
        limit_samples: Optional sample limit
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    print(f"\n{'='*80}")
    print(f"Data Preparation Pipeline")
    print(f"{'='*80}\n")
    
    # Load merged zoo
    merged_zoo_path = data_dir / "Merged zoo.csv"
    df = load_merged_zoo(merged_zoo_path, limit_samples, activation_filter)
    
    # Extract weights and metadata
    weights, weight_cols, metadata_cols = extract_weight_columns(df)
    
    # Extract labels if available
    if 'label' in df.columns:
        labels = df['label'].values
    else:
        labels = np.zeros(len(df))
    
    # Create weight pairs
    print(f"\nCreating weight pairs (overlap={overlap})...")
    x1, x2, y = create_weight_pairs(weights, labels, overlap)
    
    # Create dataloaders
    print(f"\nCreating dataloaders (batch_size={batch_size})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        x1, x2, y, batch_size=batch_size
    )
    
    # Metadata
    metadata = {
        'n_features': weights.shape[1],
        'n_samples': len(weights),
        'n_pairs': len(x1),
        'weight_cols': weight_cols,
        'metadata_cols': metadata_cols,
        'overlap': overlap,
        'activation_filter': activation_filter
    }
    
    print(f"\n{'='*80}")
    print(f"Data preparation complete!")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Pairs: {metadata['n_pairs']}")
    print(f"{'='*80}\n")
    
    return train_loader, val_loader, test_loader, metadata
