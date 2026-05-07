"""
Data preprocessing pipeline for FCL
Creates train/val/test splits with varying class overlap (2, 1, 0)
Pre-saves TensorDatasets for faster training
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import json
import ast


class DataPreprocessor:
    """
    Preprocess and save training scenarios with different overlap levels
    """
    def __init__(self, 
                 df_path: str = "./data/Merged zoo.csv",
                 scenario_path: str = "./data/Scenario",
                 overlap_levels: List[int] = [2, 1, 0],
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 batch_size: int = 20,
                 batch_limit: int = 100):
        
        self.df_path = df_path
        self.scenario_path = Path(scenario_path)
        self.overlap_levels = overlap_levels
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.batch_limit = batch_limit
        
        # Load main dataframe
        print(f"Loading data from {df_path}...")
        self.df = pd.read_csv(df_path)
        print(f"Loaded {len(self.df)} samples")
        
        # Create scenario directory
        self.scenario_path.mkdir(parents=True, exist_ok=True)
    
    def parse_label(self, label_str: str) -> set:
        """Parse label string to set of classes"""
        try:
            if isinstance(label_str, str):
                return set(ast.literal_eval(label_str))
            else:
                return set(label_str)
        except:
            return set()
    
    def compute_overlap(self, label1: set, label2: set) -> int:
        """Compute number of overlapping classes"""
        return len(label1.intersection(label2))
    
    def create_pairs_with_overlap(self, overlap: int, epoch_key: int, activ_key: int) -> List[Tuple]:
        """
        Create pairs of CNN weights with specified class overlap
        
        Returns:
            List of (idx1, idx2, label1, label2, overlap) tuples
        """
        print(f"\nCreating pairs with overlap={overlap}, epoch={epoch_key}, activ={activ_key}")
        
        # Filter by epoch and activation
        activ_map = {0: "gelu", 1: "relu", 2: "silu", 3: "leakyrelu", 4: "sigmoid", 5: "tanh"}
        activ_name = activ_map.get(activ_key, "silu")
        
        df_filtered = self.df[
            (self.df['epochCNN'] == epoch_key) & 
            (self.df['ActivationCNN'] == activ_name)
        ].copy()
        
        print(f"  Filtered to {len(df_filtered)} samples")
        
        if len(df_filtered) == 0:
            print(f"  Warning: No samples found for epoch={epoch_key}, activ={activ_name}")
            return []
        
        # Parse labels
        df_filtered['label_set'] = df_filtered['label'].apply(self.parse_label)
        
        # Create pairs
        pairs = []
        indices = df_filtered.index.tolist()
        
        print(f"  Creating pairs...")
        for i, idx1 in enumerate(tqdm(indices)):
            label1 = df_filtered.loc[idx1, 'label_set']
            
            for idx2 in indices[i+1:]:
                label2 = df_filtered.loc[idx2, 'label_set']
                
                # Compute overlap
                current_overlap = self.compute_overlap(label1, label2)
                
                if current_overlap == overlap:
                    pairs.append((idx1, idx2, label1, label2, overlap))
                
                # Limit pairs to avoid memory issues
                if len(pairs) >= 10000:
                    break
            
            if len(pairs) >= 10000:
                break
        
        print(f"  Created {len(pairs)} pairs")
        return pairs
    
    def split_pairs(self, pairs: List[Tuple]) -> Tuple[List, List, List]:
        """Split pairs into train/val/test"""
        n_total = len(pairs)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        
        # Shuffle pairs
        np.random.shuffle(pairs)
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:]
        
        print(f"  Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
        
        return train_pairs, val_pairs, test_pairs
    
    def create_tensor_batches(self, pairs: List[Tuple], data_type: str) -> Dict:
        """
        Create batched tensors from pairs
        
        Returns:
            Dictionary with loaded_list, L_ACC_list, L_indexes_list
        """
        print(f"  Creating {data_type} batches...")
        
        loaded_list = []
        L_ACC_list = []
        L_indexes_list = []
        
        # Batch pairs
        n_batches = min(len(pairs) // self.batch_size, self.batch_limit)
        
        for batch_idx in tqdm(range(n_batches)):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_pairs = pairs[start_idx:end_idx]
            
            batch_stream1 = []
            batch_stream2 = []
            batch_target = []
            batch_acc = []
            batch_indexes = []
            
            for idx1, idx2, label1, label2, overlap in batch_pairs:
                # Get weight vectors
                row1 = self.df.loc[idx1]
                row2 = self.df.loc[idx2]
                
                # Extract weights (columns 5 to 2468 based on Cols definition in meta.ipynb)
                # Assuming weight columns start after metadata columns
                weight_cols = [col for col in self.df.columns if col.startswith('weight') or col.startswith('bias')]
                
                if len(weight_cols) == 0:
                    # Fallback: assume weights are in specific column range
                    weight_cols = self.df.columns[5:2469]
                
                weights1 = row1[weight_cols].values.astype(np.float32)
                weights2 = row2[weight_cols].values.astype(np.float32)
                
                # Target: union of classes (for now, use weights1 as placeholder)
                # In practice, you'd need the actual merged weights
                target_weights = weights1  # Placeholder
                
                batch_stream1.append(weights1)
                batch_stream2.append(weights2)
                batch_target.append(target_weights)
                
                # Accuracy metadata
                acc1 = row1.get('Accuracy task1', 0.0)
                acc2 = row2.get('Accuracy task2', 0.0)
                batch_acc.append([acc1, acc2, 0.0])  # Third value for merged accuracy
                
                batch_indexes.append([idx1, idx2])
            
            # Convert to tensors
            stream1_tensor = torch.tensor(np.array(batch_stream1), dtype=torch.float32)
            stream2_tensor = torch.tensor(np.array(batch_stream2), dtype=torch.float32)
            target_tensor = torch.tensor(np.array(batch_target), dtype=torch.float32)
            
            # Stack: [batch_size, 3, 2464]
            stacked = torch.stack([stream1_tensor, stream2_tensor, target_tensor], dim=1)
            
            loaded_list.append(stacked)
            L_ACC_list.append(batch_acc)
            L_indexes_list.append(batch_indexes)
        
        return {
            'loaded_list': loaded_list,
            'L_ACC_list': L_ACC_list,
            'L_indexes_list': L_indexes_list
        }
    
    def save_scenario(self, overlap: int, epoch_key: int, activ_key: int):
        """
        Create and save complete scenario for given parameters
        """
        print(f"\n{'='*60}")
        print(f"Processing scenario: overlap={overlap}, epoch={epoch_key}, activ={activ_key}")
        print(f"{'='*60}")
        
        # Create pairs
        pairs = self.create_pairs_with_overlap(overlap, epoch_key, activ_key)
        
        if len(pairs) == 0:
            print(f"Skipping scenario - no pairs found")
            return
        
        # Split pairs
        train_pairs, val_pairs, test_pairs = self.split_pairs(pairs)
        
        # Create scenario directory
        activ_map = {0: "gelu", 1: "relu", 2: "silu", 3: "leakyrelu", 4: "sigmoid", 5: "tanh"}
        scenario_dir = self.scenario_path / f"overlapping_m{overlap}_epoch{epoch_key}_activ{activ_key}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pairs as .npy for reference
        np.save(scenario_dir / "train_pairs.npy", np.array(train_pairs, dtype=object))
        np.save(scenario_dir / "val_pairs.npy", np.array(val_pairs, dtype=object))
        np.save(scenario_dir / "test_pairs.npy", np.array(test_pairs, dtype=object))
        
        # Create and save tensor batches
        for data_type, pairs_list in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
            if len(pairs_list) == 0:
                continue
            
            batch_data = self.create_tensor_batches(pairs_list, data_type)
            
            # Save as .pt file
            save_path = scenario_dir / f"{data_type}_batches.pt"
            torch.save(batch_data, save_path)
            print(f"  Saved {save_path}")
        
        # Save metadata
        metadata = {
            'overlap': overlap,
            'epoch_key': epoch_key,
            'activ_key': activ_key,
            'activ_name': activ_map.get(activ_key, "silu"),
            'n_train_pairs': len(train_pairs),
            'n_val_pairs': len(val_pairs),
            'n_test_pairs': len(test_pairs),
            'batch_size': self.batch_size,
            'batch_limit': self.batch_limit,
        }
        
        with open(scenario_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata to {scenario_dir / 'metadata.json'}")
        print(f"✓ Scenario complete!\n")
    
    def process_all_scenarios(self, epoch_keys: List[int] = [10], 
                             activ_keys: List[int] = [2]):
        """
        Process all combinations of overlap, epoch, and activation
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)
        print(f"Overlap levels: {self.overlap_levels}")
        print(f"Epoch keys: {epoch_keys}")
        print(f"Activation keys: {activ_keys}")
        print(f"Splits: train={self.train_split}, val={self.val_split}, test={self.test_split}")
        print(f"Batch size: {self.batch_size}, Batch limit: {self.batch_limit}")
        print("="*60)
        
        total_scenarios = len(self.overlap_levels) * len(epoch_keys) * len(activ_keys)
        current = 0
        
        for overlap in self.overlap_levels:
            for epoch_key in epoch_keys:
                for activ_key in activ_keys:
                    current += 1
                    print(f"\nScenario {current}/{total_scenarios}")
                    self.save_scenario(overlap, epoch_key, activ_key)
        
        print("\n" + "="*60)
        print("ALL SCENARIOS PROCESSED!")
        print("="*60)


def verify_scenario(scenario_path: str):
    """Verify a saved scenario"""
    scenario_path = Path(scenario_path)
    
    print(f"\nVerifying scenario: {scenario_path}")
    
    # Load metadata
    with open(scenario_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"  Metadata: {metadata}")
    
    # Check files
    for data_type in ['train', 'val', 'test']:
        batch_file = scenario_path / f"{data_type}_batches.pt"
        if batch_file.exists():
            data = torch.load(batch_file)
            print(f"  {data_type}: {len(data['loaded_list'])} batches")
            if len(data['loaded_list']) > 0:
                print(f"    First batch shape: {data['loaded_list'][0].shape}")
        else:
            print(f"  {data_type}: NOT FOUND")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        df_path="./data/Merged zoo.csv",
        scenario_path="./data/Scenario",
        overlap_levels=[2, 1, 0],
        batch_size=20,
        batch_limit=50  # Limit for testing
    )
    
    # Process scenarios for epoch 10, activation silu (key=2)
    preprocessor.process_all_scenarios(
        epoch_keys=[10],
        activ_keys=[2]
    )
