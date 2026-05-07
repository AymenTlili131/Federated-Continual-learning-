#!/usr/bin/env python3
"""
Robust Checkpoint Loader with Custom Unpickler
Handles checkpoints with missing module dependencies
"""

import sys
import torch
import pickle
import io
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Add paths
PROJECT_ROOT = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks_sandbox" / "core_modules"))

# Import transformer
from Double_input_transformer import TransformerAE

# Paths
EXPERIMENTS_DIR = PROJECT_ROOT / "notebooks_sandbox" / "experiments"
DATA_DIR = PROJECT_ROOT / "notebooks_sandbox" / "cvpr_analysis_scripts" / "data"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "Merged zoo.csv"

print("="*80)
print("ROBUST CHECKPOINT LOADING AND PREDICTION GENERATION")
print("="*80)

# Custom unpickler to handle missing modules
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle missing 'config' module
        if module == 'config':
            # Return a dummy class
            return type(name, (), {})
        # Handle other missing modules
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            # Return a dummy class for any missing module
            return type(name, (), {})

def load_checkpoint_robust(checkpoint_path):
    """Load checkpoint with robust error handling"""
    try:
        # Try standard loading first
        with open(checkpoint_path, 'rb') as f:
            checkpoint = CustomUnpickler(f).load()
        return checkpoint
    except Exception as e:
        print(f"  Custom unpickler failed: {e}")
        return None

# 1. Load ground truth
print("\n1. Loading ground truth...")
df_gt = pd.read_csv(GROUND_TRUTH_PATH)
weight_columns = df_gt.columns[17:-2].tolist()
all_weights = df_gt[weight_columns].values
print(f"Ground truth shape: {all_weights.shape}")

# Create test set
n_test = 1000
np.random.seed(42)
test_indices = np.random.choice(len(all_weights), n_test, replace=False)
x1_test = all_weights[test_indices]
x2_test = all_weights[np.random.choice(len(all_weights), n_test, replace=False)]
y_test = all_weights[test_indices]

print(f"Test set: {n_test} samples")

# 2. Find checkpoints
print("\n2. Finding checkpoints...")
checkpoint_info = []
for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue
    
    parts = exp_dir.name.split('_')
    if len(parts) < 3:
        continue
    
    model_size = parts[0]
    overlap = int(parts[1].replace('overlap', ''))
    loss_name = '_'.join(parts[2:])
    
    checkpoints_dir = exp_dir / "checkpoints"
    if checkpoints_dir.exists():
        for ckpt_file in checkpoints_dir.glob("best_model.pth"):  # Only use best models
            checkpoint_info.append({
                'experiment': exp_dir.name,
                'model_size': model_size,
                'overlap': overlap,
                'loss_name': loss_name,
                'checkpoint': ckpt_file.name,
                'path': ckpt_file
            })

print(f"Found {len(checkpoint_info)} best model checkpoints")

# 3. Load and predict
def generate_predictions(checkpoint_path, x1, x2, device='cpu'):
    """Generate predictions from checkpoint"""
    try:
        # Load checkpoint
        checkpoint = load_checkpoint_robust(checkpoint_path)
        if checkpoint is None:
            return None, None, None
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Infer config from state dict
        config = {
            'max_seq_len': 50,
            'N': 1,
            'heads': 1,
            'd_model': 960,
            'd_ff': 960,
            'neck': 512,
            'dropout': 0.1
        }
        
        # Try to infer actual values
        for key, tensor in state_dict.items():
            if 'enc1.embed.neuron_l1.weight' in key:
                config['d_model'] = tensor.shape[0]
            elif 'vec2neck.weight' in key:
                config['neck'] = tensor.shape[0]
        
        # Create and load model
        model = TransformerAE(**config)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        # Generate predictions
        batch_size = 100
        all_predictions = []
        all_necks = []
        
        for i in range(0, len(x1), batch_size):
            batch_x1 = torch.from_numpy(x1[i:i+batch_size]).float().to(device)
            batch_x2 = torch.from_numpy(x2[i:i+batch_size]).float().to(device)
            
            with torch.no_grad():
                pred, neck, _, _, _ = model(batch_x1, batch_x2)
                all_predictions.append(pred.cpu().numpy())
                all_necks.append(neck.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        necks = np.vstack(all_necks)
        
        return predictions, necks, config
        
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
        return None, None, None

# 4. Generate predictions
print("\n3. Generating predictions...")
predictions_data = []
failed = []

for ckpt_info in tqdm(checkpoint_info, desc="Processing"):
    predictions, necks, config = generate_predictions(ckpt_info['path'], x1_test, x2_test)
    
    if predictions is not None:
        # Save to disk
        pred_file = DATA_DIR / f"pred_{ckpt_info['experiment']}.npz"
        np.savez_compressed(
            pred_file,
            predictions=predictions,
            necks=necks,
            x1=x1_test,
            x2=x2_test,
            y=y_test
        )
        
        # Compute metrics
        mse = float(np.mean((predictions - y_test)**2))
        mae = float(np.mean(np.abs(predictions - y_test)))
        
        predictions_data.append({
            **ckpt_info,
            'path': str(ckpt_info['path']),
            'predictions_file': str(pred_file),
            'mse': mse,
            'mae': mae,
            'config': config
        })
    else:
        failed.append(ckpt_info)

print(f"\nSuccess: {len(predictions_data)}/{len(checkpoint_info)}")
print(f"Failed: {len(failed)}")

# 5. Save metadata
if predictions_data:
    metadata = {
        'total': len(checkpoint_info),
        'successful': len(predictions_data),
        'failed': len(failed),
        'test_samples': n_test,
        'predictions': predictions_data
    }
    
    metadata_file = DATA_DIR / "predictions_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Create summary
    df_summary = pd.DataFrame(predictions_data)
    
    print("\nTop 10 by MSE:")
    top10 = df_summary.nsmallest(10, 'mse')[['loss_name', 'overlap', 'mse', 'mae']]
    print(top10.to_string())
    
    summary_file = DATA_DIR / "predictions_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSaved: {summary_file}")
    print(f"Saved: {metadata_file}")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
