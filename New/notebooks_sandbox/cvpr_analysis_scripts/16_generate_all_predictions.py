#!/usr/bin/env python3
"""
Generate Predictions from All Tournament Checkpoints
Now that checkpoint loading is fixed, generate predictions from all 54 checkpoints
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths and imports
PROJECT_ROOT = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New")
CORE_MODULES = PROJECT_ROOT / "notebooks_sandbox" / "core_modules"
sys.path.insert(0, str(CORE_MODULES))
sys.path.insert(0, str(PROJECT_ROOT))

# Import config first
from config import MODEL_CONFIGS

# Import other modules
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from Double_input_transformer import TransformerAE

# Paths
EXPERIMENTS_DIR = PROJECT_ROOT / "notebooks_sandbox" / "experiments"
DATA_DIR = PROJECT_ROOT / "notebooks_sandbox" / "cvpr_analysis_scripts" / "data"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "Merged zoo.csv"

print("="*80)
print("GENERATING PREDICTIONS FROM ALL TOURNAMENT CHECKPOINTS")
print("="*80)

# 1. Load ground truth and create test set
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

# 2. Find all checkpoints
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
        for ckpt_file in checkpoints_dir.glob("best_model.pth"):
            checkpoint_info.append({
                'experiment': exp_dir.name,
                'model_size': model_size,
                'overlap': overlap,
                'loss_name': loss_name,
                'checkpoint': ckpt_file.name,
                'path': ckpt_file
            })

print(f"Found {len(checkpoint_info)} checkpoints")

# 3. Generate predictions
def generate_predictions(checkpoint_path, x1, x2, device='cpu'):
    """Generate predictions from checkpoint"""
    try:
        # Force CPU device and set as default
        device = 'cpu'
        torch.set_default_device('cpu')
        
        # Load checkpoint - force map_location to CPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get config
        if 'config' in checkpoint:
            config_obj = checkpoint['config']
            # Convert ModelConfig object to dict
            config = {
                'max_seq_len': getattr(config_obj, 'max_seq_len', 50),
                'N': getattr(config_obj, 'N', 1),
                'heads': getattr(config_obj, 'heads', 1),
                'd_model': getattr(config_obj, 'd_model', 960),
                'd_ff': getattr(config_obj, 'd_ff', 960),
                'neck': getattr(config_obj, 'neck', 512),
                'dropout': getattr(config_obj, 'dropout', 0.1)
            }
        else:
            # Default config
            config = {
                'max_seq_len': 50,
                'N': 1,
                'heads': 1,
                'd_model': 960,
                'd_ff': 960,
                'neck': 512,
                'dropout': 0.1
            }
        
        # Create model with CPU as default device
        with torch.device('cpu'):
            model = TransformerAE(**config)
        
        # Load state dict - map all tensors to CPU
        state_dict = checkpoint['model_state_dict']
        # Create a new state dict with all tensors on CPU
        cpu_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                cpu_state_dict[k] = v.cpu()
            else:
                cpu_state_dict[k] = v
        
        model.load_state_dict(cpu_state_dict, strict=False)
        model.eval()
        model = model.cpu()
        
        # Generate predictions in batches
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
        print(f"  Error: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return None, None, None

print("\n3. Generating predictions for all checkpoints...")
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
        corr = float(np.corrcoef(predictions.flatten(), y_test.flatten())[0, 1])
        
        predictions_data.append({
            **ckpt_info,
            'path': str(ckpt_info['path']),
            'predictions_file': str(pred_file),
            'mse': mse,
            'mae': mae,
            'correlation': corr,
            'config': config
        })
    else:
        failed.append(ckpt_info)

print(f"\n✓ Success: {len(predictions_data)}/{len(checkpoint_info)}")
print(f"✗ Failed: {len(failed)}")

# 4. Save metadata and summary
if predictions_data:
    metadata = {
        'total': len(checkpoint_info),
        'successful': len(predictions_data),
        'failed': len(failed),
        'test_samples': n_test,
        'predictions': predictions_data
    }
    
    metadata_file = DATA_DIR / "checkpoint_predictions_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Create summary
    df_summary = pd.DataFrame(predictions_data)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nTop 10 by MSE:")
    top10 = df_summary.nsmallest(10, 'mse')[['loss_name', 'overlap', 'mse', 'mae', 'correlation']]
    print(top10.to_string(index=False))
    
    print("\nTop 10 by Correlation:")
    top10_corr = df_summary.nlargest(10, 'correlation')[['loss_name', 'overlap', 'mse', 'mae', 'correlation']]
    print(top10_corr.to_string(index=False))
    
    print("\nPerformance by Overlap:")
    overlap_summary = df_summary.groupby('overlap')[['mse', 'mae', 'correlation']].agg(['mean', 'std'])
    print(overlap_summary)
    
    summary_file = DATA_DIR / "checkpoint_predictions_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    
    print(f"\n✓ Saved: {metadata_file}")
    print(f"✓ Saved: {summary_file}")
    print(f"✓ Saved {len(predictions_data)} prediction files to {DATA_DIR}")

print("\n" + "="*80)
print("PREDICTION GENERATION COMPLETE")
print("="*80)
