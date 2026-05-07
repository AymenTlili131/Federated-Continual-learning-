#!/usr/bin/env python3
"""
Generate Predicted Weights from All Checkpoints
Robust checkpoint loading and prediction generation
"""

import sys
import torch
import torch.nn as nn
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

# Import transformer
from Double_input_transformer import TransformerAE

# Paths
EXPERIMENTS_DIR = PROJECT_ROOT / "notebooks_sandbox" / "experiments"
DATA_DIR = PROJECT_ROOT / "notebooks_sandbox" / "cvpr_analysis_scripts" / "data"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "Merged zoo.csv"

print("="*80)
print("GENERATING PREDICTIONS FROM CHECKPOINTS")
print("="*80)

# 1. Load ground truth and create test set
print("\n1. Loading ground truth and creating test set...")
df_gt = pd.read_csv(GROUND_TRUTH_PATH)
weight_columns = df_gt.columns[17:-2].tolist()
all_weights = df_gt[weight_columns].values
print(f"Ground truth shape: {all_weights.shape}")

# Create test set
n_test = 1000
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
    
    # Parse experiment name
    parts = exp_dir.name.split('_')
    if len(parts) < 3:
        continue
    
    model_size = parts[0]
    overlap = int(parts[1].replace('overlap', ''))
    loss_name = '_'.join(parts[2:])
    
    checkpoints_dir = exp_dir / "checkpoints"
    if checkpoints_dir.exists():
        for ckpt_file in checkpoints_dir.glob("*.pth"):
            checkpoint_info.append({
                'experiment': exp_dir.name,
                'model_size': model_size,
                'overlap': overlap,
                'loss_name': loss_name,
                'checkpoint': ckpt_file.name,
                'path': ckpt_file
            })

print(f"Found {len(checkpoint_info)} checkpoints across {len(set([c['experiment'] for c in checkpoint_info]))} experiments")

# 3. Load checkpoint and generate predictions
def load_and_predict(checkpoint_path, x1, x2, device='cpu'):
    """Load checkpoint and generate predictions with robust error handling"""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract config
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            config = checkpoint['config']
        else:
            # Infer config from state dict
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            config = {
                'max_seq_len': 50,
                'N': 1,
                'heads': 1,
                'd_model': 960,
                'd_ff': 960,
                'neck': 512,
                'dropout': 0.1
            }
            
            # Try to infer from state dict keys
            for key, tensor in state_dict.items():
                if 'enc1.embed.neuron_l1.weight' in key:
                    config['d_model'] = tensor.shape[0]
                elif 'vec2neck.weight' in key:
                    config['neck'] = tensor.shape[0]
        
        # Create model
        model = TransformerAE(**config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(device)
        
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
        print(f"  Error: {e}")
        return None, None, None

# 4. Generate predictions for all checkpoints
print("\n3. Generating predictions...")
predictions_data = []
failed_checkpoints = []

for ckpt_info in tqdm(checkpoint_info, desc="Processing checkpoints"):
    predictions, necks, config = load_and_predict(ckpt_info['path'], x1_test, x2_test)
    
    if predictions is not None:
        # Save predictions to disk
        pred_file = DATA_DIR / f"predictions_{ckpt_info['experiment']}_{ckpt_info['checkpoint'].replace('.pth', '.npz')}"
        np.savez_compressed(
            pred_file,
            predictions=predictions,
            necks=necks,
            x1=x1_test,
            x2=x2_test,
            y=y_test,
            experiment=ckpt_info['experiment'],
            loss_name=ckpt_info['loss_name'],
            overlap=ckpt_info['overlap']
        )
        
        predictions_data.append({
            **ckpt_info,
            'predictions_file': str(pred_file),
            'config': config,
            'mse': float(np.mean((predictions - y_test)**2)),
            'mae': float(np.mean(np.abs(predictions - y_test)))
        })
    else:
        failed_checkpoints.append(ckpt_info)

print(f"\nSuccessfully generated predictions for {len(predictions_data)}/{len(checkpoint_info)} checkpoints")
print(f"Failed: {len(failed_checkpoints)}")

# 5. Save metadata
print("\n4. Saving metadata...")
metadata = {
    'total_checkpoints': len(checkpoint_info),
    'successful': len(predictions_data),
    'failed': len(failed_checkpoints),
    'test_samples': n_test,
    'predictions': predictions_data
}

metadata_file = DATA_DIR / "predictions_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"Saved metadata: {metadata_file}")

# 6. Create summary
print("\n5. Creating summary...")
if predictions_data:
    df_summary = pd.DataFrame(predictions_data)
    
    # Summary by loss function
    print("\nPerformance by Loss Function (MSE):")
    loss_summary = df_summary.groupby('loss_name')['mse'].agg(['mean', 'std', 'min', 'max', 'count']).sort_values('mean')
    print(loss_summary.head(10))
    
    # Summary by overlap
    print("\nPerformance by Overlap (MSE):")
    overlap_summary = df_summary.groupby('overlap')['mse'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(overlap_summary)
    
    # Save summary
    summary_file = DATA_DIR / "predictions_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSaved summary: {summary_file}")

print("\n" + "="*80)
print("PREDICTION GENERATION COMPLETE")
print("="*80)
print(f"Generated {len(predictions_data)} prediction files")
print(f"Saved to: {DATA_DIR}")
print("="*80)
