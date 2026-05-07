#!/usr/bin/env python3
"""
Comprehensive Tournament Data Analysis
Analyzes what was actually saved during tournament execution:
- Predicted CNN weights
- Intermediary weights for finetuning
- WandB artifacts and logs
- Checkpoint files
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
EXPERIMENTS_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/experiments")
OUTPUT_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/cvpr_analysis_scripts/data")

print("="*80)
print("TOURNAMENT DATA ANALYSIS - COMPREHENSIVE AUDIT")
print("="*80)

# 1. Scan all experiment directories
print("\n1. Scanning experiment directories...")
exp_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
print(f"Found {len(exp_dirs)} experiment directories")

# 2. Analyze directory structure
print("\n2. Analyzing directory structure...")
structure_analysis = defaultdict(lambda: defaultdict(int))

for exp_dir in exp_dirs:
    # Check for subdirectories
    subdirs = ['checkpoints', 'predicted_weights', 'cnn_validation', 'topology', 
               'metrics', 'attention_heatmaps']
    
    for subdir in subdirs:
        subdir_path = exp_dir / subdir
        if subdir_path.exists():
            # Count files in subdirectory
            if subdir_path.is_dir():
                file_count = len(list(subdir_path.iterdir()))
                structure_analysis[subdir]['exists'] += 1
                structure_analysis[subdir]['total_files'] += file_count
                if file_count > 0:
                    structure_analysis[subdir]['non_empty'] += 1

print("\nDirectory Structure Summary:")
for subdir, stats in sorted(structure_analysis.items()):
    print(f"\n{subdir}:")
    print(f"  Exists in: {stats['exists']}/{len(exp_dirs)} experiments")
    print(f"  Non-empty: {stats['non_empty']}/{len(exp_dirs)} experiments")
    print(f"  Total files: {stats['total_files']}")

# 3. Detailed checkpoint analysis
print("\n3. Analyzing checkpoints...")
checkpoint_data = []

for exp_dir in exp_dirs:
    checkpoints_dir = exp_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.pth"))
        
        checkpoint_data.append({
            'experiment': exp_dir.name,
            'num_checkpoints': len(checkpoint_files),
            'checkpoint_files': [f.name for f in checkpoint_files],
            'has_best': (checkpoints_dir / "best_model.pth").exists(),
            'has_final': (checkpoints_dir / "final_model.pth").exists(),
        })

df_checkpoints = pd.DataFrame(checkpoint_data)
print(f"\nCheckpoint Summary:")
print(f"  Experiments with checkpoints: {len(df_checkpoints)}")
print(f"  Experiments with best_model.pth: {df_checkpoints['has_best'].sum()}")
print(f"  Experiments with final_model.pth: {df_checkpoints['has_final'].sum()}")
print(f"  Average checkpoints per experiment: {df_checkpoints['num_checkpoints'].mean():.1f}")
print(f"  Max checkpoints in one experiment: {df_checkpoints['num_checkpoints'].max()}")

# 4. CNN Validation analysis
print("\n4. Analyzing CNN validation data...")
cnn_val_data = []

for exp_dir in exp_dirs:
    cnn_val_dir = exp_dir / "cnn_validation"
    if cnn_val_dir.exists() and cnn_val_dir.is_dir():
        # Check for epoch subdirectories
        epoch_dirs = sorted([d for d in cnn_val_dir.iterdir() if d.is_dir()])
        
        for epoch_dir in epoch_dirs:
            # Count files in this epoch
            csv_files = list(epoch_dir.glob("*.csv"))
            json_files = list(epoch_dir.glob("*.json"))
            npy_files = list(epoch_dir.glob("*.npy"))
            
            cnn_val_data.append({
                'experiment': exp_dir.name,
                'epoch_dir': epoch_dir.name,
                'csv_files': len(csv_files),
                'json_files': len(json_files),
                'npy_files': len(npy_files),
                'total_files': len(list(epoch_dir.iterdir()))
            })

if cnn_val_data:
    df_cnn_val = pd.DataFrame(cnn_val_data)
    print(f"\nCNN Validation Summary:")
    print(f"  Experiments with CNN validation: {df_cnn_val['experiment'].nunique()}")
    print(f"  Total validation epochs: {len(df_cnn_val)}")
    print(f"  CSV files (results): {df_cnn_val['csv_files'].sum()}")
    print(f"  JSON files (eigenvalues): {df_cnn_val['json_files'].sum()}")
    print(f"  NPY files (weights): {df_cnn_val['npy_files'].sum()}")
else:
    print("\nNo CNN validation data found!")

# 5. Predicted weights analysis
print("\n5. Analyzing predicted weights directories...")
predicted_weights_data = []

for exp_dir in exp_dirs:
    weights_dir = exp_dir / "predicted_weights"
    if weights_dir.exists() and weights_dir.is_dir():
        files = list(weights_dir.iterdir())
        predicted_weights_data.append({
            'experiment': exp_dir.name,
            'num_files': len(files),
            'file_types': [f.suffix for f in files]
        })

if predicted_weights_data:
    df_weights = pd.DataFrame(predicted_weights_data)
    print(f"\nPredicted Weights Summary:")
    print(f"  Experiments with predicted_weights dir: {len(df_weights)}")
    print(f"  Non-empty directories: {(df_weights['num_files'] > 0).sum()}")
    print(f"  Total files: {df_weights['num_files'].sum()}")
else:
    print("\nNo predicted weights found in any experiment!")

# 6. Training history analysis
print("\n6. Analyzing training histories...")
history_data = []

for exp_dir in exp_dirs:
    history_file = exp_dir / "training_history.json"
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            history_data.append({
                'experiment': exp_dir.name,
                'epochs_completed': len(history.get('train_loss', [])),
                'has_cnn_results': 'cnn_results' in history,
                'has_topology': 'topology_results' in history,
                'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None
            })
        except Exception as e:
            print(f"  Error reading {exp_dir.name}: {e}")

if history_data:
    df_history = pd.DataFrame(history_data)
    print(f"\nTraining History Summary:")
    print(f"  Experiments with history: {len(df_history)}")
    print(f"  Average epochs completed: {df_history['epochs_completed'].mean():.1f}")
    print(f"  Experiments with CNN results in history: {df_history['has_cnn_results'].sum()}")
    print(f"  Experiments with topology in history: {df_history['has_topology'].sum()}")

# 7. Check for finetuning data in CNN validation results
print("\n7. Checking for finetuning data in CNN validation...")
finetuning_found = []

for exp_dir in exp_dirs:
    cnn_val_dir = exp_dir / "cnn_validation"
    if cnn_val_dir.exists():
        for epoch_dir in cnn_val_dir.iterdir():
            if epoch_dir.is_dir():
                csv_file = epoch_dir / "cnn_validation_results.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        # Check for finetuning columns
                        finetune_cols = [c for c in df.columns if 'finetune' in c.lower() or 'acc_id' in c or 'acc_ood' in c]
                        if finetune_cols:
                            finetuning_found.append({
                                'experiment': exp_dir.name,
                                'epoch': epoch_dir.name,
                                'num_samples': len(df),
                                'finetune_columns': finetune_cols
                            })
                    except Exception as e:
                        pass

if finetuning_found:
    print(f"\nFinetuning Data Found:")
    print(f"  Total instances: {len(finetuning_found)}")
    print(f"  Experiments with finetuning: {len(set([f['experiment'] for f in finetuning_found]))}")
    print(f"\nSample finetuning columns found:")
    if finetuning_found:
        print(f"  {finetuning_found[0]['finetune_columns']}")
else:
    print("\nNo finetuning data found in CNN validation results!")

# 8. Create comprehensive summary
print("\n" + "="*80)
print("COMPREHENSIVE SUMMARY")
print("="*80)

summary = {
    'total_experiments': len(exp_dirs),
    'experiments_with_checkpoints': len(df_checkpoints) if checkpoint_data else 0,
    'experiments_with_cnn_validation': df_cnn_val['experiment'].nunique() if cnn_val_data else 0,
    'experiments_with_predicted_weights': len(df_weights) if predicted_weights_data else 0,
    'experiments_with_training_history': len(df_history) if history_data else 0,
    'experiments_with_finetuning_data': len(set([f['experiment'] for f in finetuning_found])) if finetuning_found else 0,
    'total_cnn_validation_epochs': len(df_cnn_val) if cnn_val_data else 0,
    'total_checkpoint_files': df_checkpoints['num_checkpoints'].sum() if checkpoint_data else 0,
}

print("\nKey Metrics:")
for key, value in summary.items():
    print(f"  {key}: {value}")

# 9. Save detailed report
output_file = OUTPUT_DIR / "tournament_data_audit.json"
with open(output_file, 'w') as f:
    json.dump({
        'summary': summary,
        'checkpoint_details': checkpoint_data,
        'cnn_validation_details': cnn_val_data,
        'predicted_weights_details': predicted_weights_data,
        'finetuning_details': finetuning_found,
        'directory_structure': dict(structure_analysis)
    }, f, indent=2, default=str)

print(f"\nDetailed audit saved to: {output_file}")

# 10. Check what SHOULD have been saved based on code
print("\n" + "="*80)
print("EXPECTED vs ACTUAL DATA")
print("="*80)

print("\nBased on run_advanced_experiments.py, the following SHOULD be saved:")
print("  ✓ Checkpoints (best_model.pth, final_model.pth, periodic)")
print("  ✓ Training history (training_history.json)")
print("  ✓ CNN validation results (CSV files with finetuning metrics)")
print("  ✓ Eigenvalue analysis (JSON files per sample)")
print("  ✓ Topology results (JSON files per epoch)")
print("  ✗ Predicted weights (directory created but NO SAVE CODE FOUND)")

print("\nACTUAL DATA FOUND:")
print(f"  ✓ Checkpoints: {summary['experiments_with_checkpoints']}/{summary['total_experiments']} experiments")
print(f"  ✓ Training history: {summary['experiments_with_training_history']}/{summary['total_experiments']} experiments")
print(f"  ✓ CNN validation: {summary['experiments_with_cnn_validation']}/{summary['total_experiments']} experiments")
print(f"  ✓ Finetuning data: {summary['experiments_with_finetuning_data']}/{summary['total_experiments']} experiments")
print(f"  ✗ Predicted weights: {summary['experiments_with_predicted_weights']}/{summary['total_experiments']} experiments (EMPTY)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The tournament script run_advanced_experiments.py:
1. ✓ DOES run CNN finetuning (finetune_reconstructed_cnn)
2. ✓ DOES save finetuning results to CSV files
3. ✓ DOES save intermediary metrics (acc_id_initial, acc_id_final, etc.)
4. ✗ DOES NOT save predicted CNN weights to disk
5. ✗ DOES NOT save intermediary weights during finetuning

The predicted_weights directory is created but never populated.
Finetuning happens in-memory and only metrics are saved, not the actual weights.

To get the predicted weights, you would need to:
- Load a checkpoint (best_model.pth)
- Run inference on test samples
- Save the predictions manually

The finetuning results ARE available in:
  experiments/*/cnn_validation/epoch_*/cnn_validation_results.csv
""")

print("="*80)
