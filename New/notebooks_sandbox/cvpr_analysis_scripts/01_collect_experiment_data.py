#!/usr/bin/env python3
"""
Collect and aggregate experiment data from all completed runs
Creates comprehensive dataset for CVPR paper analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
EXPERIMENTS_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/experiments")
OUTPUT_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/cvpr_analysis_scripts/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def collect_experiment_metrics():
    """
    Collect comprehensive metrics from all experiment directories.
    Extracts: training history, detailed metrics CSV, topology data, CNN validation.
    """
    print("\n1. Collecting experiment metrics...")
    
    all_data = []
    exp_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(exp_dirs)} experiment directories")
    
    for exp_dir in tqdm(exp_dirs, desc="Collecting experiments"):
        # Parse experiment name
        parts = exp_dir.name.split('_')
        if len(parts) < 3:
            continue
            
        model_size = parts[0]  # tiny
        overlap_str = parts[1]  # overlap0, overlap1, overlap2
        overlap = int(overlap_str.replace('overlap', ''))
        loss_name = '_'.join(parts[2:])  # Loss function name
        
        record = {
            'model_size': model_size,
            'overlap': overlap,
            'loss_name': loss_name,
            'experiment_dir': str(exp_dir),
        }
        
        # 1. Extract training history
        history_file = exp_dir / "training_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                train_losses = history.get('train_loss', [])
                val_losses = history.get('val_loss', [])
                
                if train_losses and val_losses:
                    record['final_val_loss'] = val_losses[-1]
                    record['best_val_loss'] = min(val_losses)
                    record['final_train_loss'] = train_losses[-1]
                    record['best_train_loss'] = min(train_losses)
                    record['total_epochs'] = len(train_losses)
                    record['convergence_epoch'] = np.argmin(val_losses)
            except Exception as e:
                print(f"Error reading {history_file}: {e}")
        
        # 2. Extract detailed metrics from CSV
        metrics_csv = exp_dir / "metrics" / "test_metrics_full_and_layerwise.csv"
        if metrics_csv.exists():
            try:
                metrics_df = pd.read_csv(metrics_csv)
                
                # Compute statistics for each metric column
                metric_cols = ['euclidean', 'manhattan', 'cosine', 'frobenius', 
                              'q_quantile', 'wasserstein', 'mape', 'js_divergence', 
                              'autoregressive', 'lwln']
                
                for col in metric_cols:
                    if col in metrics_df.columns:
                        values = metrics_df[col].dropna()
                        if len(values) > 0:
                            record[f'{col}_mean'] = values.mean()
                            record[f'{col}_std'] = values.std()
                            record[f'{col}_median'] = values.median()
                            record[f'{col}_min'] = values.min()
                            record[f'{col}_max'] = values.max()
                
                # Layerwise metrics
                layer_prefixes = ['conv1_weights', 'conv1_bias', 'conv2_weights', 
                                'conv2_bias', 'fc_layer']
                for prefix in layer_prefixes:
                    cols = [c for c in metrics_df.columns if c.startswith(prefix)]
                    if cols:
                        for col in cols:
                            values = metrics_df[col].dropna()
                            if len(values) > 0:
                                record[f'{col}_mean'] = values.mean()
                                
            except Exception as e:
                print(f"Error reading {metrics_csv}: {e}")
        
        # 3. Extract topology data
        topology_dir = exp_dir / "topology"
        if topology_dir.exists():
            topology_files = sorted(topology_dir.glob("topology_epoch_*.json"))
            if topology_files:
                try:
                    # Get final epoch topology
                    with open(topology_files[-1], 'r') as f:
                        topo = json.load(f)
                    record['final_gw_distance'] = topo.get('gw_distance', np.nan)
                    
                    # Collect GW distances over time
                    gw_distances = []
                    for topo_file in topology_files:
                        with open(topo_file, 'r') as f:
                            t = json.load(f)
                            if 'gw_distance' in t:
                                gw_distances.append(t['gw_distance'])
                    
                    if gw_distances:
                        record['gw_distance_mean'] = np.mean(gw_distances)
                        record['gw_distance_std'] = np.std(gw_distances)
                        record['gw_distance_trend'] = gw_distances[-1] - gw_distances[0]
                        
                except Exception as e:
                    print(f"Error reading topology: {e}")
        
        # 4. Count CNN validation samples
        cnn_val_dir = exp_dir / "cnn_validation"
        if cnn_val_dir.exists():
            record['cnn_validation_samples'] = len(list(cnn_val_dir.glob("*.npy")))
        
        all_data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\nCollected {len(df)} experiments")
    if len(df) > 0:
        print(f"Overlaps: {sorted(df['overlap'].unique())}")
        print(f"Loss functions: {len(df['loss_name'].unique())}")
        print(f"Columns extracted: {len(df.columns)}")
    
    # Save
    output_file = OUTPUT_DIR / "experiment_metrics.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    return df

def collect_weight_representations():
    """Collect final weight representations (bottleneck features)"""
    
    all_weights = {}
    
    exp_dirs = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()]
    
    for exp_dir in tqdm(exp_dirs, desc="Collecting weight representations"):
        # Look for final checkpoint
        checkpoint_dir = exp_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue
        
        # Find final epoch checkpoint
        checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
        if not checkpoints:
            continue
        
        # Get latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
        
        # Look for corresponding bottleneck features
        epoch_num = int(latest_checkpoint.stem.split('_')[1])
        
        # Check for saved predictions/bottleneck
        predictions_file = exp_dir / f"predictions_epoch{epoch_num}.npy"
        necks_file = exp_dir / f"necks_epoch{epoch_num}.npy"
        
        if necks_file.exists():
            try:
                necks = np.load(necks_file)
                all_weights[exp_dir.name] = necks
            except Exception as e:
                print(f"Error loading {necks_file}: {e}")
    
    print(f"\nCollected weight representations for {len(all_weights)} experiments")
    
    # Save
    output_file = OUTPUT_DIR / "weight_representations.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_weights, f)
    print(f"Saved to: {output_file}")
    
    return all_weights

def collect_topology_data():
    """Collect topology analysis results"""
    
    topology_data = {}
    
    exp_dirs = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()]
    
    for exp_dir in tqdm(exp_dirs, desc="Collecting topology data"):
        topology_dir = exp_dir / "topology"
        if not topology_dir.exists():
            continue
        
        exp_topology = {}
        
        # Collect persistence diagrams
        persistence_files = list(topology_dir.glob("*_persistence.pkl"))
        for pf in persistence_files:
            try:
                with open(pf, 'rb') as f:
                    exp_topology[pf.stem] = pickle.load(f)
            except Exception as e:
                print(f"Error loading {pf}: {e}")
        
        # Collect Mapper results
        mapper_files = list(topology_dir.glob("*_mapper.pkl"))
        for mf in mapper_files:
            try:
                with open(mf, 'rb') as f:
                    exp_topology[mf.stem] = pickle.load(f)
            except Exception as e:
                print(f"Error loading {mf}: {e}")
        
        if exp_topology:
            topology_data[exp_dir.name] = exp_topology
    
    print(f"\nCollected topology data for {len(topology_data)} experiments")
    
    # Save
    output_file = OUTPUT_DIR / "topology_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(topology_data, f)
    print(f"Saved to: {output_file}")
    
    return topology_data

def main():
    print("="*80)
    print("CVPR PAPER: DATA COLLECTION")
    print("="*80)
    
    # Collect all data
    print("\n1. Collecting experiment metrics...")
    df_metrics = collect_experiment_metrics()
    
    print("\n2. Collecting weight representations...")
    weight_reps = collect_weight_representations()
    
    print("\n3. Collecting topology data...")
    topology_data = collect_topology_data()
    
    # Summary
    print("\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"Experiments: {len(df_metrics)}")
    print(f"Weight representations: {len(weight_reps)}")
    print(f"Topology datasets: {len(topology_data)}")
    print(f"\nData saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
