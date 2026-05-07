#!/usr/bin/env python3
"""
Train Special Losses: Sinkhorn Variants Only
Runs 2 Sinkhorn losses sequentially per overlap (0 → 1 → 2)

Losses:
1. Sinkhorn (full sequence)
2. LW_Sinkhorn (layerwise)

Note: Persistence losses moved to regularizers due to NaN issues and computational cost.
They are available as: MSE+0.01*PersLandscape, MAE+0.01*PersLandscape, etc.
"""

import sys
import os
from pathlib import Path
import argparse
import time
from datetime import datetime
import subprocess
import json

# Add paths
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR / "core_modules"))

def main():
    # Define the 2 Sinkhorn losses
    special_losses = [
        'Sinkhorn',
        'LW_Sinkhorn',
    ]
    
    print(f"\n{'='*80}")
    print(f"SINKHORN LOSSES TRAINING")
    print(f"2 Losses × 3 Overlaps = 6 Experiments")
    print(f"{'='*80}\n")
    
    # Path to experiment runner
    experiment_script = CURRENT_DIR / "core_modules" / "run_advanced_experiments.py"
    
    all_results = []
    
    # Process each overlap sequentially
    for overlap in [0, 1, 2]:
        print(f"\n{'='*80}")
        print(f"OVERLAP {overlap}")
        print(f"{'='*80}\n")
        
        overlap_results = []
        
        # Train all 2 losses for this overlap before moving to next
        for i, loss_name in enumerate(special_losses, 1):
            print(f"\n[{i}/2] Training: overlap{overlap}_{loss_name}")
            print(f"{'='*80}")
            
            start_time = time.time()
            
            # Build command
            cmd = [
                'conda', 'run', '-n', 'FCL', 'python3',
                str(experiment_script),
                '--single',
                '--model-size', 'tiny',
                '--overlap', str(overlap),
                '--loss', loss_name,
                '--topology-n-jobs', '1',
                '--wandb'
            ]
            
            # Run experiment
            result = subprocess.run(cmd, capture_output=False)
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✓ Completed in {elapsed/60:.1f} minutes")
                overlap_results.append({
                    'overlap': overlap,
                    'loss': loss_name,
                    'status': 'success',
                    'time': elapsed
                })
            else:
                print(f"✗ Failed after {elapsed/60:.1f} minutes")
                overlap_results.append({
                    'overlap': overlap,
                    'loss': loss_name,
                    'status': 'failed',
                    'time': elapsed
                })
        
        all_results.extend(overlap_results)
        
        # Summary for this overlap
        successes = sum(1 for r in overlap_results if r['status'] == 'success')
        failures = sum(1 for r in overlap_results if r['status'] == 'failed')
        total_time = sum(r['time'] for r in overlap_results)
        
        print(f"\n{'='*80}")
        print(f"OVERLAP {overlap} COMPLETE")
        print(f"{'='*80}")
        print(f"Successes: {successes}/2")
        print(f"Failures: {failures}/2")
        print(f"Total time: {total_time/3600:.1f} hours")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"ALL SPECIAL LOSSES COMPLETE")
    print(f"{'='*80}")
    
    total_successes = sum(1 for r in all_results if r['status'] == 'success')
    total_failures = sum(1 for r in all_results if r['status'] == 'failed')
    total_time = sum(r['time'] for r in all_results)
    
    print(f"Total experiments: 6")
    print(f"Successes: {total_successes}/6")
    print(f"Failures: {total_failures}/6")
    print(f"Total time: {total_time/3600:.1f} hours")
    
    # Save results
    results_file = CURRENT_DIR / "special_losses_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'losses': special_losses,
            'overlaps': [0, 1, 2],
            'results': all_results,
            'summary': {
                'total_experiments': 6,
                'successes': total_successes,
                'failures': total_failures,
                'total_time_hours': total_time/3600
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
