#!/usr/bin/env python3
"""
Train Tiny Models - Batch 2 (Second Half of Losses)
Runs sequentially through second half of all losses for tiny models
"""

import sys
import os
from pathlib import Path
import argparse
import time
from datetime import datetime

# Add paths
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR / "core_modules"))

from advanced_losses import get_experiment_sequence

def main():
    parser = argparse.ArgumentParser(description="Train tiny models - Batch 2")
    parser.add_argument("--overlap", type=int, required=True, help="Overlap level (0, 1, or 2)")
    args = parser.parse_args()
    
    # Get all losses and split in half
    all_losses = get_experiment_sequence()
    mid_point = len(all_losses) // 2
    batch2_losses = all_losses[mid_point:]
    
    print(f"\n{'='*80}")
    print(f"TINY MODELS - BATCH 2")
    print(f"Overlap: {args.overlap}")
    print(f"Losses: {len(batch2_losses)} (second half)")
    print(f"{'='*80}\n")
    
    # Path to experiment runner
    experiment_script = CURRENT_DIR / "core_modules" / "run_advanced_experiments.py"
    
    results = []
    for i, loss_name in enumerate(batch2_losses, 1):
        print(f"\n[{i}/{len(batch2_losses)}] Training: tiny_overlap{args.overlap}_{loss_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Build command
        cmd = [
            'conda', 'run', '-n', 'FCL', 'python3',
            str(experiment_script),
            '--single',
            '--model-size', 'tiny',
            '--overlap', str(args.overlap),
            '--loss', loss_name,
            '--topology-n-jobs', '1',
            '--wandb'
        ]
        
        # Run experiment
        import subprocess
        result = subprocess.run(cmd, capture_output=False)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Completed in {elapsed/60:.1f} minutes")
            results.append({'loss': loss_name, 'status': 'success', 'time': elapsed})
        else:
            print(f"✗ Failed after {elapsed/60:.1f} minutes")
            results.append({'loss': loss_name, 'status': 'failed', 'time': elapsed})
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH 2 COMPLETE - Overlap {args.overlap}")
    print(f"{'='*80}")
    
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = sum(1 for r in results if r['status'] == 'failed')
    total_time = sum(r['time'] for r in results)
    
    print(f"Successes: {successes}/{len(batch2_losses)}")
    print(f"Failures: {failures}/{len(batch2_losses)}")
    print(f"Total time: {total_time/3600:.1f} hours")
    
    # Save results
    import json
    results_file = CURRENT_DIR / f"batch2_overlap{args.overlap}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'overlap': args.overlap,
            'batch': 2,
            'losses': batch2_losses,
            'results': results,
            'summary': {
                'successes': successes,
                'failures': failures,
                'total_time_hours': total_time/3600
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
