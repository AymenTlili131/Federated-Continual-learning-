#!/usr/bin/env python3
"""
Parallel Training System for GPU Utilization

Runs multiple experiments in parallel to maximize GPU usage:
- 4 tiny models in parallel (~800MB each = 3.2GB)
- 2-3 small models in parallel (~1.5GB each = 3-4.5GB)
- 1-2 medium models in parallel (~3-4GB each)
- 1 large/huge model at a time

Usage:
    python parallel_training.py --model-size tiny --overlap 0 --num-parallel 4
"""

import subprocess
import multiprocessing as mp
from pathlib import Path
import argparse
import time
import json
from datetime import datetime
import sys

# Add paths for imports
CURRENT_DIR = Path(__file__).parent  # tournament_system/
NOTEBOOKS_SANDBOX = CURRENT_DIR.parent  # notebooks_sandbox/
CORE_MODULES = NOTEBOOKS_SANDBOX / "core_modules"

sys.path.insert(0, str(CORE_MODULES))

# GPU memory estimates (MB)
GPU_MEMORY_ESTIMATES = {
    'tiny': 800,
    'small': 1500,
    'medium': 3500,
    'large': 6000,
    'huge': 8000
}

# Recommended parallel counts
RECOMMENDED_PARALLEL = {
    'tiny': 4,
    'small': 3,
    'medium': 2,
    'large': 1,
    'huge': 1
}


def run_single_experiment(args_dict):
    """Run a single experiment in a subprocess"""
    model_size = args_dict['model_size']
    overlap = args_dict['overlap']
    loss_name = args_dict['loss_name']
    exp_id = args_dict['exp_id']
    
    # Build command - use absolute path to run_advanced_experiments.py
    script_dir = Path(__file__).parent
    core_modules_dir = script_dir.parent / "core_modules"
    experiment_script = core_modules_dir / "run_advanced_experiments.py"
    
    cmd = [
        'conda', 'run', '-n', 'FCL', 'python3',
        str(experiment_script),
        '--single',
        '--model-size', model_size,
        '--overlap', str(overlap),
        '--loss', loss_name,
        '--topology-n-jobs', '1',  # Sequential topology to avoid CPU overload
        '--wandb'  # Enable WandB logging
    ]
    
    # Log file
    log_dir = NOTEBOOKS_SANDBOX / 'experiment_logs'
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"parallel_{model_size}_overlap{overlap}_{loss_name}_{exp_id}_{timestamp}.log"
    
    print(f"[{exp_id}] Starting: {model_size}, overlap={overlap}, loss={loss_name}")
    print(f"[{exp_id}] Log: {log_file}")
    
    # Run experiment
    start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=NOTEBOOKS_SANDBOX,  # run from notebooks_sandbox so relative paths resolve correctly
                timeout=28800  # 8 hour timeout per experiment
            )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[{exp_id}] ✓ Completed in {duration/60:.1f} minutes")
            return {'exp_id': exp_id, 'status': 'success', 'duration': duration, 'log': str(log_file)}
        else:
            print(f"[{exp_id}] ✗ Failed with code {result.returncode}")
            return {'exp_id': exp_id, 'status': 'failed', 'returncode': result.returncode, 'log': str(log_file)}
    
    except subprocess.TimeoutExpired:
        print(f"[{exp_id}] ✗ Timeout after 2 hours")
        return {'exp_id': exp_id, 'status': 'timeout', 'log': str(log_file)}
    except Exception as e:
        print(f"[{exp_id}] ✗ Error: {e}")
        return {'exp_id': exp_id, 'status': 'error', 'error': str(e), 'log': str(log_file)}


def run_parallel_batch(model_size, overlap, losses, num_parallel):
    """Run a batch of experiments in parallel"""
    print(f"\n{'='*80}")
    print(f"PARALLEL BATCH: {model_size}, overlap={overlap}")
    print(f"Losses: {len(losses)}")
    print(f"Parallel workers: {num_parallel}")
    print(f"Estimated GPU usage: {GPU_MEMORY_ESTIMATES[model_size] * num_parallel} MB")
    print(f"{'='*80}\n")
    
    # Prepare experiment arguments
    exp_args = []
    for idx, loss_name in enumerate(losses):
        exp_args.append({
            'model_size': model_size,
            'overlap': overlap,
            'loss_name': loss_name,
            'exp_id': f"{model_size}_o{overlap}_l{idx}"
        })
    
    # Run in parallel using multiprocessing
    results = []
    with mp.Pool(processes=num_parallel) as pool:
        results = pool.map(run_single_experiment, exp_args)
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE: {successful}/{len(results)} successful, {failed} failed")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Parallel training for GPU utilization")
    parser.add_argument('--model-size', type=str, required=True,
                       choices=['tiny', 'small', 'medium', 'large', 'huge'],
                       help='Model size')
    parser.add_argument('--overlap', type=int, required=True,
                       help='Overlap level')
    parser.add_argument('--losses', nargs='+', default=None,
                       help='Loss functions to run (default: all from experiment sequence)')
    parser.add_argument('--num-parallel', type=int, default=None,
                       help='Number of parallel workers (default: auto based on model size)')
    parser.add_argument('--output-summary', type=str, default='parallel_training_summary.json',
                       help='Output summary file')
    
    args = parser.parse_args()
    
    # Auto-determine parallel count
    if args.num_parallel is None:
        args.num_parallel = RECOMMENDED_PARALLEL[args.model_size]
        print(f"Auto-selected {args.num_parallel} parallel workers for {args.model_size} models")
    
    # Get losses
    if args.losses is None:
        # Import experiment sequence from core_modules
        from advanced_losses import get_experiment_sequence
        args.losses = get_experiment_sequence()
        print(f"Using full experiment sequence: {len(args.losses)} losses")
    
    # Run parallel batch
    results = run_parallel_batch(
        args.model_size,
        args.overlap,
        args.losses,
        args.num_parallel
    )
    
    # Save summary
    summary = {
        'model_size': args.model_size,
        'overlap': args.overlap,
        'num_losses': len(args.losses),
        'num_parallel': args.num_parallel,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open(args.output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {args.output_summary}")


if __name__ == '__main__':
    main()
