#!/usr/bin/env python3
"""
TOURNAMENT-STYLE LOSS SELECTION SYSTEM

Progressive loss filtering across model sizes:
1. Tiny: Train on ALL 43 losses
2. Small: Top 10% + Bottom 5% from tiny (6-7 losses)
3. Medium: Top 10% + Bottom 5% from small (1-2 losses)
4. Large: Top 10% + Bottom 5% from medium (1-2 losses)
5. Huge: Top 10% + Bottom 5% from large (1-2 losses)

Selection criterion: MSE on test set (regardless of training loss)
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
import subprocess

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from advanced_losses import HierarchicalLossRegistry, get_experiment_sequence


# ============================================================================
# TOURNAMENT CONFIGURATION
# ============================================================================

TOURNAMENT_CONFIG = {
    'tiny': {
        'losses': 'all',  # All 43 losses
        'top_percent': 50,
        'bottom_percent': 10,
        'expected_survivors': 26  # ~22 top + 4 bottom
    },
    'small': {
        'losses': 'from_tiny',
        'top_percent': 35,
        'bottom_percent': 17.5,
        'expected_survivors': 14  # ~9 top + 5 bottom
    },
    'medium': {
        'losses': 'from_small',
        'top_percent': 12.5,
        'bottom_percent': 5,
        'expected_survivors': 3  # ~2 top + 1 bottom
    },
    'large': {
        'losses': 'from_medium',
        'top_percent': 55.5,  # 5 out of 9 = 55.5%
        'bottom_percent': 22.2,  # 2 out of 9 = 22.2%
        'expected_survivors': 7  # 5 top + 2 bottom
    },
    'huge': {
        'losses': 'from_large',
        'top_percent': 100,  # All survivors from large
        'bottom_percent': 0,
        'expected_survivors': 2  # Final 2 losses
    }
}


# ============================================================================
# LOSS SELECTION FUNCTIONS
# ============================================================================

def select_losses_from_results(results_csv, top_percent=10, bottom_percent=5):
    """
    Select top and bottom performing losses based on MSE
    
    Args:
        results_csv: Path to experiment summary CSV
        top_percent: Percentage of top performers to keep
        bottom_percent: Percentage of bottom performers to keep
    
    Returns:
        List of selected loss names
    """
    df = pd.read_csv(results_csv)
    
    # Group by loss_name and calculate mean MSE across all overlaps
    loss_performance = []
    
    for loss_name in df['loss_name'].unique():
        loss_df = df[df['loss_name'] == loss_name]
        
        # Get MSE from test metrics (should be in the results)
        if 'best_val_loss' in loss_df.columns:
            mean_mse = loss_df['best_val_loss'].mean()
            loss_performance.append({
                'loss_name': loss_name,
                'mean_mse': mean_mse,
                'n_experiments': len(loss_df)
            })
    
    # Create DataFrame and sort by MSE
    perf_df = pd.DataFrame(loss_performance)
    perf_df = perf_df.sort_values('mean_mse')
    
    # Calculate number to select
    n_losses = len(perf_df)
    n_top = max(1, int(np.ceil(n_losses * top_percent / 100)))
    n_bottom = max(1, int(np.ceil(n_losses * bottom_percent / 100)))
    
    # Select top and bottom
    top_losses_df = perf_df.head(n_top)
    bottom_losses_df = perf_df.tail(n_bottom)
    
    top_losses = top_losses_df['loss_name'].tolist()
    bottom_losses = bottom_losses_df['loss_name'].tolist()
    
    # Combine (remove duplicates if any)
    selected_losses = list(set(top_losses + bottom_losses))
    
    print(f"\n{'='*80}")
    print(f"LOSS SELECTION RESULTS")
    print(f"{'='*80}")
    print(f"Total losses evaluated: {n_losses}")
    print(f"Top {top_percent}% selected: {n_top} losses")
    print(f"Bottom {bottom_percent}% selected: {n_bottom} losses")
    print(f"Total selected: {len(selected_losses)} losses")
    print(f"\nTop performers (best MSE):")
    for i, row in top_losses_df.iterrows():
        print(f"  {i+1}. {row['loss_name']}: MSE = {row['mean_mse']:.6f}")
    print(f"\nBottom performers (worst MSE - for analysis):")
    for i, row in bottom_losses_df.iterrows():
        print(f"  {i+1}. {row['loss_name']}: MSE = {row['mean_mse']:.6f}")
    print(f"{'='*80}\n")
    
    # Save selection report
    selection_report = {
        'timestamp': datetime.now().isoformat(),
        'n_total': n_losses,
        'n_top': n_top,
        'n_bottom': n_bottom,
        'n_selected': len(selected_losses),
        'top_losses': top_losses,
        'bottom_losses': bottom_losses,
        'selected_losses': selected_losses,
        'performance': perf_df.to_dict('records')
    }
    
    return selected_losses, selection_report


# ============================================================================
# TOURNAMENT RUNNER
# ============================================================================

def run_tournament_round(
    model_size,
    losses,
    overlaps,
    epochs,
    batch_size,
    output_base_dir,
    use_wandb=True
):
    """
    Run one round of the tournament for a specific model size
    
    Args:
        model_size: Model size to train
        losses: List of loss names to evaluate
        overlaps: List of overlap values
        epochs: Number of epochs
        batch_size: Batch size
        output_base_dir: Base output directory
        use_wandb: Enable WandB logging
    
    Returns:
        Path to results summary CSV
    """
    print(f"\n{'='*80}")
    print(f"TOURNAMENT ROUND: {model_size.upper()}")
    print(f"{'='*80}")
    print(f"Losses to evaluate: {len(losses)}")
    print(f"Overlaps: {overlaps}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*80}\n")
    
    # Build command
    losses_str = ' '.join([f'"{loss}"' for loss in losses])
    overlaps_str = ' '.join(map(str, overlaps))
    
    cmd = [
        'python3', 'run_advanced_experiments.py',
        '--models', model_size,
        '--overlaps', *overlaps_str.split(),
        '--losses', *losses,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--output-dir', str(output_base_dir)
    ]
    
    if use_wandb:
        cmd.append('--wandb')
    else:
        cmd.append('--no-wandb')
    
    print(f"Running command:")
    print(f"  {' '.join(cmd)}\n")
    
    # Run experiments
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Tournament round failed for {model_size}")
        return None
    
    # Find results summary
    summary_path = Path(output_base_dir) / "advanced_experiments_summary.csv"
    
    if not summary_path.exists():
        print(f"ERROR: Results summary not found at {summary_path}")
        return None
    
    return summary_path


def run_full_tournament(
    overlaps=[0, 1, 2],
    epochs=200,
    batch_size=24,
    output_base_dir="/media/aymen/8A0CA9E80CA9CF8D/Experiments",
    use_wandb=True
):
    """
    Run complete tournament across all model sizes
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Tournament tracking
    tournament_dir = output_base_dir / "tournament_results"
    tournament_dir.mkdir(parents=True, exist_ok=True)
    
    tournament_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'overlaps': overlaps,
            'epochs': epochs,
            'batch_size': batch_size
        },
        'rounds': []
    }
    
    # Initialize log path
    log_path = tournament_dir / "tournament_log.json"
    
    # Get all losses for tiny model
    registry = HierarchicalLossRegistry()
    all_losses = get_experiment_sequence()
    
    print(f"\n{'='*80}")
    print(f"TOURNAMENT-STYLE LOSS SELECTION")
    print(f"{'='*80}")
    print(f"Starting with {len(all_losses)} losses on tiny model")
    print(f"Progressive filtering to find best 3-6 losses for huge model")
    print(f"Output directory: {output_base_dir}")
    print(f"{'='*80}\n")
    
    current_losses = all_losses
    model_sequence = ['tiny', 'small', 'medium', 'large', 'huge']
    
    for model_size in model_sequence:
        config = TOURNAMENT_CONFIG[model_size]
        
        print(f"\n{'#'*80}")
        print(f"# ROUND: {model_size.upper()} MODEL")
        print(f"# Evaluating {len(current_losses)} losses")
        print(f"{'#'*80}\n")
        
        # Run experiments for this model size
        round_output_dir = output_base_dir / f"round_{model_size}"
        summary_path = run_tournament_round(
            model_size=model_size,
            losses=current_losses,
            overlaps=overlaps,
            epochs=epochs,
            batch_size=batch_size,
            output_base_dir=round_output_dir,
            use_wandb=use_wandb
        )
        
        if summary_path is None:
            print(f"ERROR: Failed to complete round for {model_size}")
            break
        
        # Select losses for next round (unless this is the last round)
        if model_size != 'huge':
            selected_losses, selection_report = select_losses_from_results(
                summary_path,
                top_percent=config['top_percent'],
                bottom_percent=config['bottom_percent']
            )
            
            # Save selection report
            report_path = tournament_dir / f"selection_{model_size}_to_{model_sequence[model_sequence.index(model_size)+1]}.json"
            with open(report_path, 'w') as f:
                json.dump(selection_report, f, indent=2)
            
            # Update current losses for next round
            current_losses = selected_losses
            
            print(f"\nSelected {len(current_losses)} losses for next round ({model_sequence[model_sequence.index(model_size)+1]})")
            print(f"Losses: {current_losses}\n")
        
        # Log this round
        tournament_log['rounds'].append({
            'model_size': model_size,
            'n_losses': len(current_losses),
            'losses': current_losses,
            'summary_path': str(summary_path)
        })
        
        # Save tournament log
        log_path = tournament_dir / "tournament_log.json"
        with open(log_path, 'w') as f:
            json.dump(tournament_log, f, indent=2)
    
    # Final summary
    tournament_log['end_time'] = datetime.now().isoformat()
    with open(log_path, 'w') as f:
        json.dump(tournament_log, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"TOURNAMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Final losses for huge model: {current_losses}")
    print(f"Tournament log: {log_path}")
    print(f"{'='*80}\n")
    
    return tournament_log


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run tournament-style loss selection")
    parser.add_argument("--overlaps", nargs='+', type=int, default=[0, 1, 2],
                       help="Overlap levels to test")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=24,
                       help="Batch size")
    parser.add_argument("--output-dir", type=str, 
                       default="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/Experiments",
                       help="Base output directory")
    parser.add_argument("--wandb", action="store_true", default=True,
                       help="Enable WandB logging")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb",
                       help="Disable WandB logging")
    
    args = parser.parse_args()
    
    # Run tournament
    tournament_log = run_full_tournament(
        overlaps=args.overlaps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_base_dir=args.output_dir,
        use_wandb=args.wandb
    )
    
    print("\nTournament completed successfully!")
    print(f"Results saved to: {args.output_dir}/tournament_results/")


if __name__ == "__main__":
    main()
