#!/usr/bin/env python3
"""
Per-Overlap Ranking System

Ranks losses separately for each overlap tier before passing to larger models.
This accounts for the fact that different overlaps can make a single loss perform differently.

Usage:
    python per_overlap_ranking.py --model-size tiny --top-n 30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import sys

# Add paths for imports
CURRENT_DIR = Path(__file__).parent  # tournament_system/
NOTEBOOKS_SANDBOX = CURRENT_DIR.parent  # notebooks_sandbox/
CORE_MODULES = NOTEBOOKS_SANDBOX / "core_modules"

sys.path.insert(0, str(CORE_MODULES))

from multi_objective_ranking import rank_losses_multi_objective, LossPerformance


def collect_results_per_overlap(model_size: str, overlap: int, experiments_dir: Path) -> List[LossPerformance]:
    """Collect all results for a specific model size and overlap"""
    results = []
    
    # Find all experiment directories for this model/overlap
    pattern = f"{model_size}_overlap{overlap}_*"
    exp_dirs = list(experiments_dir.glob(pattern))
    
    print(f"  Found {len(exp_dirs)} experiments for {model_size}, overlap={overlap}")
    
    for exp_dir in exp_dirs:
        # Extract loss name from directory
        loss_name = exp_dir.name.replace(f"{model_size}_overlap{overlap}_", "")
        
        # Load training history
        history_file = exp_dir / "training_history.json"
        if not history_file.exists():
            continue
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Load test metrics
        metrics_file = exp_dir / "metrics" / "test_metrics_full_and_layerwise.csv"
        if not metrics_file.exists():
            continue
        
        metrics_df = pd.read_csv(metrics_file)
        
        # Create LossPerformance object
        perf = LossPerformance(
            loss_name=loss_name,
            final_val_loss=history['val_loss'][-1] if history['val_loss'] else float('inf'),
            best_val_loss=min(history['val_loss']) if history['val_loss'] else float('inf'),
            convergence_epoch=np.argmin(history['val_loss']) + 1 if history['val_loss'] else 0,
            test_mse=metrics_df['mse'].mean() if 'mse' in metrics_df.columns else float('inf'),
            test_mae=metrics_df['mae'].mean() if 'mae' in metrics_df.columns else float('inf'),
            test_cosine=metrics_df['cosine_similarity'].mean() if 'cosine_similarity' in metrics_df.columns else 0.0
        )
        
        results.append(perf)
    
    return results


def rank_per_overlap(model_size: str, overlaps: List[int], experiments_dir: Path, 
                     top_n: int = 20, bottom_n: int = 10) -> Dict[int, Dict[str, List[str]]]:
    """Rank losses separately for each overlap tier, selecting BOTH best and worst"""
    print(f"\n{'='*80}")
    print(f"PER-OVERLAP RANKING: {model_size}")
    print(f"Selecting top {top_n} AND bottom {bottom_n} performers per overlap")
    print(f"{'='*80}\n")
    
    rankings = {}
    
    for overlap in overlaps:
        print(f"\nRanking for overlap={overlap}...")
        
        # Collect results
        results = collect_results_per_overlap(model_size, overlap, experiments_dir)
        
        if not results:
            print(f"  No results found for overlap={overlap}")
            rankings[overlap] = {'top': [], 'bottom': []}
            continue
        
        # Rank using multi-objective ranking
        ranked = rank_losses_multi_objective(results, top_n=len(results))
        
        # Extract top N and bottom N
        top_losses = [r.loss_name for r in ranked[:top_n]]
        bottom_losses = [r.loss_name for r in ranked[-bottom_n:]]
        
        rankings[overlap] = {
            'top': top_losses,
            'bottom': bottom_losses,
            'combined': top_losses + bottom_losses
        }
        
        print(f"  Top {len(top_losses)} losses:")
        for i, loss_name in enumerate(top_losses[:5], 1):
            print(f"    {i}. {loss_name}")
        if len(top_losses) > 5:
            print(f"    ... and {len(top_losses) - 5} more")
        
        print(f"  Bottom {len(bottom_losses)} losses (may excel on larger models):")
        for i, loss_name in enumerate(bottom_losses[:5], 1):
            print(f"    {i}. {loss_name}")
        if len(bottom_losses) > 5:
            print(f"    ... and {len(bottom_losses) - 5} more")
        
        print(f"  Total selected: {len(top_losses) + len(bottom_losses)}")
    
    return rankings


def save_rankings(rankings: Dict[int, List[str]], output_file: Path, model_size: str):
    """Save rankings to JSON file"""
    output = {
        'model_size': model_size,
        'rankings_per_overlap': {
            str(overlap): losses
            for overlap, losses in rankings.items()
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nRankings saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Per-overlap ranking system")
    parser.add_argument('--model-size', type=str, required=True,
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model size to rank')
    parser.add_argument('--overlaps', nargs='+', type=int, default=[0, 1, 2],
                       help='Overlap levels to rank')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top losses to select per overlap')
    parser.add_argument('--bottom-n', type=int, default=10,
                       help='Number of bottom losses to select per overlap (may excel on larger models)')
    parser.add_argument('--experiments-dir', type=str,
                       default='/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/Experiments',
                       help='Experiments directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: rankings_{model_size}.json)')
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return
    
    # Rank per overlap
    rankings = rank_per_overlap(
        args.model_size,
        args.overlaps,
        experiments_dir,
        args.top_n,
        args.bottom_n
    )
    
    # Save rankings
    output_file = args.output if args.output else f"rankings_{args.model_size}.json"
    save_rankings(rankings, Path(output_file), args.model_size)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for overlap, loss_dict in rankings.items():
        total = len(loss_dict['combined'])
        print(f"Overlap {overlap}: {total} losses selected ({len(loss_dict['top'])} top + {len(loss_dict['bottom'])} bottom)")
    print(f"\nNext step: Pass these losses to next model size experiments")


if __name__ == '__main__':
    main()
