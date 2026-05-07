"""
Multi-objective ranking for loss function selection

Ranks losses based on multiple criteria:
1. Primary: Initial CNN accuracy (before finetuning)
2. Secondary: Improvement rate (speed of finetuning)
3. Tertiary: Final CNN accuracy (after finetuning)
4. Quaternary: MSE (structural similarity)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class LossPerformance:
    """Performance metrics for a single loss function"""
    loss_name: str
    mse: float
    initial_acc: float
    final_acc: float
    improvement_rate: float  # (final - initial) / n_epochs
    finetune_history: List[float]  # Accuracy at each epoch
    
    def __repr__(self):
        return (f"LossPerformance({self.loss_name}, "
                f"init_acc={self.initial_acc:.2f}%, "
                f"final_acc={self.final_acc:.2f}%, "
                f"rate={self.improvement_rate:.2f}%/epoch, "
                f"mse={self.mse:.4f})")


def compute_improvement_rate(finetune_history: List[float]) -> float:
    """
    Compute improvement rate from finetuning history
    
    Uses linear regression on accuracy vs epoch to get rate of improvement
    
    Args:
        finetune_history: [epoch_0_acc, epoch_1_acc, ..., epoch_5_acc]
    
    Returns:
        Improvement rate in accuracy points per epoch
    """
    if len(finetune_history) < 2:
        return 0.0
    
    # Linear regression: acc = rate * epoch + intercept
    epochs = np.arange(len(finetune_history))
    accs = np.array(finetune_history)
    
    # Least squares fit
    A = np.vstack([epochs, np.ones(len(epochs))]).T
    rate, intercept = np.linalg.lstsq(A, accs, rcond=None)[0]
    
    return float(rate)


def normalize_scores(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    Normalize scores to [0, 1] range
    
    Args:
        scores: Array of scores
        higher_is_better: If True, higher scores are better
    
    Returns:
        Normalized scores in [0, 1]
    """
    if len(scores) == 0:
        return scores
    
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.ones_like(scores) * 0.5
    
    normalized = (scores - min_score) / (max_score - min_score)
    
    if not higher_is_better:
        normalized = 1.0 - normalized
    
    return normalized


def rank_losses_multi_objective(
    performances: List[LossPerformance],
    weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)
) -> List[Tuple[str, float, Dict]]:
    """
    Rank losses using multi-objective optimization
    
    Criteria (in order of importance):
    1. Initial accuracy (40% weight) - Most important
    2. Improvement rate (30% weight) - Speed of finetuning
    3. Final accuracy (20% weight) - Ultimate performance
    4. MSE (10% weight) - Structural similarity
    
    Args:
        performances: List of LossPerformance objects
        weights: (w_initial, w_rate, w_final, w_mse) - must sum to 1.0
    
    Returns:
        List of (loss_name, composite_score, breakdown) sorted by score (descending)
    """
    if not performances:
        return []
    
    # Validate weights
    w_initial, w_rate, w_final, w_mse = weights
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    # Extract metrics
    loss_names = [p.loss_name for p in performances]
    initial_accs = np.array([p.initial_acc for p in performances])
    improvement_rates = np.array([p.improvement_rate for p in performances])
    final_accs = np.array([p.final_acc for p in performances])
    mses = np.array([p.mse for p in performances])
    
    # Normalize each metric to [0, 1]
    norm_initial = normalize_scores(initial_accs, higher_is_better=True)
    norm_rate = normalize_scores(improvement_rates, higher_is_better=True)
    norm_final = normalize_scores(final_accs, higher_is_better=True)
    norm_mse = normalize_scores(mses, higher_is_better=False)  # Lower MSE is better
    
    # Compute composite scores
    composite_scores = (
        w_initial * norm_initial +
        w_rate * norm_rate +
        w_final * norm_final +
        w_mse * norm_mse
    )
    
    # Create results with breakdown
    results = []
    for i, loss_name in enumerate(loss_names):
        breakdown = {
            'initial_acc': initial_accs[i],
            'improvement_rate': improvement_rates[i],
            'final_acc': final_accs[i],
            'mse': mses[i],
            'norm_initial': norm_initial[i],
            'norm_rate': norm_rate[i],
            'norm_final': norm_final[i],
            'norm_mse': norm_mse[i],
            'composite_score': composite_scores[i]
        }
        results.append((loss_name, composite_scores[i], breakdown))
    
    # Sort by composite score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def select_top_and_bottom_losses(
    ranked_losses: List[Tuple[str, float, Dict]],
    top_percent: float = 0.5,
    bottom_percent: float = 0.1
) -> Tuple[List[str], List[str]]:
    """
    Select top and bottom losses based on ranking
    
    Args:
        ranked_losses: Output from rank_losses_multi_objective
        top_percent: Percentage of top losses to select
        bottom_percent: Percentage of bottom losses to select
    
    Returns:
        (top_losses, bottom_losses) as lists of loss names
    """
    n_losses = len(ranked_losses)
    n_top = max(1, int(n_losses * top_percent))
    n_bottom = max(1, int(n_losses * bottom_percent))
    
    top_losses = [name for name, _, _ in ranked_losses[:n_top]]
    bottom_losses = [name for name, _, _ in ranked_losses[-n_bottom:]]
    
    return top_losses, bottom_losses


def create_ranking_report(
    ranked_losses: List[Tuple[str, float, Dict]],
    title: str = "Loss Function Ranking Report"
) -> pd.DataFrame:
    """
    Create a detailed ranking report as a DataFrame
    
    Args:
        ranked_losses: Output from rank_losses_multi_objective
        title: Report title
    
    Returns:
        DataFrame with ranking details
    """
    data = []
    
    for rank, (loss_name, composite_score, breakdown) in enumerate(ranked_losses, 1):
        data.append({
            'Rank': rank,
            'Loss': loss_name,
            'Composite': f"{composite_score:.4f}",
            'Initial_Acc': f"{breakdown['initial_acc']:.2f}%",
            'Improve_Rate': f"{breakdown['improvement_rate']:.2f}%/ep",
            'Final_Acc': f"{breakdown['final_acc']:.2f}%",
            'MSE': f"{breakdown['mse']:.4f}",
            'Norm_Init': f"{breakdown['norm_initial']:.3f}",
            'Norm_Rate': f"{breakdown['norm_rate']:.3f}",
            'Norm_Final': f"{breakdown['norm_final']:.3f}",
            'Norm_MSE': f"{breakdown['norm_mse']:.3f}"
        })
    
    df = pd.DataFrame(data)
    
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}\n")
    
    return df


def analyze_ranking_sensitivity(
    performances: List[LossPerformance],
    weight_configs: List[Tuple[float, float, float, float]] = None
) -> Dict:
    """
    Analyze how ranking changes with different weight configurations
    
    Args:
        performances: List of LossPerformance objects
        weight_configs: List of weight tuples to test
    
    Returns:
        Dictionary with sensitivity analysis results
    """
    if weight_configs is None:
        # Default: test different emphasis on each criterion
        weight_configs = [
            (0.4, 0.3, 0.2, 0.1),  # Balanced (default)
            (0.7, 0.15, 0.1, 0.05),  # Emphasize initial accuracy
            (0.2, 0.5, 0.2, 0.1),  # Emphasize improvement rate
            (0.2, 0.2, 0.5, 0.1),  # Emphasize final accuracy
            (0.25, 0.25, 0.25, 0.25),  # Equal weights
        ]
    
    results = {}
    
    for i, weights in enumerate(weight_configs):
        config_name = f"Config_{i+1}"
        ranked = rank_losses_multi_objective(performances, weights)
        
        results[config_name] = {
            'weights': weights,
            'top_3': [name for name, _, _ in ranked[:3]],
            'bottom_3': [name for name, _, _ in ranked[-3:]],
            'ranking': ranked
        }
    
    # Print summary
    print("\n" + "="*80)
    print("RANKING SENSITIVITY ANALYSIS")
    print("="*80)
    
    for config_name, data in results.items():
        w_init, w_rate, w_final, w_mse = data['weights']
        print(f"\n{config_name}:")
        print(f"  Weights: Init={w_init:.2f}, Rate={w_rate:.2f}, "
              f"Final={w_final:.2f}, MSE={w_mse:.2f}")
        print(f"  Top 3:    {', '.join(data['top_3'])}")
        print(f"  Bottom 3: {', '.join(data['bottom_3'])}")
    
    print("="*80 + "\n")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Create sample performances
    sample_performances = [
        LossPerformance(
            loss_name="MSE",
            mse=0.001,
            initial_acc=75.0,
            final_acc=92.0,
            improvement_rate=3.4,
            finetune_history=[75.0, 80.0, 85.0, 88.0, 90.0, 92.0]
        ),
        LossPerformance(
            loss_name="MAE",
            mse=0.002,
            initial_acc=82.0,
            final_acc=93.0,
            improvement_rate=2.2,
            finetune_history=[82.0, 85.0, 88.0, 90.0, 92.0, 93.0]
        ),
        LossPerformance(
            loss_name="Huber",
            mse=0.0015,
            initial_acc=78.0,
            final_acc=91.0,
            improvement_rate=2.6,
            finetune_history=[78.0, 82.0, 86.0, 88.0, 90.0, 91.0]
        ),
    ]
    
    # Rank losses
    ranked = rank_losses_multi_objective(sample_performances)
    
    # Create report
    df = create_ranking_report(ranked, "Sample Ranking Report")
    
    # Sensitivity analysis
    sensitivity = analyze_ranking_sensitivity(sample_performances)
