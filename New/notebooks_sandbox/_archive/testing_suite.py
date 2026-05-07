"""
Comprehensive testing suite for comparing predicted, finetuned, and ground truth weights
Includes all metrics: MSE, topology, RMT, eigenvalues, etc.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from topology_analysis import compare_weight_topology, TopologicalFeatureExtractor
from rmt_analysis import compare_weight_stages_rmt, RandomMatrixAnalyzer
from loss_functions import get_loss_function


class WeightComparator:
    """
    Compare predicted, finetuned, and ground truth weights
    """
    def __init__(self, max_homology_dim: int = 2):
        self.max_homology_dim = max_homology_dim
        self.topo_extractor = TopologicalFeatureExtractor(max_dimension=max_homology_dim)
        self.rmt_analyzer = RandomMatrixAnalyzer()
        
        # Initialize loss functions for comparison
        self.loss_functions = {
            'mse': get_loss_function('mse'),
            'mape': get_loss_function('mape'),
            'wasserstein': get_loss_function('wasserstein'),
        }
    
    def compute_basic_metrics(self, predicted: np.ndarray, 
                              ground_truth: np.ndarray) -> Dict[str, float]:
        """Compute basic distance metrics"""
        # Convert to tensors
        pred_tensor = torch.tensor(predicted, dtype=torch.float32).unsqueeze(0)
        gt_tensor = torch.tensor(ground_truth, dtype=torch.float32).unsqueeze(0)
        
        metrics = {}
        
        # L2 distance
        metrics['l2_distance'] = np.linalg.norm(predicted - ground_truth)
        
        # L1 distance
        metrics['l1_distance'] = np.sum(np.abs(predicted - ground_truth))
        
        # Cosine similarity
        cos_sim = np.dot(predicted, ground_truth) / (np.linalg.norm(predicted) * np.linalg.norm(ground_truth))
        metrics['cosine_similarity'] = cos_sim
        metrics['cosine_distance'] = 1 - cos_sim
        
        # Correlation
        metrics['pearson_correlation'] = np.corrcoef(predicted, ground_truth)[0, 1]
        metrics['spearman_correlation'] = stats.spearmanr(predicted, ground_truth)[0]
        
        # Loss function metrics
        for loss_name, loss_fn in self.loss_functions.items():
            try:
                loss_value = loss_fn(pred_tensor, gt_tensor)
                metrics[f'{loss_name}_loss'] = loss_value.item()
            except:
                pass
        
        # Statistical metrics
        metrics['mean_absolute_error'] = np.mean(np.abs(predicted - ground_truth))
        metrics['median_absolute_error'] = np.median(np.abs(predicted - ground_truth))
        metrics['max_absolute_error'] = np.max(np.abs(predicted - ground_truth))
        
        # Relative errors
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_errors = np.abs((predicted - ground_truth) / (ground_truth + 1e-10))
            metrics['mean_relative_error'] = np.mean(rel_errors)
            metrics['median_relative_error'] = np.median(rel_errors)
        
        return metrics
    
    def compare_all_stages(self, predicted: np.ndarray,
                           finetuned: np.ndarray,
                           ground_truth: np.ndarray) -> Dict:
        """
        Comprehensive comparison of all three weight stages
        
        Returns:
            Dictionary with all comparison metrics
        """
        results = {
            'basic_metrics': {},
            'topology': {},
            'rmt': {},
            'summary': {}
        }
        
        # Basic metrics for each comparison
        print("Computing basic metrics...")
        results['basic_metrics']['pred_vs_gt'] = self.compute_basic_metrics(predicted, ground_truth)
        results['basic_metrics']['finetuned_vs_gt'] = self.compute_basic_metrics(finetuned, ground_truth)
        results['basic_metrics']['pred_vs_finetuned'] = self.compute_basic_metrics(predicted, finetuned)
        
        # Topological analysis
        print("Computing topological features...")
        try:
            topo_results = compare_weight_topology(predicted, finetuned, ground_truth, 
                                                   max_dimension=self.max_homology_dim)
            results['topology'] = topo_results
        except Exception as e:
            print(f"Warning: Topology analysis failed: {e}")
            results['topology'] = {'error': str(e)}
        
        # RMT analysis
        print("Computing RMT features...")
        try:
            rmt_results = compare_weight_stages_rmt(predicted, finetuned, ground_truth)
            results['rmt'] = rmt_results
        except Exception as e:
            print(f"Warning: RMT analysis failed: {e}")
            results['rmt'] = {'error': str(e)}
        
        # Summary statistics
        results['summary'] = self._create_summary(results)
        
        return results
    
    def _create_summary(self, results: Dict) -> Dict:
        """Create summary of key metrics"""
        summary = {}
        
        # Extract key metrics
        if 'basic_metrics' in results:
            pred_gt = results['basic_metrics'].get('pred_vs_gt', {})
            ft_gt = results['basic_metrics'].get('finetuned_vs_gt', {})
            
            summary['predicted_l2_error'] = pred_gt.get('l2_distance', np.nan)
            summary['finetuned_l2_error'] = ft_gt.get('l2_distance', np.nan)
            summary['improvement_ratio'] = pred_gt.get('l2_distance', 1) / (ft_gt.get('l2_distance', 1) + 1e-10)
            
            summary['predicted_cosine_sim'] = pred_gt.get('cosine_similarity', np.nan)
            summary['finetuned_cosine_sim'] = ft_gt.get('cosine_similarity', np.nan)
        
        # RMT summary
        if 'rmt' in results and 'comparisons' in results['rmt']:
            rmt_comp = results['rmt']['comparisons']
            if 'pred_vs_gt' in rmt_comp and isinstance(rmt_comp['pred_vs_gt'], dict):
                for layer_name, metrics in rmt_comp['pred_vs_gt'].items():
                    if isinstance(metrics, dict):
                        summary[f'rmt_{layer_name}_wasserstein'] = metrics.get('wasserstein_distance', np.nan)
        
        return summary
    
    def save_results(self, results: Dict, save_path: str):
        """Save comparison results to JSON"""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {save_path}")
    
    def plot_comparison(self, predicted: np.ndarray,
                       finetuned: np.ndarray,
                       ground_truth: np.ndarray,
                       save_path: Optional[str] = None):
        """Create visualization comparing all three weight sets"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Weight distributions
        ax = axes[0, 0]
        ax.hist(predicted, bins=50, alpha=0.5, label='Predicted', density=True)
        ax.hist(finetuned, bins=50, alpha=0.5, label='Finetuned', density=True)
        ax.hist(ground_truth, bins=50, alpha=0.5, label='Ground Truth', density=True)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.set_title('Weight Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Predicted vs Ground Truth scatter
        ax = axes[0, 1]
        sample_indices = np.random.choice(len(predicted), min(1000, len(predicted)), replace=False)
        ax.scatter(ground_truth[sample_indices], predicted[sample_indices], alpha=0.5, s=1)
        ax.plot([ground_truth.min(), ground_truth.max()], 
               [ground_truth.min(), ground_truth.max()], 'r--', linewidth=2)
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title('Predicted vs Ground Truth')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Finetuned vs Ground Truth scatter
        ax = axes[0, 2]
        ax.scatter(ground_truth[sample_indices], finetuned[sample_indices], alpha=0.5, s=1)
        ax.plot([ground_truth.min(), ground_truth.max()], 
               [ground_truth.min(), ground_truth.max()], 'r--', linewidth=2)
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Finetuned')
        ax.set_title('Finetuned vs Ground Truth')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Error distributions
        ax = axes[1, 0]
        pred_errors = np.abs(predicted - ground_truth)
        ft_errors = np.abs(finetuned - ground_truth)
        ax.hist(pred_errors, bins=50, alpha=0.5, label='Predicted', density=True)
        ax.hist(ft_errors, bins=50, alpha=0.5, label='Finetuned', density=True)
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Density')
        ax.set_title('Error Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Cumulative errors
        ax = axes[1, 1]
        sorted_pred_errors = np.sort(pred_errors)
        sorted_ft_errors = np.sort(ft_errors)
        ax.plot(sorted_pred_errors, np.linspace(0, 1, len(sorted_pred_errors)), 
               label='Predicted', linewidth=2)
        ax.plot(sorted_ft_errors, np.linspace(0, 1, len(sorted_ft_errors)), 
               label='Finetuned', linewidth=2)
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Weight trajectory (sample)
        ax = axes[1, 2]
        n_samples = min(500, len(predicted))
        indices = np.linspace(0, len(predicted)-1, n_samples, dtype=int)
        ax.plot(indices, predicted[indices], 'b-', alpha=0.7, label='Predicted', linewidth=1)
        ax.plot(indices, finetuned[indices], 'g-', alpha=0.7, label='Finetuned', linewidth=1)
        ax.plot(indices, ground_truth[indices], 'r-', alpha=0.7, label='Ground Truth', linewidth=1)
        ax.set_xlabel('Weight Index')
        ax.set_ylabel('Weight Value')
        ax.set_title('Weight Trajectories (Sample)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig


def run_comprehensive_test(predicted_weights: np.ndarray,
                          finetuned_weights: np.ndarray,
                          ground_truth_weights: np.ndarray,
                          save_dir: str = "./test_results") -> Dict:
    """
    Run comprehensive testing suite
    
    Args:
        predicted_weights: Weights predicted by TransformerAE
        finetuned_weights: Weights after finetuning on actual data
        ground_truth_weights: True weights from dataset
        save_dir: Directory to save results
    
    Returns:
        Dictionary with all test results
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE WEIGHT COMPARISON TEST")
    print("="*60)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize comparator
    comparator = WeightComparator(max_homology_dim=2)
    
    # Run comparison
    results = comparator.compare_all_stages(predicted_weights, finetuned_weights, ground_truth_weights)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, value in results['summary'].items():
        if isinstance(value, float):
            print(f"{key:40s}: {value:.6f}")
        else:
            print(f"{key:40s}: {value}")
    
    # Save results
    comparator.save_results(results, save_dir / "comparison_results.json")
    
    # Create visualizations
    print("\nCreating visualizations...")
    comparator.plot_comparison(predicted_weights, finetuned_weights, ground_truth_weights,
                              save_path=save_dir / "comparison_plot.png")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print(f"Results saved to {save_dir}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    print("Testing Weight Comparison Suite")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Ground truth
    ground_truth = np.random.randn(2464) * 0.1
    
    # Predicted (with some error)
    predicted = ground_truth + np.random.randn(2464) * 0.05
    
    # Finetuned (closer to ground truth)
    finetuned = ground_truth + np.random.randn(2464) * 0.02
    
    # Run test
    results = run_comprehensive_test(predicted, finetuned, ground_truth, 
                                     save_dir="./test_results_demo")
    
    print("\nTest completed successfully!")
