"""
Utility modules for the weight-space research pipeline.
"""

from .data_loading import (
    load_merged_zoo,
    create_weight_pairs,
    create_dataloaders,
    ensure_ood_test_set
)

from .metrics import (
    compute_eigenvalues,
    compute_persistent_homology,
    compute_rmt_metrics,
    compute_wasserstein_distance
)

from .visualization import (
    create_attention_heatmap,
    plot_eigenvalue_evolution,
    plot_betti_curves,
    create_comparison_grid
)

from .wandb_logging import (
    log_attention_maps,
    log_eigenvalues_table,
    log_gif,
    log_metric_slider
)

__all__ = [
    'load_merged_zoo',
    'create_weight_pairs',
    'create_dataloaders',
    'ensure_ood_test_set',
    'compute_eigenvalues',
    'compute_persistent_homology',
    'compute_rmt_metrics',
    'compute_wasserstein_distance',
    'create_attention_heatmap',
    'plot_eigenvalue_evolution',
    'plot_betti_curves',
    'create_comparison_grid',
    'log_attention_maps',
    'log_eigenvalues_table',
    'log_gif',
    'log_metric_slider'
]
