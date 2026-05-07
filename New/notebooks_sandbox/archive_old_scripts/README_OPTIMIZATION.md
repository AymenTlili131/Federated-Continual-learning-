# FCL Optimization Framework

Comprehensive optimization framework for Federated Continual Learning with persistent homology analysis and random matrix theory metrics.

## Overview

This framework provides optimized TransformerAE models (500K - 200M+ parameters) with multiple loss functions, topological analysis, and efficient training for competing with state-of-the-art hyper-representation methods.

### Key Features

- **Multiple Model Sizes**: Tiny (500K), Small (2M), Medium (8M), Large (25M), Huge (200M+) parameters
- **7 Loss Functions**: MSE, MAPE, AUTO, LWWN, LWWN-WS, Wasserstein, Latent
- **Persistent Homology**: Persistence diagrams, Betti curves, vectorizable features
- **Random Matrix Theory**: Eigenvalue analysis, spectral density, Marchenko-Pastur comparison
- **Efficient Training**: Mixed precision, gradient checkpointing, WandB integration
- **Comprehensive Testing**: Predicted vs finetuned vs ground truth weight comparison

## Installation

```bash
# Install required packages
pip install torch torchvision numpy pandas matplotlib seaborn scipy scikit-learn tqdm wandb

# Optional: For persistent homology
pip install giotto-tda ripser persim

# Optional: For enhanced visualization
pip install plotly kaleido
```

## Quick Start

### 1. Data Preprocessing

Generate train/val/test splits with varying class overlap:

```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    df_path="./data/Merged zoo.csv",
    overlap_levels=[2, 1, 0],
    batch_size=20,
    batch_limit=100
)

preprocessor.process_all_scenarios(
    epoch_keys=[10],
    activ_keys=[2]  # silu
)
```

### 2. Run Quick Experiment

```python
from config import create_experiment_config
from optimized_models import create_model_from_config
from trainer import FCLTrainer, create_dataloaders

# Create configuration
config = create_experiment_config("small", overlap=2, primary_loss="mse")
config.training.epochs = 50

# Create model
model = create_model_from_config(config.model)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(config)

# Train
trainer = FCLTrainer(config, model)
trainer.train(train_loader, val_loader)
```

### 3. Analyze Results

```python
from testing_suite import run_comprehensive_test

# Load weights
predicted = model_output.cpu().numpy()
finetuned = finetuned_weights  # After finetuning
ground_truth = target_weights

# Run comprehensive analysis
results = run_comprehensive_test(
    predicted, finetuned, ground_truth,
    save_dir="./results/experiment_1"
)
```

## File Structure

```
New/
├── config.py                    # Experiment configurations
├── optimized_models.py          # Model architectures (tiny to huge)
├── loss_functions.py            # 7 loss function implementations
├── topology_analysis.py         # Persistent homology analysis
├── rmt_analysis.py             # Random matrix theory metrics
├── trainer.py                  # Training framework with WandB
├── testing_suite.py            # Comprehensive testing
├── data_preprocessing.py       # Data pipeline
├── notebooks_sandbox/
│   └── 00_optimized_experiments.ipynb  # Main experiment notebook
└── README_OPTIMIZATION.md      # This file
```

## Model Configurations

| Model  | Parameters | Memory (est.) | Use Case |
|--------|-----------|---------------|----------|
| Tiny   | ~500K     | ~50MB        | Rapid prototyping |
| Small  | ~2M       | ~150MB       | Fast experiments |
| Medium | ~8M       | ~500MB       | Balanced performance |
| Large  | ~25M      | ~1.5GB       | High capacity |
| Huge   | ~200M+    | ~12GB        | Comparison baseline |

## Loss Functions

### 1. MSE (Mean Squared Error)
Standard L2 loss for weight prediction.

```python
loss_fn = get_loss_function('mse')
```

### 2. MAPE (Mean Absolute Percentage Error)
Relative error metric.

```python
loss_fn = get_loss_function('mape')
```

### 3. AUTO (Autoencoder Loss)
Reconstruction + L1/L2 regularization on latent space.

```python
loss_fn = get_loss_function('auto')
```

### 4. LWWN (Layer-Wise Weighted Norm)
Different weights for different CNN layers.

```python
loss_fn = get_loss_function('lwwn')
```

### 5. LWWN-WS (with Weight Standardization)
LWWN with normalized weights.

```python
loss_fn = get_loss_function('lwwn_ws')
```

### 6. Wasserstein Distance
Earth Mover's Distance between weight distributions.

```python
loss_fn = get_loss_function('wasserstein')
```

### 7. Latent Loss
Encourages smooth and sparse latent representations.

```python
loss_fn = get_loss_function('latent')
```

### Multi-Loss
Weighted combination of multiple losses.

```python
multi_loss = MultiLoss(loss_weights={
    'mse': 1.0,
    'wasserstein': 0.05,
    'latent': 0.1
})
```

## Persistent Homology Analysis

### Compute Persistence Diagrams

```python
from topology_analysis import PersistentHomologyAnalyzer

analyzer = PersistentHomologyAnalyzer(max_dimension=2)
results = analyzer.compute_persistence_diagram(weights)

# Extract features
features = results['features']
# H0, H1, H2 features: num_features, max_lifetime, mean_lifetime, etc.
```

### Betti Curves

```python
from topology_analysis import BettiCurveAnalyzer

betti_analyzer = BettiCurveAnalyzer(n_bins=100)
betti_curves = betti_analyzer.compute_betti_curves(weights)

# Plot
betti_analyzer.plot_betti_curves(betti_curves, save_path="betti.png")
```

### Compare Weights

```python
from topology_analysis import compare_weight_topology

comparison = compare_weight_topology(
    predicted, finetuned, ground_truth,
    max_dimension=2
)
```

## Random Matrix Theory Analysis

### Eigenvalue Analysis

```python
from rmt_analysis import RandomMatrixAnalyzer

rmt_analyzer = RandomMatrixAnalyzer()
results = rmt_analyzer.analyze_all_layers(weights)

# Per-layer metrics:
# - Spectral radius
# - Condition number
# - Effective rank
# - KL divergence from Marchenko-Pastur
```

### Spectral Density

```python
# Plot spectral density with Marchenko-Pastur overlay
rmt_analyzer.plot_spectral_density(
    weight_matrix,
    layer_name="conv1_weight",
    compare_mp=True,
    save_path="spectral_density.png"
)
```

### Compare Weight Stages

```python
from rmt_analysis import compare_weight_stages_rmt

comparison = compare_weight_stages_rmt(
    predicted, finetuned, ground_truth
)

# Returns eigenvalue distribution comparisons for all layers
```

## Experiment Configurations

### Predefined Experiments

```python
from config import EXPERIMENT_SUITE

# Quick validation
config = EXPERIMENT_SUITE['quick_tiny_mse']

# Small model experiments
config = EXPERIMENT_SUITE['small_mse_overlap2']
config = EXPERIMENT_SUITE['small_wasserstein_overlap2']

# Medium model - best performers
config = EXPERIMENT_SUITE['medium_multi_overlap0']

# Large model - final comparison
config = EXPERIMENT_SUITE['large_best_overlap0']
```

### Custom Configuration

```python
from config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig

config = ExperimentConfig(
    name="custom_experiment",
    model=ModelConfig(name="custom", N=3, heads=4, d_model=128),
    data=DataConfig(overlap_levels=[1], batch_size=32),
    training=TrainingConfig(epochs=100, learning_rate=1e-4),
    loss=LossConfig(primary_loss="wasserstein"),
    metrics=MetricsConfig(track_persistent_homology=True)
)
```

## WandB Integration

```python
# Automatic logging of:
# - Training/validation losses
# - All loss function components
# - Learning rate
# - Gradient norms
# - Model architecture
# - Hyperparameters

# View at: https://wandb.ai/your-entity/fcl-optimization
```

## Comprehensive Testing Suite

```python
from testing_suite import WeightComparator

comparator = WeightComparator(max_homology_dim=2)

# Full comparison
results = comparator.compare_all_stages(
    predicted, finetuned, ground_truth
)

# Includes:
# - Basic metrics (L2, cosine, correlation)
# - All loss functions
# - Persistent homology features
# - RMT eigenvalue comparisons
# - Visualizations

# Save results
comparator.save_results(results, "comparison_results.json")
comparator.plot_comparison(predicted, finetuned, ground_truth, 
                          save_path="comparison.png")
```

## Recommended Workflow

### Phase 1: Quick Validation (1-2 hours)
1. Run tiny model with MSE loss, overlap=2
2. Verify training pipeline works
3. Check WandB logging

### Phase 2: Loss Function Comparison (4-6 hours)
1. Run small model with all 7 loss functions
2. Compare convergence and final performance
3. Identify best-performing losses

### Phase 3: Overlap Analysis (6-8 hours)
1. Run small model with overlap=[2, 1, 0]
2. Analyze OOD generalization
3. Compare topology and RMT metrics

### Phase 4: Scale Up (12-24 hours)
1. Run medium/large models with best configurations
2. Full persistent homology analysis
3. Comprehensive RMT comparison

### Phase 5: Final Comparison (24-48 hours)
1. Run huge model (original size)
2. Compare with best optimized model
3. Generate publication-quality figures
4. Prepare results for paper

## Performance Tips

### Memory Optimization
- Use gradient checkpointing for large models
- Enable mixed precision training (AMP)
- Reduce batch_limit for initial experiments
- Clear CUDA cache between experiments

### Speed Optimization
- Pre-save TensorDatasets (done in preprocessing)
- Use pin_memory=True for DataLoaders
- Increase num_workers (but watch RAM)
- Use smaller models for hyperparameter search

### GPU Utilization (RTX 5060 Ti, 16GB VRAM)
- Tiny: batch_size=64, very fast
- Small: batch_size=32, fast
- Medium: batch_size=16, moderate
- Large: batch_size=8, slower
- Huge: batch_size=2-4, slow (consider gradient accumulation)

## Troubleshooting

### Out of Memory
```python
# Reduce model size
config.model = MODEL_CONFIGS['small']

# Enable gradient checkpointing
model.use_gradient_checkpointing = True

# Reduce batch size
config.data.batch_size = 16
config.data.batch_limit = 50
```

### Slow Training
```python
# Enable mixed precision
config.training.mixed_precision = True

# Reduce logging frequency
config.training.log_every = 50

# Use smaller model
config.model = MODEL_CONFIGS['tiny']
```

### Data Not Found
```python
# Run preprocessing first
from data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
preprocessor.process_all_scenarios()
```

## Citation

If you use this framework, please cite:

```bibtex
@article{fcl_optimization_2024,
  title={Optimized Federated Continual Learning with Persistent Homology Analysis},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## References

- Hyper-Representations: https://arxiv.org/pdf/2402.18153
- Model Soups: https://arxiv.org/pdf/2209.14733
- Persistent Homology: giotto-tda documentation
- Random Matrix Theory: Marchenko-Pastur law

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
