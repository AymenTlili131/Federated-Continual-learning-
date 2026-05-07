# Weight-Space Hyper-Representation Research Pipeline

This directory contains the comprehensive research pipeline for analyzing neural network weights using transformer-based hypernetworks.

## Overview

This pipeline implements a multi-faceted approach to understanding weight-space representations, combining:

1. **Gated Attention Mechanisms** (arxiv 2505.06708) - Prevents attention collapse
2. **Topological Data Analysis** - Persistent homology of weight spaces
3. **Random Matrix Theory** - Spectral analysis of weight matrices
4. **HMMR Segmentation** - Time-series segmentation of weight evolution
5. **Weight Disentanglement** (arxiv 1912.13053) - Separating training vs generalization weights
6. **Gradient Oscillation Tracking** (arxiv 2405.20233) - Understanding optimization dynamics
7. **Weight Importance Ranking** (arxiv 2411.07191) - Clustering and ranking weight importance

## Directory Structure

```
research_scripts/
├── config.yaml                      # Main configuration file
├── 01_create_baseline.py           # Baseline checkpoint creation
├── 02_train_gated_transformer.py   # Gated attention training
├── 03_visualize_neck_evolution.py  # PCA/t-SNE/UMAP GIF creation
├── 04_scenario_training.py         # Multi-scenario training
├── 05_predict_and_finetune.py      # Weight prediction + finetuning
├── 06_track_spectral_topology.py   # Eigenvalues + persistent homology
├── 07_smooth_predictions.py        # Exponential smoothing
├── 08_rmt_analysis.py              # Random Matrix Theory analysis
├── 09_hmmr_segmentation.py         # HMMR time-series segmentation
├── 10_disentangle_weights.py       # Weight disentanglement
├── 11_generate_report.py           # Comprehensive report generation
└── utils/                           # Utility modules
    ├── data_loading.py
    ├── metrics.py
    ├── visualization.py
    └── wandb_logging.py
```

## Quick Start

### 1. Basic Usage

```bash
# Run full pipeline with medium model
./run_research_pipeline.sh medium 500 32 1e-4

# Run with large model and custom epochs
./run_research_pipeline.sh large 1000 64 5e-5
```

### 2. Individual Module Usage

```bash
# Create baseline checkpoint
python research_scripts/01_create_baseline.py \
    --model_size medium \
    --output_dir research_results/checkpoints

# Train with gated attention
python research_scripts/02_train_gated_transformer.py \
    --model_size medium \
    --baseline_checkpoint research_results/checkpoints/baseline_medium.pth \
    --num_epochs 500 \
    --enable_gated_attention \
    --track_attention_entropy

# Create evolution visualizations
python research_scripts/03_visualize_neck_evolution.py \
    --checkpoint_dir research_results/checkpoints \
    --output_dir research_results/visualizations \
    --create_gifs \
    --methods pca tsne umap
```

## Configuration

Edit `config.yaml` to customize:

- **Model sizes**: tiny, small, medium, large
- **Training parameters**: epochs, batch size, learning rate
- **Scenarios**: activations, CNN sampling epochs, overlaps
- **Metrics**: eigenvalues, persistent homology, RMT
- **Visualization**: dimensionality reduction methods, GIF settings
- **HMMR**: number of states, segmentation parameters
- **Disentanglement**: methods, gradient tracking

## Key Features

### 1. Baseline Creation

Creates properly initialized checkpoints that serve as the foundation for all experiments:

```python
# Supports multiple initialization methods
--init_method xavier_uniform  # Default
--init_method kaiming_normal
--init_method orthogonal
```

### 2. Gated Attention

Prevents attention collapse through learned gating mechanisms:

- Per-head gates control information flow
- Temperature scaling for attention scores
- Entropy monitoring for early collapse detection
- Gradient stabilization

### 3. Dimensionality Reduction Evolution

Creates GIFs showing how weight-space representations evolve:

- **PCA**: Linear projection with variance explained
- **t-SNE**: Non-linear manifold learning
- **UMAP**: Topology-preserving dimensionality reduction

Outputs:
- Individual GIFs per method
- Side-by-side comparison frames
- WandB metric sliders for interactive exploration

### 4. Scenario-Based Training

Trains across multiple scenarios:

- **Activations**: ReLU, GELU, SiLU, Tanh, LeakyReLU
- **CNN Epochs**: 10, 20, 30, 36 (different training stages)
- **Overlaps**: 0, 1, 2 (class overlap configurations)

### 5. Weight Prediction & Finetuning

For each predicted weight set:

1. Identify input model labels
2. Create union of classes from input models
3. Finetune on MNIST subset for 1, 2, 3, 4, 5 epochs
4. Save all intermediate checkpoints
5. Compare predicted vs finetuned vs ground truth

### 6. Spectral & Topological Analysis

**Eigenvalue Tracking:**
- Per-layer eigenvalue distributions
- Evolution across training epochs
- Comparison with Marchenko-Pastur law
- WandB tables with opacity for separability

**Persistent Homology:**
- Betti curves (topological features)
- Persistence diagrams
- Persistence images (vectorizable features)
- Wasserstein distances between diagrams

### 7. Exponential Smoothing

Applies smoothing to predicted weights for cleaner visualization:

```python
# Multiple alpha values for comparison
alpha_values = [0.1, 0.3, 0.5, 0.7]

# Smoothed weights
w_smooth[t] = alpha * w_pred[t] + (1 - alpha) * w_smooth[t-1]
```

### 8. Random Matrix Theory Analysis

**Batch-Level Analysis:**
- Spectral density estimation
- Eigenvalue spacing distributions
- Comparison with random matrix ensembles

**Class-Grouped Analysis:**
- Group weights by recurring class labels
- Analyze spectral properties per class
- Identify class-specific weight patterns

### 9. HMMR Segmentation

Python implementation of Hidden Markov Model Regression:

- Segments weight time-series into states
- Clusters subsequences by class labels
- Identifies temporal patterns in weight evolution
- Visualizes state transitions

### 10. Weight Disentanglement

Separates weights into training vs generalization components:

**Methods:**
- Gradient-based separation
- Fisher information matrix
- Hessian eigenvalue analysis

**Gradient Tracking:**
- Evolution over training
- Oscillation frequency analysis
- Phase transitions

### 11. Weight Importance Ranking

Implements multiple criteria for ranking weight importance:

**Importance Metrics:**
- Gradient magnitude
- Fisher information
- Taylor expansion
- Eigenvalue-based

**Clustering:**
- K-means, DBSCAN, Hierarchical
- Multiple distance metrics
- Per-layer analysis

## WandB Integration

All experiments are tracked in Weights & Biases:

### Logged Items

- **Training metrics**: Loss, accuracy, gradient norms
- **Attention maps**: Heatmaps and entropy evolution
- **Eigenvalues**: Tables with opacity for separability
- **Persistent homology**: Diagrams and Betti curves
- **Dimensionality reduction**: Interactive GIFs and sliders
- **Segmentation**: State sequences and transitions
- **Disentanglement**: Training vs generalization weights
- **Weight importance**: Rankings and clusters

### Visualization Features

- **GIF sliders**: Scrub through training epochs
- **Opacity levels**: Separate overlapping features
- **Interactive plots**: Zoom, pan, hover for details
- **Comparison views**: Side-by-side method comparisons

## Data Requirements

### Input Data

- **Merged zoo**: `data/Merged zoo.csv`
- **CNN checkpoints**: Various activation functions and epochs
- **MNIST dataset**: For finetuning experiments

### Data Splits

Configured to ensure out-of-distribution test sets:

```yaml
val_split: 0.15
test_split: 0.15

test_ood_criteria:
  min_class_distance: 2  # Minimum different classes
  max_overlap: 0.3       # Maximum overlap with training
```

## Epoch Recommendations

Based on dataset size analysis:

- **Tiny models (4M params)**: 300-500 epochs
- **Small models (8M params)**: 500-700 epochs
- **Medium models (18M params)**: 500-1000 epochs
- **Large models (43M params)**: 1000-2000 epochs

Default is set to **500 epochs** (increased from original 100) to ensure:
- Sufficient convergence
- Meaningful gradient evolution tracking
- Clear topological feature development
- Stable HMMR segmentation

## Output Structure

```
research_results/
├── checkpoints/
│   ├── baseline_medium.pth
│   ├── checkpoint_epoch_0010.pth
│   ├── checkpoint_epoch_0020.pth
│   └── ...
├── visualizations/
│   ├── neck_evolution_pca.gif
│   ├── neck_evolution_tsne.gif
│   ├── neck_evolution_umap.gif
│   └── comparison_epoch_*.png
├── metrics/
│   ├── eigenvalues_per_epoch.csv
│   ├── persistent_homology_features.csv
│   ├── rmt_analysis_results.json
│   └── weight_importance_rankings.csv
├── finetuned_weights/
│   ├── predicted_sample_001.pth
│   ├── finetuned_sample_001_epoch_1.pth
│   ├── finetuned_sample_001_epoch_2.pth
│   └── ...
├── smoothed_weights/
│   ├── alpha_0.1/
│   ├── alpha_0.3/
│   └── alpha_0.5/
├── segmentation/
│   ├── hmmr_states_5.json
│   ├── hmmr_states_10.json
│   ├── segmentation_visualization.png
│   └── class_clustering_results.json
├── disentanglement/
│   ├── training_weights.pth
│   ├── generalization_weights.pth
│   ├── gradient_evolution.csv
│   └── oscillation_analysis.json
└── research_report.html
```

## Research Questions Addressed

1. **How do weight-space representations evolve during training?**
   - Dimensionality reduction GIFs
   - Topological feature tracking
   - Eigenvalue evolution

2. **What causes attention collapse in transformer-based hypernetworks?**
   - Gated attention mechanisms
   - Entropy monitoring
   - Gradient stabilization

3. **Can we identify distinct phases in weight evolution?**
   - HMMR segmentation
   - State transition analysis
   - Class-based clustering

4. **Which weights are important for training vs generalization?**
   - Weight disentanglement
   - Fisher information analysis
   - Gradient-based separation

5. **How do different scenarios affect weight-space geometry?**
   - Multi-scenario training
   - Topological distance metrics
   - RMT analysis per scenario

6. **Can we cluster and rank weight importance?**
   - Multiple importance metrics
   - Clustering algorithms
   - Per-layer analysis

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{weight_space_research_2024,
  title={Weight-Space Hyper-Representation Research Pipeline},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## References

1. **Gated Attention**: https://arxiv.org/pdf/2505.06708
2. **HMMR**: https://github.com/fchamroukhi/HMMR_r
3. **Weight Disentanglement**: https://arxiv.org/pdf/1912.13053
4. **Gradient Oscillation**: https://arxiv.org/pdf/2405.20233
5. **Weight Importance**: https://arxiv.org/pdf/2411.07191

## License

MIT License - See LICENSE file for details
