# Notebooks Sandbox - Weight-Space Analysis

## Project Context

This project analyzes a TransformerAE that takes weights from 2 CNN models (trained on MNIST subsets) and predicts merged weights. The current implementation has limitations, and these notebooks serve as analysis tools to understand the weight space and improve future iterations.

## Data

- **Merged zoo.csv**: 36,469 rows, kaiming_uniform init, 6 activations
- **Format**: label, class indicators (0-9), activation one-hot, 2464 weight columns
- **Classes**: Pairs like [0,1], [0,2] representing MNIST digit subsets

## Notebook Index

### 01_generate_additional_zoos.ipynb
**Status**: ✓ Complete  
**Purpose**: Generate 5 additional zoo CSVs with different initializations  
**Outputs**: 
- `../data/zoo_xavier_uniform.csv`
- `../data/zoo_xavier_normal.csv`
- `../data/zoo_he_uniform.csv`
- `../data/zoo_he_normal.csv`
- `../data/zoo_orthogonal.csv`

### 02_batch_tensors_and_benchmark.ipynb
**Status**: ✓ Complete  
**Purpose**: Create pre-batched .pt tensors, merge epochs, separate test sets  
**Outputs**: `tensor_batches/*.pt`  
**Key Feature**: Benchmark speed vs CustomDataset class

### 03_checkpoint_eval_and_metrics.ipynb
**Status**: ✓ Complete  
**Purpose**: Comprehensive checkpoint evaluation with auto-dimension detection  
**Outputs**: `distances_metrics/*.csv`  
**Metrics**: All 13 distance metrics on predicted/ground truth/finetuned weights

### 04_gated_attention_and_robust_metrics.ipynb
**Status**: ✓ Complete  
**Purpose**: Implement gated attention to prevent collapse  
**Paper**: https://arxiv.org/pdf/2505.06708  
**Outputs**: `results/gated_attention_analysis.csv`

### 05_topological_analysis_multipers.ipynb
**Status**: ✓ Complete  
**Purpose**: Topological analysis with multiple libraries  
**Libraries**: multipers, GUDHI, Giotto-TDA, Mapper  
**Outputs**: `topology_analysis/figures/*.png`, `topology_analysis/*.csv`

### 06_rmt_python_reimplementation.ipynb
**Status**: ✓ Complete  
**Purpose**: Python implementation of Random Matrix Theory  
**Source**: https://github.com/RMT-TheoryAndPractice/RMT  
**Outputs**: `rmt_analysis/eigenvalue_stats.csv`

### 07_ntk_trainability_demo.ipynb
**Status**: ✓ Complete  
**Purpose**: Neural Tangent Kernel analysis  
**Outputs**: `ntk_analysis/trainability_metrics.csv`

### 08_gradient_frequency_and_superweights.ipynb
**Status**: ✓ Complete  
**Purpose**: Gradient oscillation, Fourier analysis, sparsity  
**Papers**: 
- https://arxiv.org/pdf/2405.20233 (oscillations)
- https://arxiv.org/pdf/2411.07191 (sparsity)
- https://arxiv.org/pdf/1912.13053 (disentanglement)  
**Outputs**: `gradient_analysis/*.csv`

### 09_weight_correlation_heatmaps.ipynb
**Status**: ✓ Complete  
**Purpose**: Correlation between weights and accuracy/epoch/activation  
**Priority**: HIGH  
**Outputs**: `correlation_analysis/figures/*.png`, `correlation_analysis/correlations.csv`

### 10_hmmr_application_to_testsets.ipynb
**Status**: ✓ Complete  
**Purpose**: Time-series segmentation of predicted weights  
**Outputs**: `hmmr_results/segments.csv`

### 11_multi_initialization_analysis.ipynb
**Status**: Placeholder  
**Purpose**: Compare all 6 initialization methods  
**Note**: Requires data from notebook 01

## Output Structure

```
notebooks_sandbox/
├── distances_metrics/           # CSV files with all distance metrics
│   ├── checkpoint_name_full.csv
│   └── checkpoint_name_layerwise.csv
├── correlation_analysis/        # Correlation heatmaps
│   ├── figures/
│   └── correlations.csv
├── topology_analysis/           # Topological analysis
│   ├── figures/
│   ├── persistence_diagrams/
│   └── mapper_graphs/
├── rmt_analysis/               # Random Matrix Theory
│   └── figures/
├── ntk_analysis/               # Neural Tangent Kernel
├── gradient_analysis/          # Gradient/sparsity
├── hmmr_results/              # Time-series segmentation
├── tensor_batches/            # Pre-batched .pt files
└── results/                   # Consolidated results
```

## Key Utilities

### utils_consolidated.py

**WeightDistanceMetrics**:
- 13 full-vector distance metrics
- 5 layer-wise subdistances
- CSV export functionality

**Checkpoint Utilities**:
- `detect_model_dimensions()`: Auto-detect from OrderedDict
- `load_checkpoint_auto()`: Load any checkpoint format

**Data Utilities**:
- `load_merged_zoo()`: Load and preprocess zoo CSV
- `extract_weights_from_zoo()`: Extract 2464-dim weights
- `create_weight_pairs()`: Generate training pairs

**Topology Utilities**:
- `safe_mapper_analysis()`: Robust Mapper with error handling

## Distance Metrics (13 Full + 5 Layerwise)

### Full Vector (2464 dimensions)
1. Euclidean
2. Manhattan
3. Cosine
4. Frobenius Norm
5. Q-quantile Loss
6. Norm of Jacobian
7. Fisher Information Difference
8. Contractive Loss
9. Wasserstein Distance
10. MAPE (Mean Absolute Percentage Error)
11. LWLN (Layer-wise Loss Normalization)
12. Jensen-Shannon Divergence
13. Auto-regressive Loss

### Layerwise (5 subdistances)
1. conv1_weights (2080 weights)
2. conv1_bias (26 weights)
3. conv2_weights (384 weights)
4. conv2_bias (24 weights)
5. fc_layer (remaining weights)

Each layer reports: Euclidean, Manhattan, Cosine, Relative Diff, Mean Abs Diff

## Checkpoint Structure

```
../Experiments/
└── overlapping_{overlap}_[classes]_{epoch}_{activation}_{loss}/
    ├── Tracking/
    │   └── AE_epoch_{loss}_N.csv
    └── AE_epoch_{loss}_N.pth
```

## Usage Examples

### Load and Analyze Checkpoint
```python
from utils_consolidated import load_checkpoint_auto, WeightDistanceMetrics

# Auto-load checkpoint
model, config, metadata = load_checkpoint_auto('path/to/checkpoint.pth')

# Compute metrics
calc = WeightDistanceMetrics()
metrics = calc.compute_all_full_distances(predicted_weights, ground_truth)
calc.compute_all_metrics_to_csv(predicted_weights, ground_truth, 
                                'distances_metrics/result.csv',
                                metadata={'checkpoint': 'name', 'epoch': 100})
```

### Load Zoo Data
```python
from utils_consolidated import load_merged_zoo, extract_weights_from_zoo

df = load_merged_zoo('../data/Merged zoo.csv', limit=10000)
weights, weight_cols = extract_weights_from_zoo(df)
```

## Important Notes

1. **No modifications to parent code/data**: All notebooks are self-contained
2. **CSV outputs only**: No markdown tables (easier for analysis)
3. **Error handling**: All notebooks handle edge cases (duplicate weights, insufficient samples)
4. **Merged epochs**: Training data merges epochs, test sets remain separate
5. **Finetuning**: Save predicted, ground truth, and finetuned (1-5 epochs) weights

## Running Notebooks

All notebooks are designed to run independently with full imports. Execute in order for best results:

```bash
# Generate additional zoos
jupyter notebook 01_generate_additional_zoos.ipynb

# Create batched tensors
jupyter notebook 02_batch_tensors_and_benchmark.ipynb

# Evaluate all checkpoints
jupyter notebook 03_checkpoint_eval_and_metrics.ipynb

# High priority: correlation analysis
jupyter notebook 09_weight_correlation_heatmaps.ipynb

# Advanced analyses
jupyter notebook 04_gated_attention_and_robust_metrics.ipynb
jupyter notebook 05_topological_analysis_multipers.ipynb
jupyter notebook 06_rmt_python_reimplementation.ipynb
jupyter notebook 07_ntk_trainability_demo.ipynb
jupyter notebook 08_gradient_frequency_and_superweights.ipynb
jupyter notebook 10_hmmr_application_to_testsets.ipynb
```

## Conda Environment

All notebooks use the `FCL` conda environment:
```bash
conda activate FCL
```

## References

- Gated Attention: https://arxiv.org/pdf/2505.06708
- Gradient Oscillations: https://arxiv.org/pdf/2405.20233
- Weight Sparsity: https://arxiv.org/pdf/2411.07191
- Weight Disentanglement: https://arxiv.org/pdf/1912.13053
- Multipers: https://davidlapous.github.io/multipers/
- GUDHI: https://gudhi.inria.fr/python/latest/
- Giotto-TDA: https://giotto-ai.github.io/gtda-docs/latest/
- Mapper: https://kepler-mapper.scikit-tda.org/
- RMT: https://github.com/RMT-TheoryAndPractice/RMT
