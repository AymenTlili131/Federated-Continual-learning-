# Project Organization - Notebooks Sandbox

## Overview
This folder contains self-contained analysis notebooks for the TransformerAE weight-space research project.
The transformer takes weights from 2 CNN models trained on MNIST subsets and predicts merged weights.

## Data Structure
- **Merged zoo.csv**: 36,469 rows, kaiming_uniform initialization, 6 activations (gelu, relu, silu, tanh, sigmoid, leakyrelu)
- **Columns**: label, class indicators (0-9), activation one-hot, weight columns (2464 weights)
- **Classes**: Pairs like [0,1], [0,2], etc. representing MNIST digit subsets

## Notebook Organization

### Core Analysis (Priority Order)

1. **01_generate_additional_zoos.ipynb** ✓
   - Generate 5 additional zoo CSVs with different initializations
   - Same format as Merged zoo.csv
   - Uses checkpoints folder

2. **02_batch_tensors_and_benchmark.ipynb** ✓
   - Create .pt tensor batches from Scenario data
   - Merge epochs, keep separate test sets
   - Benchmark training speed vs CustomDataset
   - Output: tensor_batches/*.pt files

3. **03_checkpoint_eval_and_metrics.ipynb** ✓
   - Iterate all .pth checkpoints in Experiments/
   - Auto-detect model dimensions from OrderedDict
   - Compute all metrics on predicted/ground truth/finetuned weights
   - Output: distances_metrics/*.csv (not markdown)

4. **09_weight_correlation_heatmaps.ipynb** ✓ HIGH PRIORITY
   - Correlation between weights and accuracy/epoch/activation
   - Generalize across zoos and layers
   - Output: correlation_analysis/figures/*.png, *.csv

### Advanced Analysis

5. **04_gated_attention_and_robust_metrics.ipynb** ✓
   - Implement https://arxiv.org/pdf/2505.06708 gated attention
   - Prevent attention collapse/gradient explosion
   - Handle edge cases (uniform attention, duplicate weights)

6. **05_topological_analysis_multipers.ipynb** ✓
   - Multipers, GUDHI, Giotto-TDA, Mapper implementations
   - Disentangle classes through topology
   - Explicit filtration functions
   - Output: topology_analysis/figures/*.png, *.csv

7. **10_hmmr_application_to_testsets.ipynb** ✓
   - Time-series segmentation of predicted weights
   - Cluster by class labels
   - Output: hmmr_results/*.csv

### Theory & Methods

8. **06_rmt_python_reimplementation.ipynb** ✓
   - Python implementation of https://github.com/RMT-TheoryAndPractice/RMT
   - Theory explanations
   - Demo on Merged zoo.csv
   - Output: rmt_analysis/

9. **07_ntk_trainability_demo.ipynb** ✓
   - Neural Tangent Kernel analysis
   - Trainability metrics
   - Output: ntk_analysis/

10. **08_gradient_frequency_and_superweights.ipynb** ✓
    - Fourier transform gradient amplification (https://arxiv.org/pdf/2405.20233)
    - Systematic sparsity (https://arxiv.org/pdf/2411.07191)
    - Weight disentanglement (https://arxiv.org/pdf/1912.13053)
    - Output: gradient_analysis/

## Output Structure

```
notebooks_sandbox/
├── 00-11_*.ipynb                    # Notebooks
├── distances_metrics/               # CSV files with all distance metrics
│   ├── checkpoint_name_metrics.csv
│   └── ...
├── correlation_analysis/            # Correlation heatmaps
│   ├── figures/*.png
│   └── correlations.csv
├── topology_analysis/               # Topological analysis results
│   ├── figures/*.png
│   ├── persistence_diagrams/
│   └── mapper_graphs/
├── rmt_analysis/                    # Random Matrix Theory
│   ├── figures/*.png
│   └── eigenvalue_stats.csv
├── ntk_analysis/                    # Neural Tangent Kernel
│   └── trainability_metrics.csv
├── gradient_analysis/               # Gradient/sparsity analysis
│   └── oscillation_metrics.csv
├── hmmr_results/                    # Time-series segmentation
│   └── segments.csv
├── tensor_batches/                  # Pre-batched .pt files
│   ├── train_overlap0_epoch_merged.pt
│   └── ...
└── results/                         # Consolidated results
    └── all_metrics_summary.csv
```

## Checkpoint Structure (Experiments/)

```
Experiments/
├── overlapping_0_[classes]_epoch_activation_loss/
│   ├── Tracking/
│   │   └── AE_epoch_loss_N.csv
│   └── AE_epoch_loss_N.pth
└── ...
```

## Key Concepts Preserved

1. **Distance Metrics**: Wasserstein, KL divergence, Frobenius, Q-quantile, MAPE, etc.
2. **Topology**: Persistent homology, Mapper, Betti numbers, filtrations
3. **RMT**: Eigenvalue distributions, Marchenko-Pastur, Wigner semicircle
4. **Gated Attention**: Prevent collapse, entropy monitoring
5. **HMMR**: Time-series segmentation, state transitions
6. **Correlation Analysis**: Weight-accuracy relationships

## Removed Redundancy

- Duplicate implementations of distance metrics → Consolidated in 03
- Multiple topology approaches → Unified in 05
- Scattered correlation code → Centralized in 09
- Redundant training scripts → Kept only benchmark in 02

## Notes

- All notebooks are self-contained with full imports
- No modifications to original code/data in parent directories
- Results saved as CSV (not markdown) for easy analysis
- Figures saved separately in appropriate folders
