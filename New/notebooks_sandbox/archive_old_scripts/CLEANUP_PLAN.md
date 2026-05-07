# Cleanup and Consolidation Plan

## Current State Analysis

### Existing Notebooks (Keep & Enhance)
1. ✓ 01_generate_additional_zoos.ipynb - Generate other initialization zoos
2. ✓ 02_batch_tensors_and_benchmark.ipynb - Batched tensors + benchmark
3. ✓ 03_checkpoint_eval_and_metrics.ipynb - Comprehensive checkpoint evaluation
4. ✓ 04_gated_attention_and_robust_metrics.ipynb - Gated attention implementation
5. ✓ 05_topological_analysis_multipers.ipynb - Topology analysis
6. ✓ 06_rmt_python_reimplementation.ipynb - RMT implementation
7. ✓ 07_ntk_trainability_demo.ipynb - NTK analysis
8. ✓ 08_gradient_frequency_and_superweights.ipynb - Gradient/sparsity
9. ✓ 09_weight_correlation_heatmaps.ipynb - Correlation analysis
10. ✓ 10_hmmr_application_to_testsets.ipynb - HMMR segmentation
11. ✓ 11_multi_initialization_analysis.ipynb - Multi-init comparison

### Files to Remove (Redundant/Superseded)
- 00_optimized_experiments.ipynb → Functionality distributed to other notebooks
- 03_checkpoint_eval_and_metrics_fresh.ipynb → Duplicate of 03
- 09_individual_neuron_correlations.ipynb → Merged into 09_weight_correlation
- distance_metrics_epoch_*.md → Will be CSV files instead
- test_*.md → Test files, not needed
- *.py files in notebooks_sandbox → Move to parent or remove

### Python Modules to Consolidate
Move to parent directory or create single utils module:
- config.py
- trainer.py
- loss_functions.py
- optimized_models.py
- data_preprocessing.py
- rmt_analysis.py
- topology_analysis.py
- distance_metrics.py (create from notebooks)

### Folders to Create
```
notebooks_sandbox/
├── distances_metrics/
├── correlation_analysis/
│   └── figures/
├── topology_analysis/
│   ├── figures/
│   ├── persistence_diagrams/
│   └── mapper_graphs/
├── rmt_analysis/
│   └── figures/
├── ntk_analysis/
├── gradient_analysis/
├── hmmr_results/
├── tensor_batches/
└── results/
```

## Action Items

### Phase 1: Structure Setup
1. Create output directories
2. Move/consolidate Python modules
3. Remove duplicate notebooks
4. Remove markdown test files

### Phase 2: Notebook Enhancement
1. Update each notebook to output CSV instead of markdown
2. Add proper error handling for edge cases
3. Ensure all imports are self-contained
4. Add progress bars and logging

### Phase 3: Integration
1. Update 02 to merge epochs, separate test sets
2. Update 03 to handle all checkpoint formats
3. Update 05 to include all topology libraries
4. Update 09 to generalize across layers

### Phase 4: Documentation
1. Add README to each output folder
2. Create master results summary notebook
3. Document all metrics and their meanings
