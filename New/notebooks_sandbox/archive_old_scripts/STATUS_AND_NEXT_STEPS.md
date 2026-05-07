# Current Status and Next Steps

## What Exists (Already Implemented)

### ✓ Completed Notebooks

1. **01_generate_additional_zoos.ipynb** - Generates 5 additional zoo CSVs with different initializations
2. **02_batch_tensors_and_benchmark.ipynb** - Creates batched .pt tensors and benchmarks speed
3. **03_checkpoint_eval_and_metrics.ipynb** - Comprehensive checkpoint evaluation with all metrics
4. **04_gated_attention_and_robust_metrics.ipynb** - Gated attention implementation
5. **05_topological_analysis_multipers.ipynb** - Topology analysis with multiple libraries
6. **06_rmt_python_reimplementation.ipynb** - RMT Python implementation
7. **07_ntk_trainability_demo.ipynb** - NTK analysis
8. **08_gradient_frequency_and_superweights.ipynb** - Gradient oscillation and sparsity
9. **09_weight_correlation_heatmaps.ipynb** - Correlation analysis (HIGH PRIORITY)
10. **10_hmmr_application_to_testsets.ipynb** - HMMR time-series segmentation
11. **11_multi_initialization_analysis.ipynb** - Multi-initialization comparison (placeholder)

### ✓ Created Infrastructure

- **utils_consolidated.py** - All distance metrics, checkpoint loading, data utilities
- **Folder structure** - All output directories created
- **Documentation** - README.md, PROJECT_ORGANIZATION.md, CLEANUP_PLAN.md

## What Needs to Be Done

### Priority 1: Update Existing Notebooks for CSV Output

All notebooks currently output to markdown or mixed formats. Need to update to CSV-only:

1. **03_checkpoint_eval_and_metrics.ipynb**
   - Change all metric outputs to CSV
   - Use `WeightDistanceMetrics.compute_all_metrics_to_csv()`
   - Save to `distances_metrics/checkpoint_name_full.csv` and `_layerwise.csv`

2. **09_weight_correlation_heatmaps.ipynb** (HIGH PRIORITY)
   - Ensure correlation matrices saved as CSV
   - Save figures to `correlation_analysis/figures/`
   - Generalize across all layers and zoos

3. **05_topological_analysis_multipers.ipynb**
   - Save Mapper results to CSV
   - Save persistence diagrams data to CSV
   - Keep figures in `topology_analysis/figures/`

4. **06, 07, 08, 10** - Update to save all metrics as CSV

### Priority 2: Implement Key Missing Features

1. **02_batch_tensors_and_benchmark.ipynb**
   - **CRITICAL**: Merge epochs during training, keep separate test sets
   - Update Scenario data structure in `../data/Scenario/`
   - Current: separate .npy per epoch
   - Target: merged training data, separate test sets per epoch

2. **03_checkpoint_eval_and_metrics.ipynb**
   - Add finetuning loop (1-5 epochs on MNIST test set)
   - Save predicted, ground truth, AND finetuned weights
   - Compute metrics on all three versions

3. **04_gated_attention_and_robust_metrics.ipynb**
   - Add handling for edge cases:
     - Uniform attention (all weights same)
     - Duplicate weights (clustering fails)
     - Insufficient unique values

### Priority 3: Consolidation and Cleanup

1. **Remove redundant files**:
   ```
   - 00_optimized_experiments.ipynb (functionality distributed)
   - 03_checkpoint_eval_and_metrics_fresh.ipynb (duplicate)
   - 09_individual_neuron_correlations.ipynb (merged into 09)
   - distance_metrics_epoch_*.md (will be CSV)
   - test_*.md (test files)
   ```

2. **Move Python modules**:
   - Keep only `utils_consolidated.py` in notebooks_sandbox
   - Move other .py files to parent or remove if redundant

3. **Clean up markdown files**:
   - Keep: README.md, STATUS_AND_NEXT_STEPS.md, PROJECT_ORGANIZATION.md
   - Remove: EXPERIMENT_GUIDE.md, IMPLEMENTATION_SUMMARY.md, etc. (redundant)

### Priority 4: Verify All Notebooks Run

Test each notebook independently:
```bash
conda activate FCL
jupyter nbconvert --to notebook --execute 01_generate_additional_zoos.ipynb
jupyter nbconvert --to notebook --execute 02_batch_tensors_and_benchmark.ipynb
# ... etc
```

## Specific Implementation Tasks

### Task 1: Update 02 for Merged Epochs

**Current behavior**: Separate .npy files per epoch in Scenario/
**Target behavior**: 
- Training data: merge all epochs into single .pt file
- Test data: separate .pt files per epoch for OOD testing

**Files to modify**:
- `02_batch_tensors_and_benchmark.ipynb`
- Output to `tensor_batches/train_overlap{N}_merged.pt`
- Output to `tensor_batches/test_overlap{N}_epoch{E}.pt`

### Task 2: Update 03 for Finetuning

**Add finetuning loop** (from meta.ipynb):
```python
for epoch_cnn in range(1, 6):  # 1-5 epochs
    # Train on MNIST test set
    train_epoch_loss, train_epoch_acc = train(model, Tr_DLr, optimizerCNN, criterion_CNN)
    # Validate
    valid_epoch_loss, valid_epoch_acc = validate(model, Ts_DL0, criterion_CNN)
    # Save weights at each epoch
    save_finetuned_weights(model, epoch_cnn)
```

**Compute metrics on**:
1. Predicted weights (direct TransformerAE output)
2. Ground truth weights (real CNN weights)
3. Finetuned weights (after 1-5 epochs of training)

### Task 3: Update 09 for Layer Generalization

**Current**: Analyzes full weight vector
**Target**: Analyze each layer separately

```python
for layer_name in ['conv1_weights', 'conv1_bias', 'conv2_weights', 'conv2_bias', 'fc_layer']:
    layer_weights = extract_layer(weights, layer_name)
    correlations = compute_correlations(layer_weights, accuracy, epoch, activation)
    save_correlation_heatmap(correlations, f'correlation_analysis/figures/{layer_name}.png')
```

### Task 4: Implement Robust Error Handling

**All notebooks need**:
```python
try:
    # Main analysis
    result = analyze_data(data)
except InsufficientDataError:
    print("Warning: Not enough unique datapoints for clustering")
    result = fallback_analysis(data)
except Exception as e:
    print(f"Error: {e}")
    result = None
```

**Specific cases**:
- Mapper: Handle < 10 samples, constant lens function
- Clustering: Handle duplicate weights, < 2 unique points
- Persistence: Handle missing ripser library
- Correlation: Handle constant columns

## Output File Naming Convention

### Distance Metrics
```
distances_metrics/
├── {checkpoint_name}_full.csv
├── {checkpoint_name}_layerwise.csv
└── summary_all_checkpoints.csv
```

### Correlation Analysis
```
correlation_analysis/
├── figures/
│   ├── conv1_weights_correlation.png
│   ├── conv2_weights_correlation.png
│   └── full_weights_correlation.png
└── correlations.csv
```

### Topology Analysis
```
topology_analysis/
├── figures/
│   ├── mapper_graph_{checkpoint}.png
│   └── persistence_diagram_{checkpoint}.png
├── mapper_results.csv
└── persistence_stats.csv
```

## Testing Checklist

- [ ] All notebooks run without errors
- [ ] All outputs are CSV (no markdown)
- [ ] Finetuning loop implemented in 03
- [ ] Merged epochs implemented in 02
- [ ] Layer-wise analysis in 09
- [ ] Error handling in all notebooks
- [ ] Redundant files removed
- [ ] Documentation updated

## Quick Start Commands

```bash
# Activate environment
conda activate FCL

# Navigate to notebooks
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Run high priority notebook
jupyter notebook 09_weight_correlation_heatmaps.ipynb

# Run full pipeline
for nb in 01 02 03 04 05 06 07 08 09 10; do
    jupyter nbconvert --to notebook --execute ${nb}_*.ipynb
done
```

## Expected Timeline

1. **Immediate** (30 min): Update 09 for CSV output and layer generalization
2. **Short-term** (2 hours): Update 02 for merged epochs, 03 for finetuning
3. **Medium-term** (4 hours): Update all notebooks for CSV output
4. **Long-term** (1 day): Test all notebooks, verify outputs, clean up

## Notes

- All notebooks already have comprehensive imports (from previous session)
- utils_consolidated.py provides all necessary helper functions
- Focus on CSV outputs for easy downstream analysis
- Maintain self-contained notebooks (no cross-dependencies)
