# Comprehensive Experiment Guide

## Quick Start

Your medium model has been trained successfully! Here's where everything is:

### 📁 Results Location

```
research_results/
├── checkpoints/
│   ├── best_medium_model.pth          ⭐ Best model (lowest validation loss)
│   ├── final_medium_model.pth         Final model at epoch 500
│   └── checkpoint_epoch_*.pth         Checkpoints every 10 epochs (50 files)
└── training_history.json              Training/validation loss curves
```

### 🔧 Fixed Issues

1. **Checkpoint Loading Error** - Fixed `weights_only=False` for PyTorch 2.6+
2. **TransformerAE Bug** - Fixed `vec2neck` layer dimension mismatch (line 346)

## 📊 New Features Implemented

### 1. Distance Metrics Module

**Full 2464-Vector Distances** (13 metrics):
- Euclidean, Manhattan, Cosine, Wasserstein
- Frobenius, Q-quantile, Jacobian norm
- Fisher Information, Contractive loss
- MAPE, LWLN, Jensen-Shannon, Auto-regressive

**Layer-wise Distances** (5 subdistances):
- conv1_weights (2080 weights)
- conv1_bias (26 weights)
- conv2_weights (384 weights)
- conv2_bias (24 weights)
- fc_layer (remaining weights)

**Usage:**
```python
from research_scripts.distance_metrics import WeightDistanceMetrics

calculator = WeightDistanceMetrics()
metrics = calculator.compute_all_metrics(weight1, weight2)

# Get markdown table
markdown_table = calculator.format_as_table(metrics)
print(markdown_table)
```

### 2. Robust Topology Analysis

**Features:**
- Mapper algorithm with comprehensive error handling
- Persistent homology computation
- Automatic fallbacks for edge cases
- Handles insufficient samples gracefully

**Usage:**
```python
from research_scripts.robust_topology_analysis import safe_compute_topology_metrics

results = safe_compute_topology_metrics(weight_matrix)
# Returns: {'mapper': {...}, 'persistence': {...}, 'errors': [...]}
```

### 3. WandB Integration

**Logs:**
- Training metrics (loss, learning rate)
- Distance tables (full + layer-wise)
- Topological metrics (Mapper, persistent homology)
- Model artifacts (checkpoints)
- Markdown tables

**Usage:**
```python
from research_scripts.wandb_integration import WandBLogger

logger = WandBLogger(
    project="weight-space-research",
    name="my_experiment",
    config={"model": "medium"},
    enabled=True
)

logger.log_distance_table(distance_metrics)
logger.log_topology_metrics(topology_results)
logger.log_training_progress(epoch, train_loss, val_loss, lr)
```

## 🚀 Running Experiments

### Single Experiment

```bash
# Run with specific configuration
./run_experiments.sh \
    --model medium \
    --overlap 2 \
    --loss mse \
    --epochs 500 \
    --wandb

# Available options:
# Models: tiny, small, medium, large
# Overlaps: 0, 1, 2
# Loss functions: mse, mape, wasserstein, contrastive, q_quantile, lwln, jensen_shannon
```

### Multiple Experiments

```bash
# Run ALL combinations (4 models × 3 overlaps × 7 losses = 84 experiments)
./run_experiments.sh --all --epochs 100 --wandb

# Dry run to see what would be executed
./run_experiments.sh --all --dry-run
```

### Direct Python Training

```bash
conda run -n FCL python3 train_with_config.py \
    --model-size medium \
    --overlap 2 \
    --loss mse \
    --epochs 500 \
    --batch-size 32 \
    --lr 1e-4 \
    --wandb \
    --wandb-name "medium_overlap2_mse"
```

## 📈 Sample Counts Per Scenario

The experiment runner automatically counts and displays samples for each scenario:

```
Scenario Samples (overlap=2):
  Activation | Epoch | Train Samples
  -----------|-------|---------------
  gelu       | 10    | 1234
  gelu       | 20    | 2345
  ...
```

## 🔍 Analyzing Results

### Load and Analyze Checkpoint

```python
import torch
from Double_input_transformer import TransformerAE
from config import MODEL_CONFIGS

# Load checkpoint
checkpoint = torch.load(
    "research_results/checkpoints/best_medium_model.pth",
    map_location='cpu',
    weights_only=False  # Important for PyTorch 2.6+
)

# Create model
config = checkpoint['config']
model = TransformerAE(
    max_seq_len=config.max_seq_len,
    N=config.N,
    heads=config.heads,
    d_model=config.d_model,
    d_ff=config.d_ff,
    neck=config.neck,
    dropout=config.dropout
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for prediction
output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
```

### Visualize Training History

```python
import json
import matplotlib.pyplot as plt

with open("research_results/training_history.json") as f:
    history = json.load(f)

plt.figure(figsize=(12, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("training_curves.png", dpi=150)
```

### Compute Distance Metrics

```python
from research_scripts.distance_metrics import WeightDistanceMetrics

# Load your weights
w1 = ...  # Shape: (2464,)
w2 = ...  # Shape: (2464,)

# Compute all metrics
calculator = WeightDistanceMetrics()
metrics = calculator.compute_all_metrics(w1, w2)

# Access results
print("Full Euclidean:", metrics['full']['euclidean'])
print("Conv1 Euclidean:", metrics['layerwise']['conv1_weights']['euclidean'])

# Generate markdown table
table = calculator.format_as_table(metrics)
with open("distance_report.md", "w") as f:
    f.write(table)
```

## 📋 Loss Functions Available

1. **MSE** - Mean Squared Error (default)
2. **MAPE** - Mean Absolute Percentage Error
3. **Wasserstein** - Earth Mover's Distance approximation
4. **Contrastive** - Contrastive loss for similarity
5. **Q-Quantile** - Quantile regression loss
6. **LWLN** - Layer-wise Loss Normalization
7. **Jensen-Shannon** - JS divergence for distributions

## 🛡️ Error Handling

All modules include comprehensive error handling:

- **Mapper Algorithm**: Handles insufficient samples, constant lens functions, clustering failures
- **Persistent Homology**: Subsamples large datasets, handles missing libraries
- **Distance Metrics**: Graceful degradation for edge cases
- **WandB Logging**: Continues even if logging fails

## 📊 Output Files

Each experiment creates:

```
experiment_results/
├── {model}_{overlap}_{loss}/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   ├── final_model.pth
│   │   └── checkpoint_epoch_*.pth
│   ├── training_history.json
│   ├── experiment_summary.md
│   └── distance_metrics_epoch_*.md
└── logs/
    └── {model}_{overlap}_{loss}.log
```

## 🎯 Next Steps

1. **Analyze trained model**:
   ```bash
   conda run -n FCL python3 load_and_analyze_results.py
   ```

2. **Run experiments with different configurations**:
   ```bash
   ./run_experiments.sh --model small --overlap 1 --loss wasserstein --wandb
   ```

3. **Compare multiple models**:
   ```bash
   ./run_experiments.sh --all --epochs 100 --wandb
   ```

4. **Export results to WandB**:
   - All metrics automatically logged if `--wandb` flag is used
   - View at: https://wandb.ai/your-username/weight-space-research

## 📚 Module Reference

### Core Modules
- `Double_input_transformer.py` - TransformerAE architecture (FIXED)
- `config.py` - Model configurations
- `train_medium_model.py` - Simple training script
- `train_with_config.py` - Advanced training with all features

### Research Modules
- `research_scripts/distance_metrics.py` - Distance calculations
- `research_scripts/robust_topology_analysis.py` - Topology metrics
- `research_scripts/wandb_integration.py` - WandB logging
- `research_scripts/utils/` - Data loading, visualization, etc.

### Experiment Scripts
- `run_experiments.sh` - Batch experiment runner
- `load_and_analyze_results.py` - Result analysis

## 🐛 Troubleshooting

**Issue**: Checkpoint loading fails
- **Fix**: Use `weights_only=False` in `torch.load()`

**Issue**: Mapper finds no clusters
- **Fix**: Automatically falls back to KMeans (handled internally)

**Issue**: Not enough samples for topology
- **Fix**: Gracefully degrades and logs warning (handled internally)

**Issue**: WandB not available
- **Fix**: Logging disabled automatically, training continues

## 📞 Support

For issues or questions:
1. Check logs in `experiment_results/logs/`
2. Review error messages (comprehensive error handling included)
3. Verify conda environment: `conda activate FCL`
