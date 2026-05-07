# Comprehensive Experiment System - Complete Guide

## ✅ What's Been Built

### Complete Integrated System
**File**: `run_comprehensive_experiments.py`

**Features**:
- ✅ All 23+ loss functions from meta.ipynb
- ✅ 10 Loss pairs (main + regularized)
- ✅ Gated attention mechanism (prevents collapse)
- ✅ Persistent homology analysis (every 50 epochs)
- ✅ RMT spectral analysis (every 50 epochs)
- ✅ NTK trainability metrics (every 100 epochs)
- ✅ Super weight identification (every 100 epochs)
- ✅ Predicted weights saving (every epoch)
- ✅ Attention heatmap visualization (every 10 epochs)
- ✅ WandB logging integration
- ✅ 500 epochs default
- ✅ Structured results output

### Fixed Issues
1. ✅ Frobenius norm error in `utils_consolidated.py`
2. ✅ Command-line argument `--models` (was `--model-sizes`)
3. ✅ Default epochs changed to 500

## 🚀 Quick Start

### Test Run (10 epochs, ~5 minutes)
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Quick test with tiny model
python3 run_comprehensive_experiments.py \
    --single \
    --models tiny \
    --overlaps 2 \
    --loss-pairs 0 \
    --epochs 10
```

### Single Full Experiment (500 epochs, ~25 hours for huge model)
```bash
# Medium model, overlap 2, MSE+LWLN loss pair
python3 run_comprehensive_experiments.py \
    --single \
    --models medium \
    --overlaps 2 \
    --loss-pairs 0 \
    --epochs 500 \
    --wandb
```

### Multiple Experiments
```bash
# Test all loss pairs on medium model
python3 run_comprehensive_experiments.py \
    --models medium \
    --overlaps 2 \
    --loss-pairs 0 1 2 3 4 \
    --epochs 500 \
    --wandb
```

### Full Suite (All combinations)
```bash
# WARNING: This will take days to complete
python3 run_comprehensive_experiments.py \
    --models tiny small medium large huge \
    --overlaps 0 1 2 \
    --loss-pairs 0 1 2 3 4 5 6 7 8 9 \
    --epochs 500 \
    --wandb
```

## 📊 Loss Pairs

The system uses **loss pairs** (main loss + regularization):

| Index | Main Loss | Regularization | Weight | Description |
|-------|-----------|----------------|--------|-------------|
| 0 | MSE | LWLN | 0.1 | MSE + layer-wise normalization |
| 1 | MAPE | JS | 0.1 | MAPE + Jensen-Shannon |
| 2 | sinkhorn | LWWS | 0.2 | Wasserstein + layer-wise Wasserstein |
| 3 | AUTO | FIM | 0.05 | Autoregressive + Fisher info |
| 4 | FFT | Mel_L2 | 0.1 | Frequency + mel-spectrogram |
| 5 | Q-quantile | log-norm | 0.05 | Quantile + log normalization |
| 6 | ws_scipy | KL | 0.1 | Wasserstein + KL divergence |
| 7 | MAE | LWLN | 0.1 | MAE + layer-wise normalization |
| 8 | Mel_FID | FFT | 0.15 | Mel-FID + frequency |
| 9 | Frobenius | log-norm | 0.05 | Frobenius + log norm |

**Example**: Loss pair 0 computes:
```python
loss = MSE(pred, target) + 0.1 * LWLN(pred, target)
```

## 📁 Output Structure

Each experiment creates a comprehensive output directory:

```
experiments/{model_size}_overlap{N}_{loss_pair_name}/
│
├── checkpoints/
│   ├── best_model.pth              # Best validation loss
│   ├── final_model.pth             # Final epoch
│   └── checkpoint_epoch_XXXX.pth   # Every 10 epochs
│
├── attention_heatmaps/              # Attention visualization
│   ├── epoch_0010_enc1.png         # Encoder 1 (all heads)
│   ├── epoch_0010_enc2.png         # Encoder 2 (all heads)
│   ├── epoch_0010_dec.png          # Decoder (all heads)
│   ├── epoch_0020_enc1.png
│   └── ...                         # Every 10 epochs
│
├── predicted_weights/               # ⭐ NEW: Predicted weights
│   ├── epoch_0001_predictions.npy  # Predictions at each epoch
│   ├── epoch_0002_predictions.npy
│   ├── ...
│   ├── epoch_0500_predictions.npy
│   └── targets.npy                 # Ground truth (saved once)
│
├── analysis/                        # ⭐ NEW: Advanced analysis
│   ├── persistent_homology.csv     # Betti numbers over time
│   ├── rmt_analysis.csv            # Spectral analysis
│   ├── ntk_analysis.csv            # NTK trainability metrics
│   └── super_weights.csv           # Super weight identification
│
├── metrics/
│   └── test_metrics_full.csv       # 13 distance metrics on test set
│
└── training_history.json            # Complete training history
```

Global summary:
```
experiments/
└── comprehensive_summary.csv        # All experiments summary
```

## 🔬 Analysis Features

### 1. Gated Attention Mechanism
**Purpose**: Prevent attention collapse by gating each attention head

**Implementation**:
- Gate network learns to weight each attention head
- Prevents uniform attention patterns
- Improves representation diversity

**Monitoring**: Check attention heatmaps for diverse patterns

### 2. Persistent Homology (Every 50 epochs)
**Purpose**: Analyze topological structure of neck representations

**Metrics Computed**:
- Betti numbers (β₀, β₁, β₂)
- Persistence diagrams
- Topological features

**Output**: `analysis/persistent_homology.csv`

**Interpretation**:
- β₀: Number of connected components
- β₁: Number of loops/holes
- β₂: Number of voids

### 3. RMT Spectral Analysis (Every 50 epochs)
**Purpose**: Analyze weight matrix eigenvalue distributions

**Metrics Computed**:
- Spectral radius
- Max/min eigenvalues
- Spectral density
- Marchenko-Pastur comparison

**Output**: `analysis/rmt_analysis.csv`

**Interpretation**:
- Spectral radius → model stability
- Eigenvalue distribution → weight initialization quality

### 4. NTK Trainability Metrics (Every 100 epochs)
**Purpose**: Measure neural tangent kernel properties

**Metrics Computed**:
- Mean gradient norm
- Max gradient norm
- Trainability score

**Output**: `analysis/ntk_analysis.csv`

**Interpretation**:
- High gradient norms → good trainability
- Low gradient norms → potential vanishing gradients

### 5. Super Weight Analysis (Every 100 epochs)
**Purpose**: Identify high-influence weights

**Metrics Computed**:
- Number of super weights (top 5%)
- Super weight ratio
- Importance threshold
- Mean/max importance

**Output**: `analysis/super_weights.csv`

**Interpretation**:
- High super weight ratio → sparse important weights
- Low ratio → distributed importance

### 6. Predicted Weights Saving (Every epoch)
**Purpose**: Track weight trajectory for HMMR segmentation

**Files**: `predicted_weights/epoch_XXXX_predictions.npy`

**Usage**:
```python
import numpy as np

# Load all predictions
predictions = []
for epoch in range(1, 501):
    pred = np.load(f"predicted_weights/epoch_{epoch:04d}_predictions.npy")
    predictions.append(pred)

# Analyze trajectory
trajectory = np.array(predictions)  # Shape: (500, n_samples, 2464)
```

## 📈 WandB Integration

When `--wandb` is enabled, logs:

**Metrics**:
- `train_loss`, `val_loss` (every epoch)
- `learning_rate` (every epoch)
- `ph/betti_0`, `ph/betti_1`, `ph/betti_2` (every 50 epochs)
- `rmt/spectral_radius` (every 50 epochs)
- `ntk/mean_grad_norm`, `ntk/trainability_score` (every 100 epochs)
- `super_weights/n_super_weights`, `super_weights/super_weight_ratio` (every 100 epochs)

**Images**:
- `attention/encoder1` (every 10 epochs)
- `attention/encoder2` (every 10 epochs)
- `attention/decoder` (every 10 epochs)

**Dashboard**: https://wandb.ai/your-username/fcl-comprehensive

## 🎯 Example Workflows

### Workflow 1: Quick Validation (10 min)
```bash
# Test system works
python3 run_comprehensive_experiments.py \
    --single --models tiny --overlaps 2 --loss-pairs 0 --epochs 10
```

### Workflow 2: Loss Function Comparison (2-3 days)
```bash
# Compare all 10 loss pairs on medium model
python3 run_comprehensive_experiments.py \
    --models medium \
    --overlaps 2 \
    --loss-pairs 0 1 2 3 4 5 6 7 8 9 \
    --epochs 500 \
    --wandb
```

### Workflow 3: Model Size Ablation (1 week)
```bash
# Test all model sizes with best loss pair
python3 run_comprehensive_experiments.py \
    --models tiny small medium large huge \
    --overlaps 2 \
    --loss-pairs 0 \
    --epochs 500 \
    --wandb
```

### Workflow 4: Overlap Study (3-4 days)
```bash
# Test all overlaps on large model
python3 run_comprehensive_experiments.py \
    --models large \
    --overlaps 0 1 2 \
    --loss-pairs 0 1 2 \
    --epochs 500 \
    --wandb
```

### Workflow 5: Full Research Suite (2-3 weeks)
```bash
# All combinations
python3 run_comprehensive_experiments.py \
    --models tiny small medium large huge \
    --overlaps 0 1 2 \
    --loss-pairs 0 1 2 3 4 5 6 7 8 9 \
    --epochs 500 \
    --wandb
```

## ⚙️ Command-Line Options

```
--models MODEL [MODEL ...]
    Model sizes to test: tiny, small, medium, large, huge
    Default: ['medium']

--overlaps OVERLAP [OVERLAP ...]
    Overlap levels: 0, 1, 2
    Default: [2]

--loss-pairs INDEX [INDEX ...]
    Loss pair indices: 0-9
    Default: [0, 1, 2]

--epochs N
    Number of training epochs
    Default: 500

--batch-size N
    Batch size for training
    Default: 32

--lr FLOAT
    Learning rate
    Default: 1e-4

--wandb
    Enable WandB logging
    Default: False

--output-dir PATH
    Base output directory
    Default: ./experiments

--single
    Run only first combination (for testing)
    Default: False
```

## 📊 Performance Estimates

| Model | Params | Batch Size | Time/Epoch | Memory | 500 Epochs |
|-------|--------|-----------|------------|--------|------------|
| tiny | 4.09M | 64 | ~30s | ~2GB | ~4hr |
| small | 8.46M | 32 | ~1min | ~4GB | ~8hr |
| medium | 18.33M | 32 | ~2min | ~6GB | ~17hr |
| large | 43.52M | 16 | ~5min | ~10GB | ~42hr |
| huge | 84.41M | 8 | ~15min | ~14GB | ~125hr |

**Full Suite Estimate** (5 models × 3 overlaps × 10 loss pairs × 500 epochs):
- Total experiments: 150
- Total epochs: 75,000
- Estimated time: **2-3 weeks** (with parallelization)
- Storage needed: **~200GB** (with all analysis)

## 🔍 Monitoring Progress

### Check Latest Results
```bash
# View most recent attention heatmaps
ls -lth experiments/*/attention_heatmaps/*.png | head -10

# Check training progress
tail -f experiments/*/training_history.json

# View analysis results
cat experiments/*/analysis/persistent_homology.csv
```

### View Predicted Weights
```python
import numpy as np
import matplotlib.pyplot as plt

# Load trajectory
exp_dir = "experiments/medium_overlap2_MSE_0.10xLWLN"
epochs = range(1, 501)
trajectory = []

for epoch in epochs:
    pred = np.load(f"{exp_dir}/predicted_weights/epoch_{epoch:04d}_predictions.npy")
    trajectory.append(pred[0])  # First sample

trajectory = np.array(trajectory)  # Shape: (500, 2464)

# Plot weight evolution
plt.figure(figsize=(12, 6))
plt.imshow(trajectory.T, aspect='auto', cmap='viridis')
plt.xlabel('Epoch')
plt.ylabel('Weight Index')
plt.title('Weight Trajectory Over Training')
plt.colorbar(label='Weight Value')
plt.savefig('weight_trajectory.png', dpi=150)
```

### Analyze Persistent Homology
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load PH results
ph_df = pd.read_csv("experiments/medium_overlap2_MSE_0.10xLWLN/analysis/persistent_homology.csv")

# Plot Betti numbers
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, betti in enumerate(['betti_0', 'betti_1', 'betti_2']):
    axes[i].plot(ph_df['epoch'], ph_df[betti])
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel(f'β{i}')
    axes[i].set_title(f'Betti Number β{i}')
    axes[i].grid(True)
plt.tight_layout()
plt.savefig('betti_numbers.png', dpi=150)
```

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size
```bash
python3 run_comprehensive_experiments.py --batch-size 16 ...
```

### Issue: Slow Training
**Solution**: Use smaller model or fewer epochs for testing
```bash
python3 run_comprehensive_experiments.py --models tiny --epochs 100 ...
```

### Issue: WandB Not Logging
**Solution**: Login to WandB first
```bash
wandb login
```

### Issue: Missing Dependencies
**Solution**: Install required packages
```bash
conda activate FCL
pip install ripser persim geomloss
```

## 📚 Next Steps

### Post-Training Analysis

1. **HMMR Segmentation** (use notebook 10):
```python
# Load all predicted weights
# Apply HMMR to find regime changes
# Identify critical training phases
```

2. **Gradient Frequency Analysis** (use notebook 08):
```python
# Analyze gradient spectra
# Identify super weights
# Study frequency components
```

3. **Multi-Initialization Comparison** (use notebook 11):
```python
# Compare across different runs
# Statistical significance testing
# Robustness analysis
```

## ✅ System Verification

**Test the complete system**:
```bash
# Quick test (5 minutes)
python3 run_comprehensive_experiments.py \
    --single --models tiny --overlaps 2 --loss-pairs 0 --epochs 10

# Check outputs
ls -R experiments/tiny_overlap2_MSE_0.10xLWLN/
```

**Expected outputs**:
- ✅ Checkpoints saved
- ✅ Attention heatmaps generated
- ✅ Predicted weights saved (10 files)
- ✅ Analysis CSVs created
- ✅ Training history JSON
- ✅ Test metrics CSV

## 🎉 Ready to Run!

The complete integrated system is ready with all features:
- ✅ 23+ loss functions with pairs
- ✅ Gated attention
- ✅ Persistent homology
- ✅ RMT analysis
- ✅ NTK metrics
- ✅ Super weight identification
- ✅ Predicted weights saving
- ✅ Structured results
- ✅ 500 epochs default

**Start with**:
```bash
python3 run_comprehensive_experiments.py \
    --single --models tiny --overlaps 2 --loss-pairs 0 --epochs 10
```
