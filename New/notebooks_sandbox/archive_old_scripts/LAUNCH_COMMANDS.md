# Launch Commands for Advanced Experiments

## 🚀 Complete System Ready

### System Features
- ✅ Hierarchical loss system (individual→layerwise→regularized→mixed)
- ✅ All 12 attention heads visualization
- ✅ Layerwise metrics tracking in CSV
- ✅ Advanced topology: Mapper, Gromov-Wasserstein, persistence images/landscapes
- ✅ WandB enabled by default
- ✅ 300 epochs default for scale experiments
- ✅ Batch size 16 optimized for memory

## 📊 Loss Hierarchy

### Level 1: Individual Losses (13 losses)
- MSE, MAE, MAPE, Quantile, Sinkhorn, FFT, MelSpec, JS, KL, Frobenius, LogNorm, FIM, AUTO

### Level 2: Layerwise Versions (14 losses)
- LW_MSE, LW_MAE, LW_MAPE, LW_Quantile, LW_Sinkhorn, LW_FFT, LW_MelSpec, LW_JS, LW_KL, LW_Frobenius, LW_LogNorm, LW_FIM, LWLN

### Level 3: Regularized Full (7 losses)
- MSE+0.05*Frobenius
- MSE+0.1*LogNorm
- MAPE+0.1*JS
- Sinkhorn+0.15*KL
- FFT+0.1*MelSpec
- Quantile+0.05*FIM
- MAE+0.05*Frobenius

### Level 4: Regularized Layerwise (5 losses)
- LW_MSE+0.05*LW_Frobenius
- LW_MSE+0.1*LW_LogNorm
- LW_MAPE+0.1*LW_JS
- LW_MAE+0.05*LW_FIM
- LW_FFT+0.1*LW_MelSpec

### Level 5: Mixed Regularization (4 losses)
- MSE+LW0.1*Frobenius+F0.05*LogNorm
- MAPE+LW0.1*JS+F0.05*KL
- Sinkhorn+LW0.15*MAE+F0.05*FIM
- FFT+LW0.1*MelSpec+F0.05*Frobenius

**Total: 43 loss configurations**

## 🎯 Recommended Launch Command

### All Overlaps and Loss Mixture at Scale

```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Full experiment suite: All overlaps (0,1,2) × All loss hierarchy
python3 run_advanced_experiments.py \
    --models medium \
    --overlaps 0 1 2 \
    --epochs 300 \
    --batch-size 16 \
    --wandb
```

**This will run**:
- 3 overlaps × 43 loss configurations = **129 experiments**
- 300 epochs each
- Batch size 16
- WandB logging enabled
- Estimated time: **~2-3 weeks** (with GPU)

## 📝 Alternative Launch Commands

### Quick Test (5 minutes)
```bash
# Test system with tiny model, 2 epochs
python3 run_advanced_experiments.py \
    --single \
    --models tiny \
    --overlaps 2 \
    --losses MSE \
    --epochs 2 \
    --batch-size 16
```

### Single Overlap, All Losses (1 week)
```bash
# Overlap 2 only, all 43 losses
python3 run_advanced_experiments.py \
    --models medium \
    --overlaps 2 \
    --epochs 300 \
    --batch-size 16 \
    --wandb
```

### All Overlaps, Key Losses Only (3-4 days)
```bash
# All overlaps, selected key losses
python3 run_advanced_experiments.py \
    --models medium \
    --overlaps 0 1 2 \
    --losses MSE LW_MSE "MSE+0.05*Frobenius" "LW_MSE+0.05*LW_Frobenius" "MSE+LW0.1*Frobenius+F0.05*LogNorm" \
    --epochs 300 \
    --batch-size 16 \
    --wandb
```

### Multiple Models, Single Overlap (1-2 weeks)
```bash
# All model sizes, overlap 2, experiment sequence
python3 run_advanced_experiments.py \
    --models tiny small medium large huge \
    --overlaps 2 \
    --epochs 300 \
    --batch-size 16 \
    --wandb
```

### Huge Model, All Overlaps (2-3 weeks)
```bash
# Huge model (~84M params), all overlaps, all losses
python3 run_advanced_experiments.py \
    --models huge \
    --overlaps 0 1 2 \
    --epochs 300 \
    --batch-size 8 \
    --wandb
```

## 🔧 Command Options

```
--models MODEL [MODEL ...]
    Model sizes: tiny, small, medium, large, huge
    Default: ['medium']

--overlaps OVERLAP [OVERLAP ...]
    Overlap levels: 0, 1, 2
    Default: [2]

--losses LOSS [LOSS ...]
    Loss names (use quotes for complex names)
    Default: Full experiment sequence (43 losses)

--epochs N
    Number of training epochs
    Default: 300

--batch-size N
    Batch size for training
    Default: 16

--wandb
    Enable WandB logging (DEFAULT: True)

--no-wandb
    Disable WandB logging

--output-dir PATH
    Base output directory
    Default: ./experiments

--single
    Run only first combination (for testing)
```

## 📊 Output Structure

Each experiment creates:
```
experiments/{model}_overlap{N}_{loss_name}/
├── checkpoints/
│   ├── best_model.pth
│   ├── final_model.pth
│   └── checkpoint_epoch_XXXX.pth
│
├── attention_heatmaps/              # ALL 12 heads visualized
│   ├── epoch_0010_enc1.png
│   ├── epoch_0010_enc2.png
│   ├── epoch_0010_dec.png
│   └── ...
│
├── predicted_weights/               # Every epoch
│   ├── epoch_0001_predictions.npy
│   ├── epoch_0002_predictions.npy
│   ├── ...
│   ├── epoch_0300_predictions.npy
│   └── targets.npy
│
├── topology/                        # Advanced topology
│   ├── topology_epoch_0050.json
│   ├── pers_image_dim_0_epoch_0050.png
│   ├── pers_image_dim_1_epoch_0050.png
│   └── ...
│
├── metrics/
│   └── test_metrics_full_and_layerwise.csv  # Both full AND layerwise
│
└── training_history.json
```

## 📈 Performance Estimates (300 epochs, batch 16)

| Model | Params | Time/Epoch | Memory | 300 Epochs | 43 Losses | 3 Overlaps |
|-------|--------|-----------|--------|------------|-----------|------------|
| tiny | 4.09M | ~30s | ~2GB | ~2.5hr | ~4.5 days | ~13 days |
| small | 8.46M | ~1min | ~4GB | ~5hr | ~9 days | ~27 days |
| medium | 18.33M | ~2min | ~6GB | ~10hr | ~18 days | ~54 days |
| large | 43.52M | ~5min | ~10GB | ~25hr | ~44 days | ~132 days |
| huge | 84.41M | ~15min | ~14GB | ~75hr | ~134 days | ~402 days |

**Recommended**: Use medium model for full suite (2-3 weeks)

## 🎯 Recommended Workflow

### Phase 1: Quick Validation (1 hour)
```bash
# Test with tiny model, 10 epochs, key losses
python3 run_advanced_experiments.py \
    --single \
    --models tiny \
    --overlaps 2 \
    --losses MSE \
    --epochs 10 \
    --batch-size 16
```

### Phase 2: Single Overlap Sweep (1 week)
```bash
# Medium model, overlap 2, all losses
python3 run_advanced_experiments.py \
    --models medium \
    --overlaps 2 \
    --epochs 300 \
    --batch-size 16 \
    --wandb
```

### Phase 3: Full Overlap Study (2-3 weeks)
```bash
# Medium model, all overlaps, all losses
python3 run_advanced_experiments.py \
    --models medium \
    --overlaps 0 1 2 \
    --epochs 300 \
    --batch-size 16 \
    --wandb
```

### Phase 4: Model Size Ablation (1-2 weeks)
```bash
# All models, overlap 2, selected losses
python3 run_advanced_experiments.py \
    --models tiny small medium large huge \
    --overlaps 2 \
    --losses MSE LW_MSE "MSE+0.05*Frobenius" \
    --epochs 300 \
    --batch-size 16 \
    --wandb
```

## 🔍 Monitoring Progress

### Check Running Experiments
```bash
# View latest results
ls -lth experiments/

# Check specific experiment
tail -f experiments/medium_overlap2_MSE/training_history.json

# View attention heatmaps
ls -lth experiments/*/attention_heatmaps/*.png | head -20

# Check topology results
cat experiments/*/topology/topology_epoch_0050.json
```

### WandB Dashboard
```
https://wandb.ai/your-username/fcl-advanced
```

## 📊 Analysis After Training

### Load Predicted Weights Trajectory
```python
import numpy as np
import matplotlib.pyplot as plt

exp_dir = "experiments/medium_overlap2_MSE"
epochs = range(1, 301)

# Load all predictions
trajectory = []
for epoch in epochs:
    pred = np.load(f"{exp_dir}/predicted_weights/epoch_{epoch:04d}_predictions.npy")
    trajectory.append(pred[0])  # First sample

trajectory = np.array(trajectory)  # Shape: (300, 2464)

# Visualize
plt.figure(figsize=(15, 6))
plt.imshow(trajectory.T, aspect='auto', cmap='viridis')
plt.xlabel('Epoch')
plt.ylabel('Weight Index')
plt.title('Weight Evolution Over Training')
plt.colorbar()
plt.savefig('weight_trajectory.png', dpi=150)
```

### Analyze Layerwise Metrics
```python
import pandas as pd

# Load metrics
metrics_df = pd.read_csv("experiments/medium_overlap2_MSE/metrics/test_metrics_full_and_layerwise.csv")

# Compare full vs layerwise
full_cols = [c for c in metrics_df.columns if not c.startswith('conv') and not c.startswith('fc')]
layerwise_cols = [c for c in metrics_df.columns if c.startswith('conv') or c.startswith('fc')]

print("Full metrics:", full_cols)
print("Layerwise metrics:", layerwise_cols)

# Analyze by layer
for layer in ['conv1_weights', 'conv1_bias', 'conv2_weights', 'conv2_bias', 'fc_layer']:
    layer_metrics = [c for c in layerwise_cols if c.startswith(layer)]
    print(f"\n{layer}:")
    print(metrics_df[layer_metrics].describe())
```

### Compare Topology Across Epochs
```python
import json
import glob

exp_dir = "experiments/medium_overlap2_MSE/topology"
topology_files = sorted(glob.glob(f"{exp_dir}/topology_epoch_*.json"))

betti_0 = []
betti_1 = []
mapper_nodes = []

for f in topology_files:
    with open(f) as fp:
        data = json.load(fp)
        betti_0.append(data['betti_numbers'].get('betti_0', 0))
        betti_1.append(data['betti_numbers'].get('betti_1', 0))
        mapper_nodes.append(data['mapper_stats'].get('mapper_n_nodes', 0))

# Plot evolution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(betti_0)
axes[0].set_title('Betti 0 (Connected Components)')
axes[1].plot(betti_1)
axes[1].set_title('Betti 1 (Loops)')
axes[2].plot(mapper_nodes)
axes[2].set_title('Mapper Graph Nodes')
plt.savefig('topology_evolution.png', dpi=150)
```

## ✅ System Verification

```bash
# Quick test (2 minutes)
python3 run_advanced_experiments.py \
    --single \
    --models tiny \
    --overlaps 2 \
    --losses MSE \
    --epochs 2 \
    --batch-size 16

# Check outputs
ls -R experiments/tiny_overlap2_MSE/
```

Expected outputs:
- ✅ Checkpoints
- ✅ Attention heatmaps (all 12 heads)
- ✅ Predicted weights (2 files)
- ✅ Topology results
- ✅ Metrics CSV (full + layerwise)
- ✅ Training history

## 🎉 Ready to Launch!

**Main command for your request**:
```bash
python3 run_advanced_experiments.py \
    --models medium \
    --overlaps 0 1 2 \
    --epochs 300 \
    --batch-size 16 \
    --wandb
```

This will run **129 experiments** (3 overlaps × 43 losses) at scale with all advanced features!
