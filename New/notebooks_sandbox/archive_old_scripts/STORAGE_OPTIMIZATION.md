# Storage Optimization for 222GB Limit

## 📊 Original Storage Estimate (BEFORE Optimization)

### Per Experiment Breakdown

**Checkpointing (200 epochs, save every 10)**:
- 20 intermediate checkpoints + best + final = 22 total

**Storage by Model Size**:
| Model | Params | Per Checkpoint | 22 Checkpoints | Other Data | Total/Exp | × 129 exps |
|-------|--------|---------------|----------------|------------|-----------|------------|
| tiny | 4M | 16 MB | 352 MB | 53 MB | 405 MB | 52 GB |
| small | 8M | 32 MB | 704 MB | 53 MB | 757 MB | 98 GB |
| medium | 18M | 72 MB | 1.58 GB | 53 MB | 1.63 GB | 210 GB |
| large | 44M | 176 MB | 3.87 GB | 53 MB | 3.92 GB | 506 GB |
| huge | 84M | 336 MB | 7.39 GB | 53 MB | 7.44 GB | 960 GB |

**TOTAL FOR 645 EXPERIMENTS: ~1.8 TB** ❌ **EXCEEDS 222GB LIMIT!**

## ✅ Optimized Storage (AFTER Optimization)

### Changes Made

1. **Checkpointing Frequency**: Every 50 epochs (was 10)
   - 200 epochs ÷ 50 = 4 intermediate checkpoints
   - Plus best + final = **6 total checkpoints** (was 22)
   - **73% reduction in checkpoint storage**

2. **Attention Heatmaps**: Every 25 epochs (was 10)
   - 200 epochs ÷ 25 = 8 saves
   - **60% reduction in attention storage**

3. **Topology Analysis**: Every 50 epochs (unchanged)
   - Still comprehensive but storage-efficient

### Optimized Storage by Model Size

| Model | Checkpoints (6×) | Attention (8×) | Topology | Weights | Metrics | Total/Exp | × 129 exps | Total |
|-------|-----------------|----------------|----------|---------|---------|-----------|------------|-------|
| tiny | 96 MB | 12 MB | 20 MB | 2 MB | 1 MB | 131 MB | 129 | 17 GB |
| small | 192 MB | 12 MB | 20 MB | 2 MB | 1 MB | 227 MB | 129 | 29 GB |
| medium | 432 MB | 12 MB | 20 MB | 2 MB | 1 MB | 467 MB | 129 | 60 GB |
| large | 1.06 GB | 12 MB | 20 MB | 2 MB | 1 MB | 1.09 GB | 129 | 141 GB |
| huge | 2.02 GB | 12 MB | 20 MB | 2 MB | 1 MB | 2.05 GB | 129 | 264 GB |

**TOTAL FOR 645 EXPERIMENTS: ~511 GB** ⚠️ **STILL OVER 222GB**

## 🎯 Recommended Strategy for 222GB Limit

### Option 1: Run Models Sequentially (RECOMMENDED)

Run one model size at a time, analyze results, then delete checkpoints before next model:

**Phase 1: Small Models (17 + 29 = 46 GB)**
```bash
./run_all_experiments.sh --models "tiny small"
# Analyze results
# Delete checkpoints, keep only metrics/analysis
```

**Phase 2: Medium Model (60 GB)**
```bash
./run_all_experiments.sh --models "medium"
# Analyze results
# Delete checkpoints
```

**Phase 3: Large Model (141 GB)**
```bash
./run_all_experiments.sh --models "large"
# Analyze results
# Delete checkpoints
```

**Phase 4: Huge Model (264 GB - SKIP or use external storage)**
```bash
# Skip huge model OR
# Use external drive/cloud storage
```

### Option 2: Further Reduce Storage

**Keep only best + final checkpoints** (no intermediate):
```python
# Modify run_advanced_experiments.py
# Remove: if epoch % 50 == 0: save checkpoint
# Keep only: best_model.pth and final_model.pth
```

**New storage per model**:
| Model | 2 Checkpoints | Other | Total/Exp | × 129 exps |
|-------|--------------|-------|-----------|------------|
| tiny | 32 MB | 35 MB | 67 MB | 8.6 GB |
| small | 64 MB | 35 MB | 99 MB | 13 GB |
| medium | 144 MB | 35 MB | 179 MB | 23 GB |
| large | 352 MB | 35 MB | 387 MB | 50 GB |
| huge | 672 MB | 35 MB | 707 MB | 91 GB |

**TOTAL: ~186 GB** ✅ **FITS IN 222GB!**

### Option 3: Reduce Number of Experiments

**Run fewer loss configurations**:
- Instead of 43 losses, run top 15-20 most important
- Reduces from 645 to ~300 experiments
- Storage: ~250 GB → ~120 GB

**Priority losses to keep**:
1. MSE (baseline)
2. LW_MSE (layerwise baseline)
3. MSE+0.05*Frobenius (regularized)
4. MAPE
5. Sinkhorn
6. FFT
7. AUTO
8. LW_MAPE
9. MSE+LW0.1*Frobenius+F0.05*LogNorm (mixed)
10. Quantile

## 🔧 Implementation

### Current Optimizations (Already Applied)

✅ Checkpoints every 50 epochs (was 10)
✅ Attention heatmaps every 25 epochs (was 10)
✅ Topology every 50 epochs

### Additional Optimization (If Needed)

To reduce to 186GB total, edit `run_advanced_experiments.py`:

```python
# Remove this block:
if epoch % 50 == 0:
    torch.save(checkpoint, checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pth")

# Keep only:
# - best_model.pth (saved when val_loss improves)
# - final_model.pth (saved at end)
```

## 📝 Storage Management Script

Create a cleanup script to manage storage between runs:

```bash
#!/bin/bash
# cleanup_checkpoints.sh

# Keep only best and final models, delete intermediate checkpoints
find experiments/*/checkpoints -name "checkpoint_epoch_*.pth" -delete

echo "Intermediate checkpoints deleted. Kept best_model.pth and final_model.pth"
```

## 🎯 Recommended Workflow for 222GB

1. **Start with current optimization** (checkpoints every 50 epochs)
   - Monitor storage usage
   - Should use ~511 GB total

2. **If storage fills up**:
   - Run models sequentially (one at a time)
   - Delete checkpoints after analyzing each model
   - Keep only metrics CSVs and analysis results

3. **If still need more space**:
   - Remove intermediate checkpoints (keep only best + final)
   - Reduces to ~186 GB total

4. **Alternative**:
   - Use external storage for checkpoints
   - Keep only analysis results locally

## 📊 Current Configuration Summary

**With Current Optimizations**:
- Checkpoints: Every 50 epochs + best + final = 6 per experiment
- Attention: Every 25 epochs = 8 saves per experiment
- Topology: Every 50 epochs = 4 saves per experiment
- Predicted weights: Every epoch = 200 files per experiment

**Estimated Storage**: ~511 GB for all 645 experiments

**Recommendation**: Run models sequentially or remove intermediate checkpoints to fit in 222GB.
