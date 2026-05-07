# Special Losses Training Guide

## 📋 Overview

Train 4 special losses (Sinkhorn + Persistence-based) across all overlaps sequentially.

**Losses:**
1. **Sinkhorn** - Optimal transport loss (full sequence)
2. **LW_Sinkhorn** - Layerwise Sinkhorn
3. **PersLandscape** - Persistence landscape loss (full sequence)
4. **LW_PersLandscape** - Layerwise persistence landscape

**Execution order:** Overlap 0 (all 4) → Overlap 1 (all 4) → Overlap 2 (all 4)

---

## 🚀 How to Run

### Single Command
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox
conda run -n FCL python3 train_special_losses.py
```

**What happens:**
1. Trains all 4 losses for overlap 0
2. Trains all 4 losses for overlap 1
3. Trains all 4 losses for overlap 2
4. Saves results to `special_losses_results.json`

---

## 📊 Expected Timeline

### Per Loss (150 epochs)
- **Overlap 0:** ~3 hours
- **Overlap 1:** ~10 hours
- **Overlap 2:** ~16 hours

### Total for 4 Losses
- **Overlap 0:** 4 × 3h = 12 hours
- **Overlap 1:** 4 × 10h = 40 hours
- **Overlap 2:** 4 × 16h = 64 hours
- **Grand Total:** ~116 hours (~4.8 days)

---

## 🔬 Persistence Losses Explained

### What is Topological Data Analysis (TDA)?

TDA extracts topological features from data:
- **Connected components** (0-dimensional features)
- **Loops/cycles** (1-dimensional features)
- **Voids** (2-dimensional features)

### Persistence Diagrams

A persistence diagram shows when topological features appear and disappear as we vary a parameter (filtration).

For weight sequences:
1. Treat weights as a 1D function
2. Build simplicial complex (chain of vertices/edges)
3. Compute persistence of connected components
4. Extract birth-death pairs

### Persistence Landscape

Converts persistence diagram to a functional representation:
- Multiple "landscape functions" (λ₁, λ₂, ..., λₖ)
- Each function captures different persistence features
- Evaluated on a grid → vector representation
- **Advantage:** Stable, vectorized, differentiable

**Our implementation:**
- 5 landscapes
- 100 resolution points
- Total: 500-dimensional vector per weight sequence

### Persistence Image

Converts persistence diagram to an image:
- Birth-death pairs → 2D points
- Apply Gaussian weighting (emphasize long-lived features)
- Discretize into pixel grid
- **Advantage:** Interpretable, stable, differentiable

**Our implementation:**
- 20×20 pixel grid
- Bandwidth = 1.0
- Weight function = persistence²
- Total: 400-dimensional vector per weight sequence

---

## 🔧 Technical Details

### Sinkhorn Loss (Fixed)

**Problem:** Device mismatch - geomloss doesn't handle GPU well

**Solution:**
```python
# Move to CPU for geomloss computation
device = pred.device
pred_cpu = pred.cpu()
target_cpu = target.cpu()

# Compute on CPU
loss_value = self.loss(pred_cpu, target_cpu)

# Move back to original device
return loss_value.to(device)
```

### Persistence Losses

**Dependencies:**
- `gudhi` - Persistence computation
- `multipers` - Multiparameter persistence (optional)

**Install:**
```bash
conda install -c conda-forge gudhi
pip install multipers
```

**Fallback:** If GUDHI not available, falls back to MSE loss

---

## 📁 Files Created

### Core Modules
- `core_modules/persistence_losses.py` - 4 new loss classes
- `core_modules/advanced_losses.py` - Updated with persistence losses

### Training Scripts
- `train_special_losses.py` - Sequential training for 4 losses

### Documentation
- `SPECIAL_LOSSES_GUIDE.md` - This file
- `TRAINING_TIME_ANALYSIS.md` - Explains time differences across overlaps

---

## 🧪 Testing

### Test Persistence Losses
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Test import
conda run -n FCL python3 -c "
from core_modules.persistence_losses import PersistenceLandscapeLoss
import torch
loss_fn = PersistenceLandscapeLoss()
pred = torch.randn(2, 2464)
target = torch.randn(2, 2464)
loss = loss_fn(pred, target)
print(f'Loss: {loss.item():.4f}')
"
```

### Test Single Experiment
```bash
# Test Sinkhorn on overlap 0
conda run -n FCL python3 core_modules/run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss Sinkhorn \
    --epochs 10
```

---

## 📊 Why Training Time Increases

From `TRAINING_TIME_ANALYSIS.md`:

### Dataset Sizes
| Overlap | Train Pairs | Ratio |
|---------|-------------|-------|
| 0       | 32,812      | 1×    |
| 1       | 130,620     | 4×    |
| 2       | 206,578     | 6.3×  |

### Training Time
- **Linear scaling** with dataset size
- ~0.3 seconds per training pair
- Overlap 2 has 6× more data → 6× longer training

**This is expected behavior!** More overlap = more valid CNN pairs = larger dataset = longer training.

---

## 🎯 Integration with Tournament

After training completes, results are saved in the same format as other experiments:
- WandB logging
- Checkpoint files
- Metrics JSON
- Can be included in ranking phase

**To include in tournament ranking:**
```bash
# Results will be in Experiments/ directory
# Use same ranking script as other losses
cd tournament_system
conda run -n FCL python3 per_overlap_ranking.py \
    --model-size tiny \
    --top-n 20 \
    --bottom-n 10 \
    --output ../rankings_tiny_with_special.json
```

---

## ✅ Summary

**4 New Losses:**
1. ✅ Sinkhorn (device-fixed)
2. ✅ LW_Sinkhorn
3. ✅ PersLandscape (GUDHI-based)
4. ✅ LW_PersLandscape

**Training:**
- Sequential per overlap (0→1→2)
- ~116 hours total (~4.8 days)
- Fully automated script

**Ready to run:**
```bash
conda run -n FCL python3 train_special_losses.py
```
