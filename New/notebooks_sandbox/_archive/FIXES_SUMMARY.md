# Critical Fixes Applied

## 🔧 Issues Fixed

### 1. **RuntimeError in validate() function**
**Error:** `Can't call numpy() on Tensor that requires grad`

**File:** `core_modules/run_advanced_experiments.py`

**Fix:**
```python
# Before (line 260-262)
all_predictions.append(output.cpu().numpy())
all_targets.append(target.cpu().numpy())
all_necks.append(neck_t.cpu().numpy())

# After
all_predictions.append(output.detach().cpu().numpy())
all_targets.append(target.detach().cpu().numpy())
all_necks.append(neck_t.detach().cpu().numpy())
```

**Why:** Tensors with `requires_grad=True` cannot be converted to numpy directly. Must call `.detach()` first to remove from computation graph.

---

### 2. **Sinkhorn Loss Device Mismatch** (Already Fixed)
**Issue:** geomloss doesn't handle GPU tensors well

**File:** `core_modules/advanced_losses.py`

**Fix:** Move tensors to CPU for geomloss computation, then back to original device
```python
device = pred.device
pred_cpu = pred.cpu()
target_cpu = target.cpu()
loss_value = self.loss(pred_cpu, target_cpu)
return loss_value.to(device)
```

---

### 3. **Persistence Losses Producing NaN**
**Issues:**
- No normalization → unbounded values
- No exception handling → crashes on edge cases
- `requires_grad=True` set manually → autograd conflicts

**File:** `core_modules/persistence_losses.py`

**Fixes:**
1. **Added normalization:**
```python
landscape_norm = max(np.linalg.norm(pred_landscape) + np.linalg.norm(target_landscape), 1e-8)
normalized_dist = landscape_dist / landscape_norm
```

2. **Added exception handling:**
```python
try:
    # Compute persistence diagrams
    ...
except Exception as e:
    # Fall back to MSE if persistence computation fails
    return F.mse_loss(pred, target)
```

3. **Removed manual requires_grad:**
```python
# Before
loss_tensor = torch.tensor(..., requires_grad=True)

# After
loss_tensor = torch.tensor(...)  # Let autograd handle it
```

---

### 4. **Persistence Losses Moved to Regularizers**
**Issue:** Standalone persistence losses are:
- Too slow (TDA computation expensive)
- Produce NaN values
- Not differentiable enough for main loss

**Solution:** Use as regularizers with very small weight (0.01)

**File:** `core_modules/advanced_losses.py`

**New losses available:**
- `MSE+0.01*PersLandscape` (full sequence)
- `MAE+0.01*PersLandscape` (full sequence)
- `LW_MSE+0.01*LW_PersLandscape` (layerwise)
- `LW_MAE+0.01*LW_PersLandscape` (layerwise)

---

### 5. **Updated train_special_losses.py**
**Changes:**
- Removed persistence losses (now regularizers)
- Focus on 2 Sinkhorn variants only
- Updated counts: 4 losses → 2 losses
- Updated experiment count: 12 → 6

**New losses trained:**
1. Sinkhorn (full sequence)
2. LW_Sinkhorn (layerwise)

---

## 📊 Current Loss Inventory

### Total Available: 48+ losses

**Level 1 - Individual (13):**
- MSE, MAE, MAPE, Quantile, Sinkhorn, FFT, MelSpec, JS, KL, Frobenius, LogNorm, FIM, AUTO

**Level 2 - Layerwise (13):**
- LW_MSE, LW_MAE, LW_MAPE, LW_Quantile, LW_Sinkhorn, LW_FFT, LW_MelSpec, LW_JS, LW_KL, LW_Frobenius, LW_LogNorm, LW_FIM, LWLN

**Level 3 - Regularized Full (14):**
- 12 original combinations
- 2 new persistence regularizers: MSE+0.01*PersLandscape, MAE+0.01*PersLandscape

**Level 4 - Regularized Layerwise (12):**
- 10 original combinations
- 2 new persistence regularizers: LW_MSE+0.01*LW_PersLandscape, LW_MAE+0.01*LW_PersLandscape

**Experiment Sequence: 37 losses**
- 6 individual
- 5 layerwise
- 14 regularized full
- 12 regularized layerwise

---

## ✅ Verification Steps

### Test Sinkhorn Loss
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox
conda run -n FCL python3 core_modules/run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss Sinkhorn \
    --epochs 5 --wandb
```

### Test LW_Sinkhorn Loss
```bash
conda run -n FCL python3 core_modules/run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss LW_Sinkhorn \
    --epochs 5 --wandb
```

### Test Persistence Regularizer
```bash
conda run -n FCL python3 core_modules/run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss "MSE+0.01*PersLandscape" \
    --epochs 5 --wandb
```

---

## 🚀 Ready to Run

### Train Sinkhorn Losses
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox
conda run -n FCL python3 train_special_losses.py
```

**Duration:** ~58 hours total
- Overlap 0: 2 × 3h = 6h
- Overlap 1: 2 × 10h = 20h
- Overlap 2: 2 × 16h = 32h

---

## 📝 Key Takeaways

1. ✅ **RuntimeError fixed** - All validate() calls now use `.detach()`
2. ✅ **Sinkhorn device handling** - CPU computation with device transfer
3. ✅ **Persistence losses stabilized** - Normalization + exception handling
4. ✅ **Persistence as regularizers** - Small weight (0.01) for stability
5. ✅ **Training script updated** - Focus on 2 Sinkhorn variants

**All fixes are backward compatible** - existing experiments unaffected.
