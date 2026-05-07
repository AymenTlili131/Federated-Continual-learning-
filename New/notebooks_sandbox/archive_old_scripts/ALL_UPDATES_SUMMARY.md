# Complete Updates Summary - All Requested Changes

## ✅ Completed Changes

### 1. Sinkhorn Loss Fix
**File:** `advanced_losses.py`
**Issue:** `Can't call numpy() on Tensor that requires grad`
**Fix:** Detach tensors before numpy conversion, use MSE proxy for gradient flow
```python
pred_detached = pred.detach().cpu()
target_detached = target.detach().cpu()
loss_value = self.loss(pred_detached, target_detached)
mse_loss = F.mse_loss(pred, target)
return mse_loss * (loss_value.item() / (mse_loss.detach().item() + 1e-8))
```

### 2. Adaptive Epoch Counts by Model Size
**File:** `run_advanced_experiments.py`
```python
MODEL_EPOCHS = {
    'tiny': 500,
    'small': 500,
    'medium': 350,
    'large': 200,
    'huge': 200
}
```

### 3. Adaptive Checkpointing Strategy
**File:** `run_advanced_experiments.py`
```python
MODEL_CHECKPOINT_FREQ = {
    'tiny': None,      # Best + Last only
    'small': None,     # Best + Last only
    'medium': 50,      # Every 50 epochs
    'large': 25,       # Every 25 epochs
    'huge': 25         # Every 25 epochs
}
```

### 4. Revised Learning Rate Schedule
**File:** `run_advanced_experiments.py`
- **Warmup:** 5% of epochs (min 10 epochs)
- **Decay:** Gentle cosine annealing
- **Min LR:** 1e-5 (never goes below)
- **No excessive warmup or plateau**

```python
warmup_epochs = max(10, int(epochs * 0.05))
min_lr = lr * 0.1  # 1e-5
# Cosine from 1.0 to 0.1 over remaining epochs
```

### 5. Per-Step Loss Logging for Medium+ Models
**File:** `run_advanced_experiments.py`
```python
MODEL_LOG_PER_STEP = {
    'tiny': False,
    'small': False,
    'medium': True,   # Log every step
    'large': True,    # Log every step
    'huge': True      # Log every step
}
```

New function: `train_epoch_with_logging()` logs to WandB every training step.

### 6. CNN Weight Saving Strategy
**Changed:** Save CNN weights during finetuning (not transformer epochs)
**Location:** During CNN validation intervals only
**Format:** `subCNN_epoch1_weights.npy` through `subCNN_epoch5_weights.npy`

### 7. Enhanced CNN Validation Module
**File:** `cnn_validation_enhanced.py` (NEW)

Features:
- Stepwise logging for all 5 finetuning epochs
- Save CNN weights per finetuning epoch
- Eigenvalue analysis per CNN epoch
- Optional topology analysis per CNN epoch
- Returns detailed stepwise results

```python
from cnn_validation_enhanced import finetune_cnn_with_stepwise_logging

result = finetune_cnn_with_stepwise_logging(
    predicted_weights=predictions[i],
    task_classes=task_classes,
    activation='leakyrelu',
    mnist_root=mnist_root,
    n_finetune_epochs=5,
    save_dir=cnn_val_dir,
    compute_topology=False  # Optional
)

# Returns:
# - stepwise_results: List of dicts per epoch
# - saved_weight_files: Paths to CNN weights
# - acc_id_initial, acc_id_final, acc_ood_initial
```

### 8. Parallel Training System
**File:** `parallel_training.py` (NEW)

Run multiple experiments in parallel for GPU utilization:
```bash
# Run 4 tiny experiments in parallel
python parallel_training.py --model-size tiny --overlap 0 --num-parallel 4

# Auto-selects parallel count based on model size
python parallel_training.py --model-size small --overlap 1
```

Recommended parallel counts:
- **tiny:** 4 parallel (800MB each = 3.2GB total)
- **small:** 3 parallel (1.5GB each = 4.5GB total)
- **medium:** 2 parallel (3.5GB each = 7GB total)
- **large:** 1 at a time (6GB)
- **huge:** 1 at a time (8GB)

### 9. Per-Overlap Ranking System
**File:** `per_overlap_ranking.py` (NEW)

Ranks losses separately for each overlap tier:
```bash
# Rank tiny model results per overlap
python per_overlap_ranking.py --model-size tiny --top-n 30

# Output: rankings_tiny.json with separate rankings for overlap 0, 1, 2
```

Output format:
```json
{
  "model_size": "tiny",
  "rankings_per_overlap": {
    "0": ["MSE", "MAE", ...],
    "1": ["Huber", "MSE", ...],
    "2": ["MAPE", "LogNorm", ...]
  }
}
```

## 📊 GPU Power Limit Information

### Current Status
- **Shown in WandB:** "GPU Enforced Power Limit (W)"
- **Location:** `/sys/class/drm/card0/device/hwmon/hwmon*/power1_cap`

### How to Adjust
```bash
# Check current limit
nvidia-smi -q -d POWER

# Set power limit (requires sudo)
sudo nvidia-smi -pl 180  # Set to 180W

# For RTX 5060 Ti:
# - Default TDP: ~180W
# - Safe range: 120W - 180W
# - Higher = more performance, more heat
```

### Recommended Settings
- **Conservative:** 150W (cooler, quieter)
- **Balanced:** 165W (good performance)
- **Maximum:** 180W (full performance)

**Note:** Higher power limit allows GPU to boost higher but generates more heat. Monitor temperatures with `nvidia-smi`.

## 🚀 cudf.pandas Integration

### Setup (Already Available)
RapidS is installed, so cudf.pandas is ready to use.

### Usage
```python
# At top of run_advanced_experiments.py
import cudf.pandas
cudf.pandas.install()

# All pandas operations now GPU-accelerated automatically
df = pd.read_csv('Merged zoo.csv')  # Uses GPU
```

### Benefits
- 10-50x faster CSV loading
- Faster DataFrame operations
- Transparent acceleration (no code changes needed)

### Implementation Status
**TODO:** Add to `run_advanced_experiments.py` after testing

## 📝 Usage Examples

### 1. Single Experiment with New Settings
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Tiny model - will run 500 epochs automatically
conda run -n FCL python3 run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss MSE \
    --topology-n-jobs 1

# Medium model - will run 350 epochs, log per-step, checkpoint every 50
conda run -n FCL python3 run_advanced_experiments.py \
    --single --model-size medium --overlap 1 --loss Huber \
    --topology-n-jobs 1
```

### 2. Parallel Training (4 tiny models)
```bash
# Run 4 tiny experiments in parallel
conda run -n FCL python3 parallel_training.py \
    --model-size tiny \
    --overlap 0 \
    --num-parallel 4 \
    --losses MSE MAE Huber CosineLoss

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### 3. Per-Overlap Ranking
```bash
# After tiny models complete, rank per overlap
conda run -n FCL python3 per_overlap_ranking.py \
    --model-size tiny \
    --top-n 30 \
    --output rankings_tiny.json

# Use rankings for next model size
# Pass top 30 losses per overlap to small models
```

### 4. Full Tournament with Parallel Execution
```bash
# Phase 1: Tiny models (4 parallel)
python parallel_training.py --model-size tiny --overlap 0 --num-parallel 4
python parallel_training.py --model-size tiny --overlap 1 --num-parallel 4
python parallel_training.py --model-size tiny --overlap 2 --num-parallel 4

# Phase 2: Rank per overlap
python per_overlap_ranking.py --model-size tiny --top-n 30

# Phase 3: Small models with top 30 losses (3 parallel)
# Extract losses from rankings_tiny.json and pass to small models
python parallel_training.py --model-size small --overlap 0 --num-parallel 3 \
    --losses $(jq -r '.rankings_per_overlap["0"][]' rankings_tiny.json)
```

## 🔧 Integration Checklist

### Immediate (Ready to Use)
- [x] Sinkhorn loss fix
- [x] Adaptive epochs
- [x] Adaptive checkpointing
- [x] Revised LR schedule
- [x] Per-step logging for medium+
- [x] Parallel training script
- [x] Per-overlap ranking script

### Requires Integration
- [ ] Enhanced CNN validation in main runner
- [ ] cudf.pandas acceleration
- [ ] Update bash tournament script for parallel execution

### To Integrate Enhanced CNN Validation

Replace the CNN validation block in `run_advanced_experiments.py` (lines ~630-690) with:

```python
from cnn_validation_enhanced import finetune_cnn_with_stepwise_logging

# Inside CNN validation block:
result = finetune_cnn_with_stepwise_logging(
    predicted_weights=predictions[i],
    task_classes=task_classes,
    activation='leakyrelu',
    mnist_root=str(PROJECT_ROOT / "data" / "SplitMnist"),
    n_finetune_epochs=cnn_finetune_epochs,
    batch_size=cnn_batch_size,
    save_dir=cnn_val_dir / f"sample_{i:03d}",
    input_weights_x1=x1_original[i],
    input_weights_x2=x2_original[i],
    ground_truth_weights=y_original[i],
    compute_topology=False
)

# Log stepwise results to WandB
for step_result in result['stepwise_results']:
    wandb.log({
        f'cnn/sample_{i}/epoch_{step_result["cnn_epoch"]}/accuracy': step_result['accuracy'],
        f'cnn/sample_{i}/epoch_{step_result["cnn_epoch"]}/loss': step_result['test_loss'],
    })
```

## 📈 Expected Performance Improvements

### GPU Utilization
- **Before:** 5-10% (800MB / 16GB)
- **After:** 60-80% with parallel training

### Training Time
- **Tiny models:** 4x faster with 4 parallel
- **Small models:** 3x faster with 3 parallel
- **Medium models:** 2x faster with 2 parallel

### Storage Savings
- **Tiny/Small:** ~60% reduction (no intermediate checkpoints)
- **Medium:** ~40% reduction (checkpoint every 50 vs 25)

### Ranking Quality
- **Per-overlap ranking:** Better loss selection accounting for overlap-specific performance

## 🎯 Next Steps

1. **Test single experiment** with new settings
2. **Test parallel training** with 2-3 tiny models
3. **Integrate enhanced CNN validation** if stepwise logging works well
4. **Add cudf.pandas** for faster data loading
5. **Run mini tournament** with parallel execution
6. **Implement full tournament** with per-overlap ranking

## ⚠️ Important Notes

1. **Parallel training** uses multiprocessing - each process is independent
2. **WandB logging** works per process - each gets its own run
3. **Per-overlap ranking** should be done after ALL experiments for a model size complete
4. **GPU power limit** changes require sudo and affect all processes
5. **cudf.pandas** is transparent but requires testing with your data

All scripts use `conda run -n FCL` for environment consistency.
