# Memory Leak Fixes and Performance Optimizations

## 🔴 Issues Identified from Tournament Crash

### 1. **PC Freeze on `large_overlap1` with `MAPE_LW0.1xJS_F0.05xKL`**
- Complex loss functions accumulate gradients and intermediate tensors
- Large model (4M+ parameters) + large overlap dataset (130K+ pairs)
- No GPU memory clearing between experiments
- Topology analysis using all CPU cores caused system freeze

### 2. **Failed Experiments**
From screenshot analysis:
- Several experiments marked as "Failed" (red)
- Some marked as "Add notes" (yellow) - likely OOM or timeout
- Last experiment froze the entire PC

### 3. **Root Causes**
- **GPU Memory Leak**: Models, optimizers, data loaders not deleted
- **CPU Overload**: Topology analysis spawning too many workers
- **Complex Losses**: Multi-component losses (MAPE + JS + KL) create large computation graphs
- **No Reproducibility**: Seeds not fixed
- **Variable Accumulation**: Large datasets kept in memory across experiments

## ✅ Fixes Implemented

### 1. **Seed Fixing for Reproducibility**
```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Called at module import
```

### 2. **GPU Memory Clearing**
**Before each experiment:**
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Using device: {device} (GPU memory cleared)")
```

**After each experiment:**
```python
# Aggressive memory cleanup
del model, optimizer, scheduler, loss_fn
del train_loader, val_loader, test_loader
del train_dataset, val_dataset, test_dataset
del x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test
del x1_train_norm, x2_train_norm, y_train_norm
del x1_val_norm, x2_val_norm, y_val_norm
del x1_test_norm, x2_test_norm, y_test_norm

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
```

### 3. **Topology CPU Worker Control**
**New CLI argument:**
```bash
--topology-n-jobs 1  # Sequential (safe for large models)
--topology-n-jobs 2  # 2 workers (balanced)
--topology-n-jobs -1 # All cores (only for small models)
```

**Implementation:**
```python
# In advanced_topology.py
def compute_mapper(..., n_jobs=1):
    clusterer=DBSCAN(eps=0.5, min_samples=3, n_jobs=n_jobs)

def compute_comprehensive_topology(..., n_jobs=1):
    mapper_results = compute_mapper(..., n_jobs=n_jobs)
```

### 4. **Import Garbage Collection**
```python
import gc  # Added at top of file
```

## 📊 Recommended Settings

### For Large Models (medium, large, huge)
```bash
--topology-n-jobs 1          # Sequential topology
--cnn-validation-samples 50  # Fewer CNN samples
--compute-topology-every 100 # Less frequent topology
```

### For Small Models (tiny, small)
```bash
--topology-n-jobs 2          # 2 workers
--cnn-validation-samples 100 # Standard samples
--compute-topology-every 50  # Standard frequency
```

### For Complex Losses (MAPE_LW, Mixed losses)
```bash
--batch-size 16              # Smaller batches
--topology-n-jobs 1          # Sequential
```

## 🚀 Updated Tournament Script

The bash script now needs to be updated to:
1. Run experiments sequentially (already does)
2. Add delays between model size changes
3. Use conservative topology settings for large models

## 🔬 Testing Strategy

### Phase 1: Single Experiment Test
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Test with tiny model, simple loss
python3 run_advanced_experiments.py \
    --single \
    --model-size tiny \
    --overlap 0 \
    --loss MSE \
    --epochs 50 \
    --topology-n-jobs 1
```

### Phase 2: Complex Loss Test
```bash
# Test with complex loss that caused crash
python3 run_advanced_experiments.py \
    --single \
    --model-size small \
    --overlap 1 \
    --loss "MAPE_LW0.1xJS_F0.05xKL" \
    --epochs 50 \
    --topology-n-jobs 1 \
    --batch-size 16
```

### Phase 3: Large Model Test
```bash
# Test with large model
python3 run_advanced_experiments.py \
    --single \
    --model-size large \
    --overlap 1 \
    --loss MSE \
    --epochs 50 \
    --topology-n-jobs 1 \
    --cnn-validation-samples 50
```

### Phase 4: Full Tournament (if above pass)
```bash
./run_full_tournament.sh --models tiny,small
```

## 📈 Memory Monitoring

### Check GPU Memory
```bash
watch -n 1 nvidia-smi
```

### Check System Memory
```bash
watch -n 1 free -h
```

### Check CPU Usage
```bash
htop
```

## ⚠️ Warning Signs

If you see:
- **GPU memory > 90%**: Reduce batch size or CNN samples
- **CPU usage > 95% all cores**: Reduce topology-n-jobs
- **System freeze**: Kill process immediately (Ctrl+C or `pkill python3`)
- **Swap usage increasing**: System running out of RAM, reduce dataset size

## 🔧 Emergency Recovery

If experiment freezes:
```bash
# Terminal 1: Monitor
watch -n 1 nvidia-smi

# Terminal 2: Kill if needed
pkill -9 python3

# Clear GPU
python3 -c "import torch; torch.cuda.empty_cache()"
```

## 📝 Changes Summary

### Files Modified
1. **`run_advanced_experiments.py`**
   - Added `set_seed(42)` for reproducibility
   - Added GPU memory clearing before/after experiments
   - Added aggressive variable deletion
   - Added `topology_n_jobs` parameter
   - Added `--topology-n-jobs` CLI argument

2. **`advanced_topology.py`**
   - Added `n_jobs` parameter to `compute_mapper()`
   - Added `n_jobs` parameter to `compute_comprehensive_topology()`
   - DBSCAN now uses configurable workers

### Backward Compatibility
- All new parameters have defaults
- Old scripts will work with default values
- `topology_n_jobs=1` is safe default (sequential)

## 🎯 Next Steps

1. **Test Phase 1-3** above to verify fixes
2. **Monitor memory** during tests
3. **Adjust settings** based on your GPU (RTX 5060 Ti)
4. **Run tournament** with conservative settings first
5. **Gradually increase** parallelization if stable

## 💾 Disk Space

Tournament generates ~150GB of data:
- Checkpoints: ~50GB
- Predicted weights: ~30GB
- Topology results: ~20GB
- CNN validation: ~20GB
- Attention maps: ~15GB
- Logs: ~15GB

Ensure sufficient disk space before running full tournament.
