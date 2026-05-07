# Sequential Batch Training Guide

## ✅ All Changes Complete

### 1. **Sinkhorn Loss Fixed**
**File:** `core_modules/advanced_losses.py`

**Problem:** Was using detached tensors, breaking gradient flow

**Solution:**
```python
def forward(self, pred, target):
    if self.loss is None:
        return F.mse_loss(pred, target)
    
    # Reshape if needed
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Use geomloss directly - it supports autograd
    return self.loss(pred, target)
```

### 2. **Tiny Epochs Reduced to 150**
```python
MODEL_EPOCHS = {
    'tiny': 150,  # was 400
    'small': 400,
    'medium': 350,
    'large': 200,
    'huge': 200
}
```

### 3. **Parallel Training Removed**
Created 2 sequential batch scripts instead:
- `train_tiny_batch1.py` - First half of losses
- `train_tiny_batch2.py` - Second half of losses

### 4. **Results Merging**
- `merge_batch_results.py` - Combines batch1 and batch2 results
- Creates merged JSON files for ranking

### 5. **Single Launch Script**
- `run_tiny_tournament.sh` - Runs everything automatically

---

## 🚀 How to Run

### Single Command
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox
./run_tiny_tournament.sh
```

This will:
1. **Overlap 0:** Launch batch1 and batch2 in parallel → wait for completion
2. **Overlap 1:** Launch batch1 and batch2 in parallel → wait for completion
3. **Overlap 2:** Launch batch1 and batch2 in parallel → wait for completion
4. **Merge:** Combine all results into unified JSON files

---

## 📊 What Happens

### Loss Distribution
All losses are split evenly:
- **Batch 1:** First half of losses (sequential)
- **Batch 2:** Second half of losses (sequential)
- **Both run in parallel** (2 processes max)

### Per Overlap
```
Overlap 0:
  ├─ batch1: losses 0-8  (9 losses) → background process
  ├─ batch2: losses 9-17 (9 losses) → background process
  └─ Wait for both to complete

Overlap 1:
  ├─ batch1: losses 0-8  (9 losses) → background process
  ├─ batch2: losses 9-17 (9 losses) → background process
  └─ Wait for both to complete

Overlap 2:
  ├─ batch1: losses 0-8  (9 losses) → background process
  ├─ batch2: losses 9-17 (9 losses) → background process
  └─ Wait for both to complete

Merge all results
```

### Output Files
```
batch1_overlap0_results.json
batch2_overlap0_results.json
tiny_overlap0_merged.json

batch1_overlap1_results.json
batch2_overlap1_results.json
tiny_overlap1_merged.json

batch1_overlap2_results.json
batch2_overlap2_results.json
tiny_overlap2_merged.json

tiny_phase1_summary.json  ← Overall summary
```

---

## 📈 Expected Timeline

### Per Overlap (with 2 parallel batches)
- **Batch 1:** ~9 losses × 150 epochs × 2 min/epoch = ~27 hours
- **Batch 2:** ~9 losses × 150 epochs × 2 min/epoch = ~27 hours
- **Running in parallel:** ~27 hours per overlap

### Total for All 3 Overlaps
- **Sequential:** 3 overlaps × 27 hours = **~81 hours (~3.4 days)**

Much faster than the original 273 experiments × 150 epochs sequentially!

---

## 🔍 Monitoring

### Check Progress
```bash
# Watch batch 1 for overlap 0
tail -f batch1_overlap0.log

# Watch batch 2 for overlap 0
tail -f batch2_overlap0.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Expected GPU Usage
- **2 tiny models in parallel:** ~1.6GB VRAM
- **Plenty of headroom** on 16GB GPU

---

## 📋 After Completion

### Check Results
```bash
# View summary
cat tiny_phase1_summary.json

# View specific overlap
cat tiny_overlap0_merged.json
```

### Run Ranking
```bash
cd tournament_system
conda run -n FCL python3 per_overlap_ranking.py \
    --model-size tiny \
    --top-n 20 \
    --bottom-n 10 \
    --output ../rankings_tiny.json
```

---

## ✅ Key Improvements

1. ✅ **Sinkhorn loss fixed** - proper gradient flow
2. ✅ **Tiny epochs: 150** - faster convergence
3. ✅ **No parallel training complexity** - simple sequential batches
4. ✅ **2 processes max** - batch1 and batch2 in parallel
5. ✅ **Automatic merging** - results combined for ranking
6. ✅ **Single launch script** - one command to run everything
7. ✅ **Clean logs** - separate log files per batch/overlap

---

## 🎯 Ready to Run

Everything is set up. Just run:

```bash
./run_tiny_tournament.sh
```

The script will handle everything and produce merged results ready for ranking!
