# Manual Training Workflow - Tiny Models

## 📋 Overview

Train tiny models in 2 separate batches, then merge results and continue tournament.

---

## 🎯 Step-by-Step Workflow

### Step 1: Train Batch 1 (First Half of Losses)

Run this for each overlap (0, 1, 2):

```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Overlap 0
conda run -n FCL python3 train_tiny_batch1.py --overlap 0

# Overlap 1
conda run -n FCL python3 train_tiny_batch1.py --overlap 1

# Overlap 2
conda run -n FCL python3 train_tiny_batch1.py --overlap 2
```

**Output files:**
- `batch1_overlap0_results.json`
- `batch1_overlap1_results.json`
- `batch1_overlap2_results.json`

**Duration:** ~9 losses × 150 epochs × 2 min/epoch = ~27 hours per overlap

---

### Step 2: Train Batch 2 (Second Half of Losses)

Run this for each overlap (0, 1, 2):

```bash
# Overlap 0
conda run -n FCL python3 train_tiny_batch2.py --overlap 0

# Overlap 1
conda run -n FCL python3 train_tiny_batch2.py --overlap 1

# Overlap 2
conda run -n FCL python3 train_tiny_batch2.py --overlap 2
```

**Output files:**
- `batch2_overlap0_results.json`
- `batch2_overlap1_results.json`
- `batch2_overlap2_results.json`

**Duration:** ~9 losses × 150 epochs × 2 min/epoch = ~27 hours per overlap

---

### Step 3: Merge Results & Continue Tournament

After **both** batch1 and batch2 complete for all overlaps:

```bash
./merge_and_continue_tournament.sh
```

**This script will:**
1. Merge batch1 and batch2 results
2. Rank tiny model results
3. Select top 20 + bottom 10 losses per overlap
4. Display selected losses for next phase

**Output files:**
- `tiny_overlap0_merged.json`
- `tiny_overlap1_merged.json`
- `tiny_overlap2_merged.json`
- `tiny_phase1_summary.json`
- `rankings_tiny.json` ← Use this for next phase

---

## ⚡ Parallel Training Option

You can run batch1 and batch2 **in parallel** to save time:

### Terminal 1:
```bash
# Train all batch1 overlaps
conda run -n FCL python3 train_tiny_batch1.py --overlap 0
conda run -n FCL python3 train_tiny_batch1.py --overlap 1
conda run -n FCL python3 train_tiny_batch1.py --overlap 2
```

### Terminal 2:
```bash
# Train all batch2 overlaps (at the same time)
conda run -n FCL python3 train_tiny_batch2.py --overlap 0
conda run -n FCL python3 train_tiny_batch2.py --overlap 1
conda run -n FCL python3 train_tiny_batch2.py --overlap 2
```

**GPU Usage:** ~1.6GB (2 tiny models in parallel)

---

## 📊 Monitoring Progress

### Check what's running:
```bash
# GPU usage
watch -n 1 nvidia-smi

# WandB dashboard
# Go to: https://wandb.ai/your-username/fcl-tournament
```

### Check results:
```bash
# View batch1 results
cat batch1_overlap0_results.json

# View batch2 results
cat batch2_overlap0_results.json

# View merged results (after step 3)
cat tiny_overlap0_merged.json

# View rankings (after step 3)
cat rankings_tiny.json
```

---

## 📈 Expected Timeline

### Sequential (one batch at a time):
- Batch1 all overlaps: ~81 hours
- Batch2 all overlaps: ~81 hours
- **Total: ~162 hours (~6.8 days)**

### Parallel (both batches simultaneously):
- Both batches all overlaps: ~81 hours
- **Total: ~81 hours (~3.4 days)**

---

## ✅ What's Fixed

1. ✅ **Sinkhorn loss** - proper gradient flow
2. ✅ **Tiny epochs: 150** - faster training
3. ✅ **No parallel complexity** - simple sequential scripts
4. ✅ **Manual control** - run batches when you want
5. ✅ **Automatic merging** - single script after training

---

## 🎯 Quick Start

**Fastest approach (parallel):**

```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Terminal 1 - Batch 1
conda run -n FCL python3 train_tiny_batch1.py --overlap 0 &
conda run -n FCL python3 train_tiny_batch1.py --overlap 1 &
conda run -n FCL python3 train_tiny_batch1.py --overlap 2 &

# Terminal 2 - Batch 2
conda run -n FCL python3 train_tiny_batch2.py --overlap 0 &
conda run -n FCL python3 train_tiny_batch2.py --overlap 1 &
conda run -n FCL python3 train_tiny_batch2.py --overlap 2 &

# Wait for all to complete, then:
./merge_and_continue_tournament.sh
```

---

## 📁 Files Summary

### Training Scripts (run manually):
- `train_tiny_batch1.py` - First half of losses
- `train_tiny_batch2.py` - Second half of losses

### Merge & Continue (run after training):
- `merge_and_continue_tournament.sh` - Merge results + rank + continue

### Helper Scripts:
- `merge_batch_results.py` - Called by merge_and_continue_tournament.sh

---

**Ready to start training!**
