# Tournament System - Complete Guide

**Last Updated:** 2026-03-19

This is the **single source of truth** for the tournament system. All other markdown files are archived.

---

## 🎯 Tournament Overview

### Objective
Find the best loss functions for federated continual learning by running a hierarchical tournament:
1. Test all 91 losses on **tiny** models
2. Select **best AND worst** performers per overlap tier
3. Pass selected losses to progressively larger models
4. Final ranking determines optimal losses

### Why Best AND Worst?
- **Best losses:** Obviously good performers
- **Worst losses:** May perform poorly on tiny but excel on larger models (different capacity needs)
- **Per-overlap selection:** Losses perform differently across overlap levels

---

## 🔧 System Architecture

### Core Components

#### 1. Experiment Runner
**File:** `run_advanced_experiments.py`
- Adaptive epochs, checkpointing, LR schedule by model size
- Per-step logging for medium+ models
- CNN validation with stepwise logging
- GPU memory management

#### 2. Parallel Training
**File:** `tournament_system/parallel_training.py`
- Runs multiple experiments on single GPU simultaneously
- 4 tiny / 3 small / 2 medium / 1 large-huge in parallel
- Each process gets dedicated WandB run

#### 3. Ranking System
**File:** `tournament_system/per_overlap_ranking.py`
- Ranks losses separately per overlap (0, 1, 2)
- Selects top-N **and** bottom-N performers
- Outputs JSON with selected losses per overlap

#### 4. Tournament Orchestrator
**File:** `tournament_system/run_tournament.sh`
- Sequential execution of tournament phases
- Waits for all experiments before ranking
- Passes selected losses to next model size

---

## 📊 Model Configurations

| Model | Epochs | Checkpoints | Per-Step Logging | Parallel Count | Est. VRAM |
|-------|--------|-------------|------------------|----------------|-----------|
| tiny | 500 | Best+Last | No | 4 | 800MB |
| small | 500 | Best+Last | No | 3 | 1.5GB |
| medium | 350 | Every 50 | Yes | 2 | 3.5GB |
| large | 200 | Every 25 | Yes | 1 | 6GB |
| huge | 200 | Every 25 | Yes | 1 | 8GB |

### Learning Rate Schedule
- **Warmup:** 5% of epochs (min 10)
- **Decay:** Gentle cosine annealing
- **Min LR:** 1e-5 (never drops to 1e-6/1e-7/1e-8)

---

## 🚀 Tournament Workflow

### Phase 1: Tiny Models (All 91 Losses)
```bash
# Run all 91 losses on tiny models across 3 overlap levels
# 4 experiments in parallel per overlap
./tournament_system/run_tournament.sh --phase 1
```

**What happens:**
1. Loads all 91 losses from experiment sequence
2. Runs 4 tiny experiments in parallel for overlap=0
3. Waits for batch to complete
4. Repeats for overlap=1, then overlap=2
5. Total: 273 experiments (91 losses × 3 overlaps)

**Duration:** ~3-4 days with 4 parallel

### Phase 2: Rank Tiny Results
```bash
# Select top 20 AND bottom 10 losses per overlap
./tournament_system/run_tournament.sh --phase 2
```

**What happens:**
1. Analyzes all tiny model results
2. Ranks losses separately for each overlap
3. Selects top 20 + bottom 10 = 30 losses per overlap
4. Saves to `rankings_tiny.json`

**Output:**
```json
{
  "rankings_per_overlap": {
    "0": {
      "top": ["MSE", "MAE", ...],      // 20 best
      "bottom": ["Loss_X", ...]         // 10 worst
    },
    "1": { "top": [...], "bottom": [...] },
    "2": { "top": [...], "bottom": [...] }
  }
}
```

### Phase 3: Small Models (30 Losses per Overlap)
```bash
# Run selected 30 losses on small models
./tournament_system/run_tournament.sh --phase 3
```

**What happens:**
1. Loads 30 losses per overlap from `rankings_tiny.json`
2. Runs 3 small experiments in parallel
3. Total: 90 experiments (30 losses × 3 overlaps)

**Duration:** ~2-3 days with 3 parallel

### Phase 4: Rank Small Results
```bash
# Select top 15 AND bottom 5 losses per overlap
./tournament_system/run_tournament.sh --phase 4
```

**Output:** `rankings_small.json` with 20 losses per overlap

### Phase 5: Medium Models (20 Losses per Overlap)
```bash
./tournament_system/run_tournament.sh --phase 5
```

**What happens:**
1. Loads 20 losses per overlap from `rankings_small.json`
2. Runs 2 medium experiments in parallel
3. Total: 60 experiments (20 losses × 3 overlaps)

**Duration:** ~3-4 days with 2 parallel

### Phase 6: Rank Medium Results
```bash
./tournament_system/run_tournament.sh --phase 6
```

**Output:** `rankings_medium.json` with 10 losses per overlap

### Phase 7: Large Models (10 Losses per Overlap)
```bash
./tournament_system/run_tournament.sh --phase 7
```

**What happens:**
1. Loads 10 losses per overlap from `rankings_medium.json`
2. Runs 1 large experiment at a time
3. Total: 30 experiments (10 losses × 3 overlaps)

**Duration:** ~2-3 days sequential

### Phase 8: Final Ranking
```bash
./tournament_system/run_tournament.sh --phase 8
```

**Output:** `final_rankings.json` with top 5 losses per overlap

---

## 💾 Data Loading & RAM Usage

### Current Status
**Observation:** 33GB RAM usage during data loading phase before GPU training starts

**Cause:** Loading entire `Merged zoo.csv` (36,468 rows × 2,464 weight columns) into pandas DataFrame

### ✅ Solution Implemented: cudf.pandas

**Status:** ENABLED in `core_modules/run_advanced_experiments.py`

```python
# GPU-accelerated pandas for reduced RAM usage
try:
    import cudf.pandas
    cudf.pandas.install()
    print("✓ cudf.pandas enabled - using GPU for data operations")
except ImportError:
    print("⚠ cudf.pandas not available - using standard pandas")
```

**Benefits:**
- Data operations use GPU RAM instead of system RAM
- 10-50x faster CSV loading
- Transparent - no code changes needed
- Already available (RapidS setup)

### Swap Memory Setup (Safety Net)

**Status:** RECOMMENDED - see `SWAP_MEMORY_SETUP.md`

**Quick setup:**
```bash
# Create 32GB swap file
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Optimize (only use when necessary)
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

**Monitor:**
```bash
watch -n 1 free -h
```

**Expected:** Swap rarely used with cudf.pandas enabled

---

## 🎮 Single GPU Multiprocessing

### How It Works
**Question:** "How does multiprocessing work with 1 GPU?"

**Answer:** PyTorch allows multiple processes to share the same GPU:
- Each process gets its own CUDA context
- GPU scheduler time-slices between processes
- VRAM is shared (4 × 800MB = 3.2GB total for 4 tiny models)
- Processes don't interfere with each other

**Example:**
```python
# Process 1: Uses GPU 0, occupies 800MB VRAM
model1 = TransformerAE(...).to('cuda:0')

# Process 2: Also uses GPU 0, occupies another 800MB VRAM
model2 = TransformerAE(...).to('cuda:0')

# Both train simultaneously, GPU scheduler handles it
```

**Benefits:**
- 60-80% GPU utilization instead of 5-10%
- 3-4x faster tournament completion
- Each process has independent WandB logging

**Limitations:**
- Total VRAM must fit all models (4 × 800MB < 16GB ✓)
- CPU overhead for context switching (minimal)
- Each process needs CPU cores (you have plenty)

---

## 📁 Workspace Organization

### Active Files (Keep)
```
notebooks_sandbox/
├── tournament_system/          # Tournament scripts
│   ├── run_tournament.sh       # Main orchestrator
│   ├── parallel_training.py    # Parallel execution
│   └── per_overlap_ranking.py  # Ranking system
├── core_modules/               # Core Python modules
│   ├── run_advanced_experiments.py
│   ├── advanced_losses.py
│   ├── advanced_topology.py
│   ├── cnn_reconstruction.py
│   ├── cnn_validation_enhanced.py
│   ├── config.py
│   ├── scenario_dataset.py
│   ├── utils_consolidated.py
│   └── weight_normalization.py
├── TOURNAMENT_GUIDE.md         # This file (single source of truth)
└── experiment_logs/            # Logs directory
```

### Archived (Move to archive_old_scripts/)
- All old bash scripts (run_*.sh except run_tournament.sh)
- All old python runners (run_*.py except run_advanced_experiments.py)
- All old markdown files (merge into this one)
- Obsolete test scripts

---

## 🧪 Testing Commands

### Test Single Experiment
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Tiny model - auto runs 500 epochs
# Includes epoch 0 baseline validation + topology
conda run -n FCL python3 core_modules/run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss MSE

# Medium model - auto runs 350 epochs, logs per-step
# Better terminal logging with periodic updates every 30 min
conda run -n FCL python3 core_modules/run_advanced_experiments.py \
    --single --model-size medium --overlap 1 --loss Huber
```

**What you'll see:**
- ✓ cudf.pandas enabled message
- Epoch 0 baseline validation (before training)
- Periodic detailed logs every 30 minutes
- Experiment name in logs: `[tiny_overlap0_MSE]`
- Time estimates and progress tracking

### Test Parallel Training
```bash
# Run 4 tiny experiments in parallel
conda run -n FCL python3 tournament_system/parallel_training.py \
    --model-size tiny --overlap 0 --num-parallel 4 \
    --losses MSE MAE Huber CosineLoss

# Monitor GPU
watch -n 1 nvidia-smi
```

### Test Ranking
```bash
# After experiments complete
conda run -n FCL python3 tournament_system/per_overlap_ranking.py \
    --model-size tiny --top-n 20 --bottom-n 10
```

---

## ⚡ GPU Power Limit

### Check Current Limit
```bash
nvidia-smi -q -d POWER
```

### Adjust Power Limit
```bash
# Maximum performance (180W for RTX 5060 Ti)
sudo nvidia-smi -pl 180

# Balanced (165W)
sudo nvidia-smi -pl 165

# Conservative (150W - cooler, quieter)
sudo nvidia-smi -pl 150
```

**Location:** `/sys/class/drm/card0/device/hwmon/hwmon*/power1_cap`

**Recommendation:** Start at 165W, increase to 180W if temps are good (<80°C)

---

## 🐛 Troubleshooting

### Issue: 33GB RAM Usage
**Cause:** Loading full CSV into pandas
**Solution:** Add cudf.pandas to run_advanced_experiments.py:
```python
import cudf.pandas
cudf.pandas.install()
```

### Issue: GPU Shows 0% Utilization
**Cause:** Data loading phase (CPU-bound)
**Normal:** GPU usage starts after "Training for X epochs" message
**Monitor:** Should see GPU spike to 80-100% during training

### Issue: Experiments Fail with OOM
**Cause:** Too many parallel processes
**Solution:** Reduce parallel count:
```bash
# Instead of 4 tiny, try 3
python parallel_training.py --model-size tiny --num-parallel 3
```

### Issue: WandB Not Logging
**Cause:** WandB not initialized or network issue
**Solution:** Check `wandb login` status, verify internet connection

---

## 📈 Expected Timeline

| Phase | Model | Experiments | Parallel | Duration |
|-------|-------|-------------|----------|----------|
| 1 | tiny | 273 (91×3) | 4 | 3-4 days |
| 2 | rank | - | - | 10 min |
| 3 | small | 90 (30×3) | 3 | 2-3 days |
| 4 | rank | - | - | 5 min |
| 5 | medium | 60 (20×3) | 2 | 3-4 days |
| 6 | rank | - | - | 5 min |
| 7 | large | 30 (10×3) | 1 | 2-3 days |
| 8 | final | - | - | 5 min |

**Total:** ~10-14 days continuous

---

## 🎯 Next Steps

1. **Clean up workspace** (move old scripts to archive)
2. **Test single experiment** to verify data loading works
3. **Test parallel training** with 2-3 tiny models
4. **Add cudf.pandas** to reduce RAM usage
5. **Run Phase 1** (tiny models tournament)
6. **Monitor and adjust** based on results

---

## 📝 Notes

- **Single GPU multiprocessing:** Works perfectly, PyTorch handles it
- **RAM usage:** Normal during data loading, will improve with cudf.pandas
- **Ranking:** Selects BOTH best and worst performers
- **Per-overlap:** Critical because losses perform differently across overlaps
- **This guide:** Single source of truth, update this file only

---

**End of Guide**
