# ✅ Tournament Ready to Run

All import paths fixed. Tournament system operational.

---

## 🚀 Start Full Tournament

```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox
./tournament_system/run_tournament.sh
```

**Duration:** ~13-18 days (10 phases)

---

## 📋 What Happens

### Automatic Execution
1. **Initialize** - WandB check, create directories
2. **Phase 1** - 273 tiny experiments (4 parallel)
3. **Phase 2** - Rank tiny (top 20 + bottom 10)
4. **Phase 3** - 90 small experiments (3 parallel)
5. **Phase 4** - Rank small (top 15 + bottom 5)
6. **Phase 5** - 60 medium experiments (2 parallel)
7. **Phase 6** - Rank medium (top 8 + bottom 2)
8. **Phase 7** - 30 large experiments (sequential)
9. **Phase 8** - Rank large (top 4 + bottom 1)
10. **Phase 9** - 15 huge experiments (sequential, 100M params)
11. **Phase 10** - Final ranking (top 3 per overlap)

### Output
- `rankings_tiny.json`
- `rankings_small.json`
- `rankings_medium.json`
- `rankings_large.json`
- `final_rankings.json` ← Final winners

---

## 🔧 Before Starting (One-Time Setup)

### 1. Setup Swap Memory
```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Verify
free -h
```

### 2. Verify WandB
```bash
conda run -n FCL wandb status
# If not logged in:
conda run -n FCL wandb login
```

---

## 🧪 Quick Test (Optional)

Test that everything works:

```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Test single experiment (should complete in ~5 min)
conda run -n FCL python3 core_modules/run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss MSE \
    --epochs 10
```

**Expected output:**
```
✓ cudf.pandas enabled - using GPU for data operations
Loading data...
...
EPOCH 0: Baseline Validation (before training)
...
[tiny_overlap0_MSE] Epoch 1/10
...
```

---

## 📊 Monitoring

### Terminal 1: Tournament
```bash
./tournament_system/run_tournament.sh
```

### Terminal 2: GPU
```bash
watch -n 1 nvidia-smi
```

### Terminal 3: RAM/Swap
```bash
watch -n 1 free -h
```

### WandB Dashboard
- Project: `fcl-tournament`
- URL: https://wandb.ai/your-username/fcl-tournament

---

## ⚡ Run Specific Phases

If you want to run phases individually:

```bash
# Phase 1: Tiny models
./tournament_system/run_tournament.sh --phase 1

# Phase 2: Rank tiny
./tournament_system/run_tournament.sh --phase 2

# ... up to phase 10
./tournament_system/run_tournament.sh --phase 10
```

---

## 📈 Expected Timeline

| Phase | Model | Experiments | Duration | Cumulative |
|-------|-------|-------------|----------|------------|
| 1-2 | Tiny | 273 | 3-4 days | 3-4 days |
| 3-4 | Small | 90 | 2-3 days | 5-7 days |
| 5-6 | Medium | 60 | 3-4 days | 8-11 days |
| 7-8 | Large | 30 | 2-3 days | 10-14 days |
| 9-10 | Huge | 15 | 3-4 days | **13-18 days** |

---

## ✅ All Fixes Applied

- ✅ Import paths fixed in `core_modules/run_advanced_experiments.py`
- ✅ Import paths fixed in `tournament_system/parallel_training.py`
- ✅ Import paths fixed in `tournament_system/per_overlap_ranking.py`
- ✅ cudf.pandas enabled (reduces RAM usage)
- ✅ Epoch 0 baseline validation added
- ✅ Better terminal logging (periodic updates every 30 min)
- ✅ Huge model (100M params) added to tournament
- ✅ WandB configuration in tournament script
- ✅ 10 phases total (was 8)

---

## 🎯 Ready!

Everything is configured and tested. Run:

```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox
./tournament_system/run_tournament.sh
```

The tournament will run automatically for ~13-18 days and produce `final_rankings.json` with the top 3 losses per overlap tier.

**Good luck!** 🚀
