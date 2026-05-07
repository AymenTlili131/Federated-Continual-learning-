# Testing Guide - Memory Leak Fixes Verification

## 🎯 Testing Strategy

Run these tests in order to verify all fixes work correctly before launching the full tournament.

## Phase 1: Simple Test (5 minutes)

### Test 1.1: Tiny Model + Simple Loss
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

python3 run_advanced_experiments.py \
    --single \
    --model-size tiny \
    --overlap 0 \
    --loss MSE \
    --epochs 50 \
    --topology-n-jobs 1 \
    --cnn-validation-samples 10
```

**Expected:**
- ✅ Completes in ~5 minutes
- ✅ GPU memory cleared message at start/end
- ✅ Seed fixed message at start
- ✅ CNN validation runs at epochs 25, 50
- ✅ Topology analysis with n_jobs=1

**Monitor:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

---

## Phase 2: Complex Loss Test (10 minutes)

### Test 2.1: The Loss That Caused Crash
```bash
python3 run_advanced_experiments.py \
    --single \
    --model-size small \
    --overlap 1 \
    --loss "MAPE_LW0.1xJS_F0.05xKL" \
    --epochs 50 \
    --topology-n-jobs 1 \
    --batch-size 16 \
    --cnn-validation-samples 20
```

**Expected:**
- ✅ Completes without crash
- ✅ GPU memory stays below 90%
- ✅ No system freeze
- ✅ Complex loss computes correctly

**If it crashes:**
- Reduce `--batch-size` to 12
- Reduce `--cnn-validation-samples` to 10
- Check GPU memory with `nvidia-smi`

---

## Phase 3: Large Model Test (15 minutes)

### Test 3.1: Large Model + Conservative Settings
```bash
python3 run_advanced_experiments.py \
    --single \
    --model-size large \
    --overlap 1 \
    --loss MSE \
    --epochs 50 \
    --topology-n-jobs 1 \
    --cnn-validation-samples 50 \
    --batch-size 24
```

**Expected:**
- ✅ Completes in ~15 minutes
- ✅ GPU memory clears between operations
- ✅ Topology runs sequentially (n_jobs=1)
- ✅ No CPU overload

**Monitor both:**
```bash
# Terminal 1: GPU
watch -n 1 nvidia-smi

# Terminal 2: CPU
htop
```

---

## Phase 4: Memory Leak Test (20 minutes)

### Test 4.1: Sequential Experiments
Run 3 experiments back-to-back to verify memory cleanup:

```bash
# Experiment 1
python3 run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss MSE \
    --epochs 25 --topology-n-jobs 1

# Experiment 2
python3 run_advanced_experiments.py \
    --single --model-size tiny --overlap 1 --loss MAE \
    --epochs 25 --topology-n-jobs 1

# Experiment 3
python3 run_advanced_experiments.py \
    --single --model-size tiny --overlap 2 --loss Huber \
    --epochs 25 --topology-n-jobs 1
```

**Expected:**
- ✅ GPU memory returns to baseline after each experiment
- ✅ Each experiment starts with "GPU memory cleared"
- ✅ No memory accumulation across experiments

**Check GPU memory:**
```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

Should see memory spike during training, then drop to baseline between experiments.

---

## Phase 5: Reproducibility Test (10 minutes)

### Test 5.1: Same Experiment Twice
```bash
# Run 1
python3 run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss MSE \
    --epochs 25 --topology-n-jobs 1 \
    --output-dir /tmp/test_run1

# Run 2
python3 run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss MSE \
    --epochs 25 --topology-n-jobs 1 \
    --output-dir /tmp/test_run2
```

**Verify reproducibility:**
```bash
# Compare final losses (should be identical)
grep "Best val loss" /tmp/test_run1/*/training_history.json
grep "Best val loss" /tmp/test_run2/*/training_history.json

# Compare predicted weights (should be very similar)
python3 -c "
import numpy as np
w1 = np.load('/tmp/test_run1/*/predicted_weights/epoch_0025_predictions.npy')
w2 = np.load('/tmp/test_run2/*/predicted_weights/epoch_0025_predictions.npy')
print(f'Max difference: {np.max(np.abs(w1 - w2))}')
print(f'Mean difference: {np.mean(np.abs(w1 - w2))}')
"
```

**Expected:**
- ✅ Final losses identical (or within 1e-6)
- ✅ Predicted weights very similar (max diff < 1e-4)

---

## Phase 6: Mini Tournament (1-2 hours)

### Test 6.1: Two Models, One Overlap, All Losses
```bash
./run_full_tournament.sh --models tiny,small --overlaps 0
```

**Expected:**
- ✅ Runs ~182 experiments (2 models × 1 overlap × 91 losses)
- ✅ Adaptive settings applied (tiny: n_jobs=2, small: n_jobs=2)
- ✅ No crashes or freezes
- ✅ Memory cleared between experiments

**Monitor progress:**
```bash
tail -f experiment_logs/tournament_*.log
```

---

## Phase 7: Full Tournament (3-5 days)

### Only run if all above tests pass!

```bash
./run_full_tournament.sh
```

**Settings applied automatically:**
- **tiny/small**: topology_jobs=2, cnn_samples=100, topo_freq=50
- **medium**: topology_jobs=1, cnn_samples=50, topo_freq=100
- **large/huge**: topology_jobs=1, cnn_samples=50, topo_freq=100

---

## 🚨 Troubleshooting

### Issue: GPU Out of Memory
```bash
# Reduce batch size
--batch-size 12

# Reduce CNN samples
--cnn-validation-samples 20

# Disable topology temporarily
--compute-topology-every 999999
```

### Issue: CPU Overload
```bash
# Force sequential topology
--topology-n-jobs 1

# Reduce data loader workers
# (edit run_advanced_experiments.py, line with num_workers=4)
```

### Issue: System Freeze
```bash
# Emergency kill
pkill -9 python3

# Clear GPU
python3 -c "import torch; torch.cuda.empty_cache()"

# Restart with more conservative settings
--topology-n-jobs 1 --batch-size 16 --cnn-validation-samples 20
```

### Issue: Disk Full
```bash
# Check space
df -h

# Clean old experiments
rm -rf /path/to/old/experiments

# Reduce checkpoint frequency
# (edit run_advanced_experiments.py, change save every 50 to 100)
```

---

## 📊 Success Criteria

Before running full tournament, verify:

- [ ] Phase 1 test passes (simple)
- [ ] Phase 2 test passes (complex loss)
- [ ] Phase 3 test passes (large model)
- [ ] Phase 4 test passes (no memory leak)
- [ ] Phase 5 test passes (reproducible)
- [ ] Phase 6 test passes (mini tournament)
- [ ] GPU memory clears between experiments
- [ ] No system freezes or crashes
- [ ] Topology runs without CPU overload

---

## 📈 Expected Timeline

| Phase | Duration | Purpose |
|-------|----------|---------|
| 1 | 5 min | Basic functionality |
| 2 | 10 min | Complex loss handling |
| 3 | 15 min | Large model stability |
| 4 | 20 min | Memory leak detection |
| 5 | 10 min | Reproducibility |
| 6 | 1-2 hours | Mini tournament |
| 7 | 3-5 days | Full tournament |

**Total testing time before full run: ~2 hours**

---

## 🎯 Quick Start

Run all tests automatically:

```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Create test script
cat > run_all_tests.sh << 'EOF'
#!/bin/bash
set -e

echo "Phase 1: Simple test..."
python3 run_advanced_experiments.py --single --model-size tiny --overlap 0 --loss MSE --epochs 50 --topology-n-jobs 1 --cnn-validation-samples 10

echo "Phase 2: Complex loss test..."
python3 run_advanced_experiments.py --single --model-size small --overlap 1 --loss "MAPE_LW0.1xJS_F0.05xKL" --epochs 50 --topology-n-jobs 1 --batch-size 16 --cnn-validation-samples 20

echo "Phase 3: Large model test..."
python3 run_advanced_experiments.py --single --model-size large --overlap 1 --loss MSE --epochs 50 --topology-n-jobs 1 --cnn-validation-samples 50

echo "All tests passed! ✅"
EOF

chmod +x run_all_tests.sh
./run_all_tests.sh
```

If all tests pass, proceed to mini tournament (Phase 6) or full tournament (Phase 7).
