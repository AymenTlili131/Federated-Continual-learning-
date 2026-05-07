# Comprehensive Updates - All Requested Changes

## 1. ✅ Sinkhorn Loss Fix
**Issue:** `Can't call numpy() on Tensor that requires grad`
**Fix:** Detach tensors before numpy conversion, use MSE proxy for gradient flow

## 2. CNN Weight Saving Strategy
**OLD:** Save transformer predictions every epoch
**NEW:** Save CNN weights during finetuning (only at validation intervals)
- At epoch 100 (validation): save `subCNN_epoch1_weights.npy` through `subCNN_epoch5_weights.npy`
- Only save during CNN validation intervals (25 or 50 epochs)

## 3. CNN Validation Granularity
**OLD:** Only final epoch 5 accuracy logged
**NEW:** Stepwise logging for all 5 finetuning epochs
- Log accuracy, loss per CNN epoch
- Eigenvalue analysis per CNN epoch
- Topology analysis per CNN epoch (optional - may be expensive)

## 4. Adaptive Checkpointing by Model Size
| Model Size | Checkpointing Strategy |
|------------|------------------------|
| tiny | Best + Last only |
| small | Best + Last only |
| medium | Every 50 epochs + Best + Last |
| large | Every 25 epochs + Best + Last |
| huge | Every 25 epochs + Best + Last |

## 5. Adaptive Epoch Counts
| Model Size | Epochs |
|------------|--------|
| tiny | 500 |
| small | 500 |
| medium | 350 |
| large | 200 |
| huge | 200 |

## 6. Revised Learning Rate Schedule
**Requirements:**
- No excessive warmup distorting training
- Gentler decay (avoid reaching 1e-6, 1e-7, 1e-8 too fast)
- Fast convergence amenable to epoch counts

**New Strategy:**
- Short warmup: 5% of epochs (25 for tiny/small, 17 for medium, 10 for large/huge)
- Cosine annealing with min_lr = 0.1 * base_lr (1e-5 minimum)
- No plateau phase (smooth decay throughout)

## 7. Per-Step Loss Logging
**Medium and larger models:** Log loss every training step to WandB
**Tiny/Small:** Log per epoch only (to reduce overhead)

## 8. Parallel Training for GPU Utilization
**Current:** 800MB / 16GB = 5% utilization
**Target:** 60-80% utilization

**Strategy:**
- Run 4 tiny experiments in parallel
- Run 2-3 small experiments in parallel
- Run 1-2 medium experiments in parallel
- Run 1 large/huge experiment at a time

**Implementation:**
- Use Python multiprocessing with GPU assignment
- Each process gets dedicated WandB run
- Synchronization barrier: wait for all tiny models before ranking

## 9. Per-Overlap Ranking System
**OLD:** Single ranking across all overlaps
**NEW:** Rank losses separately per overlap tier
- Rank for overlap=0
- Rank for overlap=1
- Rank for overlap=2
- Pass top N losses per overlap to next model size

## 10. GPU Power Limit
**Current:** Enforced Power Limit shown in WandB
**Location:** `/sys/class/drm/card0/device/hwmon/hwmon*/power1_cap`
**Adjustment:** `sudo nvidia-smi -pl <watts>` (e.g., 180W for RTX 5060 Ti)
**Safe range:** 120W - 180W (check TDP spec)

## 11. cudf.pandas Integration
**Available:** RapidS setup detected
**Usage:** Accelerate pandas operations with GPU
```python
import cudf.pandas
cudf.pandas.install()
# All pandas operations now GPU-accelerated
```

## Implementation Priority
1. ✅ Sinkhorn fix (DONE)
2. Adaptive epochs + LR schedule
3. Adaptive checkpointing
4. CNN validation granularity + weight saving
5. Per-step logging for medium+
6. Parallel training system
7. Per-overlap ranking
8. cudf.pandas integration
