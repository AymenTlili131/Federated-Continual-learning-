# Updated CNN Validation Integration Guide

## 🎯 Changes Based on Your Feedback

### ✅ Fixed Issues

1. **Batch Size**: Changed from 36 to **24** for CNN validation (matches your training setup)
2. **Test Set Criteria**: Replaced "class 0 only" with **multi-faceted OOD scoring** (15% test, 15% val, 70% train)
3. **Eigenvalue Analysis**: Expanded to track **inputs, predicted, ground truth, and all finetuning epochs**
4. **Bash Launch Script**: Created `launch_cnn_validation_experiment.sh` for easy execution
5. **Scenario Consistency**: Ensured activation+epoch consistency in training pairs

### 🎨 Creative Test Set Design

Instead of just "contains class 0", the new OOD scoring system evaluates:

**OOD Score Components** (higher = more challenging):
1. **Extreme size differences** (|len(t1) - len(t2)| ≥ 4) → +4 points
2. **Rare digits** (0, 1, 9) → +2 points
3. **Very small tasks** (len ≤ 2) or **very large** (len ≥ 8) → +2 points each
4. **Asymmetric overlaps** (overlap < 30% of smaller task) → +2 points
5. **Digit diversity** (both even and odd) → +1 point

**Result**: Top 15% by OOD score → Test set (challenging), Next 15% → Val set, Rest 70% → Train set

This ensures:
- **Larger test set** (~15% instead of ~5% with class 0 only)
- **More diverse challenges** (not just one arbitrary class)
- **Scientifically justified** (multiple difficulty dimensions)
- **Balanced coverage** (rare digits, size extremes, asymmetry)

## 📊 Eigenvalue Analysis (Comprehensive)

Now tracks eigenvalues for:

1. **Input X1** - First input weight vector
2. **Input X2** - Second input weight vector
3. **Predicted (Initial)** - Transformer's prediction before finetuning
4. **Ground Truth** - Target weights from merged zoo
5. **Finetuned Epoch 0-5** - After each finetuning epoch

This allows analysis of:
- How transformer combines inputs (X1 + X2 → Predicted)
- How close prediction is to ground truth (spectral similarity)
- How finetuning changes spectral properties (trainability)
- Convergence patterns (eigenvalue evolution)

## 🚀 Quick Test Command

### Small Test (10 minutes)
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Generate scenarios first (one-time, ~5 min)
conda activate FCL
python3 generate_scenarios.py

# Run small test
./launch_cnn_validation_experiment.sh --test
```

The `--test` flag sets:
- Model: tiny
- Epochs: 50
- CNN validation: every 25 epochs (2 validations)
- CNN samples: 10 (fast)
- Overlap: 0

### Full Single Experiment (1-2 hours)
```bash
./launch_cnn_validation_experiment.sh \
    --model-size tiny \
    --overlap 0 \
    --epochs 200 \
    --batch-size 24 \
    --lr 0.0001 \
    --loss MSE \
    --cnn-freq 25 \
    --cnn-samples 100
```

### Custom Configuration
```bash
./launch_cnn_validation_experiment.sh \
    --model-size small \
    --overlap 1 \
    --epochs 150 \
    --cnn-freq 50 \
    --cnn-samples 50 \
    --no-wandb  # Disable WandB if needed
```

## 📁 Updated File Structure

```
notebooks_sandbox/
├── cnn_reconstruction.py              # CNN reconstruction + finetuning
│   └── NEW: Batch size 24, eigenvalue tracking for all weight types
├── weight_normalization.py            # Layer-wise normalization
├── multi_objective_ranking.py         # Multi-objective loss ranking
├── generate_scenarios.py              # Scenario generation
│   └── NEW: Multi-faceted OOD scoring (15% test, 15% val, 70% train)
├── pilot_test_cnn_validation.py       # Test suite
├── launch_cnn_validation_experiment.sh # Bash launcher
│   └── NEW: Easy experiment launching with --test mode
└── UPDATED_INTEGRATION_GUIDE.md       # This file

data/
├── SplitMnist/
│   ├── train/
│   └── test/
├── Merged zoo.csv
└── Scenario/
    ├── overlapping_m0/
    │   ├── train_pairs.npy  # 70% of pairs
    │   ├── val_pairs.npy    # 15% of pairs
    │   └── test_pairs.npy   # 15% of pairs (high OOD score)
    ├── overlapping_m1/
    └── overlapping_m2/
```

## 🔧 Scenario Generation Details

### Activation + Epoch Consistency

**Important**: The scenarios themselves don't specify activation or epoch - those are **training variables** you control when running experiments.

**How it works**:
1. Scenarios define **task pairs** only: `[[task1_classes], [task2_classes]]`
2. During training, you specify: `--overlap 0` (which scenario set to use)
3. The merged zoo contains models with different activations and epochs
4. When loading training data, you filter by activation+epoch to ensure consistency

**Example**:
```python
# Load scenario
scenario = [[1,2,3], [4,5,6]]  # Task pair

# Load weights from zoo with same activation+epoch
weights_x1 = zoo[(zoo['task'] == [1,2,3]) & 
                 (zoo['activation'] == 'leakyrelu') & 
                 (zoo['epoch'] == 21)]

weights_x2 = zoo[(zoo['task'] == [4,5,6]) & 
                 (zoo['activation'] == 'leakyrelu') &  # Same activation
                 (zoo['epoch'] == 21)]                  # Same epoch

ground_truth = zoo[(zoo['task'] == [1,2,3,4,5,6]) & 
                   (zoo['activation'] == 'leakyrelu') &  # Same activation
                   (zoo['epoch'] == 21)]                  # Same epoch
```

This ensures all weights in a training sample have the same activation and epoch.

### OOD Score Distribution

Expected distribution for overlap m=0:
```
Total pairs: ~1000-2000
Train: ~1400 (70%)
Val: ~300 (15%)
Test: ~300 (15%)

Test set OOD scores: 6-12 (high challenge)
Val set OOD scores: 3-6 (medium challenge)
Train set OOD scores: 0-3 (easier, diverse)
```

## 📊 Eigenvalue Logging to WandB

```python
# In validation loop
result = finetune_reconstructed_cnn(
    predicted_weights=y_pred,
    input_weights_x1=x1,
    input_weights_x2=x2,
    ground_truth_weights=y_true,
    ...
)

# Log all eigenvalue types
eigenvalues = result['eigenvalues_analysis']

for weight_type, layer_eigenvals in eigenvalues.items():
    # weight_type: 'input_x1', 'input_x2', 'predicted_initial', 
    #              'ground_truth', 'finetuned_epoch_0', ..., 'finetuned_epoch_5'
    
    for layer_name, eigs in layer_eigenvals.items():
        wandb.log({
            f'eigenvalues/{weight_type}/{layer_name}': wandb.Histogram(eigs),
            'epoch': epoch
        })
```

## 🎯 Multi-Objective Ranking (Unchanged)

Still uses:
- **Primary (40%)**: Initial CNN accuracy
- **Secondary (30%)**: Improvement rate
- **Tertiary (20%)**: Final CNN accuracy
- **Quaternary (10%)**: MSE

## 💾 Storage Estimates

### Per Experiment
- Normalizer: ~50 KB
- Test indices: ~1 KB
- CNN validation (100 samples, 8 validations): ~80 MB
- Eigenvalues (all types, all epochs): ~50 MB
- **Total per experiment**: ~150 MB

### Tournament (273 experiments)
- CNN validation data: ~40 GB
- Original checkpoints: ~100 GB
- **Total**: ~140 GB

## ⚙️ Configuration Options

### Bash Script Options
```bash
--model-size SIZE       # tiny/small/medium/large/huge
--overlap N             # 0/1/2
--epochs N              # Training epochs
--batch-size N          # 24 recommended
--lr FLOAT              # Learning rate
--loss NAME             # Loss function
--cnn-freq N            # CNN validation frequency (25 or 50)
--cnn-samples N         # 100 for full, 10 for testing
--no-wandb              # Disable WandB
--test                  # Quick test mode
```

### Python Parameters
```python
finetune_reconstructed_cnn(
    predicted_weights=...,
    task_classes=[1,2,3,4,5],
    activation="leakyrelu",
    batch_size=24,              # NEW: 24 instead of 36
    input_weights_x1=x1,        # NEW: For eigenvalue analysis
    input_weights_x2=x2,        # NEW: For eigenvalue analysis
    ground_truth_weights=gt,    # NEW: For eigenvalue analysis
    ...
)
```

## 🔍 Verification Steps

### 1. Generate Scenarios
```bash
conda activate FCL
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox
python3 generate_scenarios.py
```

Expected output:
```
Generating scenarios for overlap m=0
Total task pairs generated: ~1500
Split statistics:
  Train pairs: ~1050 (70.0%)
  Val pairs:   ~225 (15.0%)
  Test pairs:  ~225 (15.0%) - OOD challenge
  
  Test set OOD scores: min=6, max=12, avg=8.5
```

### 2. Run Small Test
```bash
./launch_cnn_validation_experiment.sh --test
```

Should complete in ~10 minutes and produce:
- Scenario loading
- Weight normalization
- 50 epochs of training
- 2 CNN validations (epochs 25, 50)
- Eigenvalue analysis for all weight types
- WandB logging

### 3. Verify Outputs
Check that these exist:
```
Experiments/tiny_overlap0_MSE_TIMESTAMP/
├── weight_normalizer.pkl
├── cnn_validation_test_indices.npy
├── checkpoints/
├── attention_heatmaps/
├── topology/
└── cnn_validation/
    ├── epoch_25/
    │   ├── eigenvalues_input_x1.json
    │   ├── eigenvalues_input_x2.json
    │   ├── eigenvalues_predicted.json
    │   ├── eigenvalues_ground_truth.json
    │   └── eigenvalues_finetuned_epoch_*.json
    └── epoch_50/
        └── ...
```

## 🚨 Common Issues

### Issue 1: Scenarios Not Generated
```bash
# Solution: Generate manually
python3 generate_scenarios.py
```

### Issue 2: Batch Size Mismatch
```python
# OLD (wrong)
batch_size=36

# NEW (correct)
batch_size=24
```

### Issue 3: Missing Eigenvalues
```python
# Make sure to pass all weight types
result = finetune_reconstructed_cnn(
    predicted_weights=y_pred,
    input_weights_x1=x1,        # Required for eigenvalue analysis
    input_weights_x2=x2,        # Required for eigenvalue analysis
    ground_truth_weights=y_gt,  # Required for eigenvalue analysis
    ...
)
```

### Issue 4: Test Set Too Small
With new OOD scoring, test set should be ~15% of total pairs:
- m=0: ~225 test pairs (was ~50 with class 0 only)
- m=1: ~180 test pairs
- m=2: ~150 test pairs

## 📈 Expected Results

### Initial Accuracy (Before Finetuning)
- **Good losses**: 75-85%
- **Average losses**: 65-75%
- **Poor losses**: <65%

### Improvement Rate
- **Fast learners**: 3-5% per epoch
- **Average**: 2-3% per epoch
- **Slow**: <2% per epoch

### Final Accuracy (After 5 Epochs)
- **Best**: 90-95%
- **Average**: 85-90%
- **Poor**: <85%

### Eigenvalue Patterns
- **Input weights**: Broad spectrum (diverse initialization)
- **Predicted weights**: Should be similar to ground truth
- **Finetuned weights**: Spectrum should stabilize/sharpen

## 🎓 Scientific Insights

### Why Multi-Faceted OOD Scoring?

1. **Extreme size differences**: Tests generalization across task scales
2. **Rare digits**: Tests handling of underrepresented classes
3. **Small/large tasks**: Tests boundary conditions
4. **Asymmetric overlaps**: Tests partial knowledge transfer
5. **Digit diversity**: Tests balanced representation

This is more robust than arbitrary "class 0" criterion.

### Why Track All Eigenvalues?

1. **Input X1/X2**: Understand what transformer receives
2. **Predicted**: See how transformer combines inputs
3. **Ground Truth**: Know the target spectral properties
4. **Finetuned 0-5**: Track learning dynamics

Eigenvalue evolution reveals:
- **Rank collapse**: If spectrum narrows too much
- **Trainability**: If eigenvalues shift during finetuning
- **Convergence**: If spectrum stabilizes
- **Similarity**: Compare predicted vs ground truth spectra

## ✅ Pre-Launch Checklist

- [ ] Scenarios generated (`python3 generate_scenarios.py`)
- [ ] MNIST data exists (`/data/SplitMnist/`)
- [ ] Merged zoo exists (`/data/Merged zoo.csv`)
- [ ] Conda environment active (`conda activate FCL`)
- [ ] Bash script executable (`chmod +x launch_*.sh`)
- [ ] Storage available (~150 GB)
- [ ] GPU available (check `nvidia-smi`)

## 🚀 Launch Commands

### Quick Test (Recommended First)
```bash
./launch_cnn_validation_experiment.sh --test
```

### Full Experiment
```bash
./launch_cnn_validation_experiment.sh \
    --model-size tiny \
    --overlap 0 \
    --epochs 200 \
    --cnn-freq 25 \
    --cnn-samples 100
```

### Tournament (All Overlaps)
```bash
for overlap in 0 1 2; do
    ./launch_cnn_validation_experiment.sh \
        --model-size tiny \
        --overlap $overlap \
        --epochs 200 \
        --cnn-freq 50 \
        --cnn-samples 100
done
```

---

## 📝 Summary of Changes

| Aspect | Old | New |
|--------|-----|-----|
| Batch size | 36 | **24** |
| Test set criterion | Class 0 only (~5%) | **Multi-faceted OOD scoring (~15%)** |
| Eigenvalues | Initial + Final | **Input X1/X2, Predicted, GT, Finetuned 0-5** |
| Launch method | Manual Python | **Bash script with --test mode** |
| Test set size | ~50 pairs | **~225 pairs** |
| OOD dimensions | 1 (class 0) | **5 (size, rarity, extremes, asymmetry, diversity)** |

All changes implemented and ready for testing!
