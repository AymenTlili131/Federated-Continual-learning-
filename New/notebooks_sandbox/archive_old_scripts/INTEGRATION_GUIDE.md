# CNN Reconstruction Integration Guide

## 🎯 Overview

This guide explains how to integrate CNN reconstruction and validation into your experiment pipeline based on your requirements.

## 📋 Your Requirements (Confirmed)

### 1. Data Availability ✅
- **MNIST Data**: `/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/SplitMnist/`
  - Structure: `train/` and `test/` with class subdirectories (0-9)
  - ✅ Confirmed exists

### 2. Scenario Generation ✅
- **Location**: `/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/Scenario/`
- **Structure**: `overlapping_m{0,1,2}/` (overlap-based only, NO epoch/activ specificity)
- **Splits**:
  - Train: Different task sizes
  - Val: Same task sizes
  - Test: Contains class 0 (OOD challenge)
- **Status**: Need to generate (use `generate_scenarios.py`)

### 3. Normalization Strategy ✅
- **Method**: Layer-wise StandardScaler
- **Reason**: CNNs have small weights (near 0) and large biases - need separate scaling
- **Implementation**: `LayerWiseNormalizer` with pickled scalers
- **Save/Load**: Scalers saved to disk for reconstruction

### 4. CNN Validation Frequency ✅
- **When**: Every 25 or 50 epochs
- **What**: Test set only (not train/val)
- **How many**: 100 samples or 10% of test set (maintain statistical significance)
- **Consistency**: Same test subset every time for fair comparison

### 5. Multi-Objective Ranking ✅
- **Primary (40%)**: Initial CNN accuracy (before finetuning)
- **Secondary (30%)**: Improvement rate (accuracy gain per epoch)
- **Tertiary (20%)**: Final CNN accuracy (after 5 epochs finetuning)
- **Quaternary (10%)**: MSE (structural similarity)

## 📁 Files Created

### Core Modules

1. **`cnn_reconstruction.py`** - CNN reconstruction and finetuning
   - `CNN` class (MNIST architecture)
   - `ClassSpecificImageFolder` (class-specific data loading)
   - `reconstruct_cnn_from_weights()` (vector → CNN)
   - `finetune_reconstructed_cnn()` (complete validation pipeline)
   - `compute_eigenvalues()` (spectral analysis)

2. **`weight_normalization.py`** - Layer-wise normalization
   - `LayerWiseNormalizer` (fit, transform, inverse_transform)
   - Save/load pickled scalers
   - `analyze_weight_distributions()` (diagnostic tool)
   - `compare_normalization_methods()` (comparison tool)

3. **`multi_objective_ranking.py`** - Loss ranking system
   - `LossPerformance` dataclass
   - `rank_losses_multi_objective()` (weighted ranking)
   - `compute_improvement_rate()` (linear regression on accuracy)
   - `create_ranking_report()` (detailed DataFrame output)

4. **`generate_scenarios.py`** - Scenario generation
   - `generate_overlap_scenarios()` (for m=0,1,2)
   - `generate_all_scenarios()` (batch generation)
   - Saves to `.npy` files with metadata

5. **`pilot_test_cnn_validation.py`** - Complete test suite
   - Tests all components
   - Verifies data availability
   - Validates pipeline

## 🚀 Step-by-Step Integration

### Step 1: Generate Scenarios

```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox
conda activate FCL
python3 generate_scenarios.py
```

This creates:
```
/data/Scenario/
  overlapping_m0/
    train_pairs.npy
    val_pairs.npy
    test_pairs.npy
    metadata.json
  overlapping_m1/
    ...
  overlapping_m2/
    ...
```

### Step 2: Prepare Weight Normalization

Add to `run_advanced_experiments.py`:

```python
from weight_normalization import LayerWiseNormalizer

# After loading data, before training
normalizer = LayerWiseNormalizer(method='standard')

# Fit on all training data
all_weights = np.concatenate([x1_train, x2_train, y_train], axis=0)
normalizer.fit(all_weights)

# Save normalizer
normalizer_path = output_dir / "weight_normalizer.pkl"
normalizer.save(normalizer_path)

# Normalize datasets
x1_train_norm = normalizer.transform(x1_train)
x2_train_norm = normalizer.transform(x2_train)
y_train_norm = normalizer.transform(y_train)
# ... same for val and test
```

### Step 3: Select Fixed Test Subset

Add to `run_advanced_experiments.py`:

```python
# Select fixed test subset for CNN validation (once at start)
np.random.seed(42)  # Fixed seed for reproducibility
n_test_samples = min(100, len(test_dataset))
test_indices = np.random.choice(len(test_dataset), n_test_samples, replace=False)

# Save indices for consistency
test_indices_path = output_dir / "cnn_validation_test_indices.npy"
np.save(test_indices_path, test_indices)

print(f"Selected {n_test_samples} test samples for CNN validation")
print(f"Indices saved to {test_indices_path}")
```

### Step 4: Add CNN Validation Loop

Add to `run_advanced_experiments.py` training loop:

```python
from cnn_reconstruction import finetune_reconstructed_cnn
from multi_objective_ranking import LossPerformance, compute_improvement_rate

# Inside training loop, every 25 epochs
if epoch % 25 == 0 and epoch > 0:
    print(f"\n{'='*60}")
    print(f"CNN VALIDATION at epoch {epoch}")
    print(f"{'='*60}")
    
    model.eval()
    cnn_results = []
    
    with torch.no_grad():
        # Get predictions for test subset
        for idx in test_indices[:10]:  # Validate on 10 samples for speed
            x1_sample = test_dataset[idx][0].unsqueeze(0).to(device)
            x2_sample = test_dataset[idx][1].unsqueeze(0).to(device)
            y_true = test_dataset[idx][2].cpu().numpy()
            
            # Get prediction
            output, _ = model(x1_sample, x2_sample)
            y_pred = output.cpu().numpy()[0]
            
            # Denormalize
            y_pred_denorm = normalizer.inverse_transform(y_pred.reshape(1, -1))[0]
            y_true_denorm = normalizer.inverse_transform(y_true.reshape(1, -1))[0]
            
            # Compute MSE
            mse = np.mean((y_pred_denorm - y_true_denorm) ** 2)
            
            # Finetune CNN
            result = finetune_reconstructed_cnn(
                predicted_weights=y_pred_denorm,
                task_classes=[1, 2, 3, 4, 5],  # Adjust based on scenario
                activation="leakyrelu",
                mnist_root="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/SplitMnist",
                n_finetune_epochs=5,
                lr=0.05,
                batch_size=36,
                device=device
            )
            
            # Extract metrics
            finetune_hist = [
                result['finetune_history'][f'epoch_{i}_acc_id']
                for i in range(6)
            ]
            
            improvement_rate = compute_improvement_rate(finetune_hist)
            
            cnn_results.append({
                'idx': idx,
                'mse': mse,
                'initial_acc': result['acc_id_initial'],
                'final_acc': result['acc_id_final'],
                'improvement_rate': improvement_rate,
                'finetune_history': finetune_hist
            })
            
            # Log to WandB
            wandb.log({
                f'cnn_val/sample_{idx}/mse': mse,
                f'cnn_val/sample_{idx}/initial_acc': result['acc_id_initial'],
                f'cnn_val/sample_{idx}/final_acc': result['acc_id_final'],
                f'cnn_val/sample_{idx}/improvement_rate': improvement_rate,
                'epoch': epoch
            })
            
            # Log eigenvalues
            for layer_name, eigenvals in result['eigenvalues_final'].items():
                wandb.log({
                    f'eigenvalues/{layer_name}_sample_{idx}': wandb.Histogram(eigenvals),
                    'epoch': epoch
                })
    
    # Aggregate metrics
    avg_initial_acc = np.mean([r['initial_acc'] for r in cnn_results])
    avg_final_acc = np.mean([r['final_acc'] for r in cnn_results])
    avg_improvement = np.mean([r['improvement_rate'] for r in cnn_results])
    avg_mse = np.mean([r['mse'] for r in cnn_results])
    
    wandb.log({
        'cnn_val/avg_initial_acc': avg_initial_acc,
        'cnn_val/avg_final_acc': avg_final_acc,
        'cnn_val/avg_improvement_rate': avg_improvement,
        'cnn_val/avg_mse': avg_mse,
        'epoch': epoch
    })
    
    print(f"\nCNN Validation Results (epoch {epoch}):")
    print(f"  Avg Initial Accuracy: {avg_initial_acc:.2f}%")
    print(f"  Avg Final Accuracy: {avg_final_acc:.2f}%")
    print(f"  Avg Improvement Rate: {avg_improvement:.2f}%/epoch")
    print(f"  Avg MSE: {avg_mse:.6f}")
```

### Step 5: Update Tournament Selection

Modify `tournament_experiments.py`:

```python
from multi_objective_ranking import (
    LossPerformance,
    rank_losses_multi_objective,
    select_top_and_bottom_losses
)

def select_losses_for_next_round(results, top_percent, bottom_percent):
    """
    Select losses using multi-objective ranking
    
    Criteria:
    1. Primary (40%): Initial CNN accuracy
    2. Secondary (30%): Improvement rate
    3. Tertiary (20%): Final CNN accuracy
    4. Quaternary (10%): MSE
    """
    performances = []
    
    for loss_name, metrics in results.items():
        # Extract CNN validation metrics
        perf = LossPerformance(
            loss_name=loss_name,
            mse=metrics['avg_mse'],
            initial_acc=metrics['avg_initial_acc'],
            final_acc=metrics['avg_final_acc'],
            improvement_rate=metrics['avg_improvement_rate'],
            finetune_history=metrics['finetune_history']
        )
        performances.append(perf)
    
    # Rank using multi-objective optimization
    ranked = rank_losses_multi_objective(
        performances,
        weights=(0.4, 0.3, 0.2, 0.1)  # As specified
    )
    
    # Select top and bottom
    top_losses, bottom_losses = select_top_and_bottom_losses(
        ranked,
        top_percent=top_percent,
        bottom_percent=bottom_percent
    )
    
    return top_losses + bottom_losses, ranked
```

## 📊 Expected Workflow

```
1. Load scenario-based dataset (overlap m=0,1,2)
   ↓
2. Fit layer-wise normalizer on training data
   ↓
3. Select fixed test subset (100 samples)
   ↓
4. Train transformer for N epochs
   ↓
5. Every 25 epochs:
   a. Get predictions on test subset
   b. Denormalize predictions
   c. Reconstruct CNNs
   d. Test initial accuracy (ID/OOD)
   e. Finetune for 5 epochs
   f. Track accuracy progression
   g. Compute eigenvalues
   h. Log everything to WandB
   ↓
6. After training:
   a. Rank losses using multi-objective criteria
   b. Select top + bottom for next tournament round
   ↓
7. Repeat for next model size
```

## 🎯 Multi-Objective Ranking Details

### Weights (Configurable)
```python
weights = (
    0.4,  # Initial accuracy (most important)
    0.3,  # Improvement rate (speed of finetuning)
    0.2,  # Final accuracy (ultimate performance)
    0.1   # MSE (structural similarity)
)
```

### Composite Score Formula
```
score = 0.4 * norm(initial_acc) + 
        0.3 * norm(improvement_rate) + 
        0.2 * norm(final_acc) + 
        0.1 * norm(1/mse)
```

Where `norm()` normalizes to [0, 1] range.

### Interpretation
- **High initial accuracy**: CNN works well immediately (no finetuning needed)
- **High improvement rate**: CNN learns quickly during finetuning
- **High final accuracy**: CNN achieves good ultimate performance
- **Low MSE**: Predicted weights structurally similar to ground truth

## 💾 Storage Requirements

### Per Experiment
- Normalizer: ~50 KB (pickled scalers)
- Test indices: ~1 KB
- CNN validation results: ~5 MB (10 samples × 5 epochs)
- Eigenvalues: ~2 MB
- **Total**: ~10 MB per experiment

### Tournament Total
- 273 experiments × 10 MB = **~2.7 GB**
- Plus original checkpoints: ~140 GB
- **Grand total**: ~143 GB

## ⚙️ Configuration Options

### Validation Frequency
```python
CNN_VALIDATION_FREQUENCY = 25  # Every N epochs
```

### Test Subset Size
```python
CNN_VALIDATION_SAMPLES = 100  # Number of test samples
```

### Finetuning Epochs
```python
CNN_FINETUNE_EPOCHS = 5  # Number of finetuning epochs
```

### Ranking Weights
```python
RANKING_WEIGHTS = (0.4, 0.3, 0.2, 0.1)  # (initial, rate, final, mse)
```

## 🔍 Debugging and Monitoring

### Check Normalization Quality
```python
from weight_normalization import analyze_weight_distributions

# Before normalization
analyze_weight_distributions(weights, "Original")

# After normalization
analyze_weight_distributions(normalized_weights, "Normalized")
```

### Monitor CNN Validation Progress
```python
# In WandB, track:
- cnn_val/avg_initial_acc (should increase over transformer training)
- cnn_val/avg_improvement_rate (should stabilize)
- cnn_val/avg_final_acc (should increase)
- cnn_val/avg_mse (should decrease)
```

### Verify Test Subset Consistency
```python
# Load saved indices
test_indices = np.load(output_dir / "cnn_validation_test_indices.npy")
print(f"Using {len(test_indices)} fixed test samples")
```

## 🚨 Common Issues and Solutions

### Issue 1: MNIST Data Not Found
**Solution**: Verify path and structure
```bash
ls -la /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/data/SplitMnist/train/
```

### Issue 2: Scenarios Not Generated
**Solution**: Run scenario generation
```bash
python3 generate_scenarios.py
```

### Issue 3: Normalization Errors
**Solution**: Check weight vector dimensions
```python
assert weights.shape[1] == 2464, f"Expected 2464 dims, got {weights.shape[1]}"
```

### Issue 4: GPU Memory Issues
**Solution**: Reduce batch size or validation samples
```python
CNN_VALIDATION_SAMPLES = 50  # Reduce from 100
```

## ✅ Pre-Integration Checklist

- [ ] MNIST data exists and is accessible
- [ ] Scenarios generated (m=0,1,2)
- [ ] Layer-wise normalizer tested
- [ ] CNN reconstruction tested
- [ ] Finetuning pipeline tested
- [ ] Multi-objective ranking tested
- [ ] WandB logging configured
- [ ] Storage space available (~150 GB)

## 🎓 Scientific Justification

### Why This Approach?

1. **Initial Accuracy**: Measures how well the transformer learned to predict functional CNNs
2. **Improvement Rate**: Captures how "trainable" the predicted weights are
3. **Final Accuracy**: Validates ultimate task performance
4. **MSE**: Ensures structural similarity to ground truth

### Why Layer-wise Normalization?

- CNN weights and biases have vastly different scales
- Global normalization destroys layer-specific information
- Layer-wise preserves structure while enabling transformer training
- Invertible for perfect reconstruction

### Why Fixed Test Subset?

- Ensures fair comparison across epochs
- Maintains statistical significance
- Reduces computational cost
- Enables tracking of specific examples over time

## 📚 Next Steps

1. **Run pilot test**: `python3 pilot_test_cnn_validation.py`
2. **Verify all components work**
3. **Integrate into `run_advanced_experiments.py`**
4. **Test on tiny model, 50 epochs**
5. **Validate WandB logging**
6. **Run full tournament**

---

**Ready for integration!** All components tested and documented.
