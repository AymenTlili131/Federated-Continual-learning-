# Major Architecture Change Required

## 🎯 User Requirements Summary

You've requested a fundamental shift from simple weight prediction evaluation to **full CNN reconstruction and validation**. This is a significant architectural change that requires integrating the meta.ipynb pipeline into the experiment system.

## 📋 What You Want

### 1. **Scenario-Based Dataset Creation** (Not Random 70/15/15)
- Load scenarios from `Create training scenarios.ipynb`
- Train set: Different task sizes
- Val set: Same task sizes  
- Test set: Contains class 0 (OOD challenge)
- Criteria-based selection, not random splitting

### 2. **CNN Reconstruction from Predicted Weights**
- Take transformer's predicted weight vector (2464 dims)
- Reconstruct CNN architecture
- Load weights into proper layer structure
- Validate architecture integrity

### 3. **CNN Finetuning on MNIST**
- Use `ClassSpecificImageFolder` for class-specific data loading
- Train on in-distribution (ID) classes only
- Test on both ID and out-of-distribution (OOD) classes
- Track accuracy over 5 finetuning epochs
- Log step-by-step progress (epochs 0, 1, 2, 3, 4, 5)

### 4. **Ground Truth Comparison**
- Load ground truth models from `Merged zoo.csv`
- Compare reconstructed CNN to ground truth
- Track performance delta

### 5. **Eigenvalue Analysis**
- Compute eigenvalues of weight matrices
- Track before and after finetuning
- Log to WandB as histograms
- Analyze spectral properties

### 6. **Weight Normalization**
- Consider StandardScaler or MinMaxScaler
- Account for different layer means
- Preserve layer-specific information despite flattening

### 7. **Comprehensive Logging**
- Reconstructed CNN accuracy (ID and OOD)
- Finetuning progress (5 epochs)
- Eigenvalue distributions
- Ground truth comparison
- All metrics to WandB

## 📁 Files Created

### 1. `cnn_reconstruction.py` ✅
Complete CNN reconstruction and finetuning module:
- `CNN` class (MNIST architecture)
- `ClassSpecificImageFolder` (class-specific data loading)
- `reconstruct_cnn_from_weights()` (vector → CNN)
- `compute_eigenvalues()` (spectral analysis)
- `train_cnn_epoch()` (training loop)
- `validate_cnn()` (validation)
- `finetune_reconstructed_cnn()` (complete pipeline)

### 2. `scenario_dataset.py` ✅
Scenario-based dataset creation:
- `create_scenario_splits()` (load train/val/test scenarios)
- `generate_scenarios_on_fly()` (generate if files missing)
- `load_weights_from_zoo()` (ground truth loading)
- `create_scenario_dataset()` (complete dataset creation)
- Weight normalization support (StandardScaler, MinMaxScaler)

## 🔧 What Still Needs Integration

### Critical Integration Tasks

**1. Update `run_advanced_experiments.py`**
- Replace random data loading with `scenario_dataset.py`
- Add CNN reconstruction after each validation epoch
- Integrate finetuning pipeline
- Log eigenvalues to WandB
- Track ID/OOD accuracy

**2. Create Scenario Files**
- Run `Create training scenarios.ipynb` to generate scenario files
- Or use on-the-fly generation in `scenario_dataset.py`

**3. Update WandB Logging**
- Add CNN accuracy metrics (ID/OOD)
- Add finetuning progress (epochs 0-5)
- Add eigenvalue histograms
- Add ground truth comparison

**4. Add Ground Truth Loading**
- Parse `Merged zoo.csv` properly
- Match scenarios to ground truth models
- Compute performance delta

**5. Weight Normalization Decision**
- Test with/without normalization
- Compare StandardScaler vs MinMaxScaler
- Evaluate impact on reconstruction quality

## 📊 Expected Workflow

```
1. Load scenario-based dataset
   ↓
2. Train transformer on weight prediction
   ↓
3. Every N epochs:
   a. Get predicted weights
   b. Reconstruct CNN
   c. Test CNN (ID accuracy)
   d. Test CNN (OOD accuracy)
   e. Compute eigenvalues
   f. Finetune for 5 epochs
   g. Track finetuning progress
   h. Log all metrics to WandB
   ↓
4. Compare to ground truth
   ↓
5. Save results
```

## 🎯 Tournament Impact

This changes the tournament evaluation:
- **Before**: MSE on weight vectors
- **After**: CNN accuracy after reconstruction + finetuning

Tournament selection should be based on:
- **Primary**: Final CNN accuracy (after 5 finetuning epochs)
- **Secondary**: Initial CNN accuracy (before finetuning)
- **Tertiary**: OOD generalization

## 💾 Storage Impact

Adding CNN validation increases storage:
- Reconstructed CNN checkpoints
- Finetuning history (5 epochs × N samples)
- Eigenvalue data
- MNIST data loading

**Estimated additional storage per experiment**: ~500 MB
**Total for tournament**: ~140 GB additional

## ⚠️ Critical Considerations

### 1. **Dataset Size**
Your current system loads 10,000 samples randomly. With scenarios:
- Need to determine actual scenario count
- May have fewer or more samples
- Need to balance scenario coverage vs computation

### 2. **MNIST Data**
- Requires SplitMNIST dataset at `./data/SplitMnist/`
- Train and test splits
- Class-organized directories

### 3. **Computational Cost**
- CNN finetuning adds significant time
- 5 epochs × batch training × N validation samples
- May need to reduce validation frequency

### 4. **Normalization Strategy**
You raised a valid concern about layer-specific means. Options:
- **Global normalization**: StandardScaler on full 2464 vector
- **Layer-wise normalization**: Separate scaling per layer
- **No normalization**: Let gated attention handle it
- **Hybrid**: Normalize but preserve layer boundaries

## 🚀 Recommended Next Steps

### Immediate (Do This Now)
1. **Verify MNIST data exists** at `./data/SplitMnist/`
2. **Run scenario generation** or verify scenarios exist
3. **Test CNN reconstruction** with sample weights
4. **Decide on normalization** strategy

### Short-term (This Week)
1. **Integrate into `run_advanced_experiments.py`**
2. **Test full pipeline** on tiny model, 2 epochs
3. **Verify WandB logging** works correctly
4. **Validate storage** requirements

### Medium-term (Before Tournament)
1. **Run pilot experiments** (10-20 samples)
2. **Analyze reconstruction quality**
3. **Tune finetuning hyperparameters**
4. **Update tournament selection** criteria

## 📝 Code Integration Example

Here's how to integrate into `run_advanced_experiments.py`:

```python
from cnn_reconstruction import finetune_reconstructed_cnn, compute_eigenvalues
from scenario_dataset import create_scenario_dataset, get_activation_name

# Replace data loading
train_dataset, val_dataset, test_dataset, metadata = create_scenario_dataset(
    merged_zoo_path="./data/Merged zoo.csv",
    overlap=overlap,
    epoch_key=0,
    activ_key=3,  # leakyrelu
    normalize_weights=True,
    normalization_method="standardize"
)

# After validation epoch
if epoch % 10 == 0:  # Every 10 epochs
    for i in range(min(5, len(predictions))):  # Sample 5 predictions
        result = finetune_reconstructed_cnn(
            predicted_weights=predictions[i],
            task_classes=metadata['task_classes'][i],
            activation=get_activation_name(metadata['activ_key']),
            n_finetune_epochs=5
        )
        
        # Log to WandB
        wandb.log({
            f'cnn/acc_id_initial_{i}': result['acc_id_initial'],
            f'cnn/acc_id_final_{i}': result['acc_id_final'],
            f'cnn/acc_ood_initial_{i}': result['acc_ood_initial'],
            **{f'cnn/finetune_epoch_{e}_{i}': result['finetune_history'][f'epoch_{e}_acc_id'] 
               for e in range(6)}
        })
        
        # Log eigenvalues
        for layer_name, eigenvals in result['eigenvalues_final'].items():
            wandb.log({
                f'eigenvalues/{layer_name}_{i}': wandb.Histogram(eigenvals)
            })
```

## ❓ Questions to Answer

1. **Do scenario files exist?** Check `./data/Scenario/`
2. **Does MNIST data exist?** Check `./data/SplitMnist/`
3. **Normalization preference?** StandardScaler, MinMaxScaler, or none?
4. **Validation frequency?** Every N epochs for CNN reconstruction?
5. **Sample size?** How many predictions to validate per epoch?
6. **Tournament criterion?** Final CNN accuracy or initial?

## 🎓 Research Validity

This approach is **scientifically sound** because:
- Evaluates actual task performance (CNN accuracy)
- Tests generalization (OOD classes)
- Validates reconstruction quality (finetuning improvement)
- Provides interpretability (eigenvalue analysis)
- Follows meta-learning best practices

## 📚 References

Your existing notebooks:
- `meta.ipynb`: CNN reconstruction and finetuning pipeline
- `Create training scenarios.ipynb`: Scenario generation logic
- `Merged zoo.csv`: Ground truth weight repository

---

## ✅ Summary

I've created the foundational modules (`cnn_reconstruction.py`, `scenario_dataset.py`) but **full integration requires**:

1. Verifying data availability (MNIST, scenarios)
2. Deciding on normalization strategy
3. Integrating into experiment runner
4. Testing the complete pipeline
5. Updating tournament selection criteria

This is a **major architectural change** that transforms your system from weight prediction to full CNN meta-learning validation. It's the right approach for your research goals, but requires careful integration and testing.

**Ready to proceed?** Let me know:
- Do the data files exist?
- What normalization strategy do you prefer?
- Should I integrate into `run_advanced_experiments.py` now?
