# Experiment Runner - Quick Reference

## Model Sizes Available

| Model | Parameters | Layers | Heads | d_model | d_ff | Neck | Use Case |
|-------|-----------|--------|-------|---------|------|------|----------|
| tiny | ~500K | 1 | 2 | 32 | 128 | 16 | Rapid prototyping |
| small | ~2M | 2 | 4 | 64 | 256 | 32 | Fast training |
| medium | ~8M | 3 | 4 | 128 | 512 | 64 | Balanced |
| large | ~25M | 4 | 8 | 256 | 1024 | 128 | High capacity |
| huge | ~100M | 6 | 12 | 384 | 1536 | 192 | Research scale |

## Quick Start Commands

### 1. Single Quick Test (10 epochs, no WandB)
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox

# Test tiny model
./run_experiments.sh --single --models tiny --overlaps 2 --losses mse --epochs 10

# Test medium model
./run_experiments.sh --single --models medium --overlaps 2 --losses mse --epochs 50
```

### 2. Full Experiment Suite (All combinations)
```bash
# Without WandB (100 epochs each)
./run_experiments.sh --epochs 100

# With WandB logging (recommended)
./run_experiments.sh --wandb --epochs 100
```

### 3. Specific Model Size Tests
```bash
# Test all overlaps and losses for medium model
./run_experiments.sh --models medium --epochs 100 --wandb

# Test huge model only
./run_experiments.sh --models huge --overlaps "0 1 2" --losses "mse wasserstein" --epochs 100 --wandb
```

### 4. Specific Loss Function Comparison
```bash
# Compare all loss functions on medium model, overlap 2
./run_experiments.sh --models medium --overlaps 2 --losses "mse wasserstein lwwn mape" --epochs 100 --wandb
```

### 5. Overlap Comparison
```bash
# Test all overlaps with MSE loss on large model
./run_experiments.sh --models large --overlaps "0 1 2" --losses mse --epochs 100 --wandb
```

## Python Direct Usage

### Single Experiment
```python
from run_full_experiments import run_single_experiment

result = run_single_experiment(
    model_size='medium',
    overlap=2,
    loss_name='mse',
    epochs=100,
    batch_size=32,
    lr=1e-4,
    use_wandb=True,
    save_attention_every=10  # Save attention heatmaps every 10 epochs
)
```

### Full Suite
```python
from run_full_experiments import run_full_experiment_suite

results = run_full_experiment_suite(
    model_sizes=['tiny', 'small', 'medium', 'large', 'huge'],
    overlaps=[0, 1, 2],
    losses=['mse', 'wasserstein', 'lwwn', 'mape'],
    epochs=100,
    use_wandb=True,
    output_base_dir='./experiments'
)
```

## Output Structure

Each experiment creates:
```
experiments/{model_size}_overlap{N}_{loss}/
├── checkpoints/
│   ├── best_model.pth              # Best validation loss
│   ├── final_model.pth             # Final epoch
│   └── checkpoint_epoch_*.pth      # Every 10 epochs
├── attention_heatmaps/
│   ├── epoch_0010_enc1.png         # Encoder 1 attention
│   ├── epoch_0010_enc2.png         # Encoder 2 attention
│   ├── epoch_0010_dec.png          # Decoder attention
│   └── ...                         # Every 10 epochs
├── metrics/
│   └── test_metrics_full.csv       # Distance metrics on test set
└── training_history.json           # Loss curves
```

## Features Included

### ✓ Attention Heatmap Visualization
- Saved locally every 10 epochs (configurable)
- Logged to WandB at each validation step
- Separate plots for Encoder 1, Encoder 2, Decoder
- Shows all attention heads

### ✓ Distance Metrics (13 metrics)
1. Euclidean
2. Manhattan
3. Cosine
4. Frobenius Norm
5. Q-quantile Loss
6. Norm of Jacobian
7. Fisher Information Difference
8. Contractive Loss
9. Wasserstein Distance
10. MAPE
11. LWLN
12. Jensen-Shannon Divergence
13. Auto-regressive Loss

### ✓ WandB Logging
- Training/validation loss curves
- Learning rate schedule
- Attention heatmaps (all 3 types)
- Model configuration
- Hyperparameters

### ✓ Checkpointing
- Best model (lowest validation loss)
- Every 10 epochs
- Final model
- Includes optimizer state for resuming

### ✓ Finetuning (Placeholder)
- 1-5 epochs on MNIST test set
- Saves finetuned weight snapshots
- Computes metrics on finetuned weights

## Example Workflows

### Workflow 1: Quick Validation
```bash
# Test if everything works (5 minutes)
./run_experiments.sh --single --models tiny --overlaps 2 --losses mse --epochs 10
```

### Workflow 2: Model Size Comparison
```bash
# Compare all model sizes on same task (few hours)
./run_experiments.sh --models "tiny small medium large huge" \
                     --overlaps 2 \
                     --losses mse \
                     --epochs 100 \
                     --wandb
```

### Workflow 3: Loss Function Ablation
```bash
# Test all loss functions on medium model (few hours)
./run_experiments.sh --models medium \
                     --overlaps 2 \
                     --losses "mse wasserstein lwwn mape" \
                     --epochs 100 \
                     --wandb
```

### Workflow 4: Overlap Difficulty Analysis
```bash
# Test how overlap affects performance (few hours)
./run_experiments.sh --models medium \
                     --overlaps "0 1 2" \
                     --losses mse \
                     --epochs 100 \
                     --wandb
```

### Workflow 5: Full Research Suite
```bash
# Run everything (1-2 days)
./run_experiments.sh --wandb --epochs 100

# This runs:
# - 5 model sizes × 3 overlaps × 4 losses = 60 experiments
# - Each experiment: 100 epochs
# - Total: ~6000 epochs of training
```

## Monitoring Progress

### Check Running Experiment
```bash
# View latest checkpoint
ls -lth experiments/*/checkpoints/checkpoint_epoch_*.pth | head -5

# View latest attention heatmaps
ls -lth experiments/*/attention_heatmaps/*.png | head -10

# Check training history
cat experiments/medium_overlap2_mse/training_history.json | jq '.train_loss[-5:]'
```

### View WandB Dashboard
```bash
# If WandB is enabled, view at:
# https://wandb.ai/your-username/fcl-experiments
```

### Check Summary
```bash
# After completion
cat experiments/experiment_summary.csv | column -t -s,
```

## Troubleshooting

### Out of Memory
```bash
# Use smaller model or reduce batch size
./run_experiments.sh --models "tiny small" --epochs 100
```

### Slow Training
```bash
# Reduce epochs or use smaller models
./run_experiments.sh --models "tiny small medium" --epochs 50
```

### WandB Login
```bash
# First time setup
conda activate FCL
wandb login
# Then run experiments with --wandb flag
```

## Advanced Usage

### Custom Python Script
```python
import sys
sys.path.append('/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox')

from run_full_experiments import run_single_experiment
from config import MODEL_CONFIGS

# Custom configuration
result = run_single_experiment(
    model_size='huge',
    overlap=0,
    loss_name='wasserstein',
    epochs=200,
    batch_size=16,  # Reduce for huge model
    lr=5e-5,        # Lower learning rate
    use_wandb=True,
    save_attention_every=5  # More frequent attention saves
)
```

### Resume Training
```python
# Load checkpoint and continue
checkpoint = torch.load('experiments/medium_overlap2_mse/checkpoints/checkpoint_epoch_0050.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
# Continue training from start_epoch
```

## Performance Estimates (RTX 5060 Ti 16GB)

| Model | Batch Size | Time/Epoch | Memory | 100 Epochs |
|-------|-----------|------------|--------|------------|
| tiny | 64 | ~30s | ~2GB | ~50min |
| small | 32 | ~1min | ~4GB | ~1.5hr |
| medium | 32 | ~2min | ~6GB | ~3hr |
| large | 16 | ~5min | ~10GB | ~8hr |
| huge | 8 | ~15min | ~14GB | ~25hr |

## Tips

1. **Start small**: Test with `--single --models tiny --epochs 10` first
2. **Use WandB**: Essential for tracking 60+ experiments
3. **Monitor memory**: Use `nvidia-smi` to check GPU usage
4. **Save attention**: Helps debug attention collapse
5. **Check metrics**: Review `test_metrics_full.csv` for performance
6. **Compare models**: Use WandB dashboard to compare runs
7. **Backup checkpoints**: Copy `best_model.pth` files regularly

## Next Steps After Experiments

1. **Analyze Results**:
   ```bash
   jupyter notebook 03_checkpoint_eval_and_metrics.ipynb
   ```

2. **Visualize Correlations**:
   ```bash
   jupyter notebook 09_weight_correlation_heatmaps.ipynb
   ```

3. **Topology Analysis**:
   ```bash
   jupyter notebook 05_topological_analysis_multipers.ipynb
   ```

4. **Compare Initializations**:
   ```bash
   jupyter notebook 11_multi_initialization_analysis.ipynb
   ```
