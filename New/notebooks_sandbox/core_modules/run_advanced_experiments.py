#!/usr/bin/env python3
"""
ADVANCED COMPREHENSIVE EXPERIMENT SYSTEM

Complete implementation with:
- Hierarchical loss system (individual→layerwise→regularized→mixed)
- All 12 attention heads visualization
- Layerwise metrics tracking
- Advanced topology (Mapper, Gromov-Wasserstein, persistence images/landscapes)
- WandB enabled by default
- Predicted weights saving
- 300 epochs default for scale experiments
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# GPU-accelerated pandas DISABLED - causes crashes during data loading
# The cudf.pandas integration hangs/crashes when accessing GPU memory
# during DataFrame operations. Using standard pandas instead.
print("ℹ Using standard pandas (cudf.pandas disabled due to crashes)")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')
import gc

# Add paths - core_modules is one level down from notebooks_sandbox
CURRENT_DIR = Path(__file__).parent  # core_modules/
NOTEBOOKS_SANDBOX = CURRENT_DIR.parent  # notebooks_sandbox/
PROJECT_ROOT = NOTEBOOKS_SANDBOX.parent  # New/

# Add to path
sys.path.insert(0, str(PROJECT_ROOT))  # For Double_input_transformer
sys.path.insert(0, str(NOTEBOOKS_SANDBOX))  # For other modules
sys.path.insert(0, str(CURRENT_DIR))  # For core_modules

# Set seeds for reproducibility
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Import from project root
from Double_input_transformer import TransformerAE

# Import from core_modules
from config import MODEL_CONFIGS
from advanced_losses import HierarchicalLossRegistry, get_experiment_sequence
from advanced_topology import compute_comprehensive_topology, save_topology_results
from utils_consolidated import (
    WeightDistanceMetrics,
    load_merged_zoo,
    extract_weights_from_zoo,
    create_weight_pairs
)

# Adaptive configurations by model size
MODEL_EPOCHS = {
    'tiny': 150,
    'small': 400,
    'medium': 350,
    'large': 200,
    'huge': 200
}

MODEL_CHECKPOINT_FREQ = {
    'tiny': None,  # Best + Last only
    'small': None,  # Best + Last only
    'medium': 50,
    'large': 25,
    'huge': 25
}

MODEL_LOG_PER_STEP = {
    'tiny': False,
    'small': False,
    'medium': True,
    'large': True,
    'huge': True
}

# CNN validation imports from core_modules
from weight_normalization import LayerWiseNormalizer
from cnn_reconstruction import finetune_reconstructed_cnn
from multi_objective_ranking import rank_losses_multi_objective, LossPerformance

# WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: WandB not available")


# ============================================================================
# ATTENTION VISUALIZATION - ALL HEADS
# ============================================================================

def plot_all_attention_heads(attention_scores, save_path, title_prefix="", max_heads=None):
    """Plot ALL attention heads (e.g., all 12 for huge model)"""
    if not attention_scores or len(attention_scores) == 0:
        return None
    
    attn = attention_scores[0]
    if attn is None or len(attn.shape) != 4:
        return None
    
    # Average over batch dimension
    attn_avg = attn.mean(dim=0).cpu().detach().numpy()
    n_heads = attn_avg.shape[0] if max_heads is None else min(attn_avg.shape[0], max_heads)
    
    # Create grid layout
    ncols = 4
    nrows = (n_heads + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_heads):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Use golden color palette as requested
        golden_cmap = sns.dark_palette("xkcd:golden", 8, as_cmap=True)
        sns.heatmap(attn_avg[idx], ax=ax, cmap=golden_cmap, cbar=True, square=True, 
                   vmin=0, vmax=attn_avg[idx].max())
        ax.set_title(f'{title_prefix} Head {idx+1}/{n_heads}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    # Hide unused subplots
    for idx in range(n_heads, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(save_path)


def log_attention_to_wandb(attention_scores, step, prefix=""):
    """Log attention to WandB"""
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    try:
        temp_path = f"/tmp/attention_{prefix}_{step}.png"
        fig_path = plot_all_attention_heads(attention_scores, temp_path, title_prefix=prefix)
        
        if fig_path and Path(fig_path).exists():
            wandb.log({f"attention/{prefix}": wandb.Image(fig_path), "step": step})
            Path(fig_path).unlink()
    except Exception as e:
        print(f"Warning: Failed to log attention: {e}")


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for x1, x2, target in pbar:
        x1, x2, target = x1.to(device), x2.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)


def train_epoch_with_logging(model, train_loader, optimizer, loss_fn, device, epoch, total_epochs, use_wandb=False):
    """Train for one epoch with per-step logging (for medium+ models)"""
    model.train()
    total_loss = 0.0
    global_step = (epoch - 1) * len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for batch_idx, (x1, x2, target) in enumerate(pbar):
        x1, x2, target = x1.to(device), x2.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        step_loss = loss.item()
        total_loss += step_loss
        pbar.set_postfix({'loss': f'{step_loss:.6f}'})
        
        # Log to WandB every step
        if use_wandb:
            wandb.log({
                'train_loss_step': step_loss,
                'global_step': global_step + batch_idx
            })
    
    return total_loss / len(train_loader)


def validate(model, val_loader, loss_fn, device, save_attention=False, 
             attention_save_dir=None, epoch=0):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    attention_scores = {'enc1': [], 'enc2': [], 'dec': []}
    all_predictions = []
    all_targets = []
    all_necks = []
    
    with torch.no_grad():
        for batch_idx, (x1, x2, target) in enumerate(val_loader):
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            
            all_predictions.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            all_necks.append(neck_t.detach().cpu().numpy())
            
            if save_attention and batch_idx == 0:
                if scEnc1:
                    attention_scores['enc1'] = scEnc1
                if scEnc2:
                    attention_scores['enc2'] = scEnc2
                if scDec:
                    attention_scores['dec'] = scDec
    
    val_loss = total_loss / len(val_loader)
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    necks = np.concatenate(all_necks, axis=0)
    
    # Save ALL attention heads
    if save_attention and attention_save_dir:
        attention_save_dir = Path(attention_save_dir)
        attention_save_dir.mkdir(parents=True, exist_ok=True)
        
        if attention_scores['enc1']:
            path = attention_save_dir / f"epoch_{epoch:04d}_enc1.png"
            plot_all_attention_heads(attention_scores['enc1'], path, title_prefix="Encoder 1")
            log_attention_to_wandb(attention_scores['enc1'], epoch, prefix="encoder1")
        
        if attention_scores['enc2']:
            path = attention_save_dir / f"epoch_{epoch:04d}_enc2.png"
            plot_all_attention_heads(attention_scores['enc2'], path, title_prefix="Encoder 2")
            log_attention_to_wandb(attention_scores['enc2'], epoch, prefix="encoder2")
        
        if attention_scores['dec']:
            path = attention_save_dir / f"epoch_{epoch:04d}_dec.png"
            plot_all_attention_heads(attention_scores['dec'], path, title_prefix="Decoder")
            log_attention_to_wandb(attention_scores['dec'], epoch, prefix="decoder")
    
    return val_loss, predictions, targets, necks, attention_scores


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_advanced_experiment(
    model_size='medium',
    overlap=2,
    loss_name='MSE',
    epochs=None,  # Auto-determined by model size if None
    batch_size=24,
    lr=1e-4,
    output_dir=None,
    use_wandb=True,
    wandb_project="fcl-advanced",
    save_attention_every=25,
    compute_topology_every=50,
    cnn_validation_freq=25,
    cnn_validation_samples=100,
    cnn_batch_size=24,
    cnn_finetune_epochs=5,
    topology_n_jobs=1  # CPU workers for topology (1=sequential, -1=all cores)
):
    """Run advanced experiment with all features"""
    # Auto-determine epochs based on model size
    if epochs is None:
        epochs = MODEL_EPOCHS.get(model_size, 200)
    
    # Get model-specific configurations
    checkpoint_freq = MODEL_CHECKPOINT_FREQ.get(model_size, 25)
    log_per_step = MODEL_LOG_PER_STEP.get(model_size, False)
    
    # Setup device and clear GPU memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Using device: {device} (GPU memory cleared)")
    else:
        print(f"Using device: {device}")
    
    print(f"\nModel-specific config:")
    print(f"  Epochs: {epochs}")
    print(f"  Checkpoint freq: {checkpoint_freq if checkpoint_freq else 'Best+Last only'}")
    print(f"  Per-step logging: {log_per_step}")
    
    # Verify GPU availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: No GPU available, using CPU (training will be very slow!)")
    
    loss_registry = HierarchicalLossRegistry()
    
    exp_name = f"{model_size}_overlap{overlap}_{loss_name.replace('+', '_').replace('*', 'x')}"
    
    if output_dir is None:
        output_dir = Path("./experiments") / exp_name
    else:
        output_dir = Path(output_dir) / exp_name
    
    # Create output directories
    checkpoints_dir = output_dir / "checkpoints"
    attention_dir = output_dir / "attention_heatmaps"
    metrics_dir = output_dir / "metrics"
    weights_dir = output_dir / "predicted_weights"
    topology_dir = output_dir / "topology"
    
    for d in [checkpoints_dir, attention_dir, metrics_dir, weights_dir, topology_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ADVANCED EXPERIMENT: {exp_name}")
    print(f"{'='*80}\n")
    
    # Initialize WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=exp_name,
            config={
                'model_size': model_size,
                'overlap': overlap,
                'loss_name': loss_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                **MODEL_CONFIGS[model_size].__dict__
            }
        )
    
    # Load scenario-based data
    print("Loading scenario-based data...")
    scenario_dir = PROJECT_ROOT / "data" / "Scenario" / f"overlapping_m{overlap}"
    csv_path = PROJECT_ROOT / "data" / "Merged zoo.csv"
    
    # Load scenario splits
    train_pairs_file = scenario_dir / "train_pairs.npy"
    val_pairs_file = scenario_dir / "val_pairs.npy"
    test_pairs_file = scenario_dir / "test_pairs.npy"
    
    if not train_pairs_file.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_dir}. Run generate_scenarios.py first.")
    
    train_pairs = np.load(train_pairs_file, allow_pickle=True)
    val_pairs = np.load(val_pairs_file, allow_pickle=True)
    test_pairs = np.load(test_pairs_file, allow_pickle=True)
    
    print(f"  Scenario: {scenario_dir}")
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs: {len(val_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")
    
    # Load merged zoo CSV with correct column structure
    print("\nLoading merged zoo CSV...")
    # Force standard pandas (disable cudf proxy even if loaded in environment)
    os.environ['CUDF_PANDAS_MODE'] = 'disabled'
    _raw_df = pd.read_csv(csv_path)
    # Ensure we have a true pandas DataFrame (strip any cudf proxy wrapper)
    if not isinstance(_raw_df, pd.DataFrame) or type(_raw_df).__module__.startswith('cudf'):
        import io
        _raw_df = pd.DataFrame(_raw_df.to_pandas() if hasattr(_raw_df, 'to_pandas') else _raw_df)
    df = _raw_df

    # Weight columns start from index 17 onwards (0-16 are metadata)
    weight_cols = list(df.columns[17:-2])  # Exclude last 2 columns (Accuracy, epoch)

    print(f"  Total rows in zoo: {len(df)}")
    print(f"  Weight columns: {len(weight_cols)}")

    # -----------------------------------------------------------------------
    # Pre-build a fast lookup index so pair loading is O(1) per pair instead
    # of O(N) DataFrame row searches (which took 3-6 s each with cudf proxy).
    # -----------------------------------------------------------------------
    print("  Building weight lookup index (one-time cost)...")
    weight_matrix = df[weight_cols].values.astype(np.float32)   # shape (N, D)
    labels_arr    = df['label'].values
    epochs_arr    = df['epoch'].values

    # Build per-activation lookup: label_epoch_idx[activation][label][epoch] = row_index
    _label_epoch_idx = {}
    for act_col in ['leakyrelu', 'relu', 'tanh', 'sigmoid']:
        if act_col not in df.columns:
            continue
        act_mask = (df[act_col].values == 1.0)
        idx_map = {}
        for row_idx in np.where(act_mask)[0]:
            lbl = labels_arr[row_idx]
            ep  = int(epochs_arr[row_idx])
            if lbl not in idx_map:
                idx_map[lbl] = {}
            if ep not in idx_map[lbl]:       # keep first match per (label, epoch)
                idx_map[lbl][ep] = row_idx
        _label_epoch_idx[act_col] = idx_map
    print(f"  Index built: {sum(len(v) for v in _label_epoch_idx.values())} activation×label entries")

    def _lookup_weights(label_str: str, activation: str, epoch: int) -> "np.ndarray | None":
        """O(1) weight lookup using the pre-built index."""
        idx_map = _label_epoch_idx.get(activation, {}).get(label_str, {})
        for target_epoch in range(epoch, 10, -5):
            if target_epoch in idx_map:
                return weight_matrix[idx_map[target_epoch]]
        return None

    # Helper function to load weights for task pairs
    def load_weights_for_pairs(pairs, df, weight_cols, activation='leakyrelu', epoch=21):
        x1_list, x2_list, y_list, metadata_list = [], [], [], []

        print(f"  Loading {len(pairs)} pairs (indexed, fast)...")
        missing_count = 0

        for pair in tqdm(pairs, desc="  Processing pairs"):
            task1, task2 = pair
            task_combined = sorted(set(task1) | set(task2))

            task1_str        = str(task1)
            task2_str        = str(task2)
            task_combined_str = str(task_combined)

            w1        = _lookup_weights(task1_str, activation, epoch)
            w2        = _lookup_weights(task2_str, activation, epoch)
            w_combined = _lookup_weights(task_combined_str, activation, epoch)

            if w1 is not None and w2 is not None and w_combined is not None:
                x1_list.append(w1)
                x2_list.append(w2)
                y_list.append(w_combined)
                metadata_list.append({
                    'task1': task1,
                    'task2': task2,
                    'task_combined': task_combined,
                    'activation': activation,
                    'epoch': epoch
                })
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"  Warning: {missing_count} pairs not found in zoo")

        return np.array(x1_list), np.array(x2_list), np.array(y_list), metadata_list
    
    # Load scenario weights (with activation/epoch consistency)
    print("\nLoading training weights...")
    x1_train, x2_train, y_train, metadata_train = load_weights_for_pairs(
        train_pairs, df, weight_cols, activation='leakyrelu', epoch=21
    )
    
    print("\nLoading validation weights...")
    x1_val, x2_val, y_val, metadata_val = load_weights_for_pairs(
        val_pairs, df, weight_cols, activation='leakyrelu', epoch=21
    )
    
    print("\nLoading test weights...")
    x1_test, x2_test, y_test, metadata_test = load_weights_for_pairs(
        test_pairs, df, weight_cols, activation='leakyrelu', epoch=21
    )
    
    print(f"  Train samples: {len(x1_train)}")
    print(f"  Val samples: {len(x1_val)}")
    print(f"  Test samples: {len(x1_test)}")
    
    # Check if data loading succeeded
    if len(x1_train) == 0 or len(x1_val) == 0 or len(x1_test) == 0:
        raise ValueError(f"Data loading failed! Train: {len(x1_train)}, Val: {len(x1_val)}, Test: {len(x1_test)}")
    
    print(f"  Weight vector size: {x1_train.shape[1]}")
    
    # Layer-wise normalization
    print("\nApplying layer-wise weight normalization...")
    normalizer = LayerWiseNormalizer(method='standard')
    
    # Fit on training data
    normalizer.fit(y_train)
    
    # Transform all splits
    x1_train_norm = normalizer.transform(x1_train)
    x2_train_norm = normalizer.transform(x2_train)
    y_train_norm = normalizer.transform(y_train)
    
    x1_val_norm = normalizer.transform(x1_val)
    x2_val_norm = normalizer.transform(x2_val)
    y_val_norm = normalizer.transform(y_val)
    
    x1_test_norm = normalizer.transform(x1_test)
    x2_test_norm = normalizer.transform(x2_test)
    y_test_norm = normalizer.transform(y_test)
    
    # Save normalizer
    normalizer_path = output_dir / "weight_normalizer.pkl"
    normalizer.save(str(normalizer_path))
    print(f"  Normalizer saved: {normalizer_path}")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(x1_train_norm).float(),
        torch.from_numpy(x2_train_norm).float(),
        torch.from_numpy(y_train_norm).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x1_val_norm).float(),
        torch.from_numpy(x2_val_norm).float(),
        torch.from_numpy(y_val_norm).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x1_test_norm).float(),
        torch.from_numpy(x2_test_norm).float(),
        torch.from_numpy(y_test_norm).float()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    print(f"\nCreating {model_size} model...")
    config = MODEL_CONFIGS[model_size]
    model = TransformerAE(
        max_seq_len=config.max_seq_len,
        N=config.N,
        heads=config.heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        neck=config.neck,
        dropout=config.dropout
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  Attention heads: {config.heads}")
    
    # Setup training
    loss_fn = loss_registry.get_loss(loss_name)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Gentle cosine annealing scheduler with short warmup
    # Warmup: 5% of epochs, min 10 epochs
    warmup_epochs = max(10, int(epochs * 0.05))
    min_lr = lr * 0.1  # Minimum LR = 1e-5 (never goes below this)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup to max LR
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing from max to min (gentle decay)
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            # Cosine from 1.0 to 0.1 (min_lr = 0.1 * base_lr)
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\nLearning rate schedule:")
    print(f"  Warmup: {warmup_epochs} epochs (0 → {lr:.2e})")
    print(f"  Cosine decay: {epochs - warmup_epochs} epochs ({lr:.2e} → {min_lr:.2e})")
    print(f"  Min LR: {min_lr:.2e} (never goes below this)")
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Training for {epochs} epochs")
    print(f"Loss function: {loss_name}")
    print(f"{'='*80}\n")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'topology_results': []
    }
    best_val_loss = float('inf')
    
    # Track timing for periodic logging
    import time
    last_log_time = time.time()
    training_start_time = time.time()
    
    # EPOCH 0: Baseline validation before training
    print(f"\n{'='*80}")
    print(f"EPOCH 0: Baseline Validation (before training)")
    print(f"{'='*80}")
    
    model.eval()
    val_loss_epoch0, predictions_epoch0, targets_epoch0, necks_epoch0, _ = validate(
        model, val_loader, loss_fn, device,
        save_attention=False,
        attention_save_dir=None,
        epoch=0
    )
    
    print(f"Epoch 0 (baseline): Val Loss = {val_loss_epoch0:.6f}")
    
    # Run CNN validation at epoch 0
    print(f"  Running baseline CNN validation...")
    cnn_val_dir_epoch0 = output_dir / "cnn_validation" / "epoch_0000"
    cnn_val_dir_epoch0.mkdir(parents=True, exist_ok=True)
    
    # Select test subset
    test_indices_file = output_dir / "cnn_validation_test_indices.npy"
    n_test_samples = min(cnn_validation_samples, len(x1_test))
    test_indices = np.random.choice(len(x1_test), n_test_samples, replace=False)
    np.save(test_indices_file, test_indices)
    
    test_subset_x1 = torch.from_numpy(x1_test_norm[test_indices]).float().to(device)
    test_subset_x2 = torch.from_numpy(x2_test_norm[test_indices]).float().to(device)
    
    with torch.no_grad():
        predictions_norm_epoch0, _, _, _, _ = model(test_subset_x1, test_subset_x2)
        predictions_norm_epoch0 = predictions_norm_epoch0.cpu().numpy()
    
    predictions_epoch0 = normalizer.inverse_transform(predictions_norm_epoch0)
    
    # Quick CNN validation on 3 samples at epoch 0
    cnn_results_epoch0 = []
    for i in range(min(3, len(predictions_epoch0))):
        task_classes = [metadata_test[test_indices[i]]['task_combined']]
        try:
            from cnn_reconstruction import finetune_reconstructed_cnn
            result = finetune_reconstructed_cnn(
                predicted_weights=predictions_epoch0[i],
                task_classes=task_classes[0],
                activation='leakyrelu',
                mnist_root=str(PROJECT_ROOT / "data" / "SplitMnist"),
                n_finetune_epochs=cnn_finetune_epochs,
                batch_size=cnn_batch_size,
                input_weights_x1=x1_test[test_indices[i]],
                input_weights_x2=x2_test[test_indices[i]],
                ground_truth_weights=y_test[test_indices[i]]
            )
            cnn_results_epoch0.append(result)
        except Exception as e:
            print(f"    Warning: Epoch 0 CNN validation failed for sample {i}: {e}")
    
    if cnn_results_epoch0:
        avg_acc_epoch0 = np.mean([r['acc_id_final'] for r in cnn_results_epoch0])
        print(f"  Epoch 0 CNN Accuracy: {avg_acc_epoch0:.3f} (baseline)")
    
    # Run topology at epoch 0
    print(f"  Computing baseline topology...")
    topology_results_epoch0 = compute_comprehensive_topology(necks_epoch0, epoch=0, n_jobs=topology_n_jobs)
    if topology_results_epoch0:
        save_topology_results(topology_results_epoch0, topology_dir, 0)
        print(f"  Baseline topology computed")
    
    # Log epoch 0 to WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'epoch': 0,
            'val_loss': val_loss_epoch0,
            'cnn/baseline_acc': avg_acc_epoch0 if cnn_results_epoch0 else 0.0
        })
    
    print(f"\n{'='*80}")
    print(f"Starting Training from Epoch 1")
    print(f"{'='*80}\n")
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        # Train (with optional per-step logging for medium+ models)
        if log_per_step:
            train_loss = train_epoch_with_logging(model, train_loader, optimizer, loss_fn, device, epoch, epochs, use_wandb and WANDB_AVAILABLE)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, epochs)
        
        # Validate
        save_attn = (epoch % save_attention_every == 0) or (epoch == epochs)
        val_loss, predictions, targets, necks, attention_scores = validate(
            model, val_loader, loss_fn, device,
            save_attention=save_attn,
            attention_save_dir=attention_dir,
            epoch=epoch
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save predicted weights only during CNN validation (not every epoch)
        # This will be saved inside CNN validation block below
        
        # CNN Validation
        cnn_metrics = {}
        if epoch % cnn_validation_freq == 0 or epoch == epochs:
            print(f"  Running CNN validation...")
            
            # Create CNN validation directory
            cnn_val_dir = output_dir / "cnn_validation" / f"epoch_{epoch:04d}"
            cnn_val_dir.mkdir(parents=True, exist_ok=True)
            
            # Select fixed test subset (use same indices each time)
            test_indices_file = output_dir / "cnn_validation_test_indices.npy"
            if test_indices_file.exists():
                test_indices = np.load(test_indices_file)
            else:
                # First time: randomly select indices
                n_test_samples = min(cnn_validation_samples, len(x1_test))
                test_indices = np.random.choice(len(x1_test), n_test_samples, replace=False)
                np.save(test_indices_file, test_indices)
                print(f"    Selected {n_test_samples} fixed test samples for CNN validation")
            
            # Get predictions for test subset (denormalized)
            test_subset_x1 = torch.from_numpy(x1_test_norm[test_indices]).float().to(device)
            test_subset_x2 = torch.from_numpy(x2_test_norm[test_indices]).float().to(device)
            
            model.eval()
            with torch.no_grad():
                predictions_norm, _, _, _, _ = model(test_subset_x1, test_subset_x2)
                predictions_norm = predictions_norm.cpu().numpy()
            
            # Denormalize predictions and get original weights
            predictions = normalizer.inverse_transform(predictions_norm)
            x1_original = x1_test[test_indices]
            x2_original = x2_test[test_indices]
            y_original = y_test[test_indices]
            
            # Get task classes from metadata
            test_metadata_subset = [metadata_test[i] for i in test_indices]
            
            # Run CNN validation on subset
            cnn_results = []
            for i in range(min(10, len(predictions))):  # Validate on first 10 samples
                task_classes = test_metadata_subset[i]['task_combined']
                
                try:
                    result = finetune_reconstructed_cnn(
                        predicted_weights=predictions[i],
                        task_classes=task_classes,
                        activation='leakyrelu',
                        mnist_root=str(PROJECT_ROOT / "data" / "SplitMnist"),
                        n_finetune_epochs=cnn_finetune_epochs,
                        batch_size=cnn_batch_size,
                        input_weights_x1=x1_original[i],
                        input_weights_x2=x2_original[i],
                        ground_truth_weights=y_original[i]
                    )
                    
                    cnn_results.append({
                        'sample_idx': i,
                        'task_classes': task_classes,
                        'acc_id_initial': result['acc_id_initial'],
                        'acc_id_final': result['acc_id_final'],
                        'acc_ood_initial': result['acc_ood_initial'],
                        **result['finetune_history']
                    })
                    
                    # Save eigenvalues
                    eigenvalues_file = cnn_val_dir / f"sample_{i:03d}_eigenvalues.json"
                    with open(eigenvalues_file, 'w') as f:
                        # Convert numpy arrays to lists for JSON serialization
                        eigenvalues_serializable = {}
                        for key, layer_dict in result['eigenvalues_analysis'].items():
                            eigenvalues_serializable[key] = {
                                layer: eigs.tolist() if isinstance(eigs, np.ndarray) else eigs
                                for layer, eigs in layer_dict.items()
                            }
                        json.dump(eigenvalues_serializable, f, indent=2)
                    
                except Exception as e:
                    print(f"    Warning: CNN validation failed for sample {i}: {e}")
                    continue
            
            # Aggregate CNN metrics
            if cnn_results:
                avg_initial_acc = np.mean([r['acc_id_initial'] for r in cnn_results])
                avg_final_acc = np.mean([r['acc_id_final'] for r in cnn_results])
                avg_improvement = avg_final_acc - avg_initial_acc
                
                cnn_metrics = {
                    'cnn/avg_initial_acc_id': avg_initial_acc,
                    'cnn/avg_final_acc_id': avg_final_acc,
                    'cnn/avg_improvement': avg_improvement
                }
                
                # Save CNN results
                cnn_df = pd.DataFrame(cnn_results)
                cnn_df.to_csv(cnn_val_dir / "cnn_validation_results.csv", index=False)
                
                print(f"    CNN Validation: Initial={avg_initial_acc:.3f}, Final={avg_final_acc:.3f}, Δ={avg_improvement:.3f}")
        
        # Topology analysis
        topology_metrics = {}
        if epoch % compute_topology_every == 0:
            print(f"  Computing comprehensive topology (n_jobs={topology_n_jobs})...")
            topology_results = compute_comprehensive_topology(necks, epoch=epoch, n_jobs=topology_n_jobs)
            if topology_results:
                history['topology_results'].append(topology_results)
                save_topology_results(topology_results, topology_dir, epoch)
                
                # Extract ALL topology metrics for WandB
                for key in ['betti_0', 'betti_1', 'betti_2', 'mapper_n_nodes', 
                           'mapper_graph_density', 'mapper_mean_node_size', 'mapper_max_node_size',
                           'gw_distance']:
                    if key in topology_results:
                        topology_metrics[f'topology/{key}'] = topology_results[key]
                
                # Add landscape metrics
                for key, val in topology_results.items():
                    if key.startswith('landscape_'):
                        topology_metrics[f'topology/{key}'] = val
        
        # Log to WandB
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **topology_metrics,
                **cnn_metrics
            })
        
        # Terminal logging with periodic updates
        current_time = time.time()
        time_since_last_log = current_time - last_log_time
        epoch_duration = current_time - epoch_start_time
        total_elapsed = current_time - training_start_time
        
        # Always log validation epochs and every ~30 minutes
        is_validation_epoch = (epoch % cnn_validation_freq == 0) or (epoch % compute_topology_every == 0)
        is_periodic_log = time_since_last_log >= 1800  # 30 minutes
        
        if is_validation_epoch or is_periodic_log or epoch == 1 or epoch == epochs:
            elapsed_hours = total_elapsed / 3600
            remaining_epochs = epochs - epoch
            avg_epoch_time = total_elapsed / epoch
            est_remaining_hours = (remaining_epochs * avg_epoch_time) / 3600
            
            print(f"\n{'='*80}")
            print(f"[{exp_name}] Epoch {epoch:3d}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Elapsed: {elapsed_hours:.1f}h | Est. Remaining: {est_remaining_hours:.1f}h")
            print(f"  Epoch Duration: {epoch_duration:.1f}s")
            if cnn_metrics:
                print(f"  CNN Acc: {cnn_metrics.get('cnn/avg_final_acc_id', 0):.3f}")
            print(f"{'='*80}")
            last_log_time = current_time
        else:
            # Compact logging for non-validation epochs
            print(f"Epoch {epoch:3d}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Adaptive checkpointing based on model size
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        
        # Always save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoints_dir / "best_model.pth")
            print(f"  → Best model saved (val_loss: {val_loss:.6f})")
        
        # Periodic checkpoints based on model size
        if checkpoint_freq is not None and epoch % checkpoint_freq == 0:
            torch.save(checkpoint, checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pth")
            print(f"  → Checkpoint saved (epoch {epoch})")
    
    # Final checkpoint
    torch.save(checkpoint, checkpoints_dir / "final_model.pth")
    
    # Save history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss']
        }, f, indent=2)
    
    # Compute final metrics on test set
    print(f"\n{'='*80}")
    print("Computing final metrics on test set...")
    print(f"{'='*80}\n")
    
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for x1, x2, target in test_loader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            output, _, _, _, _ = model(x1, x2)
            test_predictions.append(output.cpu().numpy())
            test_targets.append(target.cpu().numpy())
    
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    # Compute BOTH full and layerwise metrics
    calc = WeightDistanceMetrics()
    all_metrics = []
    
    for i in range(min(100, len(test_predictions))):
        # Full metrics
        full_metrics = calc.compute_all_full_distances(test_predictions[i], test_targets[i])
        
        # Layerwise metrics
        layerwise_metrics = calc.compute_all_layerwise_distances(test_predictions[i], test_targets[i])
        
        # Combine
        combined_metrics = {
            'sample_idx': i,
            'model_size': model_size,
            'overlap': overlap,
            'loss_name': loss_name,
            **full_metrics,
            **layerwise_metrics
        }
        all_metrics.append(combined_metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_dir / "test_metrics_full_and_layerwise.csv", index=False)
    
    print(f"  Saved metrics to {metrics_dir / 'test_metrics_full_and_layerwise.csv'}")
    
    # Cleanup
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # Aggressive memory cleanup
    del model, optimizer, scheduler, loss_fn
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    del x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test
    del x1_train_norm, x2_train_norm, y_train_norm
    del x1_val_norm, x2_val_norm, y_val_norm
    del x1_test_norm, x2_test_norm, y_test_norm
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE: {exp_name}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Results: {output_dir}")
    print(f"  GPU memory cleared")
    print(f"{'='*80}\n")
    
    return {
        'exp_name': exp_name,
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'num_params': num_params,
        'output_dir': str(output_dir)
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run advanced FCL experiments with CNN validation")
    
    # Model configuration
    parser.add_argument("--models", nargs='+', default=None,
                       help="Model sizes to test (for batch experiments)")
    parser.add_argument("--model-size", type=str, default='medium',
                       help="Single model size (for single experiments)")
    
    # Task configuration
    parser.add_argument("--overlaps", nargs='+', type=int, default=None,
                       help="Overlap levels to test (for batch experiments)")
    parser.add_argument("--overlap", type=int, default=2,
                       help="Single overlap level (for single experiments)")
    
    # Loss configuration
    parser.add_argument("--losses", nargs='+', default=None,
                       help="Loss names to test (default: experiment sequence)")
    parser.add_argument("--loss", type=str, default='MSE',
                       help="Single loss function (for single experiments)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=200,
                       help="Epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=24,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    
    # CNN validation parameters (NEW)
    parser.add_argument("--cnn-validation-freq", type=int, default=25,
                       help="CNN validation frequency (every N epochs)")
    parser.add_argument("--cnn-validation-samples", type=int, default=100,
                       help="Number of test samples for CNN validation")
    parser.add_argument("--cnn-batch-size", type=int, default=24,
                       help="Batch size for CNN finetuning")
    parser.add_argument("--cnn-finetune-epochs", type=int, default=5,
                       help="Number of CNN finetuning epochs")
    
    # Performance parameters
    parser.add_argument("--topology-n-jobs", type=int, default=1,
                       help="CPU workers for topology analysis (1=sequential, -1=all cores)")
    
    # Logging and output
    parser.add_argument("--wandb", action="store_true", default=True,
                       help="Enable WandB logging (default: True)")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb",
                       help="Disable WandB logging")
    parser.add_argument("--output-dir", type=str, 
                       default="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/Experiments",
                       help="Base output directory")
    
    # Experiment mode
    parser.add_argument("--single", action="store_true",
                       help="Run single experiment")
    
    args = parser.parse_args()
    
    # Handle backward compatibility: convert singular to plural
    if args.models is None:
        args.models = [args.model_size]
    if args.overlaps is None:
        args.overlaps = [args.overlap]
    if args.losses is None and args.loss:
        args.losses = [args.loss]
    
    # Get loss sequence if not specified
    if args.losses is None:
        args.losses = get_experiment_sequence()
        print(f"Using experiment sequence: {len(args.losses)} losses")
    
    if args.single:
        # Run single experiment
        result = run_advanced_experiment(
            model_size=args.models[0],
            overlap=args.overlaps[0],
            loss_name=args.losses[0],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            cnn_validation_freq=args.cnn_validation_freq,
            cnn_validation_samples=args.cnn_validation_samples,
            cnn_batch_size=args.cnn_batch_size,
            cnn_finetune_epochs=args.cnn_finetune_epochs,
            topology_n_jobs=args.topology_n_jobs
        )
        print(f"\nResult: {result}")
    else:
        # Run multiple experiments
        results = []
        total_experiments = len(args.models) * len(args.overlaps) * len(args.losses)
        print(f"\n{'='*80}")
        print(f"RUNNING {total_experiments} EXPERIMENTS")
        print(f"Models: {args.models}")
        print(f"Overlaps: {args.overlaps}")
        print(f"Losses: {len(args.losses)} loss functions")
        print(f"{'='*80}\n")
        
        exp_count = 0
        for model_size in args.models:
            for overlap in args.overlaps:
                for loss_name in args.losses:
                    exp_count += 1
                    print(f"\n[{exp_count}/{total_experiments}] Starting: {model_size}, overlap={overlap}, loss={loss_name}")
                    try:
                        result = run_advanced_experiment(
                            model_size=model_size,
                            overlap=overlap,
                            loss_name=loss_name,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            use_wandb=args.wandb,
                            cnn_validation_freq=args.cnn_validation_freq,
                            cnn_validation_samples=args.cnn_validation_samples,
                            cnn_batch_size=args.cnn_batch_size,
                            cnn_finetune_epochs=args.cnn_finetune_epochs,
                            topology_n_jobs=args.topology_n_jobs
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"\nERROR in {model_size}_overlap{overlap}_{loss_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        results.append({
                            'exp_name': f"{model_size}_overlap{overlap}_{loss_name}",
                            'error': str(e)
                        })
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_path = Path(args.output_dir) / "advanced_experiments_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}")
        print(f"ALL EXPERIMENTS COMPLETE")
        print(f"Summary: {summary_path}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
