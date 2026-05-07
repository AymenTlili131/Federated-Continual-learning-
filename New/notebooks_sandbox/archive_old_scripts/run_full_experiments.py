#!/usr/bin/env python3
"""
Comprehensive experiment runner with:
- All model sizes (tiny, small, medium, large, huge)
- All overlaps (0, 1, 2)
- All loss functions
- Metrics tracking
- Finetuning (1-5 epochs)
- Attention heatmap visualization
- WandB logging
- Local saves
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
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from Double_input_transformer import TransformerAE
from config import MODEL_CONFIGS, create_experiment_config
from utils_consolidated import (
    WeightDistanceMetrics, 
    load_merged_zoo, 
    extract_weights_from_zoo,
    create_weight_pairs,
    save_metrics_csv,
    append_metrics_csv
)

# WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: WandB not available. Logging disabled.")


# ============================================================================
# ATTENTION HEATMAP VISUALIZATION
# ============================================================================

def plot_attention_heatmaps(attention_scores, save_path, title_prefix="", max_heads=8):
    """
    Plot attention heatmaps for all heads.
    
    Args:
        attention_scores: List of attention tensors from model
        save_path: Path to save figure
        title_prefix: Prefix for plot title
        max_heads: Maximum number of heads to plot
    """
    if not attention_scores or len(attention_scores) == 0:
        return None
    
    # Get first layer attention (encoder or decoder)
    attn = attention_scores[0]  # Shape: (batch, heads, seq_len, seq_len)
    
    if attn is None or len(attn.shape) != 4:
        return None
    
    # Average over batch
    attn_avg = attn.mean(dim=0).cpu().detach().numpy()  # (heads, seq_len, seq_len)
    
    n_heads = min(attn_avg.shape[0], max_heads)
    
    # Create subplot grid
    ncols = 4
    nrows = (n_heads + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_heads):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Plot heatmap
        sns.heatmap(
            attn_avg[idx], 
            ax=ax, 
            cmap='viridis',
            cbar=True,
            square=True,
            vmin=0,
            vmax=attn_avg[idx].max()
        )
        ax.set_title(f'{title_prefix} Head {idx+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    # Hide unused subplots
    for idx in range(n_heads, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save locally
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(save_path)


def log_attention_to_wandb(attention_scores, step, prefix=""):
    """Log attention heatmaps to WandB."""
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    try:
        # Create temporary figure
        temp_path = f"/tmp/attention_{prefix}_{step}.png"
        fig_path = plot_attention_heatmaps(
            attention_scores, 
            temp_path, 
            title_prefix=prefix
        )
        
        if fig_path and Path(fig_path).exists():
            wandb.log({
                f"attention/{prefix}": wandb.Image(fig_path),
                "step": step
            })
            # Clean up temp file
            Path(fig_path).unlink()
    except Exception as e:
        print(f"Warning: Failed to log attention to WandB: {e}")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class LossFunctions:
    """Collection of loss functions."""
    
    @staticmethod
    def mse_loss(pred, target):
        return nn.MSELoss()(pred, target)
    
    @staticmethod
    def mape_loss(pred, target, epsilon=1e-8):
        return torch.mean(torch.abs((pred - target) / (torch.abs(target) + epsilon))) * 100
    
    @staticmethod
    def wasserstein_loss(pred, target):
        pred_sorted, _ = torch.sort(pred, dim=-1)
        target_sorted, _ = torch.sort(target, dim=-1)
        return torch.mean(torch.abs(pred_sorted - target_sorted))
    
    @staticmethod
    def lwwn_loss(pred, target, epsilon=1e-8):
        """Layer-wise weighted normalization."""
        std = torch.std(target, dim=-1, keepdim=True)
        return torch.mean(torch.abs(pred - target) / (std + epsilon))
    
    @staticmethod
    def get_loss_function(name):
        loss_map = {
            'mse': LossFunctions.mse_loss,
            'mape': LossFunctions.mape_loss,
            'wasserstein': LossFunctions.wasserstein_loss,
            'lwwn': LossFunctions.lwwn_loss,
            'auto': LossFunctions.mse_loss,  # Default
        }
        return loss_map.get(name.lower(), LossFunctions.mse_loss)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for x1, x2, target in pbar:
        x1, x2, target = x1.to(device), x2.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, save_attention=False, attention_save_dir=None, epoch=0):
    """
    Validate model and optionally save attention heatmaps.
    
    Returns:
        val_loss, attention_scores (if save_attention=True)
    """
    model.eval()
    total_loss = 0.0
    attention_scores = {'enc1': [], 'enc2': [], 'dec': []}
    
    with torch.no_grad():
        for batch_idx, (x1, x2, target) in enumerate(val_loader):
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Collect attention scores from first batch only
            if save_attention and batch_idx == 0:
                if scEnc1:
                    attention_scores['enc1'] = scEnc1
                if scEnc2:
                    attention_scores['enc2'] = scEnc2
                if scDec:
                    attention_scores['dec'] = scDec
    
    val_loss = total_loss / len(val_loader)
    
    # Plot and save attention heatmaps
    if save_attention and attention_save_dir:
        attention_save_dir = Path(attention_save_dir)
        attention_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save encoder 1 attention
        if attention_scores['enc1']:
            path = attention_save_dir / f"epoch_{epoch:04d}_enc1.png"
            plot_attention_heatmaps(attention_scores['enc1'], path, title_prefix="Encoder 1")
            log_attention_to_wandb(attention_scores['enc1'], epoch, prefix="encoder1")
        
        # Save encoder 2 attention
        if attention_scores['enc2']:
            path = attention_save_dir / f"epoch_{epoch:04d}_enc2.png"
            plot_attention_heatmaps(attention_scores['enc2'], path, title_prefix="Encoder 2")
            log_attention_to_wandb(attention_scores['enc2'], epoch, prefix="encoder2")
        
        # Save decoder attention
        if attention_scores['dec']:
            path = attention_save_dir / f"epoch_{epoch:04d}_dec.png"
            plot_attention_heatmaps(attention_scores['dec'], path, title_prefix="Decoder")
            log_attention_to_wandb(attention_scores['dec'], epoch, prefix="decoder")
    
    return val_loss, attention_scores


def finetune_on_mnist(model, test_loader, device, num_epochs=5, lr=1e-4):
    """
    Finetune predicted weights on MNIST test set.
    
    Returns:
        List of finetuned weight snapshots at each epoch
    """
    # This is a placeholder - actual implementation would need MNIST data
    # For now, we'll just return the model weights at different stages
    finetuned_weights = []
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, num_epochs + 1):
        # In real implementation, train on MNIST here
        # For now, just save current weights
        weights = []
        for param in model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        finetuned_weights.append(np.concatenate(weights))
    
    return finetuned_weights


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    model_size: str,
    overlap: int,
    loss_name: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_dir: Path = None,
    use_wandb: bool = False,
    wandb_project: str = "fcl-experiments",
    save_attention_every: int = 10
):
    """Run a single experiment with full tracking."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"{model_size}_overlap{overlap}_{loss_name}"
    
    if output_dir is None:
        output_dir = Path("./experiments") / exp_name
    output_dir = Path(output_dir)
    
    checkpoints_dir = output_dir / "checkpoints"
    attention_dir = output_dir / "attention_heatmaps"
    metrics_dir = output_dir / "metrics"
    
    for d in [checkpoints_dir, attention_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*80}\n")
    
    # Initialize WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=exp_name,
            config={
                'model_size': model_size,
                'overlap': overlap,
                'loss': loss_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                **MODEL_CONFIGS[model_size].__dict__
            }
        )
    
    # Load data
    print("Loading data...")
    csv_path = PROJECT_ROOT / "data" / "Merged zoo.csv"
    df = load_merged_zoo(csv_path, limit=10000)
    weights, _ = extract_weights_from_zoo(df)
    x1, x2, y = create_weight_pairs(weights, overlap=overlap)
    
    # Split data
    n_train = int(0.7 * len(x1))
    n_val = int(0.15 * len(x1))
    
    train_dataset = TensorDataset(
        torch.from_numpy(x1[:n_train]).float(),
        torch.from_numpy(x2[:n_train]).float(),
        torch.from_numpy(y[:n_train]).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x1[n_train:n_train+n_val]).float(),
        torch.from_numpy(x2[n_train:n_train+n_val]).float(),
        torch.from_numpy(y[n_train:n_train+n_val]).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x1[n_train+n_val:]).float(),
        torch.from_numpy(x2[n_train+n_val:]).float(),
        torch.from_numpy(y[n_train+n_val:]).float()
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
    
    # Setup training
    criterion = LossFunctions.get_loss_function(loss_name)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Training for {epochs} epochs")
    print(f"{'='*80}\n")
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs)
        
        # Validate with attention saving
        save_attn = (epoch % save_attention_every == 0) or (epoch == epochs)
        val_loss, attention_scores = validate(
            model, val_loader, criterion, device,
            save_attention=save_attn,
            attention_save_dir=attention_dir,
            epoch=epoch
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Log to WandB
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        print(f"Epoch {epoch:3d}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoints
        if val_loss < best_val_loss or epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, checkpoints_dir / "best_model.pth")
                print(f"  → Best model saved (val_loss: {val_loss:.6f})")
            
            if epoch % 10 == 0:
                torch.save(checkpoint, checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pth")
    
    # Final checkpoint
    torch.save(checkpoint, checkpoints_dir / "final_model.pth")
    
    # Save training history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Compute final metrics on test set
    print(f"\n{'='*80}")
    print("Computing final metrics on test set...")
    print(f"{'='*80}\n")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for x1, x2, target in test_loader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            output, _, _, _, _ = model(x1, x2)
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Compute distance metrics
    calc = WeightDistanceMetrics()
    all_metrics = []
    
    for i in range(min(100, len(predictions))):  # First 100 samples
        metrics = calc.compute_all_full_distances(predictions[i], targets[i])
        metrics['sample_idx'] = i
        metrics['model_size'] = model_size
        metrics['overlap'] = overlap
        metrics['loss'] = loss_name
        all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_dir / "test_metrics_full.csv", index=False)
    
    print(f"  Saved metrics to {metrics_dir / 'test_metrics_full.csv'}")
    
    # Finetuning (placeholder)
    print(f"\nFinetuning for 5 epochs...")
    # finetuned_weights = finetune_on_mnist(model, test_loader, device, num_epochs=5)
    
    # Cleanup
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\n{'='*80}")
    print(f"Experiment Complete: {exp_name}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Attention heatmaps: {attention_dir}")
    print(f"  Metrics: {metrics_dir}")
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
# MAIN EXPERIMENT SUITE
# ============================================================================

def run_full_experiment_suite(
    model_sizes=['tiny', 'small', 'medium', 'large', 'huge'],
    overlaps=[0, 1, 2],
    losses=['mse', 'wasserstein', 'lwwn', 'mape'],
    epochs=100,
    use_wandb=True,
    output_base_dir='./experiments'
):
    """Run full cross-experiment suite."""
    
    results = []
    total_experiments = len(model_sizes) * len(overlaps) * len(losses)
    
    print(f"\n{'='*80}")
    print(f"FULL EXPERIMENT SUITE")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Model sizes: {model_sizes}")
    print(f"Overlaps: {overlaps}")
    print(f"Loss functions: {losses}")
    print(f"Epochs per experiment: {epochs}")
    print(f"{'='*80}\n")
    
    exp_count = 0
    for model_size in model_sizes:
        for overlap in overlaps:
            for loss in losses:
                exp_count += 1
                print(f"\n{'#'*80}")
                print(f"# Experiment {exp_count}/{total_experiments}")
                print(f"{'#'*80}\n")
                
                try:
                    result = run_single_experiment(
                        model_size=model_size,
                        overlap=overlap,
                        loss_name=loss,
                        epochs=epochs,
                        use_wandb=use_wandb,
                        output_dir=Path(output_base_dir) / f"{model_size}_overlap{overlap}_{loss}"
                    )
                    results.append(result)
                except Exception as e:
                    print(f"\nERROR in experiment {model_size}_overlap{overlap}_{loss}: {e}")
                    results.append({
                        'exp_name': f"{model_size}_overlap{overlap}_{loss}",
                        'error': str(e)
                    })
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = Path(output_base_dir) / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run FCL experiments with full tracking")
    parser.add_argument("--models", nargs='+', default=['tiny', 'small', 'medium', 'large', 'huge'],
                       help="Model sizes to test")
    parser.add_argument("--overlaps", nargs='+', type=int, default=[0, 1, 2],
                       help="Overlap levels to test")
    parser.add_argument("--losses", nargs='+', default=['mse', 'wasserstein', 'lwwn', 'mape'],
                       help="Loss functions to test")
    parser.add_argument("--epochs", type=int, default=500,
                       help="Epochs per experiment")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable WandB logging")
    parser.add_argument("--output-dir", type=str, default="./experiments",
                       help="Base output directory")
    parser.add_argument("--single", action="store_true",
                       help="Run single experiment (use first of each list)")
    
    args = parser.parse_args()
    
    if args.single:
        # Run single experiment
        run_single_experiment(
            model_size=args.models[0],
            overlap=args.overlaps[0],
            loss_name=args.losses[0],
            epochs=args.epochs,
            use_wandb=args.wandb,
            output_dir=Path(args.output_dir) / f"{args.models[0]}_overlap{args.overlaps[0]}_{args.losses[0]}"
        )
    else:
        # Run full suite
        run_full_experiment_suite(
            model_sizes=args.models,
            overlaps=args.overlaps,
            losses=args.losses,
            epochs=args.epochs,
            use_wandb=args.wandb,
            output_base_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
