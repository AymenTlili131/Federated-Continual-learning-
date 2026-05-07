#!/usr/bin/env python3
"""
Configurable training script with multiple loss functions and comprehensive logging.
Supports WandB integration, distance metrics, and topological analysis.
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research_scripts"))

from Double_input_transformer import TransformerAE
from config import MODEL_CONFIGS
from distance_metrics import WeightDistanceMetrics
from robust_topology_analysis import safe_compute_topology_metrics
from wandb_integration import WandBLogger, create_experiment_summary


class LossFunctions:
    """Collection of loss functions for weight prediction."""
    
    @staticmethod
    def mse_loss(pred, target):
        """Mean Squared Error."""
        return nn.MSELoss()(pred, target)
    
    @staticmethod
    def mape_loss(pred, target, epsilon=1e-8):
        """Mean Absolute Percentage Error."""
        return torch.mean(torch.abs((pred - target) / (torch.abs(target) + epsilon))) * 100
    
    @staticmethod
    def wasserstein_loss(pred, target):
        """Approximation of Wasserstein distance."""
        # Sort both tensors
        pred_sorted, _ = torch.sort(pred, dim=-1)
        target_sorted, _ = torch.sort(target, dim=-1)
        return torch.mean(torch.abs(pred_sorted - target_sorted))
    
    @staticmethod
    def contrastive_loss(pred, target, margin=1.0):
        """Contrastive loss for weight similarity."""
        distance = torch.norm(pred - target, p=2, dim=-1)
        return torch.mean(torch.pow(distance, 2))
    
    @staticmethod
    def q_quantile_loss(pred, target, q=0.5):
        """Q-quantile loss (median absolute deviation when q=0.5)."""
        errors = pred - target
        return torch.mean(torch.where(errors >= 0, q * errors, (q - 1) * errors))
    
    @staticmethod
    def lwln_loss(pred, target, epsilon=1e-8):
        """Layer-wise Loss Normalization."""
        std = torch.std(target, dim=-1, keepdim=True)
        return torch.mean(torch.abs(pred - target) / (std + epsilon))
    
    @staticmethod
    def jensen_shannon_loss(pred, target, epsilon=1e-8):
        """Jensen-Shannon divergence approximation."""
        # Normalize to probability distributions
        pred_norm = torch.abs(pred) / (torch.sum(torch.abs(pred), dim=-1, keepdim=True) + epsilon)
        target_norm = torch.abs(target) / (torch.sum(torch.abs(target), dim=-1, keepdim=True) + epsilon)
        m = 0.5 * (pred_norm + target_norm)
        
        # KL divergence
        kl_pred = torch.sum(pred_norm * torch.log((pred_norm + epsilon) / (m + epsilon)), dim=-1)
        kl_target = torch.sum(target_norm * torch.log((target_norm + epsilon) / (m + epsilon)), dim=-1)
        
        return torch.mean(0.5 * kl_pred + 0.5 * kl_target)
    
    @staticmethod
    def get_loss_function(name):
        """Get loss function by name."""
        loss_map = {
            'mse': LossFunctions.mse_loss,
            'mape': LossFunctions.mape_loss,
            'wasserstein': LossFunctions.wasserstein_loss,
            'contrastive': LossFunctions.contrastive_loss,
            'q_quantile': LossFunctions.q_quantile_loss,
            'lwln': LossFunctions.lwln_loss,
            'jensen_shannon': LossFunctions.jensen_shannon_loss
        }
        return loss_map.get(name.lower(), LossFunctions.mse_loss)


def load_and_prepare_data(csv_path, batch_size=32, overlap=2, limit_samples=10000):
    """Load Merged zoo and create weight pairs."""
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    
    if limit_samples and len(df) > limit_samples:
        df = df.sample(n=limit_samples, random_state=42)
        print(f"  Limited to {limit_samples} samples")
    
    # Extract weight columns
    weight_cols = [col for col in df.columns if col not in ['label', 'activation', 'epoch']]
    weights = df[weight_cols].values.astype(np.float32)
    
    # Ensure exactly 2464 features
    if weights.shape[1] > 2464:
        weights = weights[:, :2464]
    elif weights.shape[1] < 2464:
        padding = np.zeros((weights.shape[0], 2464 - weights.shape[1]), dtype=np.float32)
        weights = np.concatenate([weights, padding], axis=1)
    
    print(f"  Weight matrix shape: {weights.shape}")
    
    # Create pairs
    n_samples = len(weights)
    n_pairs = n_samples // 2
    
    indices = np.random.permutation(n_samples)
    idx1 = indices[:n_pairs]
    idx2 = indices[n_pairs:2*n_pairs]
    
    x1 = torch.from_numpy(weights[idx1]).float()
    x2 = torch.from_numpy(weights[idx2]).float()
    y = torch.from_numpy((weights[idx1] + weights[idx2]) / 2).float()
    
    # Split
    n_train = int(0.7 * n_pairs)
    n_val = int(0.15 * n_pairs)
    
    train_dataset = TensorDataset(x1[:n_train], x2[:n_train], y[:n_train])
    val_dataset = TensorDataset(x1[n_train:n_train+n_val], x2[n_train:n_train+n_val], y[n_train:n_train+n_val])
    test_dataset = TensorDataset(x1[n_train+n_val:], x2[n_train+n_val:], y[n_train+n_val:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
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


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x1, x2, target in val_loader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def analyze_predictions(model, test_loader, device, logger, epoch):
    """Analyze predictions with distance metrics and topology."""
    model.eval()
    
    predictions = []
    targets = []
    neck_representations = []
    
    with torch.no_grad():
        for x1, x2, target in test_loader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)
            output, neck_t, _, _, _ = model(x1, x2)
            
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            neck_representations.append(neck_t.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    neck_reps = np.concatenate(neck_representations, axis=0)
    
    # Compute distance metrics for first test sample
    if len(predictions) > 0:
        dist_calc = WeightDistanceMetrics()
        metrics = dist_calc.compute_all_metrics(predictions[0], targets[0])
        
        # Log to WandB
        logger.log_distance_table(metrics, f"distances_epoch_{epoch}")
        
        # Save markdown table
        markdown = dist_calc.format_as_table(metrics)
        logger.save_markdown_table(markdown, f"distance_metrics_epoch_{epoch}.md")
    
    # Compute topology metrics on neck representations
    if len(neck_reps) > 10:
        topology_results = safe_compute_topology_metrics(neck_reps[:100])  # Subsample for speed
        logger.log_topology_metrics(topology_results)
    
    return metrics if len(predictions) > 0 else None


def main():
    parser = argparse.ArgumentParser(description="Train TransformerAE with configurable settings")
    parser.add_argument("--model-size", type=str, default="medium", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--overlap", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiment_results")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}\n")
    print(f"Model Size:    {args.model_size}")
    print(f"Overlap:       {args.overlap}")
    print(f"Loss Function: {args.loss}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"WandB:         {args.wandb}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS[args.model_size]
    
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize WandB
    wandb_config = {
        'model_size': args.model_size,
        'overlap': args.overlap,
        'loss_function': args.loss,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        **{k: getattr(config, k) for k in ['N', 'heads', 'd_model', 'd_ff', 'neck', 'dropout']}
    }
    
    logger = WandBLogger(
        project="weight-space-research",
        name=args.wandb_name or f"{args.model_size}_overlap{args.overlap}_{args.loss}",
        config=wandb_config,
        enabled=args.wandb
    )
    
    # Load data
    csv_path = PROJECT_ROOT / "data" / "Merged zoo.csv"
    train_loader, val_loader, test_loader = load_and_prepare_data(
        csv_path, batch_size=args.batch_size, overlap=args.overlap, limit_samples=10000
    )
    
    # Create model
    print(f"\nCreating model...")
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
    criterion = LossFunctions.get_loss_function(args.loss)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Log to WandB
        logger.log_training_progress(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]['lr']
        )
        
        print(f"Epoch {epoch}/{args.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoints
        if epoch % 10 == 0 or val_loss < best_val_loss:
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
                print(f"  → New best model saved (val_loss: {val_loss:.6f})")
            
            if epoch % 10 == 0:
                torch.save(checkpoint, checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pth")
        
        # Analyze predictions every 50 epochs
        if epoch % 50 == 0:
            analyze_predictions(model, test_loader, device, logger, epoch)
    
    # Final checkpoint
    torch.save(checkpoint, checkpoints_dir / "final_model.pth")
    
    # Save history
    with open(args.output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create summary
    summary = create_experiment_summary(
        model_config=wandb_config,
        training_history=history,
        distance_metrics={},
        topology_results={}
    )
    
    logger.save_markdown_table(summary, args.output_dir / "experiment_summary.md")
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"{'='*80}\n")
    
    logger.finish()


if __name__ == "__main__":
    main()
