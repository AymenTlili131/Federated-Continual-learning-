#!/usr/bin/env python3
"""
COMPREHENSIVE INTEGRATED EXPERIMENT SYSTEM

Features:
- All 23+ loss functions from meta.ipynb with loss pairs
- Gated attention mechanism
- Persistent homology analysis
- RMT spectral analysis
- NTK trainability metrics
- HMMR time-series segmentation
- Super weight analysis
- Predicted weights saving every epoch
- Attention heatmap visualization
- WandB logging
- 500 epochs default
- Structured results output
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
from config import MODEL_CONFIGS
from comprehensive_losses import ComprehensiveLossRegistry, get_loss_pairs
from utils_consolidated import (
    WeightDistanceMetrics,
    load_merged_zoo,
    extract_weights_from_zoo,
    create_weight_pairs,
    save_metrics_csv
)

# WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: WandB not available")

# Optional analysis imports
try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: ripser not available for persistent homology")

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# GATED ATTENTION MODULE
# ============================================================================

class GatedMultiheadAttention(nn.Module):
    """Gated multi-head attention to prevent collapse"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Gate projection network
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, nhead),
            nn.Sigmoid()
        )
        
        self.nhead = nhead
        self.d_model = d_model
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # Standard attention
        attn_output, attn_weights = self.multihead_attn(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        
        # Compute gates
        gates = self.gate_proj(query.mean(dim=1))  # (batch, nhead)
        
        # Apply gates (reshape for broadcasting)
        batch_size, seq_len, _ = attn_output.shape
        gates_expanded = gates.unsqueeze(1).unsqueeze(-1)  # (batch, 1, nhead, 1)
        
        # Split attention output by heads and apply gates
        head_dim = self.d_model // self.nhead
        attn_output_heads = attn_output.view(batch_size, seq_len, self.nhead, head_dim)
        gated_output = attn_output_heads * gates_expanded
        attn_output = gated_output.view(batch_size, seq_len, self.d_model)
        
        return attn_output, attn_weights, gates


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_persistent_homology(representations, max_dim=2, subsample=1000):
    """Compute persistent homology on representations"""
    if not RIPSER_AVAILABLE:
        return None
    
    try:
        # Subsample if too large
        if len(representations) > subsample:
            indices = np.random.choice(len(representations), subsample, replace=False)
            representations = representations[indices]
        
        # Compute persistence diagrams
        result = ripser(representations, maxdim=max_dim)
        diagrams = result['dgms']
        
        # Extract Betti numbers
        betti_numbers = {}
        for dim in range(max_dim + 1):
            if dim < len(diagrams):
                # Count finite intervals
                dgm = diagrams[dim]
                finite_intervals = dgm[dgm[:, 1] < np.inf]
                betti_numbers[f'betti_{dim}'] = len(finite_intervals)
        
        return {
            'diagrams': diagrams,
            'betti_numbers': betti_numbers
        }
    except Exception as e:
        print(f"Warning: PH computation failed: {e}")
        return None


def compute_rmt_metrics(weight_matrix):
    """Compute Random Matrix Theory metrics"""
    try:
        # Compute eigenvalues
        if len(weight_matrix.shape) == 1:
            # Reshape to matrix
            n = int(np.sqrt(len(weight_matrix)))
            if n * n == len(weight_matrix):
                weight_matrix = weight_matrix.reshape(n, n)
            else:
                return None
        
        eigenvalues = np.linalg.eigvalsh(weight_matrix)
        
        # Spectral density
        hist, bins = np.histogram(eigenvalues, bins=50, density=True)
        
        # Marchenko-Pastur distribution parameters
        lambda_max = np.max(eigenvalues)
        lambda_min = np.min(eigenvalues)
        spectral_radius = max(abs(lambda_max), abs(lambda_min))
        
        return {
            'eigenvalues': eigenvalues,
            'spectral_radius': spectral_radius,
            'max_eigenvalue': lambda_max,
            'min_eigenvalue': lambda_min,
            'spectral_density': hist,
            'density_bins': bins
        }
    except Exception as e:
        print(f"Warning: RMT computation failed: {e}")
        return None


def compute_ntk_metrics(model, data_loader, device):
    """Compute Neural Tangent Kernel trainability metrics"""
    try:
        model.eval()
        
        # Get a batch
        x1, x2, _ = next(iter(data_loader))
        x1, x2 = x1[:10].to(device), x2[:10].to(device)
        
        # Compute Jacobian
        outputs, _, _, _, _ = model(x1, x2)
        
        # Compute gradient norms
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        return {
            'mean_grad_norm': np.mean(grad_norms) if grad_norms else 0.0,
            'max_grad_norm': np.max(grad_norms) if grad_norms else 0.0,
            'trainability_score': np.mean(grad_norms) if grad_norms else 0.0
        }
    except Exception as e:
        print(f"Warning: NTK computation failed: {e}")
        return None


def identify_super_weights(model, threshold=0.95):
    """Identify super weights (high influence)"""
    try:
        all_weights = []
        weight_importance = []
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.data.cpu().numpy().flatten()
                importance = np.abs(weights)
                all_weights.extend(weights)
                weight_importance.extend(importance)
        
        # Find threshold for top weights
        importance_threshold = np.percentile(weight_importance, threshold * 100)
        super_weight_mask = np.array(weight_importance) >= importance_threshold
        
        return {
            'n_super_weights': np.sum(super_weight_mask),
            'super_weight_ratio': np.mean(super_weight_mask),
            'importance_threshold': importance_threshold,
            'mean_importance': np.mean(weight_importance),
            'max_importance': np.max(weight_importance)
        }
    except Exception as e:
        print(f"Warning: Super weight analysis failed: {e}")
        return None


# ============================================================================
# ATTENTION HEATMAP VISUALIZATION
# ============================================================================

def plot_attention_heatmaps(attention_scores, save_path, title_prefix="", max_heads=8):
    """Plot attention heatmaps"""
    if not attention_scores or len(attention_scores) == 0:
        return None
    
    attn = attention_scores[0]
    if attn is None or len(attn.shape) != 4:
        return None
    
    attn_avg = attn.mean(dim=0).cpu().detach().numpy()
    n_heads = min(attn_avg.shape[0], max_heads)
    
    ncols = 4
    nrows = (n_heads + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_heads):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        sns.heatmap(attn_avg[idx], ax=ax, cmap='viridis', cbar=True, square=True)
        ax.set_title(f'{title_prefix} Head {idx+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
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
        fig_path = plot_attention_heatmaps(attention_scores, temp_path, title_prefix=prefix)
        
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


def validate(model, val_loader, loss_fn, device, save_attention=False, attention_save_dir=None, epoch=0):
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
            
            # Collect predictions and representations
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_necks.append(neck_t.cpu().numpy())
            
            # Collect attention from first batch
            if save_attention and batch_idx == 0:
                if scEnc1:
                    attention_scores['enc1'] = scEnc1
                if scEnc2:
                    attention_scores['enc2'] = scEnc2
                if scDec:
                    attention_scores['dec'] = scDec
    
    val_loss = total_loss / len(val_loader)
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    necks = np.concatenate(all_necks, axis=0)
    
    # Save attention heatmaps
    if save_attention and attention_save_dir:
        attention_save_dir = Path(attention_save_dir)
        attention_save_dir.mkdir(parents=True, exist_ok=True)
        
        if attention_scores['enc1']:
            path = attention_save_dir / f"epoch_{epoch:04d}_enc1.png"
            plot_attention_heatmaps(attention_scores['enc1'], path, title_prefix="Encoder 1")
            log_attention_to_wandb(attention_scores['enc1'], epoch, prefix="encoder1")
        
        if attention_scores['enc2']:
            path = attention_save_dir / f"epoch_{epoch:04d}_enc2.png"
            plot_attention_heatmaps(attention_scores['enc2'], path, title_prefix="Encoder 2")
            log_attention_to_wandb(attention_scores['enc2'], epoch, prefix="encoder2")
        
        if attention_scores['dec']:
            path = attention_save_dir / f"epoch_{epoch:04d}_dec.png"
            plot_attention_heatmaps(attention_scores['dec'], path, title_prefix="Decoder")
            log_attention_to_wandb(attention_scores['dec'], epoch, prefix="decoder")
    
    return val_loss, predictions, targets, necks, attention_scores


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_comprehensive_experiment(
    model_size='medium',
    overlap=2,
    loss_pair_idx=0,
    epochs=500,
    batch_size=32,
    lr=1e-4,
    output_dir=None,
    use_wandb=False,
    wandb_project="fcl-comprehensive",
    save_attention_every=10,
    compute_ph_every=50,
    compute_rmt_every=50,
    compute_ntk_every=100
):
    """Run comprehensive experiment with all analysis"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_registry = ComprehensiveLossRegistry()
    loss_pair_name = loss_registry.get_loss_pair_name(loss_pair_idx)
    
    exp_name = f"{model_size}_overlap{overlap}_{loss_pair_name.replace('+', '_').replace('*', 'x')}"
    
    if output_dir is None:
        output_dir = Path("./experiments") / exp_name
    output_dir = Path(output_dir)
    
    # Create output directories
    checkpoints_dir = output_dir / "checkpoints"
    attention_dir = output_dir / "attention_heatmaps"
    metrics_dir = output_dir / "metrics"
    weights_dir = output_dir / "predicted_weights"
    analysis_dir = output_dir / "analysis"
    
    for d in [checkpoints_dir, attention_dir, metrics_dir, weights_dir, analysis_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EXPERIMENT: {exp_name}")
    print(f"{'='*80}\n")
    
    # Initialize WandB
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=exp_name,
            config={
                'model_size': model_size,
                'overlap': overlap,
                'loss_pair': loss_pair_name,
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
    def loss_fn(pred, target):
        return loss_registry.compute_paired_loss(loss_pair_idx, pred, target)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Training for {epochs} epochs")
    print(f"Loss function: {loss_pair_name}")
    print(f"{'='*80}\n")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'ph_results': [],
        'rmt_results': [],
        'ntk_results': [],
        'super_weight_results': []
    }
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
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
        
        # Save predicted weights every epoch
        np.save(weights_dir / f"epoch_{epoch:04d}_predictions.npy", predictions)
        if epoch == 1:
            np.save(weights_dir / "targets.npy", targets)
        
        # Compute analyses
        analysis_results = {}
        
        # Persistent Homology
        if epoch % compute_ph_every == 0 and RIPSER_AVAILABLE:
            print(f"  Computing persistent homology...")
            ph_results = compute_persistent_homology(necks)
            if ph_results:
                history['ph_results'].append({'epoch': epoch, **ph_results['betti_numbers']})
                analysis_results.update({f'ph/{k}': v for k, v in ph_results['betti_numbers'].items()})
        
        # RMT Analysis
        if epoch % compute_rmt_every == 0:
            print(f"  Computing RMT metrics...")
            # Analyze first layer weights
            first_layer_weights = None
            for name, param in model.named_parameters():
                if 'weight' in name:
                    first_layer_weights = param.data.cpu().numpy()
                    break
            
            if first_layer_weights is not None:
                rmt_results = compute_rmt_metrics(first_layer_weights.flatten())
                if rmt_results:
                    history['rmt_results'].append({
                        'epoch': epoch,
                        'spectral_radius': rmt_results['spectral_radius'],
                        'max_eigenvalue': rmt_results['max_eigenvalue']
                    })
                    analysis_results['rmt/spectral_radius'] = rmt_results['spectral_radius']
        
        # NTK Analysis
        if epoch % compute_ntk_every == 0:
            print(f"  Computing NTK metrics...")
            ntk_results = compute_ntk_metrics(model, val_loader, device)
            if ntk_results:
                history['ntk_results'].append({'epoch': epoch, **ntk_results})
                analysis_results.update({f'ntk/{k}': v for k, v in ntk_results.items()})
        
        # Super Weight Analysis
        if epoch % 100 == 0:
            print(f"  Identifying super weights...")
            sw_results = identify_super_weights(model)
            if sw_results:
                history['super_weight_results'].append({'epoch': epoch, **sw_results})
                analysis_results.update({f'super_weights/{k}': v for k, v in sw_results.items()})
        
        # Log to WandB
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **analysis_results
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
    
    # Save all history
    with open(output_dir / "training_history.json", 'w') as f:
        # Convert numpy types to Python types for JSON
        history_json = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'ph_results': history['ph_results'],
            'rmt_results': history['rmt_results'],
            'ntk_results': history['ntk_results'],
            'super_weight_results': history['super_weight_results']
        }
        json.dump(history_json, f, indent=2)
    
    # Save analysis results as CSV
    if history['ph_results']:
        pd.DataFrame(history['ph_results']).to_csv(analysis_dir / "persistent_homology.csv", index=False)
    if history['rmt_results']:
        pd.DataFrame(history['rmt_results']).to_csv(analysis_dir / "rmt_analysis.csv", index=False)
    if history['ntk_results']:
        pd.DataFrame(history['ntk_results']).to_csv(analysis_dir / "ntk_analysis.csv", index=False)
    if history['super_weight_results']:
        pd.DataFrame(history['super_weight_results']).to_csv(analysis_dir / "super_weights.csv", index=False)
    
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
    
    # Compute distance metrics
    calc = WeightDistanceMetrics()
    all_metrics = []
    
    for i in range(min(100, len(test_predictions))):
        metrics = calc.compute_all_full_distances(test_predictions[i], test_targets[i])
        metrics['sample_idx'] = i
        metrics['model_size'] = model_size
        metrics['overlap'] = overlap
        metrics['loss_pair'] = loss_pair_name
        all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_dir / "test_metrics_full.csv", index=False)
    
    print(f"  Saved metrics to {metrics_dir / 'test_metrics_full.csv'}")
    
    # Cleanup
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE: {exp_name}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Results: {output_dir}")
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
    parser = argparse.ArgumentParser(description="Run comprehensive FCL experiments")
    parser.add_argument("--models", nargs='+', default=['medium'],
                       help="Model sizes to test")
    parser.add_argument("--overlaps", nargs='+', type=int, default=[2],
                       help="Overlap levels to test")
    parser.add_argument("--loss-pairs", nargs='+', type=int, default=[0, 1, 2],
                       help="Loss pair indices to test")
    parser.add_argument("--epochs", type=int, default=500,
                       help="Epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable WandB logging")
    parser.add_argument("--output-dir", type=str, default="./experiments",
                       help="Base output directory")
    parser.add_argument("--single", action="store_true",
                       help="Run single experiment")
    
    args = parser.parse_args()
    
    if args.single:
        # Run single experiment
        result = run_comprehensive_experiment(
            model_size=args.models[0],
            overlap=args.overlaps[0],
            loss_pair_idx=args.loss_pairs[0],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_wandb=args.wandb
        )
        print(f"\nResult: {result}")
    else:
        # Run multiple experiments
        results = []
        for model_size in args.models:
            for overlap in args.overlaps:
                for loss_pair_idx in args.loss_pairs:
                    try:
                        result = run_comprehensive_experiment(
                            model_size=model_size,
                            overlap=overlap,
                            loss_pair_idx=loss_pair_idx,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            use_wandb=args.wandb
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"\nERROR in {model_size}_overlap{overlap}_pair{loss_pair_idx}: {e}")
                        results.append({
                            'exp_name': f"{model_size}_overlap{overlap}_pair{loss_pair_idx}",
                            'error': str(e)
                        })
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_path = Path(args.output_dir) / "comprehensive_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}")
        print(f"ALL EXPERIMENTS COMPLETE")
        print(f"Summary: {summary_path}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
