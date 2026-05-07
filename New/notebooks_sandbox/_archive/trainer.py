"""
Comprehensive training framework for FCL with WandB integration
Supports multiple loss functions, metrics tracking, and efficient training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from config import ExperimentConfig
from loss_functions import get_loss_function, MultiLoss
from topology_analysis import compare_weight_topology, TopologicalFeatureExtractor
from rmt_analysis import compare_weight_stages_rmt, RandomMatrixAnalyzer


class FCLTrainer:
    """
    Federated Continual Learning Trainer
    """
    def __init__(self, config: ExperimentConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.device = torch.device(config.training.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup loss function
        if config.loss.use_multi_loss:
            self.criterion = MultiLoss(loss_weights=config.loss.loss_weights)
        else:
            self.criterion = get_loss_function(config.loss.primary_loss)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        # Create save directory
        self.save_dir = Path(config.save_dir) / config.name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize WandB
        self.use_wandb = WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()
        
        # Analysis tools
        if config.metrics.track_persistent_homology:
            self.topo_extractor = TopologicalFeatureExtractor(
                max_dimension=config.metrics.max_homology_dim
            )
        
        if config.metrics.track_rmt:
            self.rmt_analyzer = RandomMatrixAnalyzer()
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        elif self.config.training.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.training.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.name,
                config=self.config.to_dict()
            )
            wandb.watch(self.model, log='all', log_freq=100)
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            self.use_wandb = False
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'mse': [],
            'mape': [],
            'wasserstein': [],
            'latent': [],
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (data, acc_batch, idx_batch) in enumerate(pbar):
            # data: [Stream1, Stream2, target]
            inp1 = data[:, 0, :].to(self.device)
            inp2 = data[:, 1, :].to(self.device)
            target = data[:, 2, :].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    out, neck, scEnc1, scEnc2, scDec = self.model(inp1, inp2)
                    
                    if isinstance(self.criterion, MultiLoss):
                        loss, loss_dict = self.criterion(out, target, neck)
                        for key, value in loss_dict.items():
                            if key in epoch_metrics:
                                epoch_metrics[key].append(value)
                    else:
                        loss = self.criterion(out, target)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                out, neck, scEnc1, scEnc2, scDec = self.model(inp1, inp2)
                
                if isinstance(self.criterion, MultiLoss):
                    loss, loss_dict = self.criterion(out, target, neck)
                    for key, value in loss_dict.items():
                        if key in epoch_metrics:
                            epoch_metrics[key].append(value)
                else:
                    loss = self.criterion(out, target)
                
                loss.backward()
                
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config.training.log_every == 0:
                log_dict = {'train/batch_loss': loss.item()}
                if isinstance(self.criterion, MultiLoss):
                    for key, value in loss_dict.items():
                        log_dict[f'train/batch_{key}'] = value
                wandb.log(log_dict)
        
        # Compute epoch statistics
        results = {
            'loss': np.mean(epoch_losses),
            'loss_std': np.std(epoch_losses),
        }
        
        for key, values in epoch_metrics.items():
            if values:
                results[key] = np.mean(values)
        
        return results
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_losses = []
        val_metrics = {
            'mse': [],
            'mape': [],
            'wasserstein': [],
            'latent': [],
        }
        
        # For topology and RMT analysis
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, acc_batch, idx_batch in val_loader:
                inp1 = data[:, 0, :].to(self.device)
                inp2 = data[:, 1, :].to(self.device)
                target = data[:, 2, :].to(self.device)
                
                out, neck, scEnc1, scEnc2, scDec = self.model(inp1, inp2)
                
                if isinstance(self.criterion, MultiLoss):
                    loss, loss_dict = self.criterion(out, target, neck)
                    for key, value in loss_dict.items():
                        if key in val_metrics:
                            val_metrics[key].append(value)
                else:
                    loss = self.criterion(out, target)
                
                val_losses.append(loss.item())
                
                # Store for analysis
                all_predictions.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Compute validation statistics
        results = {
            'loss': np.mean(val_losses),
            'loss_std': np.std(val_losses),
        }
        
        for key, values in val_metrics.items():
            if values:
                results[key] = np.mean(values)
        
        # Perform topology and RMT analysis on sample
        if self.config.metrics.track_persistent_homology or self.config.metrics.track_rmt:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            # Analyze first sample
            if len(all_predictions) > 0:
                pred_sample = all_predictions[0]
                target_sample = all_targets[0]
                
                if self.config.metrics.track_persistent_homology:
                    topo_features = self.topo_extractor.extract_features(pred_sample)
                    results['topo_feature_dim'] = len(topo_features)
                
                if self.config.metrics.track_rmt:
                    rmt_results = self.rmt_analyzer.analyze_all_layers(pred_sample)
                    # Extract key metrics
                    for layer_name, layer_results in rmt_results.items():
                        if layer_results['type'] == 'weight':
                            results[f'rmt_{layer_name}_spectral_radius'] = layer_results['spectral_radius']
        
        return results
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        print(f"\nStarting training: {self.config.name}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.training.epochs}")
        print("=" * 60)
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_results = self.train_epoch(train_loader)
            
            # Validate
            val_results = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")
            print(f"  Train Loss: {train_results['loss']:.4f}")
            print(f"  Val Loss: {val_results['loss']:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': train_results['loss'],
                    'val/loss': val_results['loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time,
                }
                
                # Add all metrics
                for key, value in train_results.items():
                    if key != 'loss':
                        log_dict[f'train/{key}'] = value
                
                for key, value in val_results.items():
                    if key != 'loss':
                        log_dict[f'val/{key}'] = value
                
                wandb.log(log_dict)
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Check for improvement
            if val_results['loss'] < self.best_val_loss:
                self.best_val_loss = val_results['loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best model saved!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print("\n" + "=" * 60)
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")


def create_dataloaders(config: ExperimentConfig):
    """
    Create train/val/test dataloaders
    """
    from Double_input_transformer import CustomDataset
    
    # Create datasets
    train_dataset = CustomDataset(
        m=config.data.overlap_levels[0],
        epoch_key=config.data.epoch_key,
        activ_key=config.data.activ_key,
        batch_size=config.data.batch_size,
        batch_limit=config.data.batch_limit,
        df_path=config.data.df_path,
        data_type="train"
    )
    
    val_dataset = CustomDataset(
        m=config.data.overlap_levels[0],
        epoch_key=config.data.epoch_key,
        activ_key=config.data.activ_key,
        batch_size=config.data.batch_size,
        batch_limit=config.data.batch_limit,
        df_path=config.data.df_path,
        data_type="val"
    )
    
    test_dataset = CustomDataset(
        m=config.data.overlap_levels[0],
        epoch_key=config.data.epoch_key,
        activ_key=config.data.activ_key,
        batch_size=config.data.batch_size,
        batch_limit=config.data.batch_limit,
        df_path=config.data.df_path,
        data_type="test"
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Already batched in CustomDataset
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config import create_experiment_config
    from optimized_models import create_model_from_config
    
    # Create a test configuration
    config = create_experiment_config("tiny", 2, "mse", "test_trainer")
    config.training.epochs = 5
    
    # Create model
    model = create_model_from_config(config.model)
    
    print("Trainer test configuration:")
    print(f"  Model: {config.model.name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Loss: {config.loss.primary_loss}")
    print(f"  Device: {config.training.device}")
