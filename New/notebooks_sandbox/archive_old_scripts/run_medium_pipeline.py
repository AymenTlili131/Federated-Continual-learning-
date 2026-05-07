#!/usr/bin/env python3
"""
Streamlined runner for medium model research pipeline.
This script runs the complete pipeline with integrated data loading.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research_scripts"))

# Import utilities
from research_scripts.utils.data_loading import prepare_data_for_pipeline
from research_scripts.utils.metrics import compute_eigenvalues, compute_wasserstein_distance

# Import models
from Double_input_transformer import TransformerAE

# Try to import WandB (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available - running without experiment tracking")


class SimplifiedPipeline:
    """Streamlined research pipeline for medium model."""
    
    def __init__(
        self,
        model_size: str = "medium",
        num_epochs: int = 500,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        data_dir: Path = None,
        output_dir: Path = None,
        use_wandb: bool = True
    ):
        self.model_size = model_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_dir = data_dir or PROJECT_ROOT / "data"
        self.output_dir = output_dir or PROJECT_ROOT / "research_results"
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Create output directories
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.metrics_dir = self.output_dir / "metrics"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Import model configurations from config.py
        try:
            from config import MODEL_CONFIGS
            self.config = MODEL_CONFIGS[model_size]
        except:
            # Fallback configurations matching config.py
            self.configs = {
                "tiny": {"max_seq_len": 50, "N": 1, "heads": 2, "d_model": 32, "d_ff": 128, "neck": 16},
                "small": {"max_seq_len": 50, "N": 2, "heads": 4, "d_model": 64, "d_ff": 256, "neck": 32},
                "medium": {"max_seq_len": 50, "N": 3, "heads": 4, "d_model": 128, "d_ff": 512, "neck": 64},
                "large": {"max_seq_len": 50, "N": 4, "heads": 8, "d_model": 256, "d_ff": 1024, "neck": 128}
            }
            self.config = self.configs[model_size]
        
    def initialize_wandb(self):
        """Initialize WandB tracking."""
        if self.use_wandb:
            wandb.init(
                project="weight-space-research",
                name=f"{self.model_size}_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model_size": self.model_size,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    **self.config
                }
            )
    
    def create_model(self):
        """Create TransformerAE model."""
        print(f"\nCreating {self.model_size} model...")
        
        # Handle both ModelConfig object and dict
        if hasattr(self.config, 'max_seq_len'):
            # ModelConfig object
            model = TransformerAE(
                max_seq_len=self.config.max_seq_len,
                N=self.config.N,
                heads=self.config.heads,
                d_model=self.config.d_model,
                d_ff=self.config.d_ff,
                neck=self.config.neck,
                dropout=self.config.dropout
            )
        else:
            # Dict config
            model = TransformerAE(
                max_seq_len=self.config["max_seq_len"],
                N=self.config["N"],
                heads=self.config["heads"],
                d_model=self.config["d_model"],
                d_ff=self.config["d_ff"],
                neck=self.config["neck"],
                dropout=0.1
            )
        
        model = model.to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        return model
    
    def load_data(self):
        """Load and prepare data."""
        print(f"\nLoading data from {self.data_dir}...")
        
        try:
            train_loader, val_loader, test_loader, metadata = prepare_data_for_pipeline(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                overlap=2,
                limit_samples=5000  # Limit for faster testing
            )
            return train_loader, val_loader, test_loader, metadata
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating dummy data for testing...")
            return self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create dummy data for testing when real data unavailable."""
        n_samples = 1000
        n_features = 2464  # Each input vector is 2464 dims (26*80 + 24*16)
        
        x1 = torch.randn(n_samples, n_features)
        x2 = torch.randn(n_samples, n_features)
        y = torch.randn(n_samples, n_features)  # Output is also 2464 dims
        
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(x1, x2, y)
        
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        test_size = n_samples - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        metadata = {
            'n_features': n_features,
            'n_samples': n_samples,
            'n_pairs': n_samples,
            'source': 'dummy_data'
        }
        
        print(f"  Created dummy data: {n_samples} samples, {n_features} features")
        return train_loader, val_loader, test_loader, metadata
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
        for batch_idx, (x1, x2, target) in enumerate(pbar):
            x1, x2, target = x1.to(self.device), x2.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            # TransformerAE returns: (output, neck_t, scEnc1, scEnc2, scDec)
            output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, model, val_loader, criterion):
        """Validate model."""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x1, x2, target in val_loader:
                x1, x2, target = x1.to(self.device), x2.to(self.device), target.to(self.device)
                # TransformerAE returns: (output, neck_t, scEnc1, scEnc2, scDec)
                output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
                loss = criterion(output, target)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = self.checkpoints_dir / f"checkpoint_latest_{self.model_size}.pth"
        torch.save(checkpoint, latest_path)
    
    def run(self):
        """Run the complete pipeline."""
        print(f"\n{'='*80}")
        print(f"Weight-Space Research Pipeline - {self.model_size.upper()} Model")
        print(f"{'='*80}\n")
        print(f"Configuration:")
        print(f"  Model size: {self.model_size}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Device: {self.device}")
        print(f"  Output dir: {self.output_dir}")
        
        # Initialize WandB
        if self.use_wandb:
            self.initialize_wandb()
        
        # Load data
        train_loader, val_loader, test_loader, metadata = self.load_data()
        
        # Create model
        model = self.create_model()
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}\n")
        
        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_loss = self.validate(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            print(f"Epoch {epoch}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            if epoch % 10 == 0 or val_loss < best_val_loss:
                self.save_checkpoint(model, optimizer, epoch, train_loss, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  → New best validation loss: {best_val_loss:.6f}")
        
        # Save final checkpoint
        self.save_checkpoint(model, optimizer, self.num_epochs, train_loss, val_loss)
        
        # Save history
        history_path = self.metrics_dir / f"training_history_{self.model_size}.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}\n")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Checkpoints saved to: {self.checkpoints_dir}")
        print(f"Metrics saved to: {self.metrics_dir}")
        
        if self.use_wandb:
            wandb.finish()
        
        return model, history


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run medium model research pipeline")
    parser.add_argument("--model_size", type=str, default="medium", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SimplifiedPipeline(
        model_size=args.model_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb
    )
    
    model, history = pipeline.run()
    
    print("\n✓ Pipeline execution complete!")


if __name__ == "__main__":
    main()
