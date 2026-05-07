#!/usr/bin/env python3
"""
Train medium model using existing FCL infrastructure.
Generates data on-the-fly from Merged zoo.csv
"""

import sys
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

from Double_input_transformer import TransformerAE
from config import MODEL_CONFIGS

def load_and_prepare_data(csv_path, batch_size=32, limit_samples=10000):
    """Load Merged zoo and create weight pairs."""
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    
    if limit_samples and len(df) > limit_samples:
        df = df.sample(n=limit_samples, random_state=42)
        print(f"  Limited to {limit_samples} samples")
    
    # Extract weight columns (first 2464 columns that are numeric)
    weight_cols = [col for col in df.columns if col not in ['label', 'activation', 'epoch']]
    weights = df[weight_cols].values.astype(np.float32)
    
    # Ensure exactly 2464 features
    if weights.shape[1] > 2464:
        weights = weights[:, :2464]
    elif weights.shape[1] < 2464:
        padding = np.zeros((weights.shape[0], 2464 - weights.shape[1]), dtype=np.float32)
        weights = np.concatenate([weights, padding], axis=1)
    
    print(f"  Weight matrix shape: {weights.shape}")
    
    # Create random pairs
    n_samples = len(weights)
    n_pairs = n_samples // 2
    
    indices = np.random.permutation(n_samples)
    idx1 = indices[:n_pairs]
    idx2 = indices[n_pairs:2*n_pairs]
    
    x1 = torch.from_numpy(weights[idx1]).float()
    x2 = torch.from_numpy(weights[idx2]).float()
    y = torch.from_numpy((weights[idx1] + weights[idx2]) / 2).float()
    
    # Split into train/val/test
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

def main():
    print(f"\n{'='*80}")
    print(f"Medium Model Training - 500 Epochs")
    print(f"{'='*80}\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS["medium"]
    output_dir = PROJECT_ROOT / "research_results"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Model: medium ({config.N} layers, {config.d_model} d_model)")
    print(f"  Epochs: 500")
    print(f"  Batch size: 32")
    
    # Load data
    csv_path = PROJECT_ROOT / "data" / "Merged zoo.csv"
    train_loader, val_loader, test_loader = load_and_prepare_data(csv_path, batch_size=32, limit_samples=10000)
    
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
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    # Training loop
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(1, 501):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, 500)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch}/500 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
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
                torch.save(checkpoint, checkpoints_dir / f"best_medium_model.pth")
                print(f"  → New best model saved (val_loss: {val_loss:.6f})")
            
            if epoch % 10 == 0:
                torch.save(checkpoint, checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pth")
    
    # Save final checkpoint and history
    torch.save(checkpoint, checkpoints_dir / "final_medium_model.pth")
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
