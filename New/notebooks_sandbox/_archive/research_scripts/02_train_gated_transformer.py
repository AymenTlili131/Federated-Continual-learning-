"""
02_train_gated_transformer.py

Trains TransformerAE with gated attention mechanism for gradient stabilization
and attention collapse prevention.

Based on: https://arxiv.org/pdf/2505.06708
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from Double_input_transformer import TransformerAE


class GatedMultiheadAttention(nn.Module):
    """
    Gated multi-head attention mechanism to prevent attention collapse.
    
    Based on arxiv 2505.06708 - adds learned gates per attention head
    to dynamically control information flow and prevent attention sinking.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Standard attention components
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism - one gate per head
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, nhead),
            nn.Sigmoid()  # Gates in [0, 1]
        )
        
        # Temperature scaling (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Attended output (batch, seq_len, d_model)
            attention_weights: Optional attention weights if return_attention=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, nhead, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with temperature
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = attn_scores / self.temperature.clamp(min=0.1)  # Prevent division by zero
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # (batch, nhead, seq_len, head_dim)
        
        # Compute gates based on input
        gates = self.gate_proj(x.mean(dim=1))  # (batch, nhead)
        gates = gates.unsqueeze(-1).unsqueeze(-1)  # (batch, nhead, 1, 1)
        
        # Apply gates to attention output
        attn_output = attn_output * gates
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_weights, gates.squeeze()
        return output


class GatedTransformerAE(nn.Module):
    """TransformerAE with gated attention mechanisms."""
    
    def __init__(
        self,
        d_model: int = 960,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Use base TransformerAE but replace attention
        self.base_model = TransformerAE(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Replace transformer encoder layers with gated versions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.base_model.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Add gated attention modules
        self.gated_attentions = nn.ModuleList([
            GatedMultiheadAttention(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, return_attention: bool = False):
        """Forward pass with gated attention."""
        # Use base model's embedding
        embedded = self.base_model.embedder(x1, x2)
        
        # Apply gated transformer layers
        attention_maps = []
        gate_values = []
        
        for i, gated_attn in enumerate(self.gated_attentions):
            if return_attention:
                embedded, attn, gates = gated_attn(embedded, return_attention=True)
                attention_maps.append(attn)
                gate_values.append(gates)
            else:
                embedded = gated_attn(embedded)
        
        # Decode
        output = self.base_model.decoder(embedded)
        
        if return_attention:
            return output, attention_maps, gate_values
        return output


class AttentionEntropyMonitor:
    """Monitors attention entropy to detect attention collapse."""
    
    def __init__(self):
        self.entropy_history = []
        
    def compute_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute normalized entropy of attention weights."""
        eps = 1e-8
        attention_weights = attention_weights + eps
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        entropy = -(attention_weights * torch.log(attention_weights)).sum(dim=-1)
        seq_len = attention_weights.shape[-1]
        max_entropy = np.log(seq_len) if seq_len > 1 else 1.0
        
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else entropy
        return normalized_entropy.mean().item()
    
    def update(self, attention_weights: torch.Tensor) -> Tuple[float, str]:
        """Update entropy history and return status."""
        entropy = self.compute_entropy(attention_weights)
        self.entropy_history.append(entropy)
        
        if entropy < 0.2:
            status = "critical"
        elif entropy < 0.5:
            status = "warning"
        else:
            status = "healthy"
        
        return entropy, status


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    entropy_monitor: AttentionEntropyMonitor,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_entropy = 0.0
    total_gate_values = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (x1, x2, target) in enumerate(pbar):
        x1, x2, target = x1.to(device), x2.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with attention tracking
        output, attention_maps, gate_values = model(x1, x2, return_attention=True)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Monitor attention entropy
        if attention_maps:
            entropy, status = entropy_monitor.update(attention_maps[-1])
            total_entropy += entropy
        
        # Track gate values
        if gate_values:
            total_gate_values.append(gate_values[-1].mean().item())
        
        pbar.set_postfix({
            'loss': loss.item(),
            'entropy': entropy if attention_maps else 0.0,
            'gates': total_gate_values[-1] if total_gate_values else 0.0
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'entropy': total_entropy / len(dataloader),
        'gate_mean': np.mean(total_gate_values) if total_gate_values else 0.0,
        'gate_std': np.std(total_gate_values) if total_gate_values else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Train Gated Attention TransformerAE")
    parser.add_argument("--model_size", type=str, default="medium")
    parser.add_argument("--baseline_checkpoint", type=Path, required=True)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--wandb_project", type=str, default="weight-space-research")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--enable_gated_attention", action="store_true")
    parser.add_argument("--track_attention_entropy", action="store_true")
    parser.add_argument("--track_gradient_norms", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        config=vars(args),
        name=f"gated_transformer_{args.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"\n{'='*80}")
    print(f"Training Gated Attention TransformerAE")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Load baseline checkpoint
    print(f"\nLoading baseline checkpoint: {args.baseline_checkpoint}")
    checkpoint = torch.load(args.baseline_checkpoint)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = checkpoint["config"]
    
    if args.enable_gated_attention:
        model = GatedTransformerAE(
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"]
        )
    else:
        model = TransformerAE(
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"]
        )
    
    model = model.to(device)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    entropy_monitor = AttentionEntropyMonitor()
    
    # Training loop placeholder (needs data loading)
    print(f"\n{'='*80}")
    print(f"Training setup complete. Data loading required for actual training.")
    print(f"{'='*80}\n")
    
    wandb.finish()


if __name__ == "__main__":
    main()
