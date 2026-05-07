#!/usr/bin/env python3
"""
Load and analyze the trained medium model results.
"""

import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from Double_input_transformer import TransformerAE
from config import MODEL_CONFIGS

def load_checkpoint(checkpoint_path):
    """Load a checkpoint and return model + metadata."""
    # PyTorch 2.6+ requires weights_only=False for custom objects
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    config = checkpoint['config']
    model = TransformerAE(
        max_seq_len=config.max_seq_len,
        N=config.N,
        heads=config.heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        neck=config.neck,
        dropout=config.dropout
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Ensure model is on CPU
    model = model.cpu()
    
    return model, checkpoint

def plot_training_history(history_path, save_path=None):
    """Plot training and validation loss curves."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Medium Model Training History (500 Epochs)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    min_train_loss = min(history['train_loss'])
    min_val_loss = min(history['val_loss'])
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    stats_text = f"Min Train Loss: {min_train_loss:.6f}\n"
    stats_text += f"Min Val Loss: {min_val_loss:.6f}\n"
    stats_text += f"Final Train Loss: {final_train_loss:.6f}\n"
    stats_text += f"Final Val Loss: {final_val_loss:.6f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()
    return fig

def analyze_model_weights(model):
    """Analyze the trained model weights."""
    print("\n" + "="*80)
    print("Model Weight Analysis")
    print("="*80 + "\n")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Analyze weight statistics per layer
    print("\nWeight Statistics by Layer:")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean = param.data.mean().item()
            std = param.data.std().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()
            
            print(f"{name:50s} | Shape: {str(tuple(param.shape)):20s} | "
                  f"Mean: {mean:7.4f} | Std: {std:7.4f} | "
                  f"Range: [{min_val:7.4f}, {max_val:7.4f}]")

def predict_weights(model, x1, x2, device='cpu'):
    """Use the model to predict merged weights."""
    model.eval()
    # Ensure model and inputs are on same device
    model = model.to(device)
    x1 = x1.to(device)
    x2 = x2.to(device)
    
    with torch.no_grad():
        output, neck_t, scEnc1, scEnc2, scDec = model(x1, x2)
    
    return {
        'predicted_weights': output.cpu(),
        'neck_representation': neck_t.cpu(),
        'encoder1_attention': scEnc1,
        'encoder2_attention': scEnc2,
        'decoder_attention': scDec
    }

def main():
    results_dir = Path("research_results")
    
    print("\n" + "="*80)
    print("Medium Model Results Analysis")
    print("="*80 + "\n")
    
    # 1. Plot training history
    print("1. Plotting training history...")
    history_path = results_dir / "training_history.json"
    plot_path = results_dir / "training_curves.png"
    plot_training_history(history_path, plot_path)
    
    # 2. Load best model
    print("\n2. Loading best model...")
    best_model_path = results_dir / "checkpoints" / "best_medium_model.pth"
    model, checkpoint = load_checkpoint(best_model_path)
    
    print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Train loss: {checkpoint['train_loss']:.6f}")
    print(f"   Val loss: {checkpoint['val_loss']:.6f}")
    
    # 3. Analyze weights
    print("\n3. Analyzing model weights...")
    analyze_model_weights(model)
    
    # 4. Example prediction
    print("\n4. Testing model prediction...")
    x1_sample = torch.randn(1, 2464)
    x2_sample = torch.randn(1, 2464)
    
    results = predict_weights(model, x1_sample, x2_sample)
    
    print(f"   Input 1 shape: {x1_sample.shape}")
    print(f"   Input 2 shape: {x2_sample.shape}")
    print(f"   Predicted weights shape: {results['predicted_weights'].shape}")
    print(f"   Neck representation shape: {results['neck_representation'].shape}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80 + "\n")
    
    print("Available checkpoints:")
    checkpoints = sorted(results_dir.glob("checkpoints/checkpoint_epoch_*.pth"))
    print(f"  - {len(checkpoints)} epoch checkpoints (every 10 epochs)")
    print(f"  - best_medium_model.pth (lowest validation loss)")
    print(f"  - final_medium_model.pth (epoch 500)")

if __name__ == "__main__":
    main()
