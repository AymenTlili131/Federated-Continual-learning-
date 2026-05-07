"""
01_create_baseline.py

Creates a baseline checkpoint for the TransformerAE that will serve as the
foundation for all subsequent training experiments.

This baseline is initialized with proper weight initialization and saved
before any training occurs.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from Double_input_transformer import TransformerAE


def get_model_config(model_size: str) -> dict:
    """Get model configuration based on size."""
    configs = {
        "tiny": {
            "d_model": 256,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "params_M": 4.0
        },
        "small": {
            "d_model": 512,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "params_M": 8.3
        },
        "medium": {
            "d_model": 960,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "params_M": 18.1
        },
        "large": {
            "d_model": 1536,
            "nhead": 12,
            "num_layers": 6,
            "dim_feedforward": 3072,
            "dropout": 0.1,
            "params_M": 43.0
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    return configs[model_size]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module, init_method: str = "xavier_uniform"):
    """Initialize model weights using specified method."""
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Weight matrices
            if init_method == "xavier_uniform":
                nn.init.xavier_uniform_(param)
            elif init_method == "xavier_normal":
                nn.init.xavier_normal_(param)
            elif init_method == "kaiming_uniform":
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif init_method == "kaiming_normal":
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif init_method == "orthogonal":
                nn.init.orthogonal_(param)
        else:  # Bias vectors
            nn.init.zeros_(param)
    
    return model


def create_baseline_checkpoint(
    model_size: str,
    output_dir: Path,
    init_method: str = "xavier_uniform",
    device: str = "cuda"
) -> Path:
    """
    Create and save a baseline checkpoint.
    
    Args:
        model_size: Size of model (tiny, small, medium, large)
        output_dir: Directory to save checkpoint
        init_method: Weight initialization method
        device: Device to use (cuda/cpu)
    
    Returns:
        Path to saved checkpoint
    """
    print(f"\n{'='*80}")
    print(f"Creating Baseline Checkpoint: {model_size.upper()}")
    print(f"{'='*80}\n")
    
    # Get configuration
    config = get_model_config(model_size)
    print(f"Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    print(f"\nInitializing TransformerAE...")
    model = TransformerAE(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"]
    )
    
    # Initialize weights
    print(f"Initializing weights with method: {init_method}")
    model = initialize_weights(model, init_method)
    
    # Count parameters
    num_params = count_parameters(model)
    num_params_M = num_params / 1e6
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Parameters (M): {num_params_M:.2f}")
    print(f"  Expected (M): {config['params_M']:.2f}")
    
    # Move to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")
    
    # Create checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "model_size": model_size,
        "num_parameters": num_params,
        "init_method": init_method,
        "created_at": datetime.now().isoformat(),
        "epoch": 0,
        "training_history": []
    }
    
    # Save checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"baseline_{model_size}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    # Save config as JSON for reference
    config_path = output_dir / f"baseline_{model_size}_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "model_size": model_size,
            "config": config,
            "num_parameters": num_params,
            "num_parameters_M": num_params_M,
            "init_method": init_method,
            "created_at": checkpoint["created_at"]
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Baseline checkpoint saved:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config: {config_path}")
    print(f"{'='*80}\n")
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Create baseline checkpoint for TransformerAE"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["tiny", "small", "medium", "large"],
        help="Model size to create"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save checkpoint"
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="xavier_uniform",
        choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", 
                 "kaiming_normal", "orthogonal"],
        help="Weight initialization method"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Data directory (for validation)"
    )
    
    args = parser.parse_args()
    
    # Create baseline
    checkpoint_path = create_baseline_checkpoint(
        model_size=args.model_size,
        output_dir=args.output_dir,
        init_method=args.init_method,
        device=args.device
    )
    
    print(f"✓ Baseline checkpoint created successfully!")
    print(f"  Path: {checkpoint_path}")


if __name__ == "__main__":
    main()
