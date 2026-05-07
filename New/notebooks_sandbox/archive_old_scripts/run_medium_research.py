#!/usr/bin/env python3
"""
Run medium model research pipeline using existing FCL infrastructure.
This uses the tested trainer and config system.
"""

import sys
from pathlib import Path
import torch
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_CONFIGS, create_experiment_config
from trainer import FCLTrainer, create_dataloaders
from Double_input_transformer import TransformerAE, CustomDataset

def main():
    print(f"\n{'='*80}")
    print(f"Weight-Space Research - Medium Model (500 Epochs)")
    print(f"{'='*80}\n")
    
    # Configuration
    model_size = "medium"
    config = MODEL_CONFIGS[model_size]
    
    print(f"Model Configuration:")
    print(f"  Name: {config.name}")
    print(f"  Layers (N): {config.N}")
    print(f"  Heads: {config.heads}")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  neck: {config.neck}")
    print(f"  dropout: {config.dropout}")
    print(f"  max_seq_len: {config.max_seq_len}")
    
    # Create model
    print(f"\nCreating TransformerAE model...")
    model = TransformerAE(
        max_seq_len=config.max_seq_len,
        N=config.N,
        heads=config.heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        neck=config.neck,
        dropout=config.dropout
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Load data using CustomDataset
    print(f"\nLoading data...")
    try:
        # Try to load pre-batched data
        train_dataset = CustomDataset(
            m=2,  # overlap
            epoch_key=36,
            activ_key=0,  # gelu
            batch_size=32,
            batch_limit=100,
            data_type="train"
        )
        val_dataset = CustomDataset(
            m=2,
            epoch_key=36,
            activ_key=0,
            batch_size=32,
            batch_limit=20,
            data_type="val"
        )
        test_dataset = CustomDataset(
            m=2,
            epoch_key=36,
            activ_key=0,
            batch_size=32,
            batch_limit=20,
            data_type="test"
        )
        
        print(f"  Train batches: {len(train_dataset)}")
        print(f"  Val batches: {len(val_dataset)}")
        print(f"  Test batches: {len(test_dataset)}")
        
        # Create trainer
        print(f"\nInitializing trainer...")
        trainer = FCLTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            num_epochs=500,
            learning_rate=1e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_dir=PROJECT_ROOT / "research_results" / "checkpoints",
            use_wandb=False
        )
        
        # Train
        print(f"\n{'='*80}")
        print(f"Starting Training (500 epochs)")
        print(f"{'='*80}\n")
        
        trainer.train()
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}\n")
        
    except FileNotFoundError as e:
        print(f"\nPre-batched data not found: {e}")
        print(f"Please run data preparation first or use the notebooks.")
        print(f"\nAlternatively, you can:")
        print(f"  1. Run notebook 01_generate_additional_zoos.ipynb")
        print(f"  2. Run notebook 02_batch_tensors_and_benchmark.ipynb")
        print(f"  3. Then run this script again")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
