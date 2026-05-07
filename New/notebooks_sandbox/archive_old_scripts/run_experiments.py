"""
Main script to run FCL experiments
Supports running individual experiments or full experiment suite
"""

import argparse
import sys
from pathlib import Path
import torch

from config import (
    create_experiment_config,
    print_config_summary,
    EXPERIMENT_SUITE,
    MODEL_CONFIGS
)
from optimized_models import create_model_from_config
from trainer import FCLTrainer, create_dataloaders
from data_preprocessing import DataPreprocessor


def run_single_experiment(experiment_name: str, use_wandb: bool = True):
    """Run a single experiment by name"""
    
    if experiment_name in EXPERIMENT_SUITE:
        config = EXPERIMENT_SUITE[experiment_name]
    else:
        print(f"Unknown experiment: {experiment_name}")
        print(f"Available experiments: {list(EXPERIMENT_SUITE.keys())}")
        return
    
    print("\n" + "="*80)
    print(f"Running Experiment: {experiment_name}")
    print("="*80)
    
    print_config_summary(config)
    
    # Create model
    print("\nCreating model...")
    model = create_model_from_config(config.model)
    print(f"{model.numParams()}")
    
    # Create dataloaders
    print("\nLoading data...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
    except FileNotFoundError as e:
        print(f"\nError: Data files not found!")
        print(f"Please run data preprocessing first:")
        print(f"  python data_preprocessing.py")
        return
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = FCLTrainer(config, model)
    
    # Train
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)
    
    print("\n" + "="*80)
    print(f"Experiment {experiment_name} completed!")
    print("="*80)


def run_experiment_suite(suite_filter: str = None):
    """Run multiple experiments from the suite"""
    
    experiments_to_run = []
    
    if suite_filter:
        # Filter experiments by name pattern
        experiments_to_run = [
            (name, config) for name, config in EXPERIMENT_SUITE.items()
            if suite_filter.lower() in name.lower()
        ]
    else:
        experiments_to_run = list(EXPERIMENT_SUITE.items())
    
    print("\n" + "="*80)
    print(f"Running Experiment Suite ({len(experiments_to_run)} experiments)")
    print("="*80)
    
    for i, (exp_name, _) in enumerate(experiments_to_run):
        print(f"\n[{i+1}/{len(experiments_to_run)}] {exp_name}")
    
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    # Run each experiment
    for i, (exp_name, _) in enumerate(experiments_to_run):
        print(f"\n\n{'='*80}")
        print(f"Experiment {i+1}/{len(experiments_to_run)}")
        print(f"{'='*80}")
        
        run_single_experiment(exp_name)
        
        # Clear GPU memory between experiments
        torch.cuda.empty_cache()


def preprocess_data(overlap_levels, epoch_keys, activ_keys, batch_size, batch_limit):
    """Run data preprocessing"""
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = DataPreprocessor(
        df_path="./data/Merged zoo.csv",
        scenario_path="./data/Scenario",
        overlap_levels=overlap_levels,
        batch_size=batch_size,
        batch_limit=batch_limit
    )
    
    preprocessor.process_all_scenarios(
        epoch_keys=epoch_keys,
        activ_keys=activ_keys
    )
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)


def list_experiments():
    """List all available experiments"""
    
    print("\n" + "="*80)
    print("AVAILABLE EXPERIMENTS")
    print("="*80)
    
    for exp_name, config in EXPERIMENT_SUITE.items():
        model_config = config.model
        print(f"\n{exp_name}:")
        print(f"  Model: {model_config.name}")
        print(f"  Params: ~{model_config.get_param_count_estimate()/1e6:.1f}M")
        print(f"  Overlap: {config.data.overlap_levels[0]}")
        print(f"  Loss: {config.loss.primary_loss}")
        print(f"  Epochs: {config.training.epochs}")


def list_models():
    """List all available model configurations"""
    
    print("\n" + "="*80)
    print("AVAILABLE MODEL CONFIGURATIONS")
    print("="*80)
    
    for name, config in MODEL_CONFIGS.items():
        params = config.get_param_count_estimate()
        print(f"\n{name}:")
        print(f"  Parameters: ~{params/1e6:.1f}M")
        print(f"  Layers: {config.N}")
        print(f"  Heads: {config.heads}")
        print(f"  d_model: {config.d_model}")
        print(f"  d_ff: {config.d_ff}")
        print(f"  Neck: {config.neck}")
        print(f"  Dropout: {config.dropout}")


def main():
    parser = argparse.ArgumentParser(
        description="Run FCL experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available experiments
  python run_experiments.py --list
  
  # List available models
  python run_experiments.py --list-models
  
  # Run data preprocessing
  python run_experiments.py --preprocess
  
  # Run single experiment
  python run_experiments.py --experiment quick_tiny_mse
  
  # Run experiment suite (all small models)
  python run_experiments.py --suite small
  
  # Create custom experiment
  python run_experiments.py --custom --model small --overlap 2 --loss mse --name my_experiment
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available model configurations')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run data preprocessing')
    parser.add_argument('--experiment', type=str,
                       help='Run specific experiment by name')
    parser.add_argument('--suite', type=str,
                       help='Run experiment suite (optionally filter by name)')
    parser.add_argument('--custom', action='store_true',
                       help='Create and run custom experiment')
    
    # Custom experiment options
    parser.add_argument('--model', type=str, default='small',
                       choices=['tiny', 'small', 'medium', 'large', 'huge'],
                       help='Model size for custom experiment')
    parser.add_argument('--overlap', type=int, default=2,
                       choices=[0, 1, 2],
                       help='Class overlap level')
    parser.add_argument('--loss', type=str, default='mse',
                       choices=['mse', 'mape', 'auto', 'lwwn', 'lwwn_ws', 'wasserstein', 'latent'],
                       help='Loss function')
    parser.add_argument('--name', type=str, default=None,
                       help='Custom experiment name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    # Preprocessing options
    parser.add_argument('--overlap-levels', type=int, nargs='+', default=[2, 1, 0],
                       help='Overlap levels for preprocessing')
    parser.add_argument('--epoch-keys', type=int, nargs='+', default=[10],
                       help='Epoch keys for preprocessing')
    parser.add_argument('--activ-keys', type=int, nargs='+', default=[2],
                       help='Activation keys for preprocessing')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Batch size for preprocessing')
    parser.add_argument('--batch-limit', type=int, default=100,
                       help='Batch limit for preprocessing')
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.list:
        list_experiments()
    
    elif args.list_models:
        list_models()
    
    elif args.preprocess:
        preprocess_data(
            overlap_levels=args.overlap_levels,
            epoch_keys=args.epoch_keys,
            activ_keys=args.activ_keys,
            batch_size=args.batch_size,
            batch_limit=args.batch_limit
        )
    
    elif args.experiment:
        run_single_experiment(args.experiment)
    
    elif args.suite is not None:
        run_experiment_suite(args.suite if args.suite else None)
    
    elif args.custom:
        # Create custom experiment
        exp_name = args.name or f"custom_{args.model}_{args.loss}_overlap{args.overlap}"
        config = create_experiment_config(
            model_size=args.model,
            overlap=args.overlap,
            primary_loss=args.loss,
            experiment_name=exp_name
        )
        config.training.epochs = args.epochs
        
        # Save to suite and run
        EXPERIMENT_SUITE[exp_name] = config
        run_single_experiment(exp_name)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
