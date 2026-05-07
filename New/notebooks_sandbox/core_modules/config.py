"""
Configuration system for FCL experiments
Supports multiple model sizes, loss functions, and hyperparameters
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class ModelConfig:
    """Configuration for TransformerAE model variants"""
    name: str
    max_seq_len: int = 50
    N: int = 2  # Number of encoder/decoder layers
    heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    neck: int = 64
    dropout: float = 0.1
    
    def get_param_count_estimate(self):
        """Rough estimate of parameter count"""
        # Embedding layers
        embed_params = (16 * self.d_model * 24) + (80 * self.d_model * 26)
        # Encoder layers (2 encoders)
        enc_params = 2 * self.N * (
            4 * self.d_model * self.d_model +  # Q,K,V,O projections
            2 * self.d_model * self.d_ff +      # FFN
            4 * self.d_model                     # Layer norms
        )
        # Decoder layers
        dec_params = self.N * (
            4 * self.d_model * self.d_model +
            2 * self.d_model * self.d_ff +
            4 * self.d_model
        )
        # Neck and output
        neck_params = (2 * self.d_ff * self.neck) + (self.neck * self.d_model * self.max_seq_len)
        output_params = self.max_seq_len * self.d_model * 2464
        
        total = embed_params + enc_params + dec_params + neck_params + output_params
        return total


# Model size variants optimized for RTX 5060 Ti (16GB VRAM)
MODEL_CONFIGS = {
    # Tiny model - ~500K params - for rapid prototyping
    "tiny": ModelConfig(
        name="tiny",
        max_seq_len=50,
        N=1,
        heads=2,
        d_model=32,
        d_ff=128,
        neck=16,
        dropout=0.1
    ),
    
    # Small model - ~2M params - fast training
    "small": ModelConfig(
        name="small",
        max_seq_len=50,
        N=2,
        heads=4,
        d_model=64,
        d_ff=256,
        neck=32,
        dropout=0.1
    ),
    
    # Medium model - ~8M params - balanced
    "medium": ModelConfig(
        name="medium",
        max_seq_len=50,
        N=3,
        heads=4,
        d_model=128,
        d_ff=512,
        neck=64,
        dropout=0.15
    ),
    
    # Large model - ~25M params - high capacity
    "large": ModelConfig(
        name="large",
        max_seq_len=50,
        N=4,
        heads=8,
        d_model=256,
        d_ff=1024,
        neck=128,
        dropout=0.2
    ),
    
    # Huge model - ~100M params - high capacity research
    "huge": ModelConfig(
        name="huge",
        max_seq_len=50,
        N=6,
        heads=12,
        d_model=384,
        d_ff=1536,
        neck=192,
        dropout=0.2
    )
}


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    batch_size: int = 32
    batch_limit: int = 100
    overlap_levels: List[int] = field(default_factory=lambda: [2, 1, 0])  # Class overlap
    epoch_key: int = 10
    activ_key: int = 2  # 0:gelu, 1:relu, 2:silu, 3:leakyrelu, 4:sigmoid, 5:tanh
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    df_path: str = "./data/Merged zoo.csv"
    scenario_path: str = "./data/Scenario"
    
    def get_activ_name(self):
        activ_map = {0: "gelu", 1: "relu", 2: "silu", 3: "leakyrelu", 4: "sigmoid", 5: "tanh"}
        return activ_map.get(self.activ_key, "silu")


@dataclass
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    early_stopping_patience: int = 15
    save_every: int = 5
    log_every: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Use AMP for faster training
    num_workers: int = 4
    pin_memory: bool = True
    

@dataclass
class LossConfig:
    """Configuration for loss functions"""
    primary_loss: str = "mse"  # mse, auto, lwwn, lwwn_ws, wasserstein, latent, mape
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "reconstruction": 1.0,
        "latent": 0.1,
        "wasserstein": 0.05,
        "mape": 0.0
    })
    use_multi_loss: bool = True  # Track multiple losses simultaneously


@dataclass
class MetricsConfig:
    """Configuration for metrics and analysis"""
    track_persistent_homology: bool = True
    track_rmt: bool = True  # Random Matrix Theory metrics
    track_eigenvalues: bool = True
    track_spectral_density: bool = True
    compute_betti_curves: bool = True
    compute_persistence_diagrams: bool = True
    max_homology_dim: int = 2  # Compute H0, H1, H2
    rmt_window_size: int = 100  # For spectral analysis
    

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    metrics: MetricsConfig
    seed: int = 42
    wandb_project: str = "fcl-optimization"
    wandb_entity: Optional[str] = None
    save_dir: str = "./experiments"
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "name": self.name,
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "loss": self.loss.__dict__,
            "metrics": self.metrics.__dict__,
            "seed": self.seed
        }
    
    def save(self, path: str):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # Reconstruct nested configs
        return cls(**config_dict)


def create_experiment_config(
    model_size: str = "small",
    overlap: int = 2,
    primary_loss: str = "mse",
    experiment_name: Optional[str] = None
) -> ExperimentConfig:
    """Factory function to create experiment configurations"""
    
    if experiment_name is None:
        experiment_name = f"{model_size}_overlap{overlap}_{primary_loss}"
    
    model_config = MODEL_CONFIGS[model_size]
    data_config = DataConfig(overlap_levels=[overlap])
    training_config = TrainingConfig()
    loss_config = LossConfig(primary_loss=primary_loss)
    metrics_config = MetricsConfig()
    
    return ExperimentConfig(
        name=experiment_name,
        model=model_config,
        data=data_config,
        training=training_config,
        loss=loss_config,
        metrics=metrics_config
    )


# Predefined experiment suite
EXPERIMENT_SUITE = {
    # Quick validation experiments
    "quick_tiny_mse": create_experiment_config("tiny", 2, "mse", "quick_tiny_mse"),
    "quick_tiny_wasserstein": create_experiment_config("tiny", 2, "wasserstein", "quick_tiny_wasserstein"),
    
    # Small model experiments across loss functions
    "small_mse_overlap2": create_experiment_config("small", 2, "mse"),
    "small_mse_overlap1": create_experiment_config("small", 1, "mse"),
    "small_mse_overlap0": create_experiment_config("small", 0, "mse"),
    
    "small_wasserstein_overlap2": create_experiment_config("small", 2, "wasserstein"),
    "small_lwwn_overlap2": create_experiment_config("small", 2, "lwwn"),
    "small_mape_overlap2": create_experiment_config("small", 2, "mape"),
    
    # Medium model - best performers
    "medium_multi_overlap2": create_experiment_config("medium", 2, "mse"),
    "medium_multi_overlap1": create_experiment_config("medium", 1, "mse"),
    "medium_multi_overlap0": create_experiment_config("medium", 0, "mse"),
    
    # Large model - final comparison
    "large_best_overlap0": create_experiment_config("large", 0, "mse"),
}


def print_config_summary(config: ExperimentConfig):
    """Print a summary of the experiment configuration"""
    print(f"\n{'='*60}")
    print(f"Experiment: {config.name}")
    print(f"{'='*60}")
    print(f"\nModel: {config.model.name}")
    print(f"  - Estimated params: {config.model.get_param_count_estimate()/1e6:.2f}M")
    print(f"  - Layers: {config.model.N}, Heads: {config.model.heads}")
    print(f"  - d_model: {config.model.d_model}, d_ff: {config.model.d_ff}")
    print(f"  - Neck: {config.model.neck}, Dropout: {config.model.dropout}")
    
    print(f"\nData:")
    print(f"  - Batch size: {config.data.batch_size}")
    print(f"  - Overlap levels: {config.data.overlap_levels}")
    print(f"  - Activation: {config.data.get_activ_name()}")
    
    print(f"\nTraining:")
    print(f"  - Epochs: {config.training.epochs}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Device: {config.training.device}")
    print(f"  - Mixed precision: {config.training.mixed_precision}")
    
    print(f"\nLoss:")
    print(f"  - Primary: {config.loss.primary_loss}")
    print(f"  - Multi-loss: {config.loss.use_multi_loss}")
    
    print(f"\nMetrics:")
    print(f"  - Persistent Homology: {config.metrics.track_persistent_homology}")
    print(f"  - RMT: {config.metrics.track_rmt}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Print all model configurations
    print("\nAvailable Model Configurations:")
    print("="*60)
    for name, config in MODEL_CONFIGS.items():
        params = config.get_param_count_estimate()
        print(f"{name:10s}: {params/1e6:8.2f}M params | "
              f"N={config.N}, heads={config.heads}, "
              f"d_model={config.d_model}, d_ff={config.d_ff}")
    
    # Example experiment
    print("\n\nExample Experiment Configuration:")
    exp_config = create_experiment_config("small", 2, "mse")
    print_config_summary(exp_config)
