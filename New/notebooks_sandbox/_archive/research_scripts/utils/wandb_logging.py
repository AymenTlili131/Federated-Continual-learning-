"""WandB logging utilities."""

import wandb
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

def log_attention_maps(attention_weights: np.ndarray, epoch: int, name: str = "attention"):
    """Log attention maps to WandB."""
    wandb.log({
        f"{name}/epoch_{epoch}": wandb.Image(attention_weights),
        "epoch": epoch
    })

def log_eigenvalues_table(eigenvalues: np.ndarray, labels: List[str], opacity: float = 1.0):
    """Log eigenvalues as WandB table with opacity."""
    table = wandb.Table(columns=["index", "eigenvalue", "label", "opacity"])
    for i, (eig, label) in enumerate(zip(eigenvalues, labels)):
        table.add_data(i, float(eig), label, opacity)
    wandb.log({"eigenvalues_table": table})

def log_gif(gif_path: Path, name: str):
    """Log GIF to WandB."""
    if gif_path.exists():
        wandb.log({name: wandb.Video(str(gif_path))})

def log_metric_slider(values: List[float], name: str):
    """Log metric as slider in WandB."""
    for epoch, value in enumerate(values):
        wandb.log({name: value, "epoch": epoch})
