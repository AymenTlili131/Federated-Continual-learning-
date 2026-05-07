"""Visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from pathlib import Path

def create_attention_heatmap(attention_weights: np.ndarray, save_path: Optional[Path] = None):
    """Create attention heatmap visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis', ax=ax)
    ax.set_title('Attention Weights')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig

def plot_eigenvalue_evolution(eigenvalues_history: List, save_path: Optional[Path] = None):
    """Plot eigenvalue evolution over epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, eigs in enumerate(eigenvalues_history):
        ax.plot(eigs, alpha=0.5, label=f'Epoch {i}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue Evolution')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig

def plot_betti_curves(persistence_diagrams: List, save_path: Optional[Path] = None):
    """Plot Betti curves from persistence diagrams."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Betti Curves')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig

def create_comparison_grid(images: List, titles: List, save_path: Optional[Path] = None):
    """Create comparison grid of images."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig
