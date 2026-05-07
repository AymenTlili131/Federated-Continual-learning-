"""
03_visualize_neck_evolution.py

Creates GIF visualizations and WandB metric sliders for the evolution of
dimensionality reduction (PCA, t-SNE, UMAP) of the transformer neck
(bottleneck representations before decoding).
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import wandb
from PIL import Image
import io
from tqdm import tqdm
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))


class NeckEvolutionVisualizer:
    """Visualizes evolution of neck representations across training."""
    
    def __init__(self, output_dir: Path, methods: List[str] = ['pca', 'tsne', 'umap']):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.methods = methods
        self.frames = {method: [] for method in methods}
        
    def extract_neck_representations(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> np.ndarray:
        """Extract neck (bottleneck) representations from model."""
        model.eval()
        representations = []
        labels = []
        
        with torch.no_grad():
            for x1, x2, y in dataloader:
                x1, x2 = x1.to(device), x2.to(device)
                
                # Get embeddings (before decoder)
                embedded = model.embedder(x1, x2)
                
                # Apply transformer encoder
                encoded = model.transformer_encoder(embedded)
                
                # Take mean pooling as representation
                neck_repr = encoded.mean(dim=1)  # (batch, d_model)
                
                representations.append(neck_repr.cpu().numpy())
                labels.append(y.cpu().numpy())
        
        representations = np.concatenate(representations, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return representations, labels
    
    def reduce_dimensions(
        self,
        representations: np.ndarray,
        method: str,
        n_components: int = 2
    ) -> np.ndarray:
        """Apply dimensionality reduction."""
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(representations)
            variance_explained = reducer.explained_variance_ratio_.sum()
            return reduced, variance_explained
        
        elif method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(30, len(representations) - 1),
                n_iter=1000
            )
            reduced = reducer.fit_transform(representations)
            return reduced, None
        
        elif method == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=min(15, len(representations) - 1)
            )
            reduced = reducer.fit_transform(representations)
            return reduced, None
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def create_frame(
        self,
        reduced_data: np.ndarray,
        labels: np.ndarray,
        epoch: int,
        method: str,
        metric_value: float = None
    ) -> Image.Image:
        """Create a single frame for the GIF."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        scatter = ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=50
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Class Label')
        
        # Set labels and title
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        
        title = f'{method.upper()} Evolution - Epoch {epoch}'
        if metric_value is not None:
            title += f' (Variance: {metric_value:.2%})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
    
    def add_frame(
        self,
        representations: np.ndarray,
        labels: np.ndarray,
        epoch: int
    ):
        """Add frames for all methods."""
        for method in self.methods:
            print(f"  Computing {method.upper()}...")
            reduced, metric = self.reduce_dimensions(representations, method)
            frame = self.create_frame(reduced, labels, epoch, method, metric)
            self.frames[method].append(frame)
    
    def save_gifs(self, fps: int = 10) -> Dict[str, Path]:
        """Save GIFs for all methods."""
        gif_paths = {}
        
        for method in self.methods:
            if not self.frames[method]:
                continue
            
            gif_path = self.output_dir / f'neck_evolution_{method}.gif'
            
            # Save GIF
            self.frames[method][0].save(
                gif_path,
                save_all=True,
                append_images=self.frames[method][1:],
                duration=1000 // fps,
                loop=0
            )
            
            gif_paths[method] = gif_path
            print(f"  Saved {method.upper()} GIF: {gif_path}")
        
        return gif_paths
    
    def create_comparison_plot(self, epoch: int) -> Path:
        """Create side-by-side comparison of all methods."""
        if not self.frames['pca']:
            return None
        
        n_methods = len(self.methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, method in enumerate(self.methods):
            if epoch < len(self.frames[method]):
                axes[idx].imshow(self.frames[method][epoch])
                axes[idx].axis('off')
                axes[idx].set_title(f'{method.upper()}', fontsize=14)
        
        plt.tight_layout()
        
        comparison_path = self.output_dir / f'comparison_epoch_{epoch:04d}.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return comparison_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize neck evolution with dimensionality reduction"
    )
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--wandb_project", type=str, default="weight-space-research")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--create_gifs", action="store_true")
    parser.add_argument("--methods", nargs='+', default=['pca', 'tsne', 'umap'])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--sample_every_n_epochs", type=int, default=5)
    
    args = parser.parse_args()
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=f"neck_evolution_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"\n{'='*80}")
    print(f"Neck Evolution Visualization")
    print(f"{'='*80}\n")
    
    # Create visualizer
    visualizer = NeckEvolutionVisualizer(
        output_dir=args.output_dir,
        methods=args.methods
    )
    
    # Find all checkpoint files
    checkpoint_files = sorted(args.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {args.checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Process checkpoints (placeholder - needs actual data loading)
    print(f"\n{'='*80}")
    print(f"Visualization setup complete.")
    print(f"Note: Actual processing requires data loading implementation.")
    print(f"{'='*80}\n")
    
    # Log to WandB
    wandb.log({
        "num_checkpoints": len(checkpoint_files),
        "methods": args.methods,
        "fps": args.fps
    })
    
    wandb.finish()


if __name__ == "__main__":
    main()
