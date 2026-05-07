"""
Comprehensive WandB integration for experiment tracking.
Logs metrics, distances, topological analysis, and creates tables.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("WandB not available. Logging disabled.")


class WandBLogger:
    """Comprehensive WandB logging for research experiments."""
    
    def __init__(self, 
                 project: str = "weight-space-research",
                 entity: Optional[str] = None,
                 name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 enabled: bool = True):
        """
        Initialize WandB logger.
        
        Args:
            project: WandB project name
            entity: WandB entity (username/team)
            name: Run name
            config: Configuration dictionary
            enabled: Whether to enable logging
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        
        if self.enabled:
            try:
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    config=config or {},
                    reinit=True
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize WandB: {e}")
                self.enabled = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log scalar metrics."""
        if not self.enabled:
            return
        
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            warnings.warn(f"Failed to log metrics: {e}")
    
    def log_distance_table(self, 
                          distance_metrics: Dict,
                          table_name: str = "distance_metrics"):
        """
        Log distance metrics as WandB table.
        
        Args:
            distance_metrics: Output from WeightDistanceMetrics.compute_all_metrics()
            table_name: Name for the table
        """
        if not self.enabled:
            return
        
        try:
            # Create full distances table
            full_data = []
            for metric_name, value in distance_metrics['full'].items():
                full_data.append([metric_name, float(value) if not np.isnan(value) else None])
            
            full_table = wandb.Table(
                columns=["Metric", "Value"],
                data=full_data
            )
            wandb.log({f"{table_name}_full": full_table})
            
            # Create layer-wise distances table
            layerwise_data = []
            for layer_name, layer_metrics in distance_metrics['layerwise'].items():
                row = [layer_name]
                for metric_name in ['euclidean', 'manhattan', 'cosine', 'relative_diff', 'mean_abs_diff']:
                    value = layer_metrics.get(metric_name, np.nan)
                    row.append(float(value) if not np.isnan(value) else None)
                layerwise_data.append(row)
            
            layerwise_table = wandb.Table(
                columns=["Layer", "Euclidean", "Manhattan", "Cosine", "Relative Diff", "Mean Abs Diff"],
                data=layerwise_data
            )
            wandb.log({f"{table_name}_layerwise": layerwise_table})
            
            # Log summary statistics
            summary_metrics = {
                f"{table_name}/summary/{k}": v 
                for k, v in distance_metrics['summary'].items()
            }
            wandb.log(summary_metrics)
            
        except Exception as e:
            warnings.warn(f"Failed to log distance table: {e}")
    
    def log_topology_metrics(self, topology_results: Dict):
        """
        Log topological analysis results.
        
        Args:
            topology_results: Output from safe_compute_topology_metrics()
        """
        if not self.enabled:
            return
        
        try:
            # Log Mapper results
            if topology_results.get('mapper') and 'error' not in topology_results['mapper']:
                mapper_data = topology_results['mapper']
                wandb.log({
                    'topology/mapper/n_nodes': mapper_data.get('n_nodes', 0),
                    'topology/mapper/n_edges': mapper_data.get('n_edges', 0),
                    'topology/mapper/n_intervals': mapper_data.get('n_intervals', 0),
                    'topology/mapper/overlap': mapper_data.get('overlap', 0)
                })
                
                # Create Mapper graph table
                if mapper_data.get('nodes'):
                    node_data = [
                        [node['id'], node['size'], node.get('interval', -1), node.get('cluster', -1)]
                        for node in mapper_data['nodes']
                    ]
                    node_table = wandb.Table(
                        columns=["Node ID", "Size", "Interval", "Cluster"],
                        data=node_data
                    )
                    wandb.log({"topology/mapper/nodes": node_table})
            
            # Log Persistent Homology results
            if topology_results.get('persistence') and 'error' not in topology_results['persistence']:
                ph_data = topology_results['persistence']
                if ph_data.get('stats'):
                    ph_metrics = {
                        f'topology/persistence/{k}': v 
                        for k, v in ph_data['stats'].items()
                    }
                    wandb.log(ph_metrics)
                    
                    # Create Betti numbers table
                    betti_data = [
                        [dim, ph_data['stats'].get(f'betti_{dim}', 0)]
                        for dim in range(3)
                        if f'betti_{dim}' in ph_data['stats']
                    ]
                    if betti_data:
                        betti_table = wandb.Table(
                            columns=["Dimension", "Betti Number"],
                            data=betti_data
                        )
                        wandb.log({"topology/persistence/betti_numbers": betti_table})
            
            # Log errors if any
            if topology_results.get('errors'):
                wandb.log({
                    'topology/errors': len(topology_results['errors']),
                    'topology/error_messages': ', '.join(topology_results['errors'])
                })
                
        except Exception as e:
            warnings.warn(f"Failed to log topology metrics: {e}")
    
    def log_training_progress(self, 
                             epoch: int,
                             train_loss: float,
                             val_loss: float,
                             learning_rate: float,
                             additional_metrics: Optional[Dict] = None):
        """Log training progress metrics."""
        if not self.enabled:
            return
        
        try:
            metrics = {
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'train/learning_rate': learning_rate
            }
            
            if additional_metrics:
                metrics.update(additional_metrics)
            
            wandb.log(metrics, step=epoch)
            
        except Exception as e:
            warnings.warn(f"Failed to log training progress: {e}")
    
    def log_model_artifact(self, 
                          checkpoint_path: Path,
                          artifact_name: str = "model",
                          metadata: Optional[Dict] = None):
        """Log model checkpoint as WandB artifact."""
        if not self.enabled:
            return
        
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type='model',
                metadata=metadata or {}
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
            
        except Exception as e:
            warnings.warn(f"Failed to log model artifact: {e}")
    
    def log_markdown_table(self, markdown_text: str, key: str = "distance_table"):
        """Log markdown formatted table."""
        if not self.enabled:
            return
        
        try:
            wandb.log({key: wandb.Html(f"<pre>{markdown_text}</pre>")})
        except Exception as e:
            warnings.warn(f"Failed to log markdown table: {e}")
    
    def save_markdown_table(self, markdown_text: str, filename: str):
        """Save markdown table to file and log to WandB."""
        try:
            # Save to file
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(markdown_text)
            
            # Log to WandB
            if self.enabled:
                wandb.save(str(output_path))
                
        except Exception as e:
            warnings.warn(f"Failed to save markdown table: {e}")
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
            except Exception as e:
                warnings.warn(f"Failed to finish WandB run: {e}")


def create_experiment_summary(
    model_config: Dict,
    training_history: Dict,
    distance_metrics: Dict,
    topology_results: Dict
) -> str:
    """
    Create comprehensive experiment summary in markdown format.
    
    Returns:
        Markdown formatted summary
    """
    lines = []
    
    lines.append("# Experiment Summary")
    lines.append("")
    
    # Model configuration
    lines.append("## Model Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, value in model_config.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    
    # Training results
    lines.append("## Training Results")
    lines.append("")
    if training_history:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Epochs | {len(training_history.get('train_loss', []))} |")
        if training_history.get('train_loss'):
            lines.append(f"| Final Train Loss | {training_history['train_loss'][-1]:.6f} |")
            lines.append(f"| Best Train Loss | {min(training_history['train_loss']):.6f} |")
        if training_history.get('val_loss'):
            lines.append(f"| Final Val Loss | {training_history['val_loss'][-1]:.6f} |")
            lines.append(f"| Best Val Loss | {min(training_history['val_loss']):.6f} |")
    lines.append("")
    
    # Distance metrics
    if distance_metrics:
        lines.append("## Distance Metrics")
        lines.append("")
        lines.append("### Full Vector Distances")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for metric, value in sorted(distance_metrics.get('full', {}).items()):
            lines.append(f"| {metric} | {value:.6f} |")
        lines.append("")
        
        lines.append("### Layer-wise Distances")
        lines.append("")
        lines.append("| Layer | Euclidean | Manhattan | Cosine |")
        lines.append("|-------|-----------|-----------|--------|")
        for layer, metrics in distance_metrics.get('layerwise', {}).items():
            lines.append(
                f"| {layer} | "
                f"{metrics.get('euclidean', np.nan):.6f} | "
                f"{metrics.get('manhattan', np.nan):.6f} | "
                f"{metrics.get('cosine', np.nan):.6f} |"
            )
        lines.append("")
    
    # Topology results
    if topology_results:
        lines.append("## Topological Analysis")
        lines.append("")
        
        if topology_results.get('mapper'):
            mapper = topology_results['mapper']
            if 'error' not in mapper:
                lines.append("### Mapper Algorithm")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Nodes | {mapper.get('n_nodes', 0)} |")
                lines.append(f"| Edges | {mapper.get('n_edges', 0)} |")
                lines.append(f"| Intervals | {mapper.get('n_intervals', 0)} |")
                lines.append("")
        
        if topology_results.get('persistence'):
            ph = topology_results['persistence']
            if 'error' not in ph and ph.get('stats'):
                lines.append("### Persistent Homology")
                lines.append("")
                lines.append("| Dimension | Betti Number | Max Lifetime | Mean Lifetime |")
                lines.append("|-----------|--------------|--------------|---------------|")
                for dim in range(3):
                    if f'betti_{dim}' in ph['stats']:
                        lines.append(
                            f"| {dim} | "
                            f"{ph['stats'].get(f'betti_{dim}', 0)} | "
                            f"{ph['stats'].get(f'max_lifetime_{dim}', 0):.6f} | "
                            f"{ph['stats'].get(f'mean_lifetime_{dim}', 0):.6f} |"
                        )
                lines.append("")
    
    return "\n".join(lines)
