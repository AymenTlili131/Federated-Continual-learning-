"""
Comprehensive loss function library for FCL weight prediction
Implements: MSE, AUTO, LWWN, LWWN-WS, Wasserstein, Latent, MAPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Optional, Dict, Tuple


class MSELoss(nn.Module):
    """Standard Mean Squared Error Loss"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(predicted, target)


class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error Loss"""
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Add epsilon to avoid division by zero
        return torch.mean(torch.abs((target - predicted) / (torch.abs(target) + self.epsilon)))


class AutoencoderLoss(nn.Module):
    """
    Autoencoder-style loss with reconstruction + regularization
    Combines MSE with L1/L2 regularization on latent space
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.01, reg_type: str = "l2"):
        super().__init__()
        self.alpha = alpha  # Reconstruction weight
        self.beta = beta    # Regularization weight
        self.reg_type = reg_type
        self.mse = nn.MSELoss()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        reconstruction_loss = self.mse(predicted, target)
        
        if latent is not None:
            if self.reg_type == "l1":
                reg_loss = torch.mean(torch.abs(latent))
            else:  # l2
                reg_loss = torch.mean(latent ** 2)
            
            total_loss = self.alpha * reconstruction_loss + self.beta * reg_loss
        else:
            total_loss = reconstruction_loss
        
        return total_loss


class LWWNLoss(nn.Module):
    """
    Layer-Wise Weighted Norm Loss
    Applies different weights to different layers of the CNN
    """
    def __init__(self, layer_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        # Default layer weights for the CNN architecture
        # Based on Silu.py: Conv layers, Linear layers
        if layer_weights is None:
            self.layer_weights = {
                'conv1': 1.5,  # 200 params (8*5*5*1)
                'conv2': 1.5,  # 1200 params (6*5*5*8)
                'conv3': 1.5,  # 96 params (4*2*2*6)
                'fc1': 1.0,    # 720 params (20*36)
                'fc2': 1.0,    # 200 params (10*20)
            }
        else:
            self.layer_weights = layer_weights
        
        # Define layer boundaries in flattened weight vector (2464 total)
        self.layer_ranges = {
            'conv1': (0, 200),           # weights
            'conv1_bias': (200, 208),    # bias
            'conv2': (208, 1408),        # weights
            'conv2_bias': (1408, 1414),  # bias
            'conv3': (1414, 1510),       # weights
            'conv3_bias': (1510, 1514),  # bias
            'fc1': (1514, 2234),         # weights
            'fc1_bias': (2234, 2254),    # bias
            'fc2': (2254, 2454),         # weights
            'fc2_bias': (2454, 2464),    # bias
        }
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for layer_name, (start, end) in self.layer_ranges.items():
            # Extract layer weights
            pred_layer = predicted[:, start:end]
            target_layer = target[:, start:end]
            
            # Compute MSE for this layer
            layer_loss = F.mse_loss(pred_layer, target_layer)
            
            # Apply layer weight (use default 1.0 if not specified)
            base_layer = layer_name.replace('_bias', '')
            weight = self.layer_weights.get(base_layer, 1.0)
            
            total_loss += weight * layer_loss
        
        # Normalize by number of layers
        return total_loss / len(self.layer_ranges)


class LWWNWSLoss(nn.Module):
    """
    Layer-Wise Weighted Norm with Weight Standardization
    Normalizes weights before computing loss
    """
    def __init__(self, layer_weights: Optional[Dict[str, float]] = None, 
                 standardize: bool = True):
        super().__init__()
        self.lwwn = LWWNLoss(layer_weights)
        self.standardize = standardize
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.standardize:
            # Standardize each sample in the batch
            pred_mean = predicted.mean(dim=1, keepdim=True)
            pred_std = predicted.std(dim=1, keepdim=True) + 1e-8
            predicted_norm = (predicted - pred_mean) / pred_std
            
            target_mean = target.mean(dim=1, keepdim=True)
            target_std = target.std(dim=1, keepdim=True) + 1e-8
            target_norm = (target - target_mean) / target_std
            
            return self.lwwn(predicted_norm, target_norm)
        else:
            return self.lwwn(predicted, target)


class WassersteinLoss(nn.Module):
    """
    Wasserstein Distance Loss (Earth Mover's Distance)
    Computes 1D Wasserstein distance between weight distributions
    """
    def __init__(self, num_bins: int = 100, use_sliced: bool = True, 
                 num_projections: int = 50):
        super().__init__()
        self.num_bins = num_bins
        self.use_sliced = use_sliced
        self.num_projections = num_projections
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = predicted.size(0)
        
        if self.use_sliced:
            # Sliced Wasserstein Distance (faster, differentiable)
            return self._sliced_wasserstein(predicted, target)
        else:
            # Standard 1D Wasserstein (slower but exact)
            losses = []
            for i in range(batch_size):
                pred_sorted, _ = torch.sort(predicted[i])
                target_sorted, _ = torch.sort(target[i])
                loss = torch.mean(torch.abs(pred_sorted - target_sorted))
                losses.append(loss)
            return torch.stack(losses).mean()
    
    def _sliced_wasserstein(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Sliced Wasserstein Distance using random projections"""
        batch_size, dim = predicted.shape
        device = predicted.device
        
        # Generate random projection directions
        projections = torch.randn(self.num_projections, dim, device=device)
        projections = F.normalize(projections, dim=1)
        
        total_distance = 0.0
        
        for proj in projections:
            # Project onto random direction
            pred_proj = torch.matmul(predicted, proj)
            target_proj = torch.matmul(target, proj)
            
            # Sort projections
            pred_sorted, _ = torch.sort(pred_proj, dim=0)
            target_sorted, _ = torch.sort(target_proj, dim=0)
            
            # Compute 1D Wasserstein distance
            distance = torch.mean(torch.abs(pred_sorted - target_sorted))
            total_distance += distance
        
        return total_distance / self.num_projections


class LatentLoss(nn.Module):
    """
    Latent Space Loss
    Encourages smooth and structured latent representations
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.01):
        super().__init__()
        self.alpha = alpha  # Reconstruction weight
        self.beta = beta    # Latent smoothness weight
        self.gamma = gamma  # Latent sparsity weight
        self.mse = nn.MSELoss()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor,
                latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = self.mse(predicted, target)
        
        if latent is None:
            return recon_loss
        
        # Latent smoothness: minimize variance in latent space
        latent_mean = latent.mean(dim=0, keepdim=True)
        smoothness_loss = torch.mean((latent - latent_mean) ** 2)
        
        # Latent sparsity: encourage sparse representations
        sparsity_loss = torch.mean(torch.abs(latent))
        
        total_loss = (self.alpha * recon_loss + 
                     self.beta * smoothness_loss + 
                     self.gamma * sparsity_loss)
        
        return total_loss


class MultiLoss(nn.Module):
    """
    Combined multi-objective loss function
    Tracks and combines multiple loss functions
    """
    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # Default weights
        if loss_weights is None:
            loss_weights = {
                'mse': 1.0,
                'mape': 0.1,
                'wasserstein': 0.05,
                'latent': 0.1,
                'lwwn': 0.0,
            }
        
        self.loss_weights = loss_weights
        
        # Initialize all loss functions
        self.losses = {
            'mse': MSELoss(),
            'mape': MAPELoss(),
            'auto': AutoencoderLoss(),
            'lwwn': LWWNLoss(),
            'lwwn_ws': LWWNWSLoss(),
            'wasserstein': WassersteinLoss(),
            'latent': LatentLoss(),
        }
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor,
                latent: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all losses and return weighted sum + individual losses
        
        Returns:
            total_loss: Weighted combination of all losses
            loss_dict: Dictionary of individual loss values
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Compute each loss
        for loss_name, loss_fn in self.losses.items():
            weight = self.loss_weights.get(loss_name, 0.0)
            
            if weight > 0:
                if loss_name in ['auto', 'latent'] and latent is not None:
                    loss_value = loss_fn(predicted, target, latent)
                else:
                    loss_value = loss_fn(predicted, target)
                
                loss_dict[loss_name] = loss_value.item()
                total_loss += weight * loss_value
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def get_loss_function(loss_name: str, **kwargs):
    """
    Factory function to get loss function by name
    
    Args:
        loss_name: Name of loss function (mse, mape, auto, lwwn, lwwn_ws, wasserstein, latent, multi)
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    loss_map = {
        'mse': MSELoss,
        'mape': MAPELoss,
        'auto': AutoencoderLoss,
        'lwwn': LWWNLoss,
        'lwwn_ws': LWWNWSLoss,
        'wasserstein': WassersteinLoss,
        'latent': LatentLoss,
        'multi': MultiLoss,
    }
    
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Available: {list(loss_map.keys())}")
    
    return loss_map[loss_name](**kwargs)


if __name__ == "__main__":
    # Test all loss functions
    print("Testing Loss Functions")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    weight_dim = 2464
    latent_dim = 64
    
    predicted = torch.randn(batch_size, weight_dim)
    target = torch.randn(batch_size, weight_dim)
    latent = torch.randn(batch_size, latent_dim)
    
    # Test each loss
    losses_to_test = ['mse', 'mape', 'auto', 'lwwn', 'lwwn_ws', 'wasserstein', 'latent']
    
    for loss_name in losses_to_test:
        loss_fn = get_loss_function(loss_name)
        
        if loss_name in ['auto', 'latent']:
            loss_value = loss_fn(predicted, target, latent)
        else:
            loss_value = loss_fn(predicted, target)
        
        print(f"{loss_name:15s}: {loss_value.item():.6f}")
    
    # Test multi-loss
    print("\nTesting Multi-Loss:")
    multi_loss = MultiLoss()
    total_loss, loss_dict = multi_loss(predicted, target, latent)
    
    for name, value in loss_dict.items():
        print(f"  {name:15s}: {value:.6f}")
