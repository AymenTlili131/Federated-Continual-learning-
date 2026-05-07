"""
Comprehensive loss functions extracted from meta.ipynb
Includes all 23+ loss functions with layerwise variants and loss pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple, Optional
import warnings

try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    warnings.warn("geomloss not available. Sinkhorn loss disabled.")


# ============================================================================
# BASIC LOSSES
# ============================================================================

class MSELoss(nn.Module):
    """Mean Squared Error"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)


class MAELoss(nn.Module):
    """Mean Absolute Error (L1)"""
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.mae(pred, target)


class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        return torch.mean(torch.abs((pred - target) / (torch.abs(target) + self.epsilon))) * 100


# ============================================================================
# QUANTILE LOSSES
# ============================================================================

class QuantileLoss(nn.Module):
    """Q-quantile loss"""
    def __init__(self, q=0.5):
        super().__init__()
        self.q = q
    
    def forward(self, pred, target):
        error = target - pred
        return torch.mean(torch.maximum(self.q * error, (self.q - 1) * error))


# ============================================================================
# AUTOREGRESSIVE LOSS
# ============================================================================

class AutoregressiveMSELoss(nn.Module):
    """
    Autoregressive MSE loss with layer-wise weighting.
    Delimiters: [208, 1414, 1514, 2254, 2464] for CNN layers
    """
    def __init__(self, lambda_weights=None, epsilon=1e-8):
        super().__init__()
        if lambda_weights is None:
            lambda_weights = [10, 5, 2, 1, 0.5]
        self.lambda_weights = lambda_weights
        self.epsilon = epsilon
    
    def forward(self, predictions, targets, delimiters=None):
        if delimiters is None:
            delimiters = [208, 1414, 1514, 2254, 2464]
        
        n_chunks = len(delimiters)
        mse_losses = []
        start = 0
        
        for end in delimiters:
            pred_chunk = predictions[:, start:end]
            target_chunk = targets[:, start:end]
            mse_loss = F.mse_loss(pred_chunk, target_chunk, reduction='mean')
            mse_losses.append(mse_loss)
            start = end
        
        # Weighted autoregressive loss
        autoregressive_loss = mse_losses[0]
        for i in range(1, n_chunks):
            autoregressive_loss += (
                self.lambda_weights[i - 1] * (mse_losses[i] / (mse_losses[i - 1] + self.epsilon))
            )
        
        return autoregressive_loss


# ============================================================================
# WASSERSTEIN LOSSES
# ============================================================================

class SinkhornLoss(nn.Module):
    """Sinkhorn/Wasserstein loss using geomloss"""
    def __init__(self):
        super().__init__()
        if GEOMLOSS_AVAILABLE:
            self.loss = SamplesLoss('sinkhorn')
        else:
            self.loss = None
    
    def forward(self, pred, target):
        if self.loss is None:
            # Fallback to scipy wasserstein
            return self._scipy_wasserstein(pred, target)
        return self.loss(pred, target)
    
    def _scipy_wasserstein(self, pred, target):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        wsd_list = [wasserstein_distance(pred_np[i], target_np[i]) for i in range(pred_np.shape[0])]
        return torch.tensor(sum(wsd_list) / len(wsd_list), device=pred.device)


# WassersteinScipyLoss removed - not differentiable, no backprop support


# ============================================================================
# LAYER-WISE LOSSES
# ============================================================================

class LWLNLoss(nn.Module):
    """Layer-Wise Loss Normalization"""
    def __init__(self, delimiters=None, epsilon=1e-8):
        super().__init__()
        if delimiters is None:
            delimiters = [208, 1414, 1514, 2254, 2464]
        self.delimiters = delimiters
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        total_loss = 0.0
        start = 0
        
        for end in self.delimiters:
            pred_chunk = pred[:, start:end]
            target_chunk = target[:, start:end]
            
            # Normalize by standard deviation of target
            std = torch.std(target_chunk, dim=-1, keepdim=True)
            normalized_loss = torch.mean(torch.abs(pred_chunk - target_chunk) / (std + self.epsilon))
            total_loss += normalized_loss
            start = end
        
        return total_loss / len(self.delimiters)


class LWWSLoss(nn.Module):
    """Layer-Wise Wasserstein Loss"""
    def __init__(self, delimiters=None):
        super().__init__()
        if delimiters is None:
            delimiters = [208, 1414, 1514, 2254, 2464]
        self.delimiters = delimiters
    
    def forward(self, pred, target):
        total_loss = 0.0
        start = 0
        
        for end in self.delimiters:
            pred_chunk = pred[:, start:end]
            target_chunk = target[:, start:end]
            
            # Wasserstein per layer
            pred_np = pred_chunk.detach().cpu().numpy()
            target_np = target_chunk.detach().cpu().numpy()
            
            layer_wsd = []
            for i in range(pred_np.shape[0]):
                wsd = wasserstein_distance(pred_np[i], target_np[i])
                layer_wsd.append(wsd)
            
            total_loss += sum(layer_wsd) / len(layer_wsd)
            start = end
        
        return torch.tensor(total_loss / len(self.delimiters), device=pred.device)


# ============================================================================
# SPECTRAL/FREQUENCY LOSSES
# ============================================================================

class FFTLoss(nn.Module):
    """FFT-based frequency domain loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Compute FFT
        pred_fft = torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)
        
        # L2 loss in frequency domain
        return torch.mean(torch.abs(pred_fft - target_fft) ** 2)


class MelSpecL2Loss(nn.Module):
    """Mel-spectrogram L2 loss (adapted for weight vectors)"""
    def __init__(self, n_mels=128):
        super().__init__()
        self.n_mels = n_mels
    
    def forward(self, pred, target):
        # Simplified mel-like transformation
        # Reshape to 2D for mel-like processing
        batch_size = pred.shape[0]
        
        # Apply log-scale binning
        pred_log = torch.log(torch.abs(pred) + 1e-8)
        target_log = torch.log(torch.abs(target) + 1e-8)
        
        return F.mse_loss(pred_log, target_log)


class MelFIDLoss(nn.Module):
    """Mel-FID loss (Frechet distance in mel space)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Compute statistics
        pred_mean = torch.mean(pred, dim=0)
        target_mean = torch.mean(target, dim=0)
        
        pred_cov = torch.cov(pred.T)
        target_cov = torch.cov(target.T)
        
        # Frechet distance
        mean_diff = torch.sum((pred_mean - target_mean) ** 2)
        cov_term = torch.trace(pred_cov + target_cov - 2 * torch.sqrt(pred_cov @ target_cov))
        
        return mean_diff + cov_term


# ============================================================================
# DIVERGENCE LOSSES
# ============================================================================

class JensenShannonLoss(nn.Module):
    """Jensen-Shannon divergence"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        # Normalize to probability distributions
        pred_norm = torch.abs(pred) / (torch.sum(torch.abs(pred), dim=-1, keepdim=True) + self.epsilon)
        target_norm = torch.abs(target) / (torch.sum(torch.abs(target), dim=-1, keepdim=True) + self.epsilon)
        
        m = 0.5 * (pred_norm + target_norm)
        
        # KL divergences
        kl_pred = F.kl_div(torch.log(pred_norm + self.epsilon), m, reduction='batchmean')
        kl_target = F.kl_div(torch.log(target_norm + self.epsilon), m, reduction='batchmean')
        
        return 0.5 * (kl_pred + kl_target)


class KLDivergenceLoss(nn.Module):
    """KL Divergence"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        pred_norm = torch.abs(pred) / (torch.sum(torch.abs(pred), dim=-1, keepdim=True) + self.epsilon)
        target_norm = torch.abs(target) / (torch.sum(torch.abs(target), dim=-1, keepdim=True) + self.epsilon)
        
        return F.kl_div(torch.log(pred_norm + self.epsilon), target_norm, reduction='batchmean')


# ============================================================================
# NORM-BASED LOSSES
# ============================================================================

class FrobeniusNormLoss(nn.Module):
    """Frobenius norm loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return torch.norm(pred - target, p='fro') / pred.numel()


class LogNormLoss(nn.Module):
    """Log-normalized loss"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        log_pred = torch.log(torch.abs(pred) + self.epsilon)
        log_target = torch.log(torch.abs(target) + self.epsilon)
        return F.mse_loss(log_pred, log_target)


# ============================================================================
# INFORMATION-THEORETIC LOSSES
# ============================================================================

class FisherInformationLoss(nn.Module):
    """Fisher Information Matrix-based loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Approximate FIM as variance difference
        pred_var = torch.var(pred, dim=-1)
        target_var = torch.var(target, dim=-1)
        return torch.mean(torch.abs(pred_var - target_var))


# ============================================================================
# CONTRACTIVE LOSS
# ============================================================================

class ContractiveLoss(nn.Module):
    """Contractive Autoencoder loss"""
    def __init__(self, lambda_reg=0.05):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, W, x, recons_x, h):
        """
        Args:
            W: Weight matrix
            x: Input
            recons_x: Reconstructed input
            h: Hidden representation
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recons_x, x)
        
        # Contractive penalty
        dh = h * (1 - h)  # Derivative of activation
        w_sum = torch.sum(W ** 2, dim=1, keepdim=True)
        contractive_penalty = torch.sum(torch.mm(dh ** 2, w_sum))
        
        return recon_loss + self.lambda_reg * contractive_penalty


# ============================================================================
# LATENT SPACE LOSS
# ============================================================================

class LatentSpaceLoss(nn.Module):
    """Latent space consistency loss for TransformerAE"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, model, vec1, vec2, pred, target):
        """
        Compute latent space loss using encoder representations.
        
        Args:
            model: TransformerAE model
            vec1, vec2: Input vectors
            pred: Predicted output
            target: Ground truth target
        """
        # Get encoder representations
        z1_vec1, _ = model.enc1(vec1)
        z1_vec2, _ = model.enc1(vec2)
        z1_pred, _ = model.enc1(pred)
        z1_target, _ = model.enc1(target)
        
        z2_vec1, _ = model.enc2(vec1)
        z2_vec2, _ = model.enc2(vec2)
        z2_pred, _ = model.enc2(pred)
        z2_target, _ = model.enc2(target)
        
        # Compute neck representations
        out3_target = torch.cat([z1_target, z2_target], dim=2)
        Z_target = model.tanh(model.vec2neck(torch.sum(out3_target, dim=1, keepdim=False)))
        
        # Encoder losses
        l2_z11 = self.mse(z1_vec1, z1_target)
        l2_z12 = self.mse(z1_vec2, z1_target)
        l2_z1p = self.mse(z1_pred, z1_target)
        
        l2_z21 = self.mse(z2_vec1, z2_target)
        l2_z22 = self.mse(z2_vec2, z2_target)
        l2_z2p = self.mse(z2_pred, z2_target)
        
        # Neck loss (from forward pass)
        _, neck_pred, _, _, _ = model(vec1, vec2)
        LZ = self.mse(Z_target, neck_pred)
        
        # Combined latent loss
        loss = LZ + (2.0 * l2_z1p / (l2_z11 + l2_z12 + 1e-8)) + (2.0 * l2_z2p / (l2_z21 + l2_z22 + 1e-8))
        
        return loss


# ============================================================================
# LOSS REGISTRY AND PAIRS
# ============================================================================

class ComprehensiveLossRegistry:
    """Registry of all loss functions with pairing support"""
    
    def __init__(self):
        self.losses = {
            # Basic losses
            'MSE': MSELoss(),
            'MAE': MAELoss(),
            'MAPE': MAPELoss(),
            
            # Quantile
            'Q-quantile': QuantileLoss(q=0.5),
            
            # Autoregressive
            'AUTO': AutoregressiveMSELoss(),
            
            # Wasserstein
            'sinkhorn': SinkhornLoss(),
            'ws_scipy': WassersteinScipyLoss(scale=0.9),
            'ws_scipy_full': WassersteinScipyLoss(scale=1.0),
            
            # Layer-wise
            'LWLN': LWLNLoss(),
            'LWWS': LWWSLoss(),
            'LWWS_scipy': LWWSLoss(),
            
            # Spectral/Frequency
            'FFT': FFTLoss(),
            'Mel_L2': MelSpecL2Loss(),
            'Mel_FID': MelFIDLoss(),
            
            # Divergence
            'JS': JensenShannonLoss(),
            'KL': KLDivergenceLoss(),
            
            # Norm-based
            'Frobenius': FrobeniusNormLoss(),
            'log-norm': LogNormLoss(),
            
            # Information-theoretic
            'FIM': FisherInformationLoss(),
            
            # Contractive (requires special handling)
            # 'CAE': ContractiveLoss(),
            
            # Latent (requires model)
            # 'Latent': LatentSpaceLoss(),
        }
        
        # Loss pairs: (main_loss, regularization_loss, weight)
        self.loss_pairs = [
            ('MSE', 'LWLN', 0.1),
            ('MSE', 'Frobenius', 0.05),
            ('MAPE', 'JS', 0.1),
            ('sinkhorn', 'LWWS', 0.2),
            ('AUTO', 'FIM', 0.05),
            ('FFT', 'Mel_L2', 0.1),
            ('Q-quantile', 'log-norm', 0.05),
            ('ws_scipy', 'KL', 0.1),
            ('MAE', 'LWLN', 0.1),
            ('Mel_FID', 'FFT', 0.15),
        ]
    
    def get_loss(self, name: str):
        """Get loss function by name"""
        return self.losses.get(name, self.losses['MSE'])
    
    def get_all_loss_names(self) -> List[str]:
        """Get all available loss names"""
        return list(self.losses.keys())
    
    def compute_paired_loss(self, pair_idx: int, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss from a loss pair.
        
        Args:
            pair_idx: Index of loss pair
            pred: Predicted tensor
            target: Target tensor
        
        Returns:
            Combined loss
        """
        if pair_idx >= len(self.loss_pairs):
            pair_idx = 0
        
        main_name, reg_name, weight = self.loss_pairs[pair_idx]
        
        main_loss = self.get_loss(main_name)(pred, target)
        reg_loss = self.get_loss(reg_name)(pred, target)
        
        return main_loss + weight * reg_loss
    
    def get_loss_pair_name(self, pair_idx: int) -> str:
        """Get name of loss pair"""
        if pair_idx >= len(self.loss_pairs):
            pair_idx = 0
        main_name, reg_name, weight = self.loss_pairs[pair_idx]
        return f"{main_name}+{weight:.2f}*{reg_name}"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_loss_function(name: str):
    """Get loss function by name"""
    registry = ComprehensiveLossRegistry()
    return registry.get_loss(name)


def get_all_loss_names() -> List[str]:
    """Get all available loss function names"""
    registry = ComprehensiveLossRegistry()
    return registry.get_all_loss_names()


def get_loss_pairs() -> List[Tuple[str, str, float]]:
    """Get all loss pairs"""
    registry = ComprehensiveLossRegistry()
    return registry.loss_pairs
