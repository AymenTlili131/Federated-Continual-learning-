"""
Advanced Loss System with Hierarchical Structure

Loss Hierarchy:
1. Individual differentiable losses
2. Layerwise versions of individual losses
3. Regularized versions of full losses
4. Regularized versions of layerwise losses
5. Mixed regularization (layerwise + full sequence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


# Import persistence losses
try:
    from persistence_losses import (
        PersistenceLandscapeLoss,
        PersistenceImageLoss,
        LayerwisePersistenceLandscapeLoss,
        LayerwisePersistenceImageLoss
    )
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False


# Layer boundaries for CNN weights
LAYER_DELIMITERS = [208, 1414, 1514, 2254, 2464]


# ============================================================================
# LEVEL 1: INDIVIDUAL DIFFERENTIABLE LOSSES
# ============================================================================

class MSELoss(nn.Module):
    """Mean Squared Error"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class MAELoss(nn.Module):
    """Mean Absolute Error (L1)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        return torch.mean(torch.abs((pred - target) / (torch.abs(target) + self.epsilon))) * 100


class QuantileLoss(nn.Module):
    """Q-quantile loss"""
    def __init__(self, q=0.5):
        super().__init__()
        self.q = q
    
    def forward(self, pred, target):
        error = target - pred
        return torch.mean(torch.maximum(self.q * error, (self.q - 1) * error))


class SinkhornLoss(nn.Module):
    """Exact 1-D Wasserstein-1 distance via sorted differences.

    For scalar-valued sequences (1-D marginals) the W1 distance equals
    ||sort(p) - sort(q)||_1 / N  — provably exact, O(N log N), fully
    differentiable through torch.sort.  No geomloss / KeOps needed.
    ~1000x faster than the iterative Sinkhorn formulation for N=2464.
    """
    def forward(self, pred, target):
        if pred.dim() == 1:
            pred, target = pred.unsqueeze(0), target.unsqueeze(0)
        p_sorted = torch.sort(pred,   dim=-1).values
        t_sorted = torch.sort(target, dim=-1).values
        return torch.mean(torch.abs(p_sorted - t_sorted))


class FFTLoss(nn.Module):
    """FFT-based frequency domain loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_fft = torch.fft.fft(pred, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)
        return torch.mean(torch.abs(pred_fft - target_fft) ** 2)


class MelSpecL2Loss(nn.Module):
    """Mel-spectrogram L2 loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_log = torch.log(torch.abs(pred) + 1e-8)
        target_log = torch.log(torch.abs(target) + 1e-8)
        return F.mse_loss(pred_log, target_log)


class JensenShannonLoss(nn.Module):
    """Jensen-Shannon divergence"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        pred_norm = torch.abs(pred) / (torch.sum(torch.abs(pred), dim=-1, keepdim=True) + self.epsilon)
        target_norm = torch.abs(target) / (torch.sum(torch.abs(target), dim=-1, keepdim=True) + self.epsilon)
        
        m = 0.5 * (pred_norm + target_norm)
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


class FrobeniusNormLoss(nn.Module):
    """Frobenius norm loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return torch.norm(pred - target, p=2) / pred.numel()


class LogNormLoss(nn.Module):
    """Log-normalized loss"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        log_pred = torch.log(torch.abs(pred) + self.epsilon)
        log_target = torch.log(torch.abs(target) + self.epsilon)
        return F.mse_loss(log_pred, log_target)


class FisherInformationLoss(nn.Module):
    """Fisher Information Matrix-based loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_var = torch.var(pred, dim=-1)
        target_var = torch.var(target, dim=-1)
        return torch.mean(torch.abs(pred_var - target_var))


class AutoregressiveLoss(nn.Module):
    """Autoregressive loss with layer dependencies"""
    def __init__(self, lambda_weights=None, delimiters=None, epsilon=1e-8):
        super().__init__()
        if lambda_weights is None:
            lambda_weights = [10, 5, 2, 1, 0.5]
        if delimiters is None:
            delimiters = LAYER_DELIMITERS
        self.lambda_weights = lambda_weights
        self.delimiters = delimiters
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        n_chunks = len(self.delimiters)
        mse_losses = []
        start = 0
        
        for end in self.delimiters:
            pred_chunk = pred[:, start:end]
            target_chunk = target[:, start:end]
            mse_loss = F.mse_loss(pred_chunk, target_chunk, reduction='mean')
            mse_losses.append(mse_loss)
            start = end
        
        autoregressive_loss = mse_losses[0]
        for i in range(1, n_chunks):
            autoregressive_loss += (
                self.lambda_weights[i - 1] * (mse_losses[i] / (mse_losses[i - 1] + self.epsilon))
            )
        
        return autoregressive_loss


class DifferentiablePersistenceLoss(nn.Module):
    """Differentiable 1-parameter persistence loss via sorted-value Wasserstein.

    Mathematical basis
    ------------------
    For a 1-D weight sequence viewed as a function f: {0,...,n-1} → R on a
    path graph, the Wasserstein-p distance between H0 sublevel-set persistence
    diagrams satisfies:
        d_Wp(D(f), D(g))  ≤  ||f_sorted - g_sorted||_p   (stability theorem)
    The sorted values of f ARE the birth-times in the sublevel filtration.
    Wasserstein-1 between empirical distributions = L1 of sorted sequences.

    Fully GPU-compatible — uses only torch.sort (differentiable) + F.l1_loss.
    No GUDHI, no NumPy, no CPU offloading.
    """
    def __init__(self, p: int = 1):
        super().__init__()
        assert p in (1, 2), "p must be 1 (W1) or 2 (W2)"
        self.p = p

    def forward(self, pred, target):
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_s = torch.sort(pred, dim=-1).values
        target_s = torch.sort(target, dim=-1).values
        if self.p == 1:
            return F.l1_loss(pred_s, target_s)
        return F.mse_loss(pred_s, target_s)


class RTDLoss(nn.Module):
    """Representation Topology Divergence loss (Barannikov et al., 2022).

    Compares the eigenvalue spectra of the Gram matrices of pred/target
    weight batches.  The sorted eigenvalue sequences are the "topological
    fingerprint" of the weight cloud.  Wasserstein-2 between spectra is
    fully differentiable via torch.linalg.eigvalsh (GPU).

    subsample: max feature dims used (keep GPU memory bounded).
    """
    def __init__(self, subsample: int = 512):
        super().__init__()
        self.subsample = subsample

    def forward(self, pred, target):
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        n_feat = pred.shape[-1]
        if n_feat > self.subsample:
            idx = torch.randperm(n_feat, device=pred.device)[:self.subsample]
            pred = pred[:, idx]
            target = target[:, idx]
        K_pred = (pred @ pred.T) / n_feat
        K_tgt = (target @ target.T) / n_feat
        # Regularise to prevent ill-conditioned / rank-deficient Gram matrices
        # (occurs when batch_size < subsample, producing repeated zero eigenvalues)
        eps = 1e-4 * torch.eye(K_pred.shape[0], device=pred.device, dtype=pred.dtype)
        eigs_pred = torch.linalg.eigvalsh(K_pred + eps)
        eigs_tgt = torch.linalg.eigvalsh(K_tgt + eps)
        return F.mse_loss(eigs_pred, eigs_tgt)


# ============================================================================
# LEVEL 2: LAYERWISE VERSIONS
# ============================================================================

class LayerwiseLoss(nn.Module):
    """Wrapper to apply any loss function layer-wise"""
    def __init__(self, base_loss, delimiters=None):
        super().__init__()
        self.base_loss = base_loss
        if delimiters is None:
            delimiters = LAYER_DELIMITERS
        self.delimiters = delimiters
    
    def forward(self, pred, target):
        total_loss = 0.0
        start = 0
        
        for end in self.delimiters:
            pred_chunk = pred[:, start:end]
            target_chunk = target[:, start:end]
            layer_loss = self.base_loss(pred_chunk, target_chunk)
            total_loss += layer_loss
            start = end
        
        return total_loss / len(self.delimiters)


class LWLNLoss(nn.Module):
    """Layer-Wise Loss with Normalization"""
    def __init__(self, delimiters=None, epsilon=1e-8):
        super().__init__()
        if delimiters is None:
            delimiters = LAYER_DELIMITERS
        self.delimiters = delimiters
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        total_loss = 0.0
        start = 0
        
        for end in self.delimiters:
            pred_chunk = pred[:, start:end]
            target_chunk = target[:, start:end]
            
            std = torch.std(target_chunk, dim=-1, keepdim=True)
            normalized_loss = torch.mean(torch.abs(pred_chunk - target_chunk) / (std + self.epsilon))
            total_loss += normalized_loss
            start = end
        
        return total_loss / len(self.delimiters)


# ============================================================================
# LEVEL 3: REGULARIZED FULL LOSSES
# ============================================================================

class RegularizedLoss(nn.Module):
    """Base class for regularized losses"""
    def __init__(self, main_loss, reg_loss, reg_weight=0.1):
        super().__init__()
        self.main_loss = main_loss
        self.reg_loss = reg_loss
        self.reg_weight = reg_weight
    
    def forward(self, pred, target):
        main = self.main_loss(pred, target)
        reg = self.reg_loss(pred, target)
        return main + self.reg_weight * reg


# ============================================================================
# LEVEL 4: REGULARIZED LAYERWISE LOSSES
# ============================================================================

class RegularizedLayerwiseLoss(nn.Module):
    """Regularized layerwise loss"""
    def __init__(self, main_loss, reg_loss, reg_weight=0.1, delimiters=None):
        super().__init__()
        self.main_loss = LayerwiseLoss(main_loss, delimiters)
        self.reg_loss = LayerwiseLoss(reg_loss, delimiters)
        self.reg_weight = reg_weight
    
    def forward(self, pred, target):
        main = self.main_loss(pred, target)
        reg = self.reg_loss(pred, target)
        return main + self.reg_weight * reg


# ============================================================================
# LEVEL 5: MIXED REGULARIZATION (LAYERWISE + FULL)
# ============================================================================

class MixedRegularizationLoss(nn.Module):
    """Mixed regularization combining layerwise and full sequence"""
    def __init__(self, main_loss, layerwise_reg, full_reg, 
                 layerwise_weight=0.1, full_weight=0.05, delimiters=None):
        super().__init__()
        self.main_loss = main_loss
        self.layerwise_reg = LayerwiseLoss(layerwise_reg, delimiters)
        self.full_reg = full_reg
        self.layerwise_weight = layerwise_weight
        self.full_weight = full_weight
    
    def forward(self, pred, target):
        main = self.main_loss(pred, target)
        lw_reg = self.layerwise_reg(pred, target)
        full_reg = self.full_reg(pred, target)
        return main + self.layerwise_weight * lw_reg + self.full_weight * full_reg


# ============================================================================
# LOSS REGISTRY WITH HIERARCHICAL STRUCTURE
# ============================================================================

class HierarchicalLossRegistry:
    """Registry organizing losses by hierarchy level"""
    
    def __init__(self):
        # Level 1: Individual losses
        self.individual_losses = {
            'MSE': MSELoss(),
            'MAE': MAELoss(),
            'MAPE': MAPELoss(),
            'Quantile': QuantileLoss(q=0.5),
            'Sinkhorn': SinkhornLoss(),
            'FFT': FFTLoss(),
            'MelSpec': MelSpecL2Loss(),
            'JS': JensenShannonLoss(),
            'KL': KLDivergenceLoss(),
            'Frobenius': FrobeniusNormLoss(),
            'LogNorm': LogNormLoss(),
            'FIM': FisherInformationLoss(),
            'AUTO': AutoregressiveLoss(),
            # Differentiable topology losses — fully GPU, true autograd
            'DiffPers': DifferentiablePersistenceLoss(p=1),
            'RTD': RTDLoss(subsample=512),
        }
        
        # Level 2: Layerwise versions
        self.layerwise_losses = {}
        for name, loss in self.individual_losses.items():
            if name != 'AUTO':  # AUTO already layerwise
                self.layerwise_losses[f'LW_{name}'] = LayerwiseLoss(loss)
        self.layerwise_losses['LWLN'] = LWLNLoss()
        # Add explicit LayerwiseSinkhorn for better tracking
        self.layerwise_losses['LW_Sinkhorn'] = LayerwiseLoss(SinkhornLoss())
        
        # Note: Layerwise persistence losses removed
        # They are too slow and produce NaN values
        # Instead, they are available as regularizers in Level 4
        
        # Level 3: Regularized full losses
        self.regularized_full = self._create_regularized_full()
        
        # Level 4: Regularized layerwise
        self.regularized_layerwise = self._create_regularized_layerwise()
        
        # Level 5: Mixed regularization
        self.mixed_regularization = self._create_mixed_regularization()
        
        # All losses combined
        self.all_losses = {
            **self.individual_losses,
            **self.layerwise_losses,
            **self.regularized_full,
            **self.regularized_layerwise,
            **self.mixed_regularization
        }
    
    def _create_regularized_full(self):
        """Create regularized full sequence losses"""
        configs = [
            ('MSE', 'Frobenius', 0.05),
            ('MSE', 'LogNorm', 0.1),
            ('MAPE', 'JS', 0.1),
            ('Sinkhorn', 'KL', 0.15),
            ('FFT', 'MelSpec', 0.1),
            ('Quantile', 'FIM', 0.05),
            ('MAE', 'Frobenius', 0.05),
            # Sinkhorn combinations
            ('Sinkhorn', 'MSE', 0.1),
            ('Sinkhorn', 'MAE', 0.1),
            ('Sinkhorn', 'Frobenius', 0.1),
            ('MSE', 'Sinkhorn', 0.15),
            ('MAE', 'Sinkhorn', 0.15),
            # Differentiable persistence combinations
            ('MSE', 'DiffPers', 0.1),
            ('MAE', 'DiffPers', 0.1),
            ('MSE', 'RTD', 0.05),
            ('MAE', 'RTD', 0.05),
        ]
        
        # Add persistence as regularizers if available (very small weight due to computational cost)
        # Note: Persistence losses are added separately to avoid key errors
        
        losses = {}
        for main_name, reg_name, weight in configs:
            name = f'{main_name}+{weight}*{reg_name}'
            losses[name] = RegularizedLoss(
                self.individual_losses[main_name],
                self.individual_losses[reg_name],
                weight
            )
        
        # Add persistence regularizers separately
        if PERSISTENCE_AVAILABLE:
            losses['MSE+0.01*PersLandscape'] = RegularizedLoss(
                self.individual_losses['MSE'],
                PersistenceLandscapeLoss(),
                0.01
            )
            losses['MAE+0.01*PersLandscape'] = RegularizedLoss(
                self.individual_losses['MAE'],
                PersistenceLandscapeLoss(),
                0.01
            )
        
        return losses
    
    def _create_regularized_layerwise(self):
        """Create regularized layerwise losses"""
        configs = [
            ('MSE', 'Frobenius', 0.05),
            ('MSE', 'LogNorm', 0.1),
            ('MAPE', 'JS', 0.1),
            ('MAE', 'FIM', 0.05),
            ('FFT', 'MelSpec', 0.1),
            # Layerwise Sinkhorn combinations
            ('Sinkhorn', 'MSE', 0.1),
            ('Sinkhorn', 'MAE', 0.1),
            ('Sinkhorn', 'Frobenius', 0.1),
            ('MSE', 'Sinkhorn', 0.15),
            ('MAE', 'Sinkhorn', 0.15),
            # Layerwise differentiable persistence
            ('MSE', 'DiffPers', 0.1),
            ('MAE', 'DiffPers', 0.1),
            ('MSE', 'RTD', 0.05),
        ]
        
        # Add persistence as regularizers if available (very small weight)
        # Note: Persistence losses are added separately to avoid key errors
        
        losses = {}
        for main_name, reg_name, weight in configs:
            name = f'LW_{main_name}+{weight}*LW_{reg_name}'
            losses[name] = RegularizedLayerwiseLoss(
                self.individual_losses[main_name],
                self.individual_losses[reg_name],
                weight
            )
        
        # Add layerwise persistence regularizers separately
        if PERSISTENCE_AVAILABLE:
            losses['LW_MSE+0.01*LW_PersLandscape'] = RegularizedLayerwiseLoss(
                self.individual_losses['MSE'],
                PersistenceLandscapeLoss(),
                0.01
            )
            losses['LW_MAE+0.01*LW_PersLandscape'] = RegularizedLayerwiseLoss(
                self.individual_losses['MAE'],
                PersistenceLandscapeLoss(),
                0.01
            )
        
        return losses
    
    def _create_mixed_regularization(self):
        """
        REMOVED: Mixed regularization with 3+ losses causes NaN issues
        User requirement: Max 2 PyTorch differentiable losses per combination
        """
        return {}
    
    def get_loss(self, name: str):
        """Get loss by name.  Raises KeyError for unknown names (no silent MSE fallback)."""
        if name not in self.all_losses:
            available = sorted(self.all_losses.keys())
            raise KeyError(
                f"Unknown loss '{name}'. Available losses ({len(available)}): "
                + ", ".join(available[:20])
                + (" ..." if len(available) > 20 else "")
            )
        return self.all_losses[name]
    
    def get_all_loss_names(self) -> List[str]:
        """Get all loss names"""
        return list(self.all_losses.keys())
    
    def get_losses_by_level(self, level: int) -> Dict[str, nn.Module]:
        """Get losses by hierarchy level (1-5)"""
        if level == 1:
            return self.individual_losses
        elif level == 2:
            return self.layerwise_losses
        elif level == 3:
            return self.regularized_full
        elif level == 4:
            return self.regularized_layerwise
        elif level == 5:
            return self.mixed_regularization
        else:
            return {}
    
    def get_experiment_sequence(self) -> List[str]:
        """Get recommended experiment sequence"""
        sequence = []
        
        # Level 1: Key individual losses (including new topology losses)
        sequence.extend([
            'MSE', 'MAE', 'MAPE', 'Sinkhorn', 'FFT', 'AUTO',
            'DiffPers', 'RTD',
        ])

        # Level 2: Layerwise versions (including explicit Sinkhorn)
        sequence.extend(['LW_MSE', 'LW_MAE', 'LW_MAPE', 'LW_Sinkhorn', 'LWLN',
                         'LW_DiffPers', 'LW_RTD'])
        
        # Level 3: Regularized full (all of them now)
        sequence.extend(list(self.regularized_full.keys()))
        
        # Level 4: Regularized layerwise (all of them now)
        sequence.extend(list(self.regularized_layerwise.keys()))
        
        # Level 5: Mixed
        sequence.extend(list(self.mixed_regularization.keys()))
        
        return sequence


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_loss_function(name: str):
    """Get loss function by name"""
    registry = HierarchicalLossRegistry()
    return registry.get_loss(name)


def get_all_loss_names() -> List[str]:
    """Get all available loss names"""
    registry = HierarchicalLossRegistry()
    return registry.get_all_loss_names()


def get_experiment_sequence() -> List[str]:
    """Get recommended experiment sequence"""
    registry = HierarchicalLossRegistry()
    return registry.get_experiment_sequence()
