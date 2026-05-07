"""
CNN Reconstruction and Finetuning Module

Reconstructs CNNs from predicted weight vectors and validates them
through finetuning on MNIST with class-specific data loaders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from pathlib import Path
from collections import OrderedDict
import copy
from typing import Dict, List, Tuple, Optional

# Layer boundaries for CNN weight reconstruction
LAYER_BOUNDARIES = {
    'conv1_weight': (0, 200),      # [8, 1, 5, 5]
    'conv1_bias': (200, 208),      # [8]
    'conv2_weight': (208, 1408),   # [6, 8, 5, 5]
    'conv2_bias': (1408, 1414),    # [6]
    'conv3_weight': (1414, 1510),  # [4, 6, 2, 2]
    'conv3_bias': (1510, 1514),    # [4]
    'fc1_weight': (1514, 2234),    # [20, 36]
    'fc1_bias': (2234, 2254),      # [20]
    'fc2_weight': (2254, 2454),    # [10, 20]
    'fc2_bias': (2454, 2464),      # [10]
}


class CNN(nn.Module):
    """
    Simple CNN for MNIST classification
    Architecture: Conv(8) -> Pool -> Conv(6) -> Pool -> Conv(4) -> FC(20) -> FC(10)
    """
    def __init__(self, channels_in=1, nlin="leakyrelu", dropout=0.0, init_type="kaiming_uniform"):
        super().__init__()
        self.module_list = nn.ModuleList()
        
        # Layer 1: Conv2d(1, 8, 5) + MaxPool + Activation
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        
        # Layer 2: Conv2d(8, 6, 5) + MaxPool + Activation
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        
        # Layer 3: Conv2d(6, 4, 2) + Activation
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        
        # Flatten
        self.module_list.append(nn.Flatten())
        
        # FC1: Linear(36, 20) + Activation
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        
        # FC2: Linear(20, 10)
        self.module_list.append(nn.Linear(20, 10))
        
        self.initialize_weights(init_type)
    
    def initialize_weights(self, init_type):
        for m in self.module_list:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                elif init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                elif init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                elif init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                elif init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                elif init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)
    
    def get_nonlin(self, nlin):
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        elif nlin == "relu":
            return nn.ReLU()
        elif nlin == "tanh":
            return nn.Tanh()
        elif nlin == "sigmoid":
            return nn.Sigmoid()
        elif nlin == "silu":
            return nn.SiLU()
        elif nlin == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x


class ClassSpecificImageFolder(ImageFolder):
    """ImageFolder that drops specific classes"""
    def __init__(self, root, dropped_classes=None, **kwargs):
        self.dropped_classes = dropped_classes or []
        super().__init__(root, **kwargs)
    
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in Path(directory).iterdir() if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"No valid classes in {directory}")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def reconstruct_cnn_from_weights(weight_vector: np.ndarray, activation: str = "leakyrelu") -> CNN:
    """
    Reconstruct CNN from flattened weight vector
    
    Args:
        weight_vector: Flattened weights (2464 dimensions)
        activation: Activation function name
    
    Returns:
        CNN model with loaded weights
    """
    # Create empty CNN
    model = CNN(channels_in=1, nlin=activation, dropout=0.0, init_type="kaiming_uniform")
    
    # Build state dict
    checkpoint = OrderedDict()
    checkpoint["module_list.0.weight"] = torch.tensor(
        np.array(weight_vector[0:200]).reshape([8, 1, 5, 5])
    ).float()
    checkpoint["module_list.0.bias"] = torch.tensor(
        np.array(weight_vector[200:208]).reshape([8])
    ).float()
    
    checkpoint["module_list.3.weight"] = torch.tensor(
        np.array(weight_vector[208:1408]).reshape([6, 8, 5, 5])
    ).float()
    checkpoint["module_list.3.bias"] = torch.tensor(
        np.array(weight_vector[1408:1414]).reshape([6])
    ).float()
    
    checkpoint["module_list.6.weight"] = torch.tensor(
        np.array(weight_vector[1414:1510]).reshape([4, 6, 2, 2])
    ).float()
    checkpoint["module_list.6.bias"] = torch.tensor(
        np.array(weight_vector[1510:1514]).reshape([4])
    ).float()
    
    checkpoint["module_list.9.weight"] = torch.tensor(
        np.array(weight_vector[1514:2234]).reshape([20, 36])
    ).float()
    checkpoint["module_list.9.bias"] = torch.tensor(
        np.array(weight_vector[2234:2254]).reshape([20])
    ).float()
    
    checkpoint["module_list.11.weight"] = torch.tensor(
        np.array(weight_vector[2254:2454]).reshape([10, 20])
    ).float()
    checkpoint["module_list.11.bias"] = torch.tensor(
        np.array(weight_vector[2454:2464]).reshape([10])
    ).float()
    
    # Load weights
    model.load_state_dict(checkpoint)
    
    return model


def compute_eigenvalues(model: CNN) -> Dict[str, np.ndarray]:
    """
    Compute eigenvalues of weight matrices for each layer
    
    Returns:
        Dictionary mapping layer names to eigenvalue arrays
    """
    eigenvalues_dict = {}
    
    for name, param in model.named_parameters():
        if len(param.shape) > 1:  # Only weight matrices, not biases
            weights = param.detach().cpu()
            weights = weights.view(weights.shape[0], -1)
            weights = weights.T @ weights  # Gram matrix
            
            eigenvalues = torch.linalg.eigvals(weights).numpy()
            eigenvalues_dict[name] = eigenvalues.real
    
    return eigenvalues_dict


def train_cnn_epoch(model, train_loader, optimizer, criterion, device):
    """Train CNN for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate_cnn(model, test_loader, criterion, device):
    """Validate CNN"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = total_loss / len(test_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def finetune_reconstructed_cnn(
    predicted_weights: np.ndarray,
    task_classes: List[int],
    activation: str = "leakyrelu",
    mnist_root: str = "./data/SplitMnist",
    n_finetune_epochs: int = 5,
    lr: float = 0.05,
    batch_size: int = 24,
    device: str = "cuda",
    input_weights_x1: Optional[np.ndarray] = None,
    input_weights_x2: Optional[np.ndarray] = None,
    ground_truth_weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Finetune reconstructed CNN and track accuracy
    
    Args:
        predicted_weights: Predicted weight vector from transformer
        task_classes: List of class labels for this task
        activation: CNN activation function
        mnist_root: Path to MNIST data
        n_finetune_epochs: Number of finetuning epochs
        lr: Learning rate
        batch_size: Batch size (16 or 24 recommended)
        device: Device to use
        input_weights_x1: Optional input weight vector 1 (for eigenvalue analysis)
        input_weights_x2: Optional input weight vector 2 (for eigenvalue analysis)
        ground_truth_weights: Optional ground truth weights (for comparison)
    
    Returns:
        Dictionary with finetuning results and eigenvalue analysis
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Reconstruct CNN
    model = reconstruct_cnn_from_weights(predicted_weights, activation)
    model = model.to(device)
    
    # Determine OOD classes (classes not in task)
    all_classes = list(range(10))
    ood_classes = [c for c in all_classes if c not in task_classes]
    
    # Create data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(1)
    ])
    
    # In-distribution (ID) test set
    test_id = ClassSpecificImageFolder(
        root=f"{mnist_root}/test/",
        dropped_classes=[str(c) for c in ood_classes],
        transform=transform
    )
    test_id_loader = DataLoader(test_id, batch_size=batch_size, shuffle=False)
    
    # Out-of-distribution (OOD) test set (if applicable)
    test_ood_loader = None
    if len(task_classes) < 10:
        test_ood = ClassSpecificImageFolder(
            root=f"{mnist_root}/test/",
            dropped_classes=[str(c) for c in task_classes],
            transform=transform
        )
        test_ood_loader = DataLoader(test_ood, batch_size=batch_size, shuffle=False)
    
    # Training set
    train_dataset = ClassSpecificImageFolder(
        root=f"{mnist_root}/train/",
        dropped_classes=[str(c) for c in ood_classes],
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initial validation (before finetuning)
    criterion = CrossEntropyLoss()
    _, acc_id_initial = validate_cnn(model, test_id_loader, criterion, device)
    
    acc_ood_initial = None
    if test_ood_loader:
        _, acc_ood_initial = validate_cnn(model, test_ood_loader, criterion, device)
    
    # Compute eigenvalues for all weight types
    eigenvalues_analysis = {
        'predicted_initial': compute_eigenvalues(model)
    }
    
    # Eigenvalues for input weights (if provided)
    if input_weights_x1 is not None:
        cnn_x1 = reconstruct_cnn_from_weights(input_weights_x1, activation)
        eigenvalues_analysis['input_x1'] = compute_eigenvalues(cnn_x1)
    
    if input_weights_x2 is not None:
        cnn_x2 = reconstruct_cnn_from_weights(input_weights_x2, activation)
        eigenvalues_analysis['input_x2'] = compute_eigenvalues(cnn_x2)
    
    if ground_truth_weights is not None:
        cnn_gt = reconstruct_cnn_from_weights(ground_truth_weights, activation)
        eigenvalues_analysis['ground_truth'] = compute_eigenvalues(cnn_gt)
    
    # Setup optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-3, max_lr=0.1, 
        step_size_up=400, mode="triangular2", cycle_momentum=False
    )
    
    # Finetune with eigenvalue tracking at each epoch
    finetune_history = {
        'epoch_0_acc_id': acc_id_initial,
        'epoch_0_acc_ood': acc_ood_initial,
    }
    
    # Track eigenvalues at each finetuning epoch
    eigenvalues_analysis['finetuned_epoch_0'] = compute_eigenvalues(model)
    
    for epoch in range(1, n_finetune_epochs + 1):
        train_loss, train_acc = train_cnn_epoch(model, train_loader, optimizer, criterion, device)
        _, acc_id = validate_cnn(model, test_id_loader, criterion, device)
        
        finetune_history[f'epoch_{epoch}_acc_id'] = acc_id
        finetune_history[f'epoch_{epoch}_train_loss'] = train_loss
        finetune_history[f'epoch_{epoch}_train_acc'] = train_acc
        
        if test_ood_loader:
            _, acc_ood = validate_cnn(model, test_ood_loader, criterion, device)
            finetune_history[f'epoch_{epoch}_acc_ood'] = acc_ood
        
        # Compute eigenvalues after this epoch
        eigenvalues_analysis[f'finetuned_epoch_{epoch}'] = compute_eigenvalues(model)
        
        scheduler.step()
    
    return {
        'finetune_history': finetune_history,
        'eigenvalues_analysis': eigenvalues_analysis,
        'reconstructed_model': model,
        'acc_id_initial': acc_id_initial,
        'acc_ood_initial': acc_ood_initial,
        'acc_id_final': finetune_history[f'epoch_{n_finetune_epochs}_acc_id'],
    }
