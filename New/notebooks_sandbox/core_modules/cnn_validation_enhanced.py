"""
Enhanced CNN Validation with Stepwise Logging and Weight Saving

This module extends the CNN validation to:
1. Save CNN weights at each finetuning epoch (not transformer epochs)
2. Log accuracy/loss per CNN finetuning epoch
3. Perform eigenvalue analysis per CNN epoch
4. Optionally perform topology analysis per CNN epoch
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from cnn_reconstruction import CNN, reconstruct_cnn_from_weights, LAYER_BOUNDARIES
from cnn_reconstruction import ClassSpecificImageFolder
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam


def compute_eigenvalues_for_cnn(cnn_model: nn.Module) -> Dict[str, np.ndarray]:
    """Compute eigenvalues for each layer in CNN"""
    eigenvalues = {}
    
    for name, param in cnn_model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            # Reshape to 2D if needed
            weight_matrix = param.data.cpu().numpy()
            if len(weight_matrix.shape) > 2:
                # Flatten conv weights
                weight_matrix = weight_matrix.reshape(weight_matrix.shape[0], -1)
            
            # Compute eigenvalues
            try:
                eigs = np.linalg.eigvalsh(weight_matrix @ weight_matrix.T)
                eigenvalues[name] = eigs
            except:
                eigenvalues[name] = np.array([])
    
    return eigenvalues


def finetune_cnn_with_stepwise_logging(
    predicted_weights: np.ndarray,
    task_classes: List[int],
    activation: str,
    mnist_root: str,
    n_finetune_epochs: int = 5,
    batch_size: int = 24,
    save_dir: Optional[Path] = None,
    input_weights_x1: Optional[np.ndarray] = None,
    input_weights_x2: Optional[np.ndarray] = None,
    ground_truth_weights: Optional[np.ndarray] = None,
    compute_topology: bool = False
) -> Dict:
    """
    Finetune CNN with stepwise logging and weight saving
    
    Returns:
        Dictionary with:
        - stepwise_results: List of dicts per epoch with acc, loss, eigenvalues
        - saved_weight_files: List of paths to saved CNN weights
        - acc_id_initial, acc_id_final, acc_ood_initial (for compatibility)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reconstruct CNN from predicted weights
    cnn = reconstruct_cnn_from_weights(predicted_weights, activation=activation)
    cnn = cnn.to(device)
    
    # Create data loaders
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = ClassSpecificImageFolder(
        root=Path(mnist_root) / "train",
        classes=task_classes,
        transform=transform
    )
    test_dataset = ClassSpecificImageFolder(
        root=Path(mnist_root) / "test",
        classes=task_classes,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Evaluate initial accuracy
    cnn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc_id_initial = correct / total if total > 0 else 0.0
    
    # Finetuning with stepwise logging
    optimizer = Adam(cnn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    stepwise_results = []
    saved_weight_files = []
    
    for epoch in range(1, n_finetune_epochs + 1):
        # Train
        cnn.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate
        cnn.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = cnn(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total if total > 0 else 0.0
        avg_test_loss = test_loss / len(test_loader)
        
        # Compute eigenvalues
        eigenvalues = compute_eigenvalues_for_cnn(cnn)
        
        # Save CNN weights for this epoch
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract weights to numpy array
            cnn_weights = []
            for name, param in cnn.named_parameters():
                cnn_weights.append(param.data.cpu().numpy().flatten())
            cnn_weights_array = np.concatenate(cnn_weights)
            
            weight_file = save_dir / f"subCNN_epoch{epoch}_weights.npy"
            np.save(weight_file, cnn_weights_array)
            saved_weight_files.append(str(weight_file))
            
            # Save eigenvalues
            eigenvalues_file = save_dir / f"subCNN_epoch{epoch}_eigenvalues.json"
            eigenvalues_serializable = {
                name: eigs.tolist() if isinstance(eigs, np.ndarray) else eigs
                for name, eigs in eigenvalues.items()
            }
            with open(eigenvalues_file, 'w') as f:
                json.dump(eigenvalues_serializable, f, indent=2)
        
        # Store stepwise results
        epoch_result = {
            'cnn_epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'accuracy': acc,
            'eigenvalues_summary': {
                name: {
                    'mean': float(np.mean(eigs)) if len(eigs) > 0 else 0.0,
                    'max': float(np.max(eigs)) if len(eigs) > 0 else 0.0,
                    'min': float(np.min(eigs)) if len(eigs) > 0 else 0.0
                }
                for name, eigs in eigenvalues.items()
            }
        }
        
        stepwise_results.append(epoch_result)
        
        print(f"    CNN Epoch {epoch}/{n_finetune_epochs}: Loss={avg_test_loss:.4f}, Acc={acc:.4f}")
    
    # Final accuracy
    acc_id_final = stepwise_results[-1]['accuracy']
    
    # OOD accuracy (evaluate on all classes not in task_classes)
    all_classes = list(range(10))
    ood_classes = [c for c in all_classes if c not in task_classes]
    
    if ood_classes:
        ood_dataset = ClassSpecificImageFolder(
            root=Path(mnist_root) / "test",
            classes=ood_classes,
            transform=transform
        )
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        cnn.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in ood_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc_ood_initial = correct / total if total > 0 else 0.0
    else:
        acc_ood_initial = 0.0
    
    return {
        'stepwise_results': stepwise_results,
        'saved_weight_files': saved_weight_files,
        'acc_id_initial': acc_id_initial,
        'acc_id_final': acc_id_final,
        'acc_ood_initial': acc_ood_initial,
        'task_classes': task_classes,
        'n_epochs': n_finetune_epochs
    }
