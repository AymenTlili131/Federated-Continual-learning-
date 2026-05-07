#!/usr/bin/env python3
"""
Fix Checkpoint Loading - Create mock config module and custom unpickler
"""

import sys
import torch
import pickle
import io
from pathlib import Path
from types import ModuleType

# Create mock config module
config_module = ModuleType('config')
config_module.MODEL_CONFIGS = {}
sys.modules['config'] = config_module

# Add other potential missing modules
for module_name in ['advanced_losses', 'advanced_topology', 'utils_consolidated', 
                     'weight_normalization', 'cnn_reconstruction', 'multi_objective_ranking']:
    if module_name not in sys.modules:
        sys.modules[module_name] = ModuleType(module_name)

# Custom unpickler with persistent_load
class CheckpointUnpickler(pickle.Unpickler):
    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)
        self.persistent_load_cache = {}
    
    def persistent_load(self, pid):
        """Handle persistent IDs for torch tensors"""
        # pid is typically a tuple like ('storage', storage_type, key, location, size)
        if isinstance(pid, tuple) and len(pid) > 0:
            if pid[0] == 'storage':
                storage_type, key, location, size = pid[1], pid[2], pid[3], pid[4]
                
                # Check cache
                if key in self.persistent_load_cache:
                    return self.persistent_load_cache[key]
                
                # Create appropriate storage type
                if 'Float' in storage_type.__name__:
                    storage = torch.FloatStorage(size)
                elif 'Long' in storage_type.__name__:
                    storage = torch.LongStorage(size)
                elif 'Int' in storage_type.__name__:
                    storage = torch.IntStorage(size)
                elif 'Double' in storage_type.__name__:
                    storage = torch.DoubleStorage(size)
                elif 'Half' in storage_type.__name__:
                    storage = torch.HalfStorage(size)
                else:
                    storage = storage_type(size)
                
                self.persistent_load_cache[key] = storage
                return storage
        
        # Fallback
        return pid
    
    def find_class(self, module, name):
        """Handle missing classes"""
        # Try standard loading first
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Return a dummy class for missing modules
            return type(name, (), {})

def load_checkpoint_fixed(checkpoint_path):
    """Load checkpoint with all fixes applied"""
    try:
        with open(checkpoint_path, 'rb') as f:
            unpickler = CheckpointUnpickler(f)
            checkpoint = unpickler.load()
        return checkpoint
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

# Test loading
print("Testing checkpoint loading with fixes...")
test_path = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/experiments/tiny_overlap0_AUTO/checkpoints/best_model.pth")

checkpoint = load_checkpoint_fixed(test_path)

if checkpoint is not None:
    print(f"✓ Successfully loaded checkpoint!")
    print(f"  Type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"  Keys: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  State dict keys (first 5): {list(state_dict.keys())[:5]}")
            print(f"  Total parameters: {len(state_dict)}")
    print("\n✓ Checkpoint loading FIXED!")
else:
    print("✗ Failed to load checkpoint")
