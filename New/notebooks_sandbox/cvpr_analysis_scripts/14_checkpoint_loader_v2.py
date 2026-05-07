#!/usr/bin/env python3
"""
Checkpoint Loader V2 - Handle old pickle protocols and missing modules
"""

import sys
import torch
import pickle
from pathlib import Path
from types import ModuleType
import warnings
warnings.filterwarnings('ignore')

# Create all potentially missing modules BEFORE any imports
missing_modules = ['config', 'advanced_losses', 'advanced_topology', 'utils_consolidated',
                   'weight_normalization', 'cnn_reconstruction', 'multi_objective_ranking']

for module_name in missing_modules:
    if module_name not in sys.modules:
        mock_module = ModuleType(module_name)
        # Add some common attributes
        mock_module.MODEL_CONFIGS = {}
        mock_module.__dict__.update({
            'HierarchicalLossRegistry': type('HierarchicalLossRegistry', (), {}),
            'get_experiment_sequence': lambda: [],
            'compute_comprehensive_topology': lambda *args, **kwargs: {},
            'save_topology_results': lambda *args, **kwargs: None,
            'WeightDistanceMetrics': type('WeightDistanceMetrics', (), {}),
            'load_merged_zoo': lambda *args, **kwargs: {},
            'extract_weights_from_zoo': lambda *args, **kwargs: [],
            'create_weight_pairs': lambda *args, **kwargs: [],
            'LayerWiseNormalizer': type('LayerWiseNormalizer', (), {}),
            'finetune_reconstructed_cnn': lambda *args, **kwargs: {},
            'rank_losses_multi_objective': lambda *args, **kwargs: {},
            'LossPerformance': type('LossPerformance', (), {}),
        })
        sys.modules[module_name] = mock_module

# Now add project paths
PROJECT_ROOT = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks_sandbox"))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks_sandbox" / "core_modules"))

# Import transformer
from Double_input_transformer import TransformerAE

# Custom pickle module that handles old protocols
class CustomPickle:
    @staticmethod
    def load(file, **kwargs):
        # Use pickle with fix_imports for old protocols
        return pickle.load(file, fix_imports=True, encoding='bytes', errors='ignore')
    
    @staticmethod
    def loads(data, **kwargs):
        return pickle.loads(data, fix_imports=True, encoding='bytes', errors='ignore')

def load_checkpoint_robust(checkpoint_path):
    """Load checkpoint with maximum compatibility"""
    try:
        # Method 1: Try standard torch.load first
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            return checkpoint, "standard"
        except Exception as e1:
            pass
        
        # Method 2: Try with custom pickle module
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', 
                                   pickle_module=CustomPickle, weights_only=False)
            return checkpoint, "custom_pickle"
        except Exception as e2:
            pass
        
        # Method 3: Manual unpickling with maximum compatibility
        try:
            import io
            with open(checkpoint_path, 'rb') as f:
                # Read the file
                data = f.read()
                
                # Try to unpickle with different protocols
                for protocol in [0, 1, 2, 3, 4, 5]:
                    try:
                        checkpoint = pickle.loads(data, fix_imports=True, encoding='latin1')
                        return checkpoint, f"manual_protocol_{protocol}"
                    except:
                        continue
        except Exception as e3:
            pass
        
        # Method 4: Try loading just the state dict using torch internals
        try:
            import zipfile
            import tempfile
            
            # PyTorch checkpoints are zip files
            with zipfile.ZipFile(checkpoint_path, 'r') as z:
                # Extract to temp location
                with tempfile.TemporaryDirectory() as tmpdir:
                    z.extractall(tmpdir)
                    tmppath = Path(tmpdir)
                    
                    # Try to load data.pkl
                    data_pkl = tmppath / 'data.pkl'
                    if data_pkl.exists():
                        with open(data_pkl, 'rb') as f:
                            checkpoint = pickle.load(f, encoding='latin1', fix_imports=True)
                        return checkpoint, "zip_extraction"
        except Exception as e4:
            pass
        
        return None, "all_methods_failed"
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return None, "fatal_error"

# Test on all checkpoints
print("="*80)
print("TESTING CHECKPOINT LOADING ON ALL CHECKPOINTS")
print("="*80)

EXPERIMENTS_DIR = PROJECT_ROOT / "notebooks_sandbox" / "experiments"

checkpoint_files = []
for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue
    checkpoints_dir = exp_dir / "checkpoints"
    if checkpoints_dir.exists():
        for ckpt_file in checkpoints_dir.glob("best_model.pth"):
            checkpoint_files.append({
                'experiment': exp_dir.name,
                'path': ckpt_file
            })

print(f"\nFound {len(checkpoint_files)} checkpoints to test")

successful = []
failed = []

for i, ckpt_info in enumerate(checkpoint_files[:5]):  # Test first 5
    print(f"\n[{i+1}/{len(checkpoint_files)}] Testing: {ckpt_info['experiment']}")
    checkpoint, method = load_checkpoint_robust(ckpt_info['path'])
    
    if checkpoint is not None:
        print(f"  ✓ SUCCESS using method: {method}")
        print(f"    Type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"    Keys: {list(checkpoint.keys())[:5]}")
        successful.append({**ckpt_info, 'method': method})
    else:
        print(f"  ✗ FAILED: {method}")
        failed.append(ckpt_info)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Successful: {len(successful)}/{len(checkpoint_files[:5])}")
print(f"Failed: {len(failed)}/{len(checkpoint_files[:5])}")

if successful:
    print(f"\n✓ Found working method: {successful[0]['method']}")
    print("Can proceed with prediction generation!")
else:
    print("\n✗ All methods failed")
    print("Need alternative approach")
