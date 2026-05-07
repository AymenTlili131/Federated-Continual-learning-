#!/usr/bin/env python3
"""
Final Checkpoint Loader - Import actual config module from core_modules
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add core_modules to path FIRST
PROJECT_ROOT = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New")
CORE_MODULES = PROJECT_ROOT / "notebooks_sandbox" / "core_modules"
sys.path.insert(0, str(CORE_MODULES))
sys.path.insert(0, str(PROJECT_ROOT))

# Now import config and other modules
try:
    from config import MODEL_CONFIGS
    print("✓ Imported config.MODEL_CONFIGS")
except ImportError as e:
    print(f"✗ Failed to import config: {e}")
    # Create mock if needed
    from types import ModuleType
    config = ModuleType('config')
    config.MODEL_CONFIGS = {}
    sys.modules['config'] = config
    print("✓ Created mock config module")

# Import other potentially needed modules
try:
    from advanced_losses import HierarchicalLossRegistry
    print("✓ Imported advanced_losses")
except:
    print("⚠ Could not import advanced_losses (may not be needed)")

# Import transformer
import torch
import numpy as np
from Double_input_transformer import TransformerAE

print("\n" + "="*80)
print("TESTING CHECKPOINT LOADING WITH PROPER ENVIRONMENT")
print("="*80)

# Test loading
EXPERIMENTS_DIR = PROJECT_ROOT / "notebooks_sandbox" / "experiments"

test_checkpoints = []
for exp_dir in sorted(EXPERIMENTS_DIR.iterdir())[:5]:
    if not exp_dir.is_dir():
        continue
    ckpt_dir = exp_dir / "checkpoints"
    if ckpt_dir.exists():
        best_model = ckpt_dir / "best_model.pth"
        if best_model.exists():
            test_checkpoints.append({
                'experiment': exp_dir.name,
                'path': best_model
            })

print(f"\nTesting {len(test_checkpoints)} checkpoints...")

successful = 0
failed = 0

for ckpt_info in test_checkpoints:
    try:
        print(f"\n[{ckpt_info['experiment']}]")
        checkpoint = torch.load(ckpt_info['path'], map_location='cpu', weights_only=False)
        
        print(f"  ✓ Loaded successfully!")
        print(f"    Keys: {list(checkpoint.keys())}")
        
        if 'config' in checkpoint:
            print(f"    Config type: {type(checkpoint['config'])}")
            print(f"    Config keys: {list(checkpoint['config'].keys())[:5]}")
        
        if 'model_state_dict' in checkpoint:
            print(f"    State dict params: {len(checkpoint['model_state_dict'])}")
        
        successful += 1
        
    except Exception as e:
        print(f"  ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
        failed += 1

print("\n" + "="*80)
print(f"RESULTS: {successful}/{len(test_checkpoints)} successful")
print("="*80)

if successful > 0:
    print("\n✓✓✓ CHECKPOINT LOADING FIXED! ✓✓✓")
    print("Can now proceed with prediction generation!")
else:
    print("\n✗ Still failing - need different approach")
