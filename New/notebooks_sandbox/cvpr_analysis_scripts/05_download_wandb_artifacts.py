#!/usr/bin/env python3
"""
Download WandB artifacts and figures for CVPR paper
"""

import wandb
import os
from pathlib import Path
import json
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
FIGURES_DIR = Path("/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/CVPR 2026/figures/wandb")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def download_wandb_runs():
    """Download figures and artifacts from WandB runs"""
    
    # Initialize WandB API
    api = wandb.Api()
    
    # Get project (adjust project name as needed)
    try:
        # Try to get runs from the project
        # You may need to adjust the entity/project path
        runs = api.runs("federated-continual-learning")  # Adjust project name
        
        print(f"Found {len(runs)} runs")
        
        downloaded_count = 0
        
        for run in tqdm(runs[:50], desc="Downloading WandB artifacts"):  # Limit to 50 most recent
            try:
                # Get run name and metadata
                run_name = run.name
                run_id = run.id
                
                # Create directory for this run
                run_dir = FIGURES_DIR / run_name
                run_dir.mkdir(exist_ok=True)
                
                # Download logged images
                for file in run.files():
                    if file.name.endswith(('.png', '.jpg', '.pdf', '.svg')):
                        try:
                            file.download(root=str(run_dir), replace=True)
                            downloaded_count += 1
                        except Exception as e:
                            print(f"Error downloading {file.name}: {e}")
                
                # Save run config and summary
                config_file = run_dir / "config.json"
                with open(config_file, 'w') as f:
                    json.dump(dict(run.config), f, indent=2)
                
                summary_file = run_dir / "summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(dict(run.summary), f, indent=2)
                
            except Exception as e:
                print(f"Error processing run {run.name}: {e}")
        
        print(f"\nDownloaded {downloaded_count} files from WandB")
        
    except Exception as e:
        print(f"Error accessing WandB: {e}")
        print("Make sure you're logged in: wandb login")
        return False
    
    return True

def organize_figures_by_type():
    """Organize downloaded figures by type"""
    
    # Create organized directories
    dirs = {
        'loss_curves': FIGURES_DIR / 'loss_curves',
        'accuracy_plots': FIGURES_DIR / 'accuracy_plots',
        'topology': FIGURES_DIR / 'topology',
        'attention': FIGURES_DIR / 'attention',
        'other': FIGURES_DIR / 'other'
    }
    
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    
    # Move files to appropriate directories
    for run_dir in FIGURES_DIR.iterdir():
        if not run_dir.is_dir() or run_dir.name in dirs.keys():
            continue
        
        for file in run_dir.glob("**/*.png"):
            filename = file.name.lower()
            
            if 'loss' in filename or 'train' in filename:
                shutil.copy(file, dirs['loss_curves'] / file.name)
            elif 'accuracy' in filename or 'acc' in filename:
                shutil.copy(file, dirs['accuracy_plots'] / file.name)
            elif 'topology' in filename or 'persistence' in filename or 'mapper' in filename:
                shutil.copy(file, dirs['topology'] / file.name)
            elif 'attention' in filename or 'head' in filename:
                shutil.copy(file, dirs['attention'] / file.name)
            else:
                shutil.copy(file, dirs['other'] / file.name)
    
    print("\nOrganized figures by type:")
    for name, d in dirs.items():
        count = len(list(d.glob("*.png")))
        print(f"  {name}: {count} files")

def main():
    print("="*80)
    print("DOWNLOADING WANDB ARTIFACTS FOR CVPR PAPER")
    print("="*80)
    
    # Check if wandb is logged in
    try:
        api = wandb.Api()
        print("\nWandB API initialized successfully")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        print("Please login to WandB: wandb login")
        return
    
    # Download runs
    print("\n1. Downloading WandB runs...")
    success = download_wandb_runs()
    
    if success:
        # Organize figures
        print("\n2. Organizing figures by type...")
        organize_figures_by_type()
    
    print("\n" + "="*80)
    print("WANDB DOWNLOAD COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
