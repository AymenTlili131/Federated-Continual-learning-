#!/bin/bash
# Run all CVPR analysis scripts in sequence

echo "=========================================="
echo "CVPR PAPER: RUNNING ALL ANALYSES"
echo "=========================================="

cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/cvpr_analysis_scripts

# 1. Collect experiment data
echo ""
echo "1. Collecting experiment data..."
conda run -n FCL python3 01_collect_experiment_data.py

# 2. Statistical analysis
echo ""
echo "2. Running statistical analysis..."
conda run -n FCL python3 02_statistical_analysis.py

# 3. Eigenvalue analysis
echo ""
echo "3. Running eigenvalue analysis..."
conda run -n FCL python3 03_eigenvalue_analysis.py

# 4. Topological analysis
echo ""
echo "4. Running topological analysis..."
conda run -n FCL python3 04_topological_analysis.py

# 5. Download WandB artifacts (optional - requires login)
echo ""
echo "5. Downloading WandB artifacts (optional)..."
echo "Skipping WandB download - run manually if needed"
# conda run -n FCL python3 05_download_wandb_artifacts.py

echo ""
echo "=========================================="
echo "ALL ANALYSES COMPLETE"
echo "=========================================="
echo "Figures saved to: ../CVPR 2026/figures/"
echo "Data saved to: ./data/"
