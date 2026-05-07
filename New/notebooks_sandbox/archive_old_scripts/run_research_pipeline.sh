#!/bin/bash
################################################################################
# Weight-Space Hyper-Representation Research Pipeline
# 
# This script orchestrates a comprehensive research pipeline for analyzing
# neural network weights using transformer-based hypernetworks.
#
# Key Components:
# 1. Baseline checkpoint creation
# 2. Gated attention TransformerAE training (medium/large)
# 3. WandB visualization (PCA/t-SNE/UMAP evolution as GIFs)
# 4. Scenario-based training (activations + CNN sampling epochs)
# 5. Weight prediction + finetuning pipeline
# 6. Eigenvalue & persistent homology tracking
# 7. Exponential smoothing for visualization
# 8. Random Matrix Theory analysis
# 9. HMMR time-series segmentation
# 10. Weight disentanglement (training vs generalization)
# 11. Gradient evolution & oscillation tracking
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New"
SCRIPTS_DIR="${PROJECT_ROOT}/research_scripts"
DATA_DIR="${PROJECT_ROOT}/data"
RESULTS_DIR="${PROJECT_ROOT}/research_results"
CHECKPOINTS_DIR="${RESULTS_DIR}/checkpoints"
WANDB_DIR="${RESULTS_DIR}/wandb"

# Configuration
MODEL_SIZE="${1:-medium}"  # medium or large
NUM_EPOCHS="${2:-500}"     # Default 500 epochs (increased from 100)
BATCH_SIZE="${3:-32}"
LEARNING_RATE="${4:-1e-4}"
WANDB_PROJECT="weight-space-research"
WANDB_ENTITY="${WANDB_ENTITY:-}"  # Set via environment variable

# Logging
LOG_FILE="${RESULTS_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1" | tee -a "$LOG_FILE"
}

check_dependencies() {
    log_step "Checking dependencies..."
    
    # Check Python packages
    python3 -c "import torch; import wandb; import numpy; import pandas; import scipy" 2>/dev/null || {
        log_error "Missing required Python packages. Please install requirements."
        exit 1
    }
    
    # Check CUDA availability
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        log_info "CUDA is available"
    else
        log_warn "CUDA not available, will use CPU (slower)"
    fi
    
    log_info "All dependencies satisfied"
}

create_directories() {
    log_step "Creating directory structure..."
    
    mkdir -p "$SCRIPTS_DIR"
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$CHECKPOINTS_DIR"
    mkdir -p "$WANDB_DIR"
    mkdir -p "${RESULTS_DIR}/visualizations"
    mkdir -p "${RESULTS_DIR}/metrics"
    mkdir -p "${RESULTS_DIR}/finetuned_weights"
    mkdir -p "${RESULTS_DIR}/segmentation"
    mkdir -p "${RESULTS_DIR}/disentanglement"
    
    log_info "Directory structure created"
}

################################################################################
# Pipeline Steps
################################################################################

step_1_create_baseline() {
    log_step "Step 1: Creating baseline checkpoint..."
    
    python3 "${SCRIPTS_DIR}/01_create_baseline.py" \
        --model_size "$MODEL_SIZE" \
        --output_dir "$CHECKPOINTS_DIR" \
        --data_dir "$DATA_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "Baseline checkpoint created: ${CHECKPOINTS_DIR}/baseline_${MODEL_SIZE}.pth"
}

step_2_train_gated_transformer() {
    log_step "Step 2: Training Gated Attention TransformerAE..."
    
    python3 "${SCRIPTS_DIR}/02_train_gated_transformer.py" \
        --model_size "$MODEL_SIZE" \
        --baseline_checkpoint "${CHECKPOINTS_DIR}/baseline_${MODEL_SIZE}.pth" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --output_dir "$CHECKPOINTS_DIR" \
        --enable_gated_attention \
        --track_attention_entropy \
        --track_gradient_norms \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "Gated TransformerAE training complete"
}

step_3_visualize_evolution() {
    log_step "Step 3: Creating dimensionality reduction visualizations..."
    
    python3 "${SCRIPTS_DIR}/03_visualize_neck_evolution.py" \
        --checkpoint_dir "$CHECKPOINTS_DIR" \
        --output_dir "${RESULTS_DIR}/visualizations" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --create_gifs \
        --methods pca tsne umap \
        --fps 10 \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "Visualizations created and uploaded to WandB"
}

step_4_scenario_training() {
    log_step "Step 4: Running scenario-based training..."
    
    python3 "${SCRIPTS_DIR}/04_scenario_training.py" \
        --baseline_checkpoint "${CHECKPOINTS_DIR}/baseline_${MODEL_SIZE}.pth" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "${RESULTS_DIR}/scenarios" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --all_activations \
        --cnn_sampling_epochs 10,20,30,36 \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "Scenario-based training complete"
}

step_5_predict_and_finetune() {
    log_step "Step 5: Weight prediction and finetuning pipeline..."
    
    python3 "${SCRIPTS_DIR}/05_predict_and_finetune.py" \
        --checkpoint_dir "$CHECKPOINTS_DIR" \
        --test_data_dir "$DATA_DIR" \
        --output_dir "${RESULTS_DIR}/finetuned_weights" \
        --finetune_epochs 1,2,3,4,5 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --save_all_checkpoints \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "Weight prediction and finetuning complete"
}

step_6_track_eigenvalues_homology() {
    log_step "Step 6: Tracking eigenvalues and persistent homology..."
    
    python3 "${SCRIPTS_DIR}/06_track_spectral_topology.py" \
        --checkpoint_dir "$CHECKPOINTS_DIR" \
        --finetuned_dir "${RESULTS_DIR}/finetuned_weights" \
        --output_dir "${RESULTS_DIR}/metrics" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --compute_eigenvalues \
        --compute_persistent_homology \
        --opacity_levels 0.3,0.5,0.7,1.0 \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "Spectral and topological analysis complete"
}

step_7_exponential_smoothing() {
    log_step "Step 7: Applying exponential smoothing to predicted weights..."
    
    python3 "${SCRIPTS_DIR}/07_smooth_predictions.py" \
        --predictions_dir "${RESULTS_DIR}/finetuned_weights" \
        --output_dir "${RESULTS_DIR}/smoothed_weights" \
        --alpha 0.1,0.3,0.5 \
        --create_comparison_plots \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "Exponential smoothing applied"
}

step_8_rmt_analysis() {
    log_step "Step 8: Random Matrix Theory analysis..."
    
    python3 "${SCRIPTS_DIR}/08_rmt_analysis.py" \
        --checkpoint_dir "$CHECKPOINTS_DIR" \
        --finetuned_dir "${RESULTS_DIR}/finetuned_weights" \
        --output_dir "${RESULTS_DIR}/metrics" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --batch_analysis \
        --class_grouped_analysis \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "RMT analysis complete"
}

step_9_hmmr_segmentation() {
    log_step "Step 9: HMMR time-series segmentation..."
    
    python3 "${SCRIPTS_DIR}/09_hmmr_segmentation.py" \
        --predictions_dir "${RESULTS_DIR}/finetuned_weights" \
        --output_dir "${RESULTS_DIR}/segmentation" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --num_states 5,10,15 \
        --cluster_subsequences \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "HMMR segmentation complete"
}

step_10_weight_disentanglement() {
    log_step "Step 10: Weight disentanglement analysis..."
    
    python3 "${SCRIPTS_DIR}/10_disentangle_weights.py" \
        --checkpoint_dir "$CHECKPOINTS_DIR" \
        --finetuned_dir "${RESULTS_DIR}/finetuned_weights" \
        --output_dir "${RESULTS_DIR}/disentanglement" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --track_gradients \
        --track_oscillations \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "Weight disentanglement complete"
}

step_11_generate_report() {
    log_step "Step 11: Generating comprehensive research report..."
    
    python3 "${SCRIPTS_DIR}/11_generate_report.py" \
        --results_dir "$RESULTS_DIR" \
        --output_file "${RESULTS_DIR}/research_report.html" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        2>&1 | tee -a "$LOG_FILE"
    
    log_info "Research report generated: ${RESULTS_DIR}/research_report.html"
}

################################################################################
# Main Pipeline Execution
################################################################################

main() {
    echo "================================================================================"
    echo "Weight-Space Hyper-Representation Research Pipeline"
    echo "================================================================================"
    echo "Model Size: $MODEL_SIZE"
    echo "Epochs: $NUM_EPOCHS"
    echo "Batch Size: $BATCH_SIZE"
    echo "Learning Rate: $LEARNING_RATE"
    echo "Results Directory: $RESULTS_DIR"
    echo "Log File: $LOG_FILE"
    echo "================================================================================"
    echo ""
    
    # Preliminary checks
    check_dependencies
    create_directories
    
    # Execute pipeline steps
    step_1_create_baseline
    step_2_train_gated_transformer
    step_3_visualize_evolution
    step_4_scenario_training
    step_5_predict_and_finetune
    step_6_track_eigenvalues_homology
    step_7_exponential_smoothing
    step_8_rmt_analysis
    step_9_hmmr_segmentation
    step_10_weight_disentanglement
    step_11_generate_report
    
    echo ""
    echo "================================================================================"
    log_info "Pipeline execution complete!"
    echo "================================================================================"
    echo "Results available at: $RESULTS_DIR"
    echo "WandB project: $WANDB_PROJECT"
    echo "Log file: $LOG_FILE"
    echo "================================================================================"
}

# Run main pipeline
main "$@"
