#!/bin/bash
################################################################################
# RUN ALL EXPERIMENTS - COMPREHENSIVE LAUNCH SCRIPT
#
# This script runs all combinations of:
# - Models: medium (or specify others)
# - Overlaps: 0, 1, 2
# - All 43 loss configurations (hierarchical system)
# - 300 epochs, batch size 16
# - WandB logging enabled
#
# Usage:
#   ./run_all_experiments.sh                    # Run all with medium model
#   ./run_all_experiments.sh --models tiny      # Run all with tiny model
#   ./run_all_experiments.sh --quick-test       # Quick test (2 epochs)
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
MODELS="tiny small medium large huge"
OVERLAPS="0 1 2"
EPOCHS=200
BATCH_SIZE=24
WANDB="--wandb"
CONDA_ENV="FCL"

# Parse arguments
QUICK_TEST=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --overlaps)
            OVERLAPS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB="--no-wandb"
            shift
            ;;
        --quick-test)
            QUICK_TEST=true
            EPOCHS=2
            MODELS="tiny"
            OVERLAPS="2"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODEL          Model sizes (default: all)"
            echo "                          Options: tiny, small, medium, large, huge"
            echo "  --overlaps \"N M\"        Overlap levels (default: \"0 1 2\")"
            echo "  --epochs N              Number of epochs (default: 200)"
            echo "  --batch-size N          Batch size (default: 24)"
            echo "  --no-wandb              Disable WandB logging"
            echo "  --quick-test            Quick test (2 epochs, tiny model)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all (all models, 3 overlaps, 200 epochs)"
            echo "  $0 --models tiny --quick-test        # Quick test"
            echo "  $0 --models \"medium large\"          # Multiple models"
            echo "  $0 --overlaps \"2\" --epochs 100      # Single overlap, 100 epochs"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}                  ${GREEN}COMPREHENSIVE EXPERIMENT LAUNCHER${NC}                        ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Models:      ${GREEN}${MODELS}${NC}"
echo -e "  Overlaps:    ${GREEN}${OVERLAPS}${NC}"
echo -e "  Epochs:      ${GREEN}${EPOCHS}${NC}"
echo -e "  Batch size:  ${GREEN}${BATCH_SIZE}${NC}"
echo -e "  WandB:       ${GREEN}$([ "$WANDB" = "--wandb" ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Quick test:  ${GREEN}$([ "$QUICK_TEST" = true ] && echo "Yes" || echo "No")${NC}"
echo ""

# Calculate total experiments
MODEL_COUNT=$(echo $MODELS | wc -w)
OVERLAP_COUNT=$(echo $OVERLAPS | wc -w)
# 43 loss configurations in hierarchical system
LOSS_COUNT=43
TOTAL_EXPERIMENTS=$((MODEL_COUNT * OVERLAP_COUNT * LOSS_COUNT))

echo -e "${YELLOW}Experiment Count:${NC}"
echo -e "  Models:      ${MODEL_COUNT}"
echo -e "  Overlaps:    ${OVERLAP_COUNT}"
echo -e "  Losses:      ${LOSS_COUNT} (hierarchical system)"
echo -e "  ${GREEN}Total:       ${TOTAL_EXPERIMENTS} experiments${NC}"
echo ""

# Estimate time (200 epochs)
if [ "$EPOCHS" -eq 200 ]; then
    if [[ "$MODELS" == *"huge"* ]] && [[ "$MODELS" == *"large"* ]] && [[ "$MODELS" == *"medium"* ]]; then
        TIME_EST="~60-80 days (all models)"
    elif [[ "$MODELS" == *"tiny"* ]] && [[ "$MODELS" == *"small"* ]] && [[ "$MODELS" == *"medium"* ]]; then
        TIME_EST="~20-25 days (small-medium models)"
    elif [[ "$MODELS" == *"tiny"* ]]; then
        TIME_EST="~3 days"
    elif [[ "$MODELS" == *"small"* ]]; then
        TIME_EST="~6 days"
    elif [[ "$MODELS" == *"medium"* ]]; then
        TIME_EST="~12 days"
    elif [[ "$MODELS" == *"large"* ]]; then
        TIME_EST="~30 days"
    elif [[ "$MODELS" == *"huge"* ]]; then
        TIME_EST="~90 days"
    else
        TIME_EST="~2-3 weeks"
    fi
else
    TIME_EST="Varies by epochs"
fi

echo -e "${YELLOW}Estimated Time:${NC} ${TIME_EST}"
echo ""

# Confirm unless quick test
if [ "$QUICK_TEST" = false ]; then
    echo -e "${YELLOW}This will run ${TOTAL_EXPERIMENTS} experiments.${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted.${NC}"
        exit 1
    fi
fi

# Activate conda environment
echo -e "${BLUE}Activating conda environment: ${CONDA_ENV}${NC}"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Check if script exists
SCRIPT_PATH="run_advanced_experiments.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: $SCRIPT_PATH not found!${NC}"
    exit 1
fi

# Create log directory
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

echo -e "${GREEN}Starting experiments...${NC}"
echo -e "Logging to: ${LOG_FILE}"
echo ""

# Build command
CMD="python3 $SCRIPT_PATH --models $MODELS --overlaps $OVERLAPS --epochs $EPOCHS --batch-size $BATCH_SIZE $WANDB"

echo -e "${BLUE}Command:${NC}"
echo "  $CMD"
echo ""

# Run experiments
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}                         EXPERIMENTS RUNNING                                ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Run with logging
$CMD 2>&1 | tee $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}                         EXPERIMENTS COMPLETE                               ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All experiments completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Results:${NC}"
    echo -e "  Experiments:  ${TOTAL_EXPERIMENTS}"
    echo -e "  Log file:     ${LOG_FILE}"
    echo -e "  Output dir:   ./experiments/"
    echo -e "  Summary:      ./experiments/advanced_experiments_summary.csv"
    echo ""
    
    # Show summary if exists
    SUMMARY_FILE="experiments/advanced_experiments_summary.csv"
    if [ -f "$SUMMARY_FILE" ]; then
        echo -e "${YELLOW}Summary Preview:${NC}"
        head -n 10 "$SUMMARY_FILE" | column -t -s,
        echo ""
    fi
    
    echo -e "${GREEN}Next steps:${NC}"
    echo -e "  1. Check WandB dashboard for visualizations"
    echo -e "  2. Analyze results: experiments/advanced_experiments_summary.csv"
    echo -e "  3. View attention heatmaps: experiments/*/attention_heatmaps/"
    echo -e "  4. Examine topology: experiments/*/topology/"
    echo -e "  5. Load predicted weights: experiments/*/predicted_weights/"
else
    echo -e "${RED}✗ Experiments failed with exit code: $EXIT_CODE${NC}"
    echo -e "${YELLOW}Check log file for details: ${LOG_FILE}${NC}"
    exit $EXIT_CODE
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
