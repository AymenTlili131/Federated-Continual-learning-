#!/bin/bash
################################################################################
# Comprehensive FCL Experiment Runner
# Runs cross-experiments with all model sizes, overlaps, and loss functions
# Includes metrics tracking, finetuning, attention visualization, and WandB logging
################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="FCL"
OUTPUT_DIR="${SCRIPT_DIR}/experiments"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_header() {
    echo ""
    echo -e "${CYAN}================================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================================================================${NC}"
    echo ""
}

# Parse arguments
MODELS="tiny small medium large huge"
OVERLAPS="0 1 2"
LOSSES="mse wasserstein lwwn mape"
EPOCHS=100
USE_WANDB=false
SINGLE_EXP=false

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
        --losses)
            LOSSES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --single)
            SINGLE_EXP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models SIZES      Model sizes (default: tiny small medium large huge)"
            echo "  --overlaps LEVELS   Overlap levels (default: 0 1 2)"
            echo "  --losses FUNCS      Loss functions (default: mse wasserstein lwwn mape)"
            echo "  --epochs N          Epochs per experiment (default: 100)"
            echo "  --wandb             Enable WandB logging"
            echo "  --single            Run single experiment (first of each list)"
            echo "  --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  # Run full suite with WandB"
            echo "  $0 --wandb --epochs 100"
            echo ""
            echo "  # Run single quick test"
            echo "  $0 --single --models tiny --overlaps 2 --losses mse --epochs 10"
            echo ""
            echo "  # Run specific subset"
            echo "  $0 --models \"medium large\" --overlaps \"0 2\" --losses \"mse wasserstein\""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_header "FCL Comprehensive Experiment Runner"

log_info "Configuration:"
echo "  Models:   $MODELS"
echo "  Overlaps: $OVERLAPS"
echo "  Losses:   $LOSSES"
echo "  Epochs:   $EPOCHS"
echo "  WandB:    $USE_WANDB"
echo "  Single:   $SINGLE_EXP"
echo "  Output:   $OUTPUT_DIR"
echo ""

# Activate conda environment
log_info "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Check Python script exists
PYTHON_SCRIPT="${SCRIPT_DIR}/run_full_experiments.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Make script executable
chmod +x "$PYTHON_SCRIPT"

# Build command
CMD="python3 $PYTHON_SCRIPT"
CMD="$CMD --model-sizes $MODELS"
CMD="$CMD --overlaps $OVERLAPS"
CMD="$CMD --losses $LOSSES"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --output-dir $OUTPUT_DIR"

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --wandb"
fi

if [ "$SINGLE_EXP" = true ]; then
    CMD="$CMD --single"
fi

# Run experiments
print_header "Starting Experiments"

log_info "Command: $CMD"
echo ""

START_TIME=$(date +%s)

if $CMD; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_header "Experiments Complete!"
    
    log_success "Total duration: ${DURATION}s ($((DURATION/60))m)"
    log_info "Results saved to: $OUTPUT_DIR"
    
    # Show summary
    if [ -f "$OUTPUT_DIR/experiment_summary.csv" ]; then
        echo ""
        log_info "Experiment Summary:"
        echo ""
        head -20 "$OUTPUT_DIR/experiment_summary.csv" | column -t -s,
    fi
    
    exit 0
else
    log_error "Experiments failed!"
    exit 1
fi
