#!/bin/bash

################################################################################
# CNN Validation Experiment Launcher
# 
# Runs transformer training with CNN reconstruction and validation
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
CONDA_ENV="FCL"
PROJECT_ROOT="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New"
SCRIPT_DIR="${PROJECT_ROOT}/notebooks_sandbox"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/Experiments"

# Experiment parameters (can be overridden via command line)
MODEL_SIZE="tiny"
OVERLAP=0
EPOCHS=200
BATCH_SIZE=24
LR=0.0001
LOSS_NAME="MSE"
CNN_VALIDATION_FREQ=25
CNN_VALIDATION_SAMPLES=100
WANDB_ENABLED="--wandb"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --overlap)
            OVERLAP="$2"
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
        --lr)
            LR="$2"
            shift 2
            ;;
        --loss)
            LOSS_NAME="$2"
            shift 2
            ;;
        --cnn-freq)
            CNN_VALIDATION_FREQ="$2"
            shift 2
            ;;
        --cnn-samples)
            CNN_VALIDATION_SAMPLES="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB_ENABLED=""
            shift
            ;;
        --test)
            # Quick test mode
            MODEL_SIZE="tiny"
            EPOCHS=50
            CNN_VALIDATION_FREQ=25
            CNN_VALIDATION_SAMPLES=10
            echo -e "${YELLOW}TEST MODE: Reduced parameters${NC}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-size SIZE       Model size (tiny/small/medium/large/huge) [default: tiny]"
            echo "  --overlap N             Task overlap (0/1/2) [default: 0]"
            echo "  --epochs N              Training epochs [default: 200]"
            echo "  --batch-size N          Batch size [default: 24]"
            echo "  --lr FLOAT              Learning rate [default: 0.0001]"
            echo "  --loss NAME             Loss function name [default: MSE]"
            echo "  --cnn-freq N            CNN validation frequency (epochs) [default: 25]"
            echo "  --cnn-samples N         CNN validation samples [default: 100]"
            echo "  --no-wandb              Disable WandB logging"
            echo "  --test                  Quick test mode (50 epochs, 10 samples)"
            echo "  --help                  Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}           ${GREEN}CNN VALIDATION EXPERIMENT LAUNCHER${NC}                             ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Model size:          ${GREEN}${MODEL_SIZE}${NC}"
echo -e "  Task overlap:        ${GREEN}${OVERLAP}${NC}"
echo -e "  Training epochs:     ${GREEN}${EPOCHS}${NC}"
echo -e "  Batch size:          ${GREEN}${BATCH_SIZE}${NC}"
echo -e "  Learning rate:       ${GREEN}${LR}${NC}"
echo -e "  Loss function:       ${GREEN}${LOSS_NAME}${NC}"
echo -e "  CNN validation freq: ${GREEN}${CNN_VALIDATION_FREQ} epochs${NC}"
echo -e "  CNN validation samples: ${GREEN}${CNN_VALIDATION_SAMPLES}${NC}"
echo -e "  WandB logging:       ${GREEN}$([ -n "$WANDB_ENABLED" ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Output directory:    ${GREEN}${OUTPUT_DIR}${NC}"
echo ""

# Check conda environment
echo -e "${YELLOW}Checking environment...${NC}"
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo -e "${RED}ERROR: Conda environment '${CONDA_ENV}' not found${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Conda environment '${CONDA_ENV}' found"

# Check data directories
echo -e "${YELLOW}Checking data directories...${NC}"

if [ ! -d "${DATA_DIR}/SplitMnist/train" ]; then
    echo -e "${RED}ERROR: MNIST training data not found at ${DATA_DIR}/SplitMnist/train${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} MNIST data found"

if [ ! -f "${DATA_DIR}/Merged zoo.csv" ]; then
    echo -e "${RED}ERROR: Merged zoo.csv not found at ${DATA_DIR}/Merged zoo.csv${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Merged zoo.csv found"

# Check if scenarios exist, generate if not
SCENARIO_DIR="${DATA_DIR}/Scenario/overlapping_m${OVERLAP}"
if [ ! -d "${SCENARIO_DIR}" ] || [ ! -f "${SCENARIO_DIR}/train_pairs.npy" ]; then
    echo -e "${YELLOW}Scenarios not found. Generating...${NC}"
    conda run -n ${CONDA_ENV} python3 ${SCRIPT_DIR}/generate_scenarios.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Scenario generation failed${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Scenarios generated"
else
    echo -e "  ${GREEN}✓${NC} Scenarios found at ${SCENARIO_DIR}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo -e "  ${GREEN}✓${NC} Output directory ready"

# Create experiment name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="${MODEL_SIZE}_overlap${OVERLAP}_${LOSS_NAME}_${TIMESTAMP}"
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"

echo ""
echo -e "${YELLOW}Experiment:${NC}"
echo -e "  Name: ${GREEN}${EXP_NAME}${NC}"
echo -e "  Directory: ${GREEN}${EXP_DIR}${NC}"
echo ""

# Prepare command
CMD="conda run -n ${CONDA_ENV} python3 ${SCRIPT_DIR}/run_advanced_experiments.py \
    --single \
    --model-size ${MODEL_SIZE} \
    --overlap ${OVERLAP} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --loss ${LOSS_NAME} \
    --output-dir ${EXP_DIR} \
    --cnn-validation-freq ${CNN_VALIDATION_FREQ} \
    --cnn-validation-samples ${CNN_VALIDATION_SAMPLES} \
    ${WANDB_ENABLED}"

# Create log directory
LOG_DIR="${SCRIPT_DIR}/experiment_logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

echo -e "${YELLOW}Command:${NC}"
echo -e "  ${BLUE}${CMD}${NC}"
echo ""
echo -e "${YELLOW}Log file:${NC}"
echo -e "  ${GREEN}${LOG_FILE}${NC}"
echo ""

# Confirmation
echo -e "${YELLOW}Ready to launch experiment. Continue? [y/N]${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${RED}Experiment cancelled${NC}"
    exit 0
fi

# Launch experiment
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}                         EXPERIMENT RUNNING                                 ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Run with tee to show output and save to log
eval ${CMD} 2>&1 | tee ${LOG_FILE}

EXIT_CODE=${PIPESTATUS[0]}

# Summary
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                         EXPERIMENT COMPLETE                                ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}✓ Experiment completed successfully${NC}"
    echo -e "  Results: ${EXP_DIR}"
    echo -e "  Log: ${LOG_FILE}"
else
    echo -e "${RED}✗ Experiment failed with exit code ${EXIT_CODE}${NC}"
    echo -e "  Check log: ${LOG_FILE}"
fi

echo ""
