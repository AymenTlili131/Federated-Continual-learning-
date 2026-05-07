#!/bin/bash

################################################################################
# Complete CNN Validation Experiment Runner
# 
# Runs all overlaps (0, 1, 2) with CNN reconstruction and validation
# Generates scenarios beforehand and runs experiments sequentially
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
CONDA_ENV="FCL"
PROJECT_ROOT="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New"
SCRIPT_DIR="${PROJECT_ROOT}/notebooks_sandbox"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/Experiments"

# Experiment parameters
MODEL_SIZE="tiny"
OVERLAPS=(0 1 2)  # Run all overlaps
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
            OVERLAPS=(0)  # Only test overlap 0
            echo -e "${YELLOW}TEST MODE: Reduced parameters, overlap 0 only${NC}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Runs CNN validation experiments for all overlaps (0, 1, 2)"
            echo ""
            echo "Options:"
            echo "  --model-size SIZE       Model size (tiny/small/medium/large/huge) [default: tiny]"
            echo "  --epochs N              Training epochs [default: 200]"
            echo "  --batch-size N          Batch size [default: 24]"
            echo "  --lr FLOAT              Learning rate [default: 0.0001]"
            echo "  --loss NAME             Loss function name [default: MSE]"
            echo "  --cnn-freq N            CNN validation frequency (epochs) [default: 25]"
            echo "  --cnn-samples N         CNN validation samples [default: 100]"
            echo "  --no-wandb              Disable WandB logging"
            echo "  --test                  Quick test mode (50 epochs, overlap 0 only, 10 samples)"
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
echo -e "${CYAN}║${NC}           ${GREEN}COMPLETE CNN VALIDATION EXPERIMENT RUNNER${NC}                      ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Model size:          ${GREEN}${MODEL_SIZE}${NC}"
echo -e "  Overlaps:            ${GREEN}${OVERLAPS[@]}${NC}"
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

# Activate conda environment for all subsequent commands
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to activate conda environment${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Conda environment '${CONDA_ENV}' activated"

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

# Generate all scenarios beforehand
echo -e "${YELLOW}Generating scenarios for all overlaps...${NC}"

SCENARIOS_NEEDED=false
for overlap in "${OVERLAPS[@]}"; do
    SCENARIO_DIR="${DATA_DIR}/Scenario/overlapping_m${overlap}"
    if [ ! -d "${SCENARIO_DIR}" ] || [ ! -f "${SCENARIO_DIR}/train_pairs.npy" ]; then
        SCENARIOS_NEEDED=true
        break
    fi
done

if [ "$SCENARIOS_NEEDED" = true ]; then
    echo -e "  ${YELLOW}Scenarios not found. Generating...${NC}"
    python3 ${SCRIPT_DIR}/generate_scenarios.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Scenario generation failed${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Scenarios generated successfully"
else
    echo -e "  ${GREEN}✓${NC} All scenarios already exist"
    
    # Print scenario statistics
    for overlap in "${OVERLAPS[@]}"; do
        SCENARIO_DIR="${DATA_DIR}/Scenario/overlapping_m${overlap}"
        if [ -f "${SCENARIO_DIR}/metadata.json" ]; then
            echo -e "  ${CYAN}Overlap m=${overlap}:${NC}"
            python3 -c "import json; d=json.load(open('${SCENARIO_DIR}/metadata.json')); print(f\"    Train: {d['train_pairs']:,}, Val: {d['val_pairs']:,}, Test: {d['test_pairs']:,}\")"
        fi
    done
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo -e "  ${GREEN}✓${NC} Output directory ready"

# Create log directory
LOG_DIR="${SCRIPT_DIR}/experiment_logs"
mkdir -p ${LOG_DIR}

# Summary file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="${OUTPUT_DIR}/experiment_summary_${TIMESTAMP}.txt"

echo ""
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║${NC}                    STARTING EXPERIMENTS                                    ${MAGENTA}║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Initialize summary
echo "CNN Validation Experiments - ${TIMESTAMP}" > ${SUMMARY_FILE}
echo "Model: ${MODEL_SIZE}, Epochs: ${EPOCHS}, Loss: ${LOSS_NAME}" >> ${SUMMARY_FILE}
echo "CNN Validation: Every ${CNN_VALIDATION_FREQ} epochs, ${CNN_VALIDATION_SAMPLES} samples" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

# Run experiments for each overlap
TOTAL_OVERLAPS=${#OVERLAPS[@]}
CURRENT=0
FAILED_EXPERIMENTS=()

for overlap in "${OVERLAPS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  EXPERIMENT ${CURRENT}/${TOTAL_OVERLAPS}: Overlap m=${overlap}                                        ${GREEN}║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Create experiment name
    EXP_NAME="${MODEL_SIZE}_overlap${overlap}_${LOSS_NAME}_${TIMESTAMP}"
    EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
    
    echo -e "  Name:      ${CYAN}${EXP_NAME}${NC}"
    echo -e "  Directory: ${CYAN}${EXP_DIR}${NC}"
    echo -e "  Log:       ${CYAN}${LOG_FILE}${NC}"
    echo ""
    
    # Build command
    CMD="python3 ${SCRIPT_DIR}/run_advanced_experiments.py \
        --single \
        --model-size ${MODEL_SIZE} \
        --overlap ${overlap} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --loss ${LOSS_NAME} \
        --output-dir ${EXP_DIR} \
        --cnn-validation-freq ${CNN_VALIDATION_FREQ} \
        --cnn-validation-samples ${CNN_VALIDATION_SAMPLES} \
        ${WANDB_ENABLED}"
    
    echo -e "${BLUE}Command:${NC}"
    echo -e "  ${CMD}"
    echo ""
    
    # Run experiment
    START_TIME=$(date +%s)
    
    eval ${CMD} 2>&1 | tee ${LOG_FILE}
    EXIT_CODE=${PIPESTATUS[0]}
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))
    
    # Record result
    echo "" >> ${SUMMARY_FILE}
    echo "Experiment ${CURRENT}/${TOTAL_OVERLAPS}: ${EXP_NAME}" >> ${SUMMARY_FILE}
    echo "  Overlap: m=${overlap}" >> ${SUMMARY_FILE}
    echo "  Duration: ${DURATION_MIN}m ${DURATION_SEC}s" >> ${SUMMARY_FILE}
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo -e "${GREEN}✓ Experiment completed successfully (${DURATION_MIN}m ${DURATION_SEC}s)${NC}"
        echo "  Status: SUCCESS" >> ${SUMMARY_FILE}
        echo "  Output: ${EXP_DIR}" >> ${SUMMARY_FILE}
    else
        echo -e "${RED}✗ Experiment failed with exit code ${EXIT_CODE}${NC}"
        echo "  Status: FAILED (exit code ${EXIT_CODE})" >> ${SUMMARY_FILE}
        echo "  Log: ${LOG_FILE}" >> ${SUMMARY_FILE}
        FAILED_EXPERIMENTS+=("${EXP_NAME}")
    fi
    
    echo "  Log: ${LOG_FILE}" >> ${SUMMARY_FILE}
    echo "----------------------------------------" >> ${SUMMARY_FILE}
done

# Final summary
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                    ALL EXPERIMENTS COMPLETE                                ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Summary:${NC}"
echo -e "  Total experiments: ${TOTAL_OVERLAPS}"
echo -e "  Successful: $((TOTAL_OVERLAPS - ${#FAILED_EXPERIMENTS[@]}))"
echo -e "  Failed: ${#FAILED_EXPERIMENTS[@]}"

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed experiments:${NC}"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo -e "  - ${exp}"
    done
fi

echo ""
echo -e "  Summary file: ${GREEN}${SUMMARY_FILE}${NC}"
echo -e "  Logs directory: ${GREEN}${LOG_DIR}${NC}"
echo -e "  Output directory: ${GREEN}${OUTPUT_DIR}${NC}"
echo ""

# Deactivate conda environment
conda deactivate

if [ ${#FAILED_EXPERIMENTS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All experiments completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some experiments failed. Check logs for details.${NC}"
    exit 1
fi
