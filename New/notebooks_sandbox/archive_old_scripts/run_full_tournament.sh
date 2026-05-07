#!/bin/bash

################################################################################
# Full Tournament Runner - All Models, Overlaps, and Losses
# 
# Runs complete tournament across:
# - All model sizes (tiny, small, medium, large, huge)
# - All overlaps (0, 1, 2)
# - All loss functions (91 losses from experiment sequence)
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

# Tournament parameters
MODEL_SIZES=("tiny" "small" "medium" "large" "huge")
OVERLAPS=(0 1 2)
EPOCHS=200
BATCH_SIZE=24
LR=0.0001
CNN_VALIDATION_FREQ=25
CNN_VALIDATION_SAMPLES=100
WANDB_ENABLED="--wandb"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            IFS=',' read -ra MODEL_SIZES <<< "$2"
            shift 2
            ;;
        --overlaps)
            IFS=',' read -ra OVERLAPS <<< "$2"
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
            MODEL_SIZES=("tiny")
            OVERLAPS=(0)
            EPOCHS=50
            CNN_VALIDATION_FREQ=25
            CNN_VALIDATION_SAMPLES=10
            echo -e "${YELLOW}TEST MODE: tiny model, overlap 0 only, 50 epochs${NC}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Runs full tournament across all models, overlaps, and losses"
            echo ""
            echo "Options:"
            echo "  --models M1,M2,...      Model sizes (tiny,small,medium,large,huge) [default: all]"
            echo "  --overlaps O1,O2,...    Overlaps (0,1,2) [default: all]"
            echo "  --epochs N              Training epochs [default: 200]"
            echo "  --batch-size N          Batch size [default: 24]"
            echo "  --lr FLOAT              Learning rate [default: 0.0001]"
            echo "  --cnn-freq N            CNN validation frequency [default: 25]"
            echo "  --cnn-samples N         CNN validation samples [default: 100]"
            echo "  --no-wandb              Disable WandB logging"
            echo "  --test                  Quick test mode (tiny, overlap 0, 50 epochs)"
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
echo -e "${CYAN}║${NC}           ${GREEN}FULL TOURNAMENT RUNNER${NC}                                         ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Print configuration
echo -e "${YELLOW}Tournament Configuration:${NC}"
echo -e "  Model sizes:         ${GREEN}${MODEL_SIZES[@]}${NC}"
echo -e "  Overlaps:            ${GREEN}${OVERLAPS[@]}${NC}"
echo -e "  Training epochs:     ${GREEN}${EPOCHS}${NC}"
echo -e "  Batch size:          ${GREEN}${BATCH_SIZE}${NC}"
echo -e "  Learning rate:       ${GREEN}${LR}${NC}"
echo -e "  CNN validation freq: ${GREEN}${CNN_VALIDATION_FREQ} epochs${NC}"
echo -e "  CNN validation samples: ${GREEN}${CNN_VALIDATION_SAMPLES}${NC}"
echo -e "  WandB logging:       ${GREEN}$([ -n "$WANDB_ENABLED" ] && echo "Enabled" || echo "Disabled")${NC}"
echo ""

# Calculate total experiments
# Note: Losses are determined by get_experiment_sequence() in advanced_losses.py (91 losses)
TOTAL_MODELS=${#MODEL_SIZES[@]}
TOTAL_OVERLAPS=${#OVERLAPS[@]}
ESTIMATED_LOSSES=91  # From experiment sequence

TOTAL_EXPERIMENTS=$((TOTAL_MODELS * TOTAL_OVERLAPS * ESTIMATED_LOSSES))

echo -e "${YELLOW}Estimated experiments:${NC}"
echo -e "  Models: ${TOTAL_MODELS}"
echo -e "  Overlaps: ${TOTAL_OVERLAPS}"
echo -e "  Losses: ~${ESTIMATED_LOSSES} (from experiment sequence)"
echo -e "  ${MAGENTA}Total: ~${TOTAL_EXPERIMENTS} experiments${NC}"
echo -e "  ${MAGENTA}Estimated time: ~$((TOTAL_EXPERIMENTS * 3)) hours (~$((TOTAL_EXPERIMENTS * 3 / 24)) days)${NC}"
echo ""

# Check conda environment
echo -e "${YELLOW}Checking environment...${NC}"
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo -e "${RED}ERROR: Conda environment '${CONDA_ENV}' not found${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Conda environment '${CONDA_ENV}' found"

# Activate conda environment
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
    echo -e "${RED}ERROR: MNIST training data not found${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} MNIST data found"

if [ ! -f "${DATA_DIR}/Merged zoo.csv" ]; then
    echo -e "${RED}ERROR: Merged zoo.csv not found${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Merged zoo.csv found"

# Generate all scenarios beforehand
echo -e "${YELLOW}Checking scenarios...${NC}"

SCENARIOS_NEEDED=false
for overlap in "${OVERLAPS[@]}"; do
    SCENARIO_DIR="${DATA_DIR}/Scenario/overlapping_m${overlap}"
    if [ ! -d "${SCENARIO_DIR}" ] || [ ! -f "${SCENARIO_DIR}/train_pairs.npy" ]; then
        SCENARIOS_NEEDED=true
        break
    fi
done

if [ "$SCENARIOS_NEEDED" = true ]; then
    echo -e "  ${YELLOW}Generating scenarios...${NC}"
    python3 ${SCRIPT_DIR}/generate_scenarios.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Scenario generation failed${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Scenarios generated"
else
    echo -e "  ${GREEN}✓${NC} All scenarios exist"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Create log directory
LOG_DIR="${SCRIPT_DIR}/experiment_logs"
mkdir -p ${LOG_DIR}

# Summary file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="${OUTPUT_DIR}/tournament_summary_${TIMESTAMP}.txt"

echo ""
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║${NC}                    STARTING TOURNAMENT                                     ${MAGENTA}║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Confirmation
echo -e "${YELLOW}This will run ${TOTAL_EXPERIMENTS} experiments. Continue? [y/N]${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${RED}Tournament cancelled${NC}"
    conda deactivate
    exit 0
fi

# Initialize summary
echo "Full Tournament - ${TIMESTAMP}" > ${SUMMARY_FILE}
echo "Models: ${MODEL_SIZES[@]}" >> ${SUMMARY_FILE}
echo "Overlaps: ${OVERLAPS[@]}" >> ${SUMMARY_FILE}
echo "Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}, LR: ${LR}" >> ${SUMMARY_FILE}
echo "CNN Validation: Every ${CNN_VALIDATION_FREQ} epochs, ${CNN_VALIDATION_SAMPLES} samples" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

# Run tournament
CURRENT=0
FAILED_EXPERIMENTS=()
SUCCESSFUL_EXPERIMENTS=0

for model_size in "${MODEL_SIZES[@]}"; do
    for overlap in "${OVERLAPS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}  BATCH ${CURRENT}/${TOTAL_MODELS}x${TOTAL_OVERLAPS}: ${model_size} / overlap=${overlap}                                   ${GREEN}║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        # Adaptive settings based on model size
        TOPOLOGY_JOBS=1
        CNN_SAMPLES=${CNN_VALIDATION_SAMPLES}
        TOPO_FREQ=50
        
        case ${model_size} in
            tiny|small)
                TOPOLOGY_JOBS=2
                CNN_SAMPLES=${CNN_VALIDATION_SAMPLES}
                TOPO_FREQ=50
                echo -e "${CYAN}  Settings: topology_jobs=2, cnn_samples=${CNN_SAMPLES}, topo_freq=50${NC}"
                ;;
            medium)
                TOPOLOGY_JOBS=1
                CNN_SAMPLES=$((CNN_VALIDATION_SAMPLES / 2))
                TOPO_FREQ=100
                echo -e "${YELLOW}  Settings: topology_jobs=1, cnn_samples=${CNN_SAMPLES}, topo_freq=100 (conservative)${NC}"
                ;;
            large|huge)
                TOPOLOGY_JOBS=1
                CNN_SAMPLES=$((CNN_VALIDATION_SAMPLES / 2))
                TOPO_FREQ=100
                echo -e "${RED}  Settings: topology_jobs=1, cnn_samples=${CNN_SAMPLES}, topo_freq=100 (very conservative)${NC}"
                ;;
        esac
        
        # Build command - this will run ALL losses from experiment sequence
        CMD="python3 ${SCRIPT_DIR}/run_advanced_experiments.py \
            --models ${model_size} \
            --overlaps ${overlap} \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --lr ${LR} \
            --output-dir ${OUTPUT_DIR} \
            --cnn-validation-freq ${CNN_VALIDATION_FREQ} \
            --cnn-validation-samples ${CNN_SAMPLES} \
            --topology-n-jobs ${TOPOLOGY_JOBS} \
            ${WANDB_ENABLED}"
        
        LOG_FILE="${LOG_DIR}/tournament_${model_size}_overlap${overlap}_${TIMESTAMP}.log"
        
        echo -e "${BLUE}Command:${NC}"
        echo -e "  ${CMD}"
        echo -e "${BLUE}Log:${NC} ${LOG_FILE}"
        echo ""
        
        # Run batch
        START_TIME=$(date +%s)
        
        eval ${CMD} 2>&1 | tee ${LOG_FILE}
        EXIT_CODE=${PIPESTATUS[0]}
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        DURATION_MIN=$((DURATION / 60))
        DURATION_SEC=$((DURATION % 60))
        
        # Record result
        echo "" >> ${SUMMARY_FILE}
        echo "Batch ${CURRENT}: ${model_size} / overlap=${overlap}" >> ${SUMMARY_FILE}
        echo "  Duration: ${DURATION_MIN}m ${DURATION_SEC}s" >> ${SUMMARY_FILE}
        
        if [ ${EXIT_CODE} -eq 0 ]; then
            echo -e "${GREEN}✓ Batch completed (${DURATION_MIN}m ${DURATION_SEC}s)${NC}"
            echo "  Status: SUCCESS" >> ${SUMMARY_FILE}
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + ESTIMATED_LOSSES))
        else
            echo -e "${RED}✗ Batch failed with exit code ${EXIT_CODE}${NC}"
            echo "  Status: FAILED (exit code ${EXIT_CODE})" >> ${SUMMARY_FILE}
            FAILED_EXPERIMENTS+=("${model_size}_overlap${overlap}")
        fi
        
        echo "  Log: ${LOG_FILE}" >> ${SUMMARY_FILE}
        echo "----------------------------------------" >> ${SUMMARY_FILE}
    done
done

# Final summary
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                    TOURNAMENT COMPLETE                                     ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Summary:${NC}"
echo -e "  Total batches: $((TOTAL_MODELS * TOTAL_OVERLAPS))"
echo -e "  Successful batches: $((TOTAL_MODELS * TOTAL_OVERLAPS - ${#FAILED_EXPERIMENTS[@]}))"
echo -e "  Failed batches: ${#FAILED_EXPERIMENTS[@]}"
echo -e "  Estimated successful experiments: ${SUCCESSFUL_EXPERIMENTS}"

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed batches:${NC}"
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
    echo -e "${GREEN}✓ Full tournament completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some batches failed. Check logs for details.${NC}"
    exit 1
fi
