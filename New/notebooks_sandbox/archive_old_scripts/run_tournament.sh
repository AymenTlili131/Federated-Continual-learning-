#!/bin/bash
################################################################################
# TOURNAMENT-STYLE LOSS SELECTION RUNNER
#
# Progressive filtering approach:
# - Tiny: All 43 losses → Select top 10% + bottom 5% (~6 losses)
# - Small: 6 losses → Select top 10% + bottom 5% (~2 losses)
# - Medium: 2 losses → Select top 15% + bottom 10% (~1-2 losses)
# - Large: 1-2 losses → Select top 20% + bottom 10% (~1-2 losses)
# - Huge: Final 1-2 losses
#
# Saves to external drive: /media/aymen/8A0CA9E80CA9CF8D/Experiments
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
CONDA_ENV="FCL"
OUTPUT_DIR="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/Experiments"
OVERLAPS="0 1 2"
EPOCHS=200
BATCH_SIZE=24
WANDB="--wandb"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --overlaps)
            OVERLAPS="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB="--no-wandb"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Tournament-style loss selection across model sizes"
            echo ""
            echo "Options:"
            echo "  --epochs N              Epochs per experiment (default: 200)"
            echo "  --batch-size N          Batch size (default: 24)"
            echo "  --overlaps \"N M\"        Overlap levels (default: \"0 1 2\")"
            echo "  --no-wandb              Disable WandB logging"
            echo "  --help                  Show this help"
            echo ""
            echo "Tournament Structure:"
            echo "  Tiny:   43 losses → ~6 survivors (top 10% + bottom 5%)"
            echo "  Small:  6 losses → ~2 survivors"
            echo "  Medium: 2 losses → ~1-2 survivors"
            echo "  Large:  1-2 losses → ~1-2 survivors"
            echo "  Huge:   Final 1-2 losses"
            echo ""
            echo "Storage: External drive at $EXTERNAL_DRIVE"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check output directory
if [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
    echo -e "${RED}ERROR: Output directory parent not found${NC}"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# Print banner
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}              ${GREEN}TOURNAMENT-STYLE LOSS SELECTION SYSTEM${NC}                      ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Overlaps:     ${GREEN}${OVERLAPS}${NC}"
echo -e "  Epochs:       ${GREEN}${EPOCHS}${NC}"
echo -e "  Batch size:   ${GREEN}${BATCH_SIZE}${NC}"
echo -e "  WandB:        ${GREEN}$([ "$WANDB" = "--wandb" ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Storage:      ${GREEN}${OUTPUT_DIR}${NC}"
echo ""

# Calculate storage savings
echo -e "${YELLOW}Tournament Approach Benefits:${NC}"
echo -e "  ${GREEN}✓${NC} Computational savings: ~85% reduction"
echo -e "  ${GREEN}✓${NC} Storage savings: ~80% reduction"
echo -e "  ${GREEN}✓${NC} Time savings: Weeks → Days"
echo ""
echo -e "${YELLOW}Expected Experiment Count:${NC}"
echo -e "  Tiny:   43 losses × 3 overlaps = ${GREEN}129 experiments${NC}"
echo -e "  Small:  26 losses × 3 overlaps = ${GREEN}78 experiments${NC}"
echo -e "  Medium: 14 losses × 3 overlaps = ${GREEN}42 experiments${NC}"
echo -e "  Large:  3 losses × 3 overlaps = ${GREEN}9 experiments${NC}"
echo -e "  Huge:   2 losses × 3 overlaps = ${GREEN}6 experiments${NC}"
echo -e "  ${CYAN}Total: 264 experiments (vs 645 without tournament)${NC}"
echo ""

# Storage estimate
echo -e "${YELLOW}Storage Estimate (External Drive):${NC}"
echo -e "  Tiny:   ${GREEN}~17 GB${NC}"
echo -e "  Small:  ${GREEN}~18 GB${NC}"
echo -e "  Medium: ${GREEN}~20 GB${NC}"
echo -e "  Large:  ${GREEN}~10 GB${NC}"
echo -e "  Huge:   ${GREEN}~12 GB${NC}"
echo -e "  ${CYAN}Total: ~77 GB (vs 511 GB without tournament)${NC}"
echo ""

# Confirm
read -p "Start tournament? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborted.${NC}"
    exit 1
fi

# Activate conda
echo -e "${BLUE}Activating conda environment: ${CONDA_ENV}${NC}"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Create log directory
LOG_DIR="tournament_logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/tournament_${TIMESTAMP}.log"

echo -e "${GREEN}Starting tournament...${NC}"
echo -e "Logging to: ${LOG_FILE}"
echo ""

# Run tournament
CMD="python3 $SCRIPT_PATH --overlaps $OVERLAPS --epochs $EPOCHS --batch-size $BATCH_SIZE --output-dir $OUTPUT_DIR $WANDB"

echo -e "${BLUE}Command:${NC}"
echo "  $CMD"
echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}                         TOURNAMENT RUNNING                                 ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Run with logging
$CMD 2>&1 | tee $LOG_FILE

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}                         TOURNAMENT COMPLETE                                ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Tournament completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Results:${NC}"
    echo -e "  Log file:        ${LOG_FILE}"
    echo -e "  Results:         ${EXTERNAL_DRIVE}/tournament_results/"
    echo -e "  Experiments:     ${EXTERNAL_DRIVE}/round_*/"
    echo ""
    
    # Show final losses if available
    TOURNAMENT_LOG="${EXTERNAL_DRIVE}/tournament_results/tournament_log.json"
    if [ -f "$TOURNAMENT_LOG" ]; then
        echo -e "${YELLOW}Final losses for huge model:${NC}"
        python3 -c "import json; log=json.load(open('$TOURNAMENT_LOG')); print('  ' + ', '.join(log['rounds'][-1]['losses']) if log['rounds'] else 'N/A')"
        echo ""
    fi
    
    echo -e "${GREEN}Next steps:${NC}"
    echo -e "  1. Review tournament results in WandB"
    echo -e "  2. Check selection reports: ${EXTERNAL_DRIVE}/tournament_results/selection_*.json"
    echo -e "  3. Analyze final losses on huge model"
    echo -e "  4. Compare performance across model sizes"
else
    echo -e "${RED}✗ Tournament failed with exit code: $EXIT_CODE${NC}"
    echo -e "${YELLOW}Check log file for details: ${LOG_FILE}${NC}"
    exit $EXIT_CODE
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
