#!/bin/bash
# Merge Batch Results and Continue Tournament
# Run this AFTER both batch1 and batch2 training scripts complete

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_phase() {
    echo -e "${BLUE}[PHASE]${NC} $1"
}

main() {
    log_info "=========================================="
    log_info "MERGE RESULTS & CONTINUE TOURNAMENT"
    log_info "=========================================="
    
    # Step 1: Merge batch results
    log_phase "Step 1: Merging batch results..."
    conda run -n FCL python3 "$SCRIPT_DIR/merge_batch_results.py"
    
    if [ $? -ne 0 ]; then
        log_error "Failed to merge results. Check that both batch1 and batch2 completed."
        exit 1
    fi
    
    log_info "✓ Results merged successfully"
    echo ""
    
    # Step 2: Rank tiny results
    log_phase "Step 2: Ranking tiny model results..."
    conda run -n FCL python3 "$SCRIPT_DIR/tournament_system/per_overlap_ranking.py" \
        --model-size tiny \
        --top-n 20 \
        --bottom-n 10 \
        --output "$SCRIPT_DIR/rankings_tiny.json"
    
    if [ $? -ne 0 ]; then
        log_error "Failed to rank results"
        exit 1
    fi
    
    log_info "✓ Tiny models ranked"
    log_info "Rankings saved to: rankings_tiny.json"
    echo ""
    
    # Step 3: Display selected losses for next phase
    log_phase "Step 3: Extracting selected losses for small models..."
    
    if [ ! -f "$SCRIPT_DIR/rankings_tiny.json" ]; then
        log_error "rankings_tiny.json not found!"
        exit 1
    fi
    
    # Extract and display selected losses per overlap
    for overlap in 0 1 2; do
        log_info "Overlap $overlap - Selected losses:"
        
        # Extract combined losses (top + bottom)
        losses=$(python3 -c "
import json
with open('$SCRIPT_DIR/rankings_tiny.json', 'r') as f:
    data = json.load(f)
    combined = data['rankings_per_overlap']['$overlap']['combined']
    print(', '.join(combined))
")
        
        echo "  $losses"
    done
    
    echo ""
    log_info "=========================================="
    log_info "PHASE 1 COMPLETE - TINY MODELS"
    log_info "=========================================="
    log_info "Next: Run small model experiments with selected losses"
    log_info "Use rankings_tiny.json to get loss lists for each overlap"
}

main "$@"
