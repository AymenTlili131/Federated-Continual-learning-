#!/bin/bash
# Run Tiny Model Tournament - Sequential Batches
# Launches batch1 and batch2 in parallel, then merges results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_phase() {
    echo -e "${BLUE}[PHASE]${NC} $1"
}

# Main execution
main() {
    log_info "=========================================="
    log_info "TINY MODEL TOURNAMENT - SEQUENTIAL BATCHES"
    log_info "=========================================="
    
    # Run all 3 overlaps
    for overlap in 0 1 2; do
        log_phase "Starting overlap $overlap"
        
        # Launch batch1 and batch2 in parallel
        log_info "Launching batch 1 (background)..."
        conda run -n FCL python3 "$SCRIPT_DIR/train_tiny_batch1.py" \
            --overlap $overlap \
            > "$SCRIPT_DIR/batch1_overlap${overlap}.log" 2>&1 &
        BATCH1_PID=$!
        
        log_info "Launching batch 2 (background)..."
        conda run -n FCL python3 "$SCRIPT_DIR/train_tiny_batch2.py" \
            --overlap $overlap \
            > "$SCRIPT_DIR/batch2_overlap${overlap}.log" 2>&1 &
        BATCH2_PID=$!
        
        log_info "Batch 1 PID: $BATCH1_PID"
        log_info "Batch 2 PID: $BATCH2_PID"
        log_info "Both batches running in parallel..."
        
        # Wait for both to complete
        wait $BATCH1_PID
        BATCH1_EXIT=$?
        
        wait $BATCH2_PID
        BATCH2_EXIT=$?
        
        if [ $BATCH1_EXIT -eq 0 ] && [ $BATCH2_EXIT -eq 0 ]; then
            log_info "✓ Both batches completed successfully for overlap $overlap"
        else
            log_warn "⚠ Some batches failed for overlap $overlap (batch1: $BATCH1_EXIT, batch2: $BATCH2_EXIT)"
        fi
        
        log_info "Completed overlap $overlap"
        echo ""
    done
    
    # Merge all results
    log_phase "Merging all batch results..."
    conda run -n FCL python3 "$SCRIPT_DIR/merge_batch_results.py"
    
    log_info "=========================================="
    log_info "TINY TOURNAMENT COMPLETE!"
    log_info "=========================================="
    log_info "Results merged and ready for ranking"
    log_info "Next step: Run ranking script"
}

main "$@"
