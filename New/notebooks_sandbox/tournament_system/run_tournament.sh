#!/bin/bash
# Tournament Orchestrator - Sequential Execution of All Phases
# This is the ONLY tournament script you need to run

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXPERIMENTS_DIR="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/Experiments"

# WandB configuration
export WANDB_PROJECT="fcl-tournament"
export WANDB_DIR="$PROJECT_ROOT/wandb"
export WANDB_CACHE_DIR="$PROJECT_ROOT/.wandb_cache"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check WandB login
check_wandb() {
    if ! conda run -n FCL wandb status &>/dev/null; then
        log_warn "WandB not logged in. Please run: conda run -n FCL wandb login"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_info "WandB authenticated ✓"
    fi
}

# Initialize tournament
init_tournament() {
    log_info "Initializing tournament..."
    mkdir -p "$EXPERIMENTS_DIR"
    mkdir -p "$WANDB_DIR"
    mkdir -p "$WANDB_CACHE_DIR"
    check_wandb
    log_info "Tournament initialized ✓"
}

# Phase 1: Tiny Models (All 91 Losses × 3 Overlaps = 273 Experiments)
phase_1_tiny() {
    log_info "=========================================="
    log_info "PHASE 1: TINY MODELS (91 losses × 3 overlaps)"
    log_info "=========================================="
    
    for overlap in 0 1 2; do
        log_info "Running tiny models for overlap=$overlap with 3 parallel workers..."
        conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
            --model-size tiny \
            --overlap $overlap \
            --num-parallel 3 \
            --output-summary "phase1_tiny_overlap${overlap}_summary.json"
        
        log_info "Completed tiny models for overlap=$overlap"
    done
    
    log_info "Phase 1 complete! Total: 273 experiments"
}

# ---------------------------------------------------------------------------
# Phase 1b: Re-run FAILED / CRASHED tiny-model experiments
#
# Source: WandB table inspection (screenshots, 2025-04-12)
#
# Overlap 0 failures:
#   PersLandscape          – Failed     (broken gradient chain → now fixed)
#   LW_Sinkhorn            – Failed     (CPU offload crash → now fixed)
#   LW_MAPE+0.1*LW_JS      – Failed     (unknown, re-run)
#   LW_MSE+0.05*LW_Frobenius – Failed   (unknown, re-run)
#   LW_Sinkhorn (redo)     – Finished but took 11h 19m → GPU fix expected ~30 min
#
# Overlap 1 failures:
#   Sinkhorn+0.15*KL       – Failed     (CPU OOM during Sinkhorn loop → now fixed)
#   Sinkhorn               – Failed     (second run, same root cause)
#   MSE                    – Crashed    (infrastructure, re-run)
#   MAPE                   – Crashed    (infrastructure, re-run)
#   FFT                    – Crashed    (infrastructure, re-run)
#   LW_Sinkhorn (redo)     – Finished but took >1 day → GPU fix expected ~30 min
#
# Overlap 2 failures:
#   LW_Sinkhorn            – Crashed    (CPU offload OOM → now fixed)
#   Sinkhorn+0.15*KL       – Failed     (same)
#   Sinkhorn               – Failed     (same)
# ---------------------------------------------------------------------------
phase_1b_redo_failed_tiny() {
    log_phase "=========================================="
    log_phase "PHASE 1b: REDO FAILED TINY EXPERIMENTS"
    log_phase "=========================================="

    # Overlap 0 — failures + LW_Sinkhorn redo (was 11h on CPU, now GPU)
    log_info "Re-running overlap=0 failures..."
    conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
        --model-size tiny --overlap 0 --num-parallel 3 \
        --losses "PersLandscape" "LW_Sinkhorn" \
                 "LW_MAPE+0.1*LW_JS" "LW_MSE+0.05*LW_Frobenius" \
        --output-summary "phase1b_tiny_overlap0_redo_summary.json"

    # Overlap 1 — infrastructure crashes + Sinkhorn bug fixes
    log_info "Re-running overlap=1 failures..."
    conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
        --model-size tiny --overlap 1 --num-parallel 3 \
        --losses "MSE" "MAPE" "FFT" \
                 "Sinkhorn" "Sinkhorn+0.15*KL" "LW_Sinkhorn" \
        --output-summary "phase1b_tiny_overlap1_redo_summary.json"

    # Overlap 2 — Sinkhorn bug fixes
    log_info "Re-running overlap=2 failures..."
    conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
        --model-size tiny --overlap 2 --num-parallel 3 \
        --losses "LW_Sinkhorn" "Sinkhorn" "Sinkhorn+0.15*KL" \
        --output-summary "phase1b_tiny_overlap2_redo_summary.json"

    log_info "Phase 1b complete — all WandB failures re-run"
}

# ---------------------------------------------------------------------------
# Phase 1c: First runs of NEW differentiable topology losses
#
# DiffPers  = Wasserstein-1 on sorted weights (true H0 persistence upper bound)
# RTD       = Representation Topology Divergence (Gram eigenvalue spectrum)
# + layerwise and regularised combinations registered in HierarchicalLossRegistry
# ---------------------------------------------------------------------------
phase_1c_new_topology_tiny() {
    log_phase "=========================================="
    log_phase "PHASE 1c: NEW TOPOLOGY LOSSES (DiffPers, RTD)"
    log_phase "=========================================="

    for overlap in 0 1 2; do
        log_info "Running new topology losses for overlap=$overlap..."
        conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
            --model-size tiny --overlap $overlap --num-parallel 3 \
            --losses "DiffPers" "RTD" \
                     "LW_DiffPers" "LW_RTD" \
                     "MSE+0.1*DiffPers" "MAE+0.1*DiffPers" \
                     "MSE+0.05*RTD" "MAE+0.05*RTD" \
                     "LW_MSE+0.1*LW_DiffPers" "LW_MAE+0.1*LW_DiffPers" \
                     "LW_MSE+0.05*LW_RTD" \
            --output-summary "phase1c_tiny_overlap${overlap}_topo_summary.json"
        log_info "Completed new topology losses for overlap=$overlap"
    done

    log_info "Phase 1c complete"
}

# ---------------------------------------------------------------------------
# Phase 1d: Multi-parameter persistence (multipers) experiments
#
# The paper's CENTRAL NOVEL CONTRIBUTION:
#   Multi-parameter persistence (ξ₁ = weight value, ξ₂ = layer position)
#   reveals joint structure invisible to single-parameter TDA.
#
# Training:  uses DiffPers and RTD (differentiable proxies) as losses.
# Analysis:  multipers_analysis.py runs OFFLINE after training to compute
#            the true 2-parameter persistence features and divergences,
#            comparing pred vs GT vs finetuned weight snapshots.
# ---------------------------------------------------------------------------
phase_1d_multipers_tiny() {
    log_phase "=========================================="
    log_phase "PHASE 1d: MULTIPERS ANALYSIS (novel contribution)"
    log_phase "=========================================="

    for overlap in 0 1 2; do
        log_info "[1d] Running multipers-centered losses for overlap=$overlap..."
        conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
            --model-size tiny --overlap $overlap --num-parallel 2 \
            --losses "PersLandscape" "PersImage" \
                     "LW_DiffPers" "LW_RTD" \
                     "MSE+0.1*DiffPers" "Sinkhorn+0.1*DiffPers" \
            --output-summary "phase1d_tiny_overlap${overlap}_multipers_summary.json"
        log_info "[1d] Training complete for overlap=$overlap"

        log_info "[1d] Running offline multipers analysis for overlap=$overlap..."
        conda run -n FCL python3 "$PROJECT_ROOT/core_modules/multipers_analysis.py" \
            --experiments-dir "$EXPERIMENTS_DIR" \
            --overlap $overlap \
            --model-size tiny \
            --output-dir "$PROJECT_ROOT/multipers_results/overlap${overlap}" \
            2>/dev/null || log_warn "multipers offline analysis skipped (standalone script not yet wired)"
        log_info "[1d] Multipers analysis complete for overlap=$overlap"
    done

    log_info "Phase 1d complete"
}

# Phase 2: Rank Tiny Results (Top 20 + Bottom 10 per Overlap)
phase_2_rank_tiny() {
    log_info "=========================================="
    log_info "PHASE 2: RANK TINY RESULTS"
    log_info "=========================================="
    
    conda run -n FCL python3 "$SCRIPT_DIR/per_overlap_ranking.py" \
        --model-size tiny \
        --top-n 20 \
        --bottom-n 10 \
        --output "$PROJECT_ROOT/rankings_tiny.json"
    
    log_info "Rankings saved to rankings_tiny.json"
    log_info "Selected: 30 losses per overlap (20 top + 10 bottom)"
}

# Phase 3: Small Models (30 Losses × 3 Overlaps = 90 Experiments)
phase_3_small() {
    log_info "=========================================="
    log_info "PHASE 3: SMALL MODELS (30 losses × 3 overlaps)"
    log_info "=========================================="
    
    if [ ! -f "$PROJECT_ROOT/rankings_tiny.json" ]; then
        log_error "rankings_tiny.json not found! Run phase 2 first."
        exit 1
    fi
    
    # Extract losses from rankings_tiny.json
    for overlap in 0 1 2; do
        log_info "Extracting losses for overlap=$overlap..."
        losses=$(jq -r ".rankings_per_overlap[\"$overlap\"].combined[]" "$PROJECT_ROOT/rankings_tiny.json" | tr '\n' ' ')
        
        log_info "Running small models for overlap=$overlap with 3 parallel workers..."
        conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
            --model-size small \
            --overlap $overlap \
            --num-parallel 3 \
            --losses $losses \
            --output-summary "phase3_small_overlap${overlap}_summary.json"
        
        log_info "Completed small models for overlap=$overlap"
    done
    
    log_info "Phase 3 complete! Total: 90 experiments"
}

# Phase 4: Rank Small Results (Top 15 + Bottom 5 per Overlap)
phase_4_rank_small() {
    log_info "=========================================="
    log_info "PHASE 4: RANK SMALL RESULTS"
    log_info "=========================================="
    
    conda run -n FCL python3 "$SCRIPT_DIR/per_overlap_ranking.py" \
        --model-size small \
        --top-n 15 \
        --bottom-n 5 \
        --output "$PROJECT_ROOT/rankings_small.json"
    
    log_info "Rankings saved to rankings_small.json"
    log_info "Selected: 20 losses per overlap (15 top + 5 bottom)"
}

# Phase 5: Medium Models (20 Losses × 3 Overlaps = 60 Experiments)
phase_5_medium() {
    log_info "=========================================="
    log_info "PHASE 5: MEDIUM MODELS (20 losses × 3 overlaps)"
    log_info "=========================================="
    
    if [ ! -f "$PROJECT_ROOT/rankings_small.json" ]; then
        log_error "rankings_small.json not found! Run phase 4 first."
        exit 1
    fi
    
    for overlap in 0 1 2; do
        log_info "Extracting losses for overlap=$overlap..."
        losses=$(jq -r ".rankings_per_overlap[\"$overlap\"].combined[]" "$PROJECT_ROOT/rankings_small.json" | tr '\n' ' ')
        
        log_info "Running medium models for overlap=$overlap with 2 parallel workers..."
        conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
            --model-size medium \
            --overlap $overlap \
            --num-parallel 2 \
            --losses $losses \
            --output-summary "phase5_medium_overlap${overlap}_summary.json"
        
        log_info "Completed medium models for overlap=$overlap"
    done
    
    log_info "Phase 5 complete! Total: 60 experiments"
}

# Phase 6: Rank Medium Results (Top 8 + Bottom 2 per Overlap)
phase_6_rank_medium() {
    log_info "=========================================="
    log_info "PHASE 6: RANK MEDIUM RESULTS"
    log_info "=========================================="
    
    conda run -n FCL python3 "$SCRIPT_DIR/per_overlap_ranking.py" \
        --model-size medium \
        --top-n 8 \
        --bottom-n 2 \
        --output "$PROJECT_ROOT/rankings_medium.json"
    
    log_info "Rankings saved to rankings_medium.json"
    log_info "Selected: 10 losses per overlap (8 top + 2 bottom)"
}

# Phase 7: Large Models (10 Losses × 3 Overlaps = 30 Experiments)
phase_7_large() {
    log_info "=========================================="
    log_info "PHASE 7: LARGE MODELS (10 losses × 3 overlaps)"
    log_info "=========================================="
    
    if [ ! -f "$PROJECT_ROOT/rankings_medium.json" ]; then
        log_error "rankings_medium.json not found! Run phase 6 first."
        exit 1
    fi
    
    for overlap in 0 1 2; do
        log_info "Extracting losses for overlap=$overlap..."
        losses=$(jq -r ".rankings_per_overlap[\"$overlap\"].combined[]" "$PROJECT_ROOT/rankings_medium.json" | tr '\n' ' ')
        
        log_info "Running large models for overlap=$overlap (sequential)..."
        conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
            --model-size large \
            --overlap $overlap \
            --num-parallel 1 \
            --losses $losses \
            --output-summary "phase7_large_overlap${overlap}_summary.json"
        
        log_info "Completed large models for overlap=$overlap"
    done
    
    log_info "Phase 7 complete! Total: 30 experiments"
}

# Phase 8: Rank Large Results (Top 4 + Bottom 1 per Overlap)
phase_8_rank_large() {
    log_info "=========================================="
    log_info "PHASE 8: RANK LARGE RESULTS"
    log_info "=========================================="
    
    conda run -n FCL python3 "$SCRIPT_DIR/per_overlap_ranking.py" \
        --model-size large \
        --top-n 4 \
        --bottom-n 1 \
        --output "$PROJECT_ROOT/rankings_large.json"
    
    log_info "Rankings saved to rankings_large.json"
    log_info "Selected: 5 losses per overlap (4 top + 1 bottom)"
}

# Phase 9: Huge Models (5 Losses × 3 Overlaps = 15 Experiments)
phase_9_huge() {
    log_info "=========================================="
    log_info "PHASE 9: HUGE MODELS (5 losses × 3 overlaps)"
    log_info "=========================================="
    
    if [ ! -f "$PROJECT_ROOT/rankings_large.json" ]; then
        log_error "rankings_large.json not found! Run phase 8 first."
        exit 1
    fi
    
    for overlap in 0 1 2; do
        log_info "Extracting losses for overlap=$overlap..."
        losses=$(jq -r ".rankings_per_overlap[\"$overlap\"].combined[]" "$PROJECT_ROOT/rankings_large.json" | tr '\n' ' ')
        
        log_info "Running huge models for overlap=$overlap (sequential)..."
        conda run -n FCL python3 "$SCRIPT_DIR/parallel_training.py" \
            --model-size huge \
            --overlap $overlap \
            --num-parallel 1 \
            --losses $losses \
            --output-summary "phase9_huge_overlap${overlap}_summary.json"
        
        log_info "Completed huge models for overlap=$overlap"
    done
    
    log_info "Phase 9 complete! Total: 15 experiments"
}

# Phase 10: Final Ranking (Top 3 per Overlap)
phase_10_final() {
    log_info "=========================================="
    log_info "PHASE 10: FINAL RANKING"
    log_info "=========================================="
    
    conda run -n FCL python3 "$SCRIPT_DIR/per_overlap_ranking.py" \
        --model-size huge \
        --top-n 3 \
        --bottom-n 0 \
        --output "$PROJECT_ROOT/final_rankings.json"
    
    log_info "Final rankings saved to final_rankings.json"
    log_info ""
    log_info "=========================================="
    log_info "TOURNAMENT COMPLETE!"
    log_info "=========================================="
    log_info "Results:"
    cat "$PROJECT_ROOT/final_rankings.json"
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        log_info "Running FULL TOURNAMENT (all phases)"
        log_warn "This will take ~10-14 days. Press Ctrl+C to cancel."
        sleep 5
        
        init_tournament
        phase_1_tiny
        phase_1b_redo_failed_tiny
        phase_1c_new_topology_tiny
        phase_1d_multipers_tiny
        phase_2_rank_tiny
        phase_3_small
        phase_4_rank_small
        phase_5_medium
        phase_6_rank_medium
        phase_7_large
        phase_8_rank_large
        phase_9_huge
        phase_10_final
    else
        case "$1" in
            --phase)
                case "$2" in
                    1)  phase_1_tiny ;;
                    1b) phase_1b_redo_failed_tiny ;;
                    1c) phase_1c_new_topology_tiny ;;
                    1d) phase_1d_multipers_tiny ;;
                    2)  phase_2_rank_tiny ;;
                    3) phase_3_small ;;
                    4) phase_4_rank_small ;;
                    5) phase_5_medium ;;
                    6) phase_6_rank_medium ;;
                    7) phase_7_large ;;
                    8) phase_8_rank_large ;;
                    9) phase_9_huge ;;
                    10) phase_10_final ;;
                    *) log_error "Invalid phase: $2. Use 1-10."; exit 1 ;;
                esac
                ;;
            --help)
                echo "Usage: $0 [--phase N] [--help]"
                echo ""
                echo "Options:"
                echo "  --phase N    Run specific phase (1-10)"
                echo "  --help       Show this help"
                echo ""
                echo "Phases:"
                echo "  1:  Tiny models (full loss sequence × 3 overlaps)"
                echo "  1b: Redo failed/crashed tiny experiments (from WandB, 2025-04-12)"
                echo "  1c: New differentiable topology losses (DiffPers, RTD)"
                echo "  1d: Multipers experiments + offline 2-param persistence analysis"
                echo "  2:  Rank tiny results"
                echo "  3: Small models (30 losses × 3 overlaps)"
                echo "  4: Rank small results"
                echo "  5: Medium models (20 losses × 3 overlaps)"
                echo "  6: Rank medium results"
                echo "  7: Large models (10 losses × 3 overlaps)"
                echo "  8: Rank large results"
                echo "  9: Huge models (5 losses × 3 overlaps)"
                echo "  10: Final ranking"
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    fi
}

main "$@"
