# Workspace Cleanup Summary

## New Structure

```
notebooks_sandbox/
├── tournament_system/              # Tournament execution
│   ├── run_tournament.sh          # Main orchestrator (ONLY script to run)
│   ├── parallel_training.py       # Parallel execution engine
│   └── per_overlap_ranking.py     # Ranking system (top + bottom)
│
├── core_modules/                   # Core Python modules
│   ├── run_advanced_experiments.py
│   ├── advanced_losses.py
│   ├── advanced_topology.py
│   ├── cnn_reconstruction.py
│   ├── cnn_validation_enhanced.py
│   ├── config.py
│   ├── scenario_dataset.py
│   ├── utils_consolidated.py
│   ├── weight_normalization.py
│   └── multi_objective_ranking.py
│
├── archive_old_scripts/            # Old/obsolete scripts (archived)
│   └── (all old .sh, .py, .md files)
│
├── TOURNAMENT_GUIDE.md             # SINGLE source of truth
│
└── (notebooks, data dirs, logs remain as-is)
```

## What Was Moved

### To `tournament_system/`
- `parallel_training.py` - Parallel execution
- `per_overlap_ranking.py` - Ranking with top+bottom selection
- `run_tournament.sh` - Main orchestrator (NEW)

### To `core_modules/`
- All core Python modules needed for experiments
- `run_advanced_experiments.py` - Main experiment runner
- Loss, topology, CNN, config modules

### To `archive_old_scripts/`
- All old bash scripts (run_*.sh, launch_*.sh)
- All old Python runners (run_*.py, train_*.py, etc.)
- All old markdown files (merged into TOURNAMENT_GUIDE.md)

## How to Use

### Run Full Tournament
```bash
cd /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox
./tournament_system/run_tournament.sh
```

### Run Specific Phase
```bash
./tournament_system/run_tournament.sh --phase 1  # Tiny models
./tournament_system/run_tournament.sh --phase 2  # Rank tiny
# ... etc
```

### Test Single Experiment
```bash
conda run -n FCL python3 core_modules/run_advanced_experiments.py \
    --single --model-size tiny --overlap 0 --loss MSE
```

## Cleanup Complete ✓
