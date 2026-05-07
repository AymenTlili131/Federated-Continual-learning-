#!/usr/bin/env bash
# NO set -e: continue past individual failures, log them at the end
echo "=== Overlap 2: 21 curated experiments to run ==="

EXPS="/home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/experiments"
RUN="conda run -n FCL python3 /home/aymen/Documents/GitHub/Federated-Continual-learning-/New/notebooks_sandbox/core_modules/run_advanced_experiments.py"
FAILED=()

# Fetch all finished WandB run names once.
WANDB_DONE=$(mktemp)
trap 'rm -f "$WANDB_DONE"' EXIT
echo "Fetching finished runs from WandB..."
conda run -n FCL python3 - <<'PYEOF' 2>/dev/null > "$WANDB_DONE"
import wandb
api = wandb.Api()
for r in api.runs('aymentlili/fcl-advanced'):
    if r.state == 'finished':
        print(r.name)
PYEOF
echo "  → $(wc -l < "$WANDB_DONE") finished runs found on WandB"

# Skip if local final_model.pth exists OR WandB marks the run as finished.
run_exp() {
    local idx="$1" loss="$2"
    local dir_suffix
    dir_suffix=$(echo "$loss" | sed 's/+/_/g; s/\*\([0-9]\)/\1x/g; s/\*/x/g')
    local exp_name="tiny_overlap2_${dir_suffix}"
    if [ -f "$EXPS/${exp_name}/checkpoints/final_model.pth" ] || grep -qxF "$exp_name" "$WANDB_DONE"; then
        echo "✓ SKIP [${idx}/21] ${loss}  (already done)"
        return 0
    fi
    echo "--- [${idx}/21] overlap2: ${loss} ---"
    $RUN --single --model-size tiny --overlap 2 --loss "$loss" \
         --epochs 200 --topology-n-jobs 1 --wandb \
         --output-dir "$EXPS" \
    || { echo "✗ FAILED: ${loss}"; FAILED+=("$loss"); }
}

#run_exp  1  "DiffPers"
#run_exp  2  "RTD"
#run_exp  3  "LW_DiffPers"
run_exp  4  "LW_RTD"
run_exp  5  "MSE+0.1*DiffPers"
run_exp  6  "MAE+0.1*DiffPers"
run_exp  7  "MSE+0.05*RTD"
run_exp  8  "MAE+0.05*RTD"
run_exp  9  "MSE+0.01*PersLandscape"
run_exp 10  "LW_MSE+0.1*LW_DiffPers"
run_exp 11  "LW_MAE+0.1*LW_DiffPers"
run_exp 12  "LW_MSE+0.05*LW_RTD"
run_exp 13  "LW_MSE+0.01*LW_PersLandscape"
run_exp 14  "MelSpec"
run_exp 15  "LW_FFT"
run_exp 16  "LW_FFT+0.1*LW_MelSpec"
run_exp 17  "Sinkhorn+0.15*KL"
run_exp 18  "MSE+0.15*Sinkhorn"
run_exp 19  "Sinkhorn+0.1*MAE"
run_exp 20  "Quantile"
run_exp 21  "FIM"

echo ""
echo "=== Overlap 2 complete ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "✗ Failed experiments (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "    $f"; done
    exit 1
else
    echo "✓ All experiments completed or already done."
fi
