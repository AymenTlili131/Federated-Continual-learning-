#!/usr/bin/env python3
"""
finish_missing.py
-----------------
Compares the expected full experiment set against what has best_model.pth
locally (and optionally WandB cloud). Prints exact commands to run any
that are missing or crashed.

Also handles the WandB duplicate-run problem: when the same experiment name
appears multiple times, only the latest finished run is kept.

Usage:
  # Print status + missing commands
  conda run -n FCL python3 scripts/finish_missing.py

  # Also check WandB cloud for latest run state
  conda run -n FCL python3 scripts/finish_missing.py --check-wandb

  # Generate a bash script to run all missing experiments
  conda run -n FCL python3 scripts/finish_missing.py --generate-script missing_runs.sh
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

NOTEBOOKS_SANDBOX = Path(__file__).parent.parent
EXPERIMENTS       = NOTEBOOKS_SANDBOX / 'experiments'
CORE_MODULES      = NOTEBOOKS_SANDBOX / 'core_modules'
TOURNAMENT_DIR    = NOTEBOOKS_SANDBOX / 'tournament_system'
RUN_SCRIPT        = CORE_MODULES / 'run_advanced_experiments.py'

# ──────────────────────────────────────────────────────────────────────────────
# Full expected loss set = everything registered in HierarchicalLossRegistry
# ──────────────────────────────────────────────────────────────────────────────

def _get_all_registered_losses() -> List[str]:
    try:
        sys.path.insert(0, str(CORE_MODULES))
        sys.path.insert(0, str(NOTEBOOKS_SANDBOX))
        sys.path.insert(0, str(NOTEBOOKS_SANDBOX.parent))
        from advanced_losses import HierarchicalLossRegistry
        reg = HierarchicalLossRegistry()
        return sorted(reg.all_losses.keys())
    except Exception as e:
        print(f"WARNING: could not load loss registry ({e}), using fallback list")
        return [
            'AUTO', 'DiffPers', 'FFT', 'FFT+0.1*MelSpec', 'FIM', 'Frobenius',
            'JS', 'KL', 'LWLN', 'LW_DiffPers', 'LW_FFT', 'LW_FFT+0.1*LW_MelSpec',
            'LW_FIM', 'LW_Frobenius', 'LW_JS', 'LW_KL', 'LW_LogNorm', 'LW_MAE',
            'LW_MAE+0.01*LW_PersLandscape', 'LW_MAE+0.05*LW_FIM',
            'LW_MAE+0.1*LW_DiffPers', 'LW_MAE+0.15*LW_Sinkhorn', 'LW_MAPE',
            'LW_MAPE+0.1*LW_JS', 'LW_MSE', 'LW_MSE+0.01*LW_PersLandscape',
            'LW_MSE+0.05*LW_Frobenius', 'LW_MSE+0.05*LW_RTD',
            'LW_MSE+0.1*LW_DiffPers', 'LW_MSE+0.1*LW_LogNorm',
            'LW_MSE+0.15*LW_Sinkhorn', 'LW_MelSpec', 'LW_Quantile', 'LW_RTD',
            'LW_Sinkhorn', 'LW_Sinkhorn+0.1*LW_Frobenius', 'LW_Sinkhorn+0.1*LW_MAE',
            'LW_Sinkhorn+0.1*LW_MSE', 'LogNorm', 'MAE', 'MAE+0.01*PersLandscape',
            'MAE+0.05*Frobenius', 'MAE+0.05*RTD', 'MAE+0.1*DiffPers',
            'MAE+0.15*Sinkhorn', 'MAPE', 'MAPE+0.1*JS', 'MSE',
            'MSE+0.01*PersLandscape', 'MSE+0.05*Frobenius', 'MSE+0.05*RTD',
            'MSE+0.1*DiffPers', 'MSE+0.1*LogNorm', 'MSE+0.15*Sinkhorn', 'MelSpec',
            'Quantile', 'Quantile+0.05*FIM', 'RTD', 'Sinkhorn',
            'Sinkhorn+0.1*Frobenius', 'Sinkhorn+0.1*MAE', 'Sinkhorn+0.1*MSE',
            'Sinkhorn+0.15*KL',
        ]

ALL_EXPECTED_LOSSES = _get_all_registered_losses()

MODEL_SIZES = ['tiny']
OVERLAPS    = [0, 1, 2]


# ──────────────────────────────────────────────────────────────────────────────
# WandB duplicate-run resolution
# ──────────────────────────────────────────────────────────────────────────────

def get_latest_wandb_run(runs, name: str):
    """Given list of WandB runs, return the most recent finished run with this name."""
    matching = [r for r in runs if r.name == name]
    if not matching:
        return None
    # Sort by created_at descending, prefer finished > crashed
    def sort_key(r):
        state_order = {'finished': 0, 'running': 1, 'crashed': 2, 'failed': 2}
        try:
            from dateutil.parser import parse as _parse
            ts = _parse(str(r.created_at)).timestamp()
        except Exception:
            ts = 0
        return (state_order.get(r.state, 3), -ts)
    return sorted(matching, key=sort_key)[0]


def check_wandb_status(
    project: str = 'fcl-advanced',
    entity: str  = 'aymentlili',
    after_date: str = '2026-03-20',
) -> Dict[str, str]:
    """Query WandB and return {exp_name: state} for runs after after_date."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping cloud check")
        return {}

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}"))
    print(f"  Cloud: {len(runs)} total runs in {entity}/{project}")

    from datetime import datetime, timezone
    cutoff = datetime.fromisoformat(after_date).replace(tzinfo=timezone.utc)

    status: Dict[str, str] = {}
    duplicate_names: Dict[str, int] = {}

    for run in runs:
        try:
            created = run.created_at
            if hasattr(created, 'timestamp'):
                pass
            else:
                from dateutil.parser import parse
                created = parse(str(created)).replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if created < cutoff:
            continue  # Skip old debug runs

        name = run.name
        duplicate_names[name] = duplicate_names.get(name, 0) + 1

        # Only keep latest per name
        latest = get_latest_wandb_run(runs, name)
        if latest and latest.id == run.id:
            status[name] = run.state

    dupes = {k: v for k, v in duplicate_names.items() if v > 1}
    if dupes:
        print(f"\n  Duplicate run names (only latest kept): {len(dupes)}")
        for name, count in sorted(dupes.items()):
            print(f"    {name}: {count} runs")

    return status


# ──────────────────────────────────────────────────────────────────────────────
# Check local experiment status
# ──────────────────────────────────────────────────────────────────────────────

def check_local_status() -> Dict[str, str]:
    """Return {exp_name: 'ok'|'missing'} for all expected experiments."""
    status = {}
    for model_size in MODEL_SIZES:
        for overlap in OVERLAPS:
            for loss in ALL_EXPECTED_LOSSES:
                name = f"{model_size}_overlap{overlap}_{loss}"
                exp_dir  = EXPERIMENTS / name
                ckpt     = exp_dir / 'checkpoints' / 'best_model.pth'
                hist     = exp_dir / 'training_history.json'

                if ckpt.exists():
                    # Check if training completed (val_loss present)
                    if hist.exists():
                        try:
                            import json as _json
                            h = _json.load(open(hist))
                            vl = h.get('val_loss', [])
                            status[name] = 'ok' if vl else 'incomplete'
                        except Exception:
                            status[name] = 'ok'
                    else:
                        status[name] = 'ok'
                else:
                    status[name] = 'missing'
    return status


# ──────────────────────────────────────────────────────────────────────────────
# Generate run commands
# ──────────────────────────────────────────────────────────────────────────────

def make_run_command(model_size: str, overlap: int, loss: str) -> str:
    return (
        f"conda run -n FCL python3 {RUN_SCRIPT} "
        f"--single --model-size {model_size} --overlap {overlap} --loss \"{loss}\" "
        f"--epochs 200 --topology-n-jobs 1 --wandb "
        f"--output-dir {EXPERIMENTS}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-wandb',    action='store_true',
                        help='Also query WandB cloud for run states')
    parser.add_argument('--after-date',     default='2026-03-20',
                        help='Ignore WandB runs before this date (YYYY-MM-DD)')
    parser.add_argument('--generate-script', type=str, default=None,
                        help='Write missing run commands to this bash file')
    parser.add_argument('--model-sizes',    nargs='+', default=MODEL_SIZES)
    parser.add_argument('--overlaps',       nargs='+', type=int, default=OVERLAPS)
    args = parser.parse_args()

    print("=== Checking local experiment status ===\n")
    local = check_local_status()

    cloud = {}
    if args.check_wandb:
        print("\n=== Checking WandB cloud (after %s) ===" % args.after_date)
        cloud = check_wandb_status(after_date=args.after_date)

    # Collate
    missing_cmds = []
    print("\n" + "=" * 70)
    print(f"{'EXPERIMENT':<55} {'LOCAL':>8}  {'CLOUD':>10}")
    print("-" * 70)

    counts = {'ok': 0, 'missing': 0, 'incomplete': 0, 'crashed_cloud': 0}
    for model_size in args.model_sizes:
        for overlap in args.overlaps:
            for loss in ALL_EXPECTED_LOSSES:
                name  = f"{model_size}_overlap{overlap}_{loss}"
                local_st = local.get(name, 'missing')
                cloud_st = cloud.get(name, '–')

                phase = loss_to_phase(loss)
                marker = ''
                if local_st == 'ok':
                    counts['ok'] += 1
                    marker = '✓'
                elif cloud_st == 'finished':
                    counts['ok'] += 1
                    marker = '≈'  # on cloud, not local
                else:
                    if cloud_st in ('crashed', 'failed'):
                        counts['crashed_cloud'] += 1
                    elif local_st == 'incomplete':
                        counts['incomplete'] += 1
                    else:
                        counts['missing'] += 1
                    missing_cmds.append((name, model_size, overlap, loss, phase))
                    marker = '✗'

                if marker != '✓':  # only print non-OK
                    print(f"{marker} {name:<53} {local_st:>8}  {cloud_st:>10}  [{phase}]")

    print("=" * 70)
    print(f"\nSummary: {counts['ok']} OK, {counts['missing']} missing, "
          f"{counts['incomplete']} incomplete, {counts['crashed_cloud']} crashed on cloud")

    if missing_cmds:
        print(f"\n\n{'='*70}")
        print(f"RUN COMMANDS for {len(missing_cmds)} missing/failed experiments")
        print(f"{'='*70}\n")

        # Group by overlap for parallel terminal use
        by_overlap = {}
        for name, ms, ov, loss, phase in missing_cmds:
            by_overlap.setdefault(ov, []).append((ms, loss, phase))

        all_lines = ["#!/usr/bin/env bash", "set -e",
                     f"NB={NOTEBOOKS_SANDBOX}", ""]

        for ov in sorted(by_overlap.keys()):
            print(f"# ── Overlap {ov} ──")
            all_lines.append(f"# Overlap {ov}")
            for ms, loss, phase in by_overlap[ov]:
                cmd = make_run_command(ms, ov, loss)
                print(cmd)
                all_lines.append(cmd)
            print()
            all_lines.append("")

        if args.generate_script:
            script_path = Path(args.generate_script)
            script_path.write_text('\n'.join(all_lines))
            script_path.chmod(0o755)
            print(f"Script written → {script_path}")
            print(f"Run with: bash {script_path}")
    else:
        print("\nAll expected experiments are complete!")

    # Save status JSON
    status_out = NOTEBOOKS_SANDBOX / 'paper_results' / 'experiment_status.json'
    status_out.parent.mkdir(parents=True, exist_ok=True)
    with open(status_out, 'w') as f:
        json.dump({
            'local':   local,
            'cloud':   cloud,
            'missing': [x[0] for x in missing_cmds],
            'counts':  counts,
        }, f, indent=2)
    print(f"\nStatus JSON → {status_out}")


if __name__ == '__main__':
    main()
