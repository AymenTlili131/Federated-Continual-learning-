#!/usr/bin/env python3
"""
tournament_select.py
--------------------
After fair_evaluation.py finishes, select which losses advance to each stage.

Selection rule (per overlap):
  tiny  → small :  top 30% + bottom 10%  (best carry forward + worst get 2nd chance)
  small → medium:  top 20% + bottom 5%
  medium→ large :  top 10% + bottom 2.5%
  large results are final.

Primary ranking metric: cnn_improvement (finetuned acc - initial acc)
Fallback (if CNN not run): cosine similarity to ground-truth weights.

Usage:
  # See who advances from tiny→small (reads fair_eval_all.csv)
  conda run -n FCL python3 scripts/tournament_select.py --stage tiny

  # Generate small-model run commands
  conda run -n FCL python3 scripts/tournament_select.py --stage tiny \
      --generate-script missing_runs_small_overlap{N}.sh

  # After small finishes (reads fair_eval_small.csv)
  conda run -n FCL python3 scripts/tournament_select.py --stage small
"""

import sys
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

NOTEBOOKS_SANDBOX = Path(__file__).parent.parent
EXPERIMENTS       = NOTEBOOKS_SANDBOX / 'experiments'
CORE_MODULES      = NOTEBOOKS_SANDBOX / 'core_modules'
PAPER_RESULTS     = NOTEBOOKS_SANDBOX / 'paper_results'
RUN_SCRIPT        = CORE_MODULES / 'run_advanced_experiments.py'

STAGE_CONFIG = {
    'tiny': {
        'top_pct': 0.30, 'bot_pct': 0.10,
        'next_size': 'small',
        'next_epochs': 300,
        'eval_csv': PAPER_RESULTS / 'fair_eval' / 'fair_eval_all.csv',
    },
    'small': {
        'top_pct': 0.20, 'bot_pct': 0.05,
        'next_size': 'medium',
        'next_epochs': 300,
        'eval_csv': PAPER_RESULTS / 'fair_eval' / 'fair_eval_small.csv',
    },
    'medium': {
        'top_pct': 0.10, 'bot_pct': 0.025,
        'next_size': 'large',
        'next_epochs': 300,
        'eval_csv': PAPER_RESULTS / 'fair_eval' / 'fair_eval_medium.csv',
    },
}

RANK_COL_PRIORITY = ['cnn_accuracy', 'cnn_baseline', 'mse', 'cnn_improvement', 'cosine']


def load_eval(stage: str) -> pd.DataFrame:
    cfg = STAGE_CONFIG[stage]
    csv = cfg['eval_csv']
    if not csv.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {csv}\n"
            f"Run fair_evaluation.py --model-size {stage} first."
        )
    return pd.read_csv(csv)


def rank_col(df: pd.DataFrame) -> str:
    for col in RANK_COL_PRIORITY:
        if col in df.columns and df[col].notna().any():
            return col
    raise ValueError(f"No ranking column found in CSV. Expected one of {RANK_COL_PRIORITY}")


def select_advances(df_ov: pd.DataFrame, top_pct: float, bot_pct: float, col: str) -> pd.DataFrame:
    n = len(df_ov)
    n_top = max(1, round(n * top_pct))
    n_bot = max(1, round(n * bot_pct))

    ascending = (col == 'mse')   # lower mse = better
    ranked = df_ov.sort_values(col, ascending=ascending).reset_index(drop=True)

    top = ranked.head(n_top).copy(); top['selection'] = 'top'
    bot = ranked.tail(n_bot).copy(); bot['selection'] = 'bottom'

    # Avoid double-counting if top/bottom overlap (small n)
    combined = pd.concat([top, bot]).drop_duplicates(subset=['loss_name'])
    return combined


def make_cmd(model_size: str, overlap: int, loss: str, epochs: int) -> str:
    return (
        f"conda run -n FCL python3 {RUN_SCRIPT} "
        f"--single --model-size {model_size} --overlap {overlap} --loss \"{loss}\" "
        f"--epochs {epochs} --topology-n-jobs 1 --wandb "
        f"--output-dir {EXPERIMENTS}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['tiny', 'small', 'medium'],
                        help='Current finished stage to select from')
    parser.add_argument('--overlaps', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--generate-script', type=str, default=None,
                        help='Write run commands to this file (use {overlap} in name)')
    args = parser.parse_args()

    cfg  = STAGE_CONFIG[args.stage]
    df   = load_eval(args.stage)
    col  = rank_col(df)
    print(f"Ranking metric: {col}")
    print(f"Selection: top {cfg['top_pct']*100:.0f}% + bottom {cfg['bot_pct']*100:.0f}%")
    print(f"Next stage model: {cfg['next_size']} ({cfg['next_epochs']} epochs)\n")

    rankings_out = {}
    all_selected = []

    for ov in args.overlaps:
        sub = df[df['overlap'] == ov].copy()
        if sub.empty:
            print(f"Overlap {ov}: no data — skipping")
            continue

        sub = sub.dropna(subset=[col])
        selected = select_advances(sub, cfg['top_pct'], cfg['bot_pct'], col)
        all_selected.append(selected.assign(overlap=ov))

        n_top = (selected['selection'] == 'top').sum()
        n_bot = (selected['selection'] == 'bottom').sum()
        print(f"=== Overlap {ov} → {len(selected)} advances ({n_top} top, {n_bot} bottom) ===")
        print(selected[['loss_name', 'selection', col]].to_string(index=False))
        print()

        rankings_out[str(ov)] = {
            'top':    selected[selected['selection'] == 'top']['loss_name'].tolist(),
            'bottom': selected[selected['selection'] == 'bottom']['loss_name'].tolist(),
            'combined': selected['loss_name'].tolist(),
        }

    # Save rankings JSON
    PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = PAPER_RESULTS / f'rankings_{args.stage}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'stage': args.stage,
            'next_model_size': cfg['next_size'],
            'ranking_metric': col,
            'rankings_per_overlap': rankings_out,
        }, f, indent=2)
    print(f"Rankings → {json_path}")

    # Generate per-overlap run scripts
    if args.generate_script and all_selected:
        combined_df = pd.concat(all_selected, ignore_index=True)
        for ov in args.overlaps:
            ov_sel = combined_df[combined_df['overlap'] == ov]
            if ov_sel.empty:
                continue

            script_name = args.generate_script.format(overlap=ov) \
                if '{overlap}' in args.generate_script \
                else args.generate_script.replace('.sh', f'_overlap{ov}.sh')

            lines = ['#!/usr/bin/env bash', 'set -e',
                     f'echo "=== {cfg[\"next_size\"]} overlap{ov}: {len(ov_sel)} experiments ==="', '']
            for _, row in ov_sel.iterrows():
                loss = row['loss_name']
                lines.append(f'echo "--- {cfg[\"next_size\"]}_overlap{ov}_{loss} ---"')
                lines.append(make_cmd(cfg['next_size'], ov, loss, cfg['next_epochs']))
                lines.append('')

            Path(script_name).write_text('\n'.join(lines))
            os.chmod(script_name, 0o755)
            print(f"Script → {script_name}")


if __name__ == '__main__':
    main()
