"""Learning-curve aggregator for the headline + ablation tables .

Reads the CSVs written by HCMARLLogger for every (method, seed) combination
under matrix['headline'] AND every (rung, seed) combination under
matrix['ablation'] in config/experiment_matrix.yaml, and produces a
summary of IQM + stratified bootstrap 95% CI on cumulative_reward at
each anchor step (default 500K / 1M / 2M / 3M).

Headline runs are read from logs/{method}/seed_{seed}/training_log.csv.
Ablation runs are read from logs/ablation_{rung}/seed_{seed}/training_log.csv
(scripts/run_ablations.py launches with --run-name ablation_{rung} so the
two grids never collide on the mappo subdir).

Usage:
    python scripts/aggregate_learning_curves.py \
        --matrix config/experiment_matrix.yaml \
        --logs-root logs \
        --out aggregated_results.json

Exit codes:
    0 — aggregation produced a table for every expected (run, seed, anchor).
    2 — at least one CSV is missing or an anchor step is not reached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcmarl.aggregation import aggregate_by_method


def score_at_step(csv_path: str, anchor_step: int) -> float:
    """Return cumulative_reward at the last episode whose global_step
    is <= anchor_step. Raises FileNotFoundError / KeyError / ValueError
    so the caller can surface an actionable error instead of silent NaN."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "global_step" not in df.columns:
        raise KeyError(f"{csv_path}: missing 'global_step' column")
    if "cumulative_reward" not in df.columns:
        raise KeyError(f"{csv_path}: missing 'cumulative_reward' column")
    mask = df["global_step"] <= anchor_step
    if not mask.any():
        raise ValueError(
            f"{csv_path}: no episodes reach anchor step {anchor_step:,} "
            f"(max observed: {int(df['global_step'].max()):,})"
        )
    return float(df.loc[mask, "cumulative_reward"].iloc[-1])


def _aggregate_grid(
    grid_label: str,
    runs: Dict[str, str],
    seeds: List[int],
    anchors: List[int],
    logs_root: str,
    errors: List[str],
) -> Dict[str, Dict]:
    """Aggregate one named grid (headline OR ablation).

    runs: {display_name: log_subdir} — display_name is what shows up in
        the printout, log_subdir is the directory under logs_root.
    """
    by_anchor: Dict[str, Dict] = {}
    for anchor in anchors:
        scores: Dict[str, np.ndarray] = {}
        for display_name, log_subdir in runs.items():
            vec = []
            for seed in seeds:
                csv_path = os.path.join(
                    logs_root, log_subdir, f"seed_{seed}", "training_log.csv",
                )
                try:
                    vec.append(score_at_step(csv_path, anchor))
                except (FileNotFoundError, KeyError, ValueError) as e:
                    errors.append(
                        f"[{grid_label} anchor {anchor}] {display_name} "
                        f"seed {seed}: {e}"
                    )
            scores[display_name] = np.asarray(vec, dtype=np.float64)
        by_anchor[str(anchor)] = aggregate_by_method(scores)
    return by_anchor


def aggregate(matrix_path: str, logs_root: str) -> Dict:
    with open(matrix_path, encoding="utf-8") as f:
        matrix = yaml.safe_load(f)

    anchors: List[int] = list(matrix["curve_anchors_steps"])

    # Headline grid: logs land at logs/{method_key}/seed_{seed}/.
    headline = matrix.get("headline") or {}
    h_seeds: List[int] = list(headline.get("seeds") or [])
    h_methods: Dict[str, Dict] = headline.get("methods") or {}
    headline_runs: Dict[str, str] = {m: m for m in h_methods}

    # Ablation grid: logs land at logs/ablation_{rung}/seed_{seed}/
    # because run_ablations.py passes --run-name ablation_{rung}.
    ablation = matrix.get("ablation") or {}
    a_seeds: List[int] = list(ablation.get("seeds") or [])
    ablation_runs: Dict[str, str] = {
        r["name"]: f"ablation_{r['name']}" for r in (ablation.get("rungs") or [])
    }

    report: Dict = {
        "anchors": anchors,
        "headline_seeds": h_seeds,
        "ablation_seeds": a_seeds,
        "by_anchor": {},
        "by_anchor_ablation": {},
    }
    errors: List[str] = []

    if headline_runs and h_seeds:
        report["by_anchor"] = _aggregate_grid(
            "headline", headline_runs, h_seeds, anchors, logs_root, errors,
        )
    if ablation_runs and a_seeds:
        report["by_anchor_ablation"] = _aggregate_grid(
            "ablation", ablation_runs, a_seeds, anchors, logs_root, errors,
        )

    report["errors"] = errors
    report["complete"] = (len(errors) == 0)
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="config/experiment_matrix.yaml")
    p.add_argument("--logs-root", default="logs")
    p.add_argument("--out", default="aggregated_results.json")
    args = p.parse_args()

    report = aggregate(args.matrix, args.logs_root)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if report["by_anchor"]:
        print("\n###  HEADLINE GRID  ###")
        for anchor_str, by_method in report["by_anchor"].items():
            print(f"\n=== Anchor: {int(anchor_str):,} steps ===")
            for method, stats in by_method.items():
                print(f"  {method:18s}  IQM={stats['iqm']:>8.2f}  "
                      f"95% CI=[{stats['ci_lo']:>8.2f}, {stats['ci_hi']:>8.2f}]  "
                      f"(n={stats['n_seeds']})")

    if report["by_anchor_ablation"]:
        print("\n###  ABLATION LADDER  ###")
        for anchor_str, by_rung in report["by_anchor_ablation"].items():
            print(f"\n=== Anchor: {int(anchor_str):,} steps ===")
            for rung, stats in by_rung.items():
                print(f"  {rung:18s}  IQM={stats['iqm']:>8.2f}  "
                      f"95% CI=[{stats['ci_lo']:>8.2f}, {stats['ci_hi']:>8.2f}]  "
                      f"(n={stats['n_seeds']})")

    if report["errors"]:
        print("\nERRORS:")
        for e in report["errors"]:
            print(f"  - {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
