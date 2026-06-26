"""Plateau check for a training run's cumulative_reward curve.

Reads `logs/<method>/<run-name>/training_log.csv`, computes the rolling mean
of `cumulative_reward` over the last 500K and last 200K env-steps, and emits
a verdict:

  ratio = mean_last_200K / mean_last_500K

  ratio > 1.02   -> STILL_CLIMBING  (run full 5M steps)
  0.98 <= r <= 1.02 -> PLATEAU      (3M steps is enough, 1M buffer)
  ratio < 0.98   -> REGRESSING      (debug before committing the batch)

Usage:
  python scripts/check_plateau.py logs/hcmarl/watch_1m/training_log.csv
  python scripts/check_plateau.py logs/hcmarl/watch_1m/training_log.csv --metric cumulative_reward

Exit codes:
  0  PLATEAU
  1  STILL_CLIMBING
  2  REGRESSING
  3  not enough data (run was too short to decide)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


WINDOW_LONG = 500_000
WINDOW_SHORT = 200_000


def load_rows(csv_path: Path, metric: str) -> list[tuple[int, float]]:
    """Return [(global_step, metric_value), ...] sorted by global_step."""
    rows: list[tuple[int, float]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if metric not in (reader.fieldnames or []):
            raise SystemExit(
                f"column '{metric}' not in CSV header {reader.fieldnames!r}"
            )
        if "global_step" not in (reader.fieldnames or []):
            raise SystemExit(
                f"column 'global_step' not in CSV header {reader.fieldnames!r}"
            )
        for r in reader:
            try:
                step = int(r["global_step"])
                val = float(r[metric])
            except (TypeError, ValueError):
                continue
            rows.append((step, val))
    rows.sort(key=lambda x: x[0])
    return rows


def window_mean(rows: list[tuple[int, float]], max_step: int, window: int) -> float | None:
    lo = max_step - window
    vals = [v for s, v in rows if s >= lo]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", type=Path, help="path to training_log.csv")
    ap.add_argument("--metric", default="cumulative_reward",
                    help="column to assess (default: cumulative_reward)")
    ap.add_argument("--window-long", type=int, default=WINDOW_LONG,
                    help=f"long window in steps (default {WINDOW_LONG})")
    ap.add_argument("--window-short", type=int, default=WINDOW_SHORT,
                    help=f"short window in steps (default {WINDOW_SHORT})")
    args = ap.parse_args()

    if not args.csv_path.exists():
        print(f"ERROR: CSV not found at {args.csv_path}", file=sys.stderr)
        return 3

    rows = load_rows(args.csv_path, args.metric)
    if not rows:
        print("ERROR: CSV has no usable rows", file=sys.stderr)
        return 3

    max_step = rows[-1][0]
    n_rows = len(rows)
    print(f"CSV:          {args.csv_path}")
    print(f"Metric:       {args.metric}")
    print(f"Rows:         {n_rows}")
    print(f"Max step:     {max_step:,}")

    if max_step < args.window_long:
        print(
            f"NOT_ENOUGH_DATA: max_step {max_step:,} < long window {args.window_long:,}. "
            f"Let the run reach at least {args.window_long:,} steps before checking."
        )
        return 3

    long_mean = window_mean(rows, max_step, args.window_long)
    short_mean = window_mean(rows, max_step, args.window_short)
    if long_mean is None or short_mean is None:
        print("NOT_ENOUGH_DATA: empty window(s)", file=sys.stderr)
        return 3

    ratio = short_mean / long_mean if long_mean != 0 else float("inf")

    print(f"Last {args.window_long:,}:  mean={long_mean:.4f}")
    print(f"Last {args.window_short:,}:  mean={short_mean:.4f}")
    print(f"Ratio:        {ratio:.4f}")
    print()

    if ratio > 1.02:
        print("VERDICT: STILL_CLIMBING -> run full 5M steps.")
        return 1
    if ratio < 0.98:
        print("VERDICT: REGRESSING -> debug before committing the 45-run batch.")
        return 2
    print("VERDICT: PLATEAU -> 3M steps is enough (1M buffer).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
