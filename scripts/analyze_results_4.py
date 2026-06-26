"""EXP4 stage 1 — quantitative analysis of Results 1 / Results 2 / Results 3.

For each results folder, parse every artefact (training_log.csv, summary.json,
mmicrl_results.json, config.yaml) and extract / compute every quantity:

  Per-seed:
    - final_window_mean      (mean over last 500 episodes, per numeric column)
    - best_value             (max for reward/safety/fairness; min for cost/violation/peak_fatigue)
    - trajectory_bin50       (per-50-episode mean, for learning-curve plots)

  Per-method (cross-seed):
    - mean, std, min, max
    - IQM + 95% stratified-bootstrap CI (Agarwal et al. 2021, via hcmarl/aggregation.py)
    - learning curve: per-bin IQM + 95% bootstrap CI ribbon

  Pairwise (every method pair):
    - probability of improvement P(X > Y) with 95% bootstrap CI
    - mean-difference + 95% bootstrap CI

  EXP3 Part 1: extract K_discovered, MI, ARI, NMI, confusion matrix, K-selection
                per-seed scores, theta_per_type, type_proportions.

  EXP3 Part 2: identical pipeline to EXP1, with synthetic flag set on every
                output and an explicit "is_synthetic": true on every JSON.

Outputs to Results 4/Result {1,2,3} Analysis/:
  _quant_analysis.json         (every number, machine-readable)
  _metrics_table.csv           (flat per-method summary)
  _learning_curves.json        (per-method binned learning curves with CIs)
  _pairwise_comparisons.json   (PoI + mean-diff CIs for every method pair)

Stage 2 (visualize_results_4.py) reads these and emits PNGs.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcmarl.aggregation import iqm, stratified_bootstrap_iqm_ci  # noqa: E402

RESULTS_4 = ROOT / "Results 4"
RESULTS_4.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
FINAL_WINDOW_EPISODES = 500    # last 500 episodes -> "final-window mean"
BIN_SIZE_EPISODES = 50         # learning-curve bin
N_BOOTSTRAP = 10_000

CSV_COLUMNS_25 = [
    "actor_loss", "constraint_recovery_time", "cost_critic_loss",
    "cost_ema", "critic_loss", "cumulative_cost", "cumulative_reward",
    "ecbf_interventions", "entropy", "episode", "forced_rest_rate",
    "global_step", "jain_fairness", "lambda", "lazy_agent_flag",
    "peak_fatigue", "per_agent_entropy_mean", "per_agent_entropy_min",
    "policy_loss", "safety_autonomy_index", "safety_rate",
    "tasks_completed", "value_loss", "violation_rate", "wall_time",
]

# direction["higher_is_better"] -> "best" = max; else "best" = min
HIGHER_IS_BETTER = {
    "actor_loss":              False,   # closer to 0 is better; both directions feasible
    "cost_ema":                False,
    "critic_loss":             False,
    "cumulative_cost":         False,
    "cumulative_reward":       True,
    "ecbf_interventions":      False,   # fewer is better (less safety-filter firing)
    "forced_rest_rate":        None,    # diagnostic, not optimized
    "jain_fairness":           True,
    "lazy_agent_flag":         False,
    "peak_fatigue":            False,
    "per_agent_entropy_mean":  None,
    "per_agent_entropy_min":   None,
    "safety_autonomy_index":   None,
    "safety_rate":             True,
    "tasks_completed":         True,
    "value_loss":              False,
    "violation_rate":          False,
    "wall_time":               None,
    "lambda":                  None,
    "cost_critic_loss":        False,
    "policy_loss":             False,
    "constraint_recovery_time":None,
    "entropy":                 None,
    "episode":                 None,
    "global_step":             None,
}

NUMERIC_COLUMNS = [c for c in CSV_COLUMNS_25 if c not in ("episode", "global_step")]
LEARNING_CURVE_METRICS = [
    "cumulative_reward", "safety_rate", "violation_rate", "peak_fatigue",
    "ecbf_interventions", "tasks_completed", "jain_fairness",
    "per_agent_entropy_mean", "actor_loss", "critic_loss",
]


# ------------------------------------------------------------------
# CSV reader (no pandas dependency)
# ------------------------------------------------------------------
def read_training_log(path: Path) -> Dict[str, np.ndarray]:
    """Return {column_name: float_array} with empty cells as NaN."""
    cols: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row")
        for c in reader.fieldnames:
            cols[c] = []
        for row in reader:
            for c in reader.fieldnames:
                v = row.get(c, "").strip()
                if v == "":
                    cols[c].append(float("nan"))
                else:
                    try:
                        cols[c].append(float(v))
                    except ValueError:
                        cols[c].append(float("nan"))
    return {c: np.asarray(v, dtype=np.float64) for c, v in cols.items()}


def all_nan(arr: np.ndarray) -> bool:
    return arr.size == 0 or bool(np.all(np.isnan(arr)))


# ------------------------------------------------------------------
# Per-seed reductions
# ------------------------------------------------------------------
def per_seed_final_window_mean(arr: np.ndarray, window: int = FINAL_WINDOW_EPISODES) -> float:
    """Mean over the last `window` non-NaN values."""
    if arr.size == 0:
        return float("nan")
    tail = arr[-window:] if arr.size >= window else arr
    valid = tail[~np.isnan(tail)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def per_seed_best(arr: np.ndarray, higher_is_better: Optional[bool]) -> float:
    """Best value along the full trajectory, direction-aware. Returns NaN if all-NaN."""
    if all_nan(arr) or higher_is_better is None:
        return float("nan")
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan")
    return float(valid.max() if higher_is_better else valid.min())


def per_seed_bin_means(arr: np.ndarray, bin_size: int = BIN_SIZE_EPISODES) -> np.ndarray:
    """Bin into chunks of `bin_size` episodes; return per-bin mean (NaN-aware)."""
    n = arr.size
    n_bins = n // bin_size
    if n_bins == 0:
        return np.full(1, float(np.nanmean(arr)) if not all_nan(arr) else float("nan"))
    out = np.zeros(n_bins)
    for i in range(n_bins):
        chunk = arr[i * bin_size:(i + 1) * bin_size]
        valid = chunk[~np.isnan(chunk)]
        out[i] = float(valid.mean()) if valid.size > 0 else float("nan")
    return out


# ------------------------------------------------------------------
# Cross-seed reductions
# ------------------------------------------------------------------
def cross_seed_summary(scores: np.ndarray) -> Dict[str, float]:
    """Mean / std / min / max / IQM / 95% bootstrap CI on IQM, NaN-aware."""
    arr = np.asarray(scores, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "iqm": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_seeds": 0}
    iqm_pt = iqm(finite)
    lo, hi = stratified_bootstrap_iqm_ci(finite, n_resamples=N_BOOTSTRAP)
    return {
        "mean":   float(finite.mean()),
        "std":    float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min":    float(finite.min()),
        "max":    float(finite.max()),
        "iqm":    float(iqm_pt),
        "ci_lo":  float(lo),
        "ci_hi":  float(hi),
        "n_seeds": int(finite.size),
    }


def per_bin_iqm_ci(per_seed_bins: np.ndarray) -> Dict[str, np.ndarray]:
    """per_seed_bins shape (n_seeds, n_bins). Return {iqm, ci_lo, ci_hi} arrays of length n_bins."""
    n_seeds, n_bins = per_seed_bins.shape
    iqm_arr = np.zeros(n_bins)
    lo_arr  = np.zeros(n_bins)
    hi_arr  = np.zeros(n_bins)
    for j in range(n_bins):
        col = per_seed_bins[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            iqm_arr[j] = lo_arr[j] = hi_arr[j] = float("nan")
            continue
        iqm_arr[j] = iqm(finite)
        lo, hi = stratified_bootstrap_iqm_ci(finite, n_resamples=2000)  # smaller B for per-bin
        lo_arr[j] = lo; hi_arr[j] = hi
    return {"iqm": iqm_arr, "ci_lo": lo_arr, "ci_hi": hi_arr}


# ------------------------------------------------------------------
# Pairwise comparisons
# ------------------------------------------------------------------
def probability_of_improvement(x: np.ndarray, y: np.ndarray, higher_is_better: bool) -> float:
    """P(X > Y) when higher is better, P(X < Y) otherwise. Ties count 0.5."""
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=np.float64); y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    if higher_is_better:
        wins = float(np.sum(x[:, None] > y[None, :]) + 0.5 * np.sum(x[:, None] == y[None, :]))
    else:
        wins = float(np.sum(x[:, None] < y[None, :]) + 0.5 * np.sum(x[:, None] == y[None, :]))
    return wins / (x.size * y.size)


def pairwise_bootstrap(
    x: np.ndarray, y: np.ndarray, higher_is_better: bool,
    n_boot: int = 5_000, seed: int = 4271326,
) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=np.float64); y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0 or higher_is_better is None:
        return {"poi": float("nan"), "poi_ci_lo": float("nan"), "poi_ci_hi": float("nan"),
                "mean_diff": float("nan"), "diff_ci_lo": float("nan"), "diff_ci_hi": float("nan")}
    poi_pt = probability_of_improvement(x, y, higher_is_better)
    mean_diff_pt = float(x.mean() - y.mean())
    rng = np.random.default_rng(seed)
    pois = np.empty(n_boot); diffs = np.empty(n_boot)
    nx, ny = x.size, y.size
    for b in range(n_boot):
        bx = x[rng.integers(0, nx, nx)]
        by = y[rng.integers(0, ny, ny)]
        pois[b]  = probability_of_improvement(bx, by, higher_is_better)
        diffs[b] = float(bx.mean() - by.mean())
    return {
        "poi": poi_pt,
        "poi_ci_lo": float(np.percentile(pois, 2.5)),
        "poi_ci_hi": float(np.percentile(pois, 97.5)),
        "mean_diff": mean_diff_pt,
        "diff_ci_lo": float(np.percentile(diffs, 2.5)),
        "diff_ci_hi": float(np.percentile(diffs, 97.5)),
    }


# ------------------------------------------------------------------
# Per-folder pipeline
# ------------------------------------------------------------------
def analyze_method_grid(
    grid_root: Path,
    method_subdirs: Dict[str, Path],   # {label: dir} where dir contains seed_<n>/
    is_synthetic: bool = False,
) -> Dict:
    """Analyze a {method/rung -> seed dir} grid. Returns the full quant dict."""
    # 1. Load every seed for every method
    print(f"  Loading {grid_root}...")
    method_data: Dict[str, Dict] = {}
    for method, mdir in method_subdirs.items():
        seeds_data = {}
        seed_dirs = sorted([d for d in mdir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
        for sdir in seed_dirs:
            seed_id = int(sdir.name.split("_")[1])
            log_path = sdir / "training_log.csv"
            sum_path = sdir / "summary.json"
            mmi_path = sdir / "mmicrl" / "mmicrl_results.json"
            if not log_path.exists():
                continue
            tlog = read_training_log(log_path)
            summary = json.load(open(sum_path)) if sum_path.exists() else {}
            mmicrl = json.load(open(mmi_path)) if mmi_path.exists() else None
            seeds_data[seed_id] = {
                "tlog": tlog, "summary": summary, "mmicrl": mmicrl,
            }
        method_data[method] = seeds_data
        print(f"    {method}: {len(seeds_data)} seeds loaded")

    # 2. Per-seed reductions per metric
    summary_per_method: Dict[str, Dict] = {}
    for method, seeds in method_data.items():
        # Skip if no seeds
        if not seeds:
            summary_per_method[method] = {"n_seeds": 0}
            continue
        # Determine n_bins from longest seed tlog
        n_bins_max = max(t["tlog"][CSV_COLUMNS_25[9]].size // BIN_SIZE_EPISODES
                         for t in seeds.values()
                         if CSV_COLUMNS_25[9] in t["tlog"])
        per_metric: Dict[str, Dict] = {}
        for col in NUMERIC_COLUMNS:
            hib = HIGHER_IS_BETTER.get(col)
            per_seed_finals = []
            per_seed_bests  = []
            per_seed_bins   = []
            for sid in sorted(seeds.keys()):
                arr = seeds[sid]["tlog"].get(col)
                if arr is None or all_nan(arr):
                    per_seed_finals.append(float("nan"))
                    per_seed_bests.append(float("nan"))
                    per_seed_bins.append(np.full(n_bins_max, float("nan")))
                    continue
                per_seed_finals.append(per_seed_final_window_mean(arr))
                per_seed_bests.append(per_seed_best(arr, hib))
                bins = per_seed_bin_means(arr)
                if bins.size < n_bins_max:
                    bins = np.concatenate([bins, np.full(n_bins_max - bins.size, float("nan"))])
                else:
                    bins = bins[:n_bins_max]
                per_seed_bins.append(bins)
            per_seed_finals = np.asarray(per_seed_finals)
            per_seed_bests  = np.asarray(per_seed_bests)
            per_seed_bins   = np.stack(per_seed_bins, axis=0) if per_seed_bins else np.zeros((0, n_bins_max))
            per_metric[col] = {
                "higher_is_better": hib,
                "per_seed_final_window_mean": per_seed_finals.tolist(),
                "per_seed_best":              per_seed_bests.tolist(),
                "final_window_mean_xseed":    cross_seed_summary(per_seed_finals),
                "best_xseed":                 cross_seed_summary(per_seed_bests),
            }
            # Learning-curve binned IQM+CI (only for the 10 key metrics, to save space)
            if col in LEARNING_CURVE_METRICS:
                bin_stats = per_bin_iqm_ci(per_seed_bins)
                per_metric[col]["learning_curve"] = {
                    "bin_size_episodes": BIN_SIZE_EPISODES,
                    "bin_centers_episode": [(i + 0.5) * BIN_SIZE_EPISODES for i in range(n_bins_max)],
                    "iqm":   bin_stats["iqm"].tolist(),
                    "ci_lo": bin_stats["ci_lo"].tolist(),
                    "ci_hi": bin_stats["ci_hi"].tolist(),
                }
        # summary.json fields aggregated across seeds
        sum_fields = {}
        for sid in sorted(seeds.keys()):
            for k, v in seeds[sid]["summary"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    sum_fields.setdefault(k, []).append(float(v))
        summary_xseed = {k: cross_seed_summary(np.asarray(v))
                         for k, v in sum_fields.items()}
        # mmicrl summary (HCMARL / hcmarl_with_mmicrl arms only)
        mmicrl_xseed: Dict[str, List] = {}
        for sid in sorted(seeds.keys()):
            mmi = seeds[sid].get("mmicrl")
            if mmi is None:
                continue
            mmicrl_xseed.setdefault("n_demonstrations", []).append(mmi.get("n_demonstrations"))
            mmicrl_xseed.setdefault("n_types_discovered", []).append(mmi.get("n_types_discovered"))
            mmicrl_xseed.setdefault("mutual_information", []).append(mmi.get("mutual_information"))
            mmicrl_xseed.setdefault("mi_collapsed",       []).append(int(bool(mmi.get("mi_collapsed", False))))
            tp = mmi.get("type_proportions") or []
            mmicrl_xseed.setdefault("max_type_proportion", []).append(float(max(tp)) if tp else float("nan"))
            mmicrl_xseed.setdefault("min_type_proportion", []).append(float(min(tp)) if tp else float("nan"))
        mmicrl_summary = None
        if mmicrl_xseed:
            mmicrl_summary = {}
            for k, v in mmicrl_xseed.items():
                clean = [x for x in v if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
                if not clean:
                    continue
                mmicrl_summary[k] = cross_seed_summary(np.asarray(clean, dtype=np.float64))
        summary_per_method[method] = {
            "n_seeds": len(seeds),
            "seed_ids": sorted(seeds.keys()),
            "per_metric_training_log": per_metric,
            "summary_json_xseed": summary_xseed,
            "mmicrl_xseed": mmicrl_summary,
        }

    # 3. Pairwise comparisons (best_reward + final_window_mean reward + safety_rate)
    pairwise: Dict[str, Dict] = {}
    methods = list(method_data.keys())
    pair_metrics = ["cumulative_reward", "safety_rate", "violation_rate", "peak_fatigue"]
    for i, ma in enumerate(methods):
        for mb in methods[i + 1:]:
            key = f"{ma}__vs__{mb}"
            pairwise[key] = {}
            for col in pair_metrics:
                if "per_metric_training_log" not in summary_per_method.get(ma, {}):
                    continue
                if "per_metric_training_log" not in summary_per_method.get(mb, {}):
                    continue
                xa = np.asarray(summary_per_method[ma]["per_metric_training_log"][col]["per_seed_final_window_mean"])
                xb = np.asarray(summary_per_method[mb]["per_metric_training_log"][col]["per_seed_final_window_mean"])
                hib = HIGHER_IS_BETTER.get(col)
                pairwise[key][f"{col}__final_window"] = pairwise_bootstrap(xa, xb, hib)
                # also compare best
                ya = np.asarray(summary_per_method[ma]["per_metric_training_log"][col]["per_seed_best"])
                yb = np.asarray(summary_per_method[mb]["per_metric_training_log"][col]["per_seed_best"])
                pairwise[key][f"{col}__best"] = pairwise_bootstrap(ya, yb, hib)

    return {
        "is_synthetic": is_synthetic,
        "n_methods": len(method_data),
        "methods": list(method_data.keys()),
        "per_method": summary_per_method,
        "pairwise": pairwise,
    }


# ------------------------------------------------------------------
# Flat metrics-table CSV writer
# ------------------------------------------------------------------
def write_metrics_table(quant: Dict, out_path: Path):
    rows: List[Dict] = []
    for method, info in quant["per_method"].items():
        if "per_metric_training_log" not in info:
            continue
        n_seeds = info["n_seeds"]
        for col, m in info["per_metric_training_log"].items():
            rows.append({
                "method_or_rung": method,
                "metric": col,
                "n_seeds": n_seeds,
                "higher_is_better": m["higher_is_better"],
                "final_window_mean_iqm": m["final_window_mean_xseed"]["iqm"],
                "final_window_mean_ci_lo": m["final_window_mean_xseed"]["ci_lo"],
                "final_window_mean_ci_hi": m["final_window_mean_xseed"]["ci_hi"],
                "final_window_mean_mean": m["final_window_mean_xseed"]["mean"],
                "final_window_mean_std":  m["final_window_mean_xseed"]["std"],
                "best_iqm":   m["best_xseed"]["iqm"],
                "best_ci_lo": m["best_xseed"]["ci_lo"],
                "best_ci_hi": m["best_xseed"]["ci_hi"],
                "best_mean":  m["best_xseed"]["mean"],
                "best_std":   m["best_xseed"]["std"],
            })
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("(no rows)\n"); return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_learning_curves(quant: Dict, out_path: Path):
    """Trim learning-curve arrays into a slim json (separate file to keep main JSON small)."""
    out: Dict = {}
    for method, info in quant["per_method"].items():
        if "per_metric_training_log" not in info:
            continue
        out[method] = {}
        for col, m in info["per_metric_training_log"].items():
            if "learning_curve" in m:
                out[method][col] = m["learning_curve"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def strip_learning_curves(quant: Dict) -> Dict:
    """Return a copy of `quant` without the learning_curve arrays (for the main JSON)."""
    out = json.loads(json.dumps(quant))   # deep copy via JSON round-trip; arrays are lists
    for method, info in out["per_method"].items():
        if "per_metric_training_log" not in info:
            continue
        for col, m in info["per_metric_training_log"].items():
            m.pop("learning_curve", None)
    return out


# ------------------------------------------------------------------
# EXP3 PART 1 — single-shot K-discovery analysis
# ------------------------------------------------------------------
def analyze_exp3_part1(part1_dir: Path) -> Dict:
    """Pull every number out of the K=3 and K=1 MMICRL JSONs."""
    out: Dict = {"is_synthetic": False, "regimes": {}}
    for regime in ("k3", "k1"):
        path = part1_dir / f"synthetic_{regime}_mmicrl.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        mmi = d.get("mmicrl_results", {})
        gtm = d.get("ground_truth_metrics", {})
        out["regimes"][regime.upper()] = {
            "regime": d.get("regime"),
            "n_workers": d.get("n_workers"),
            "n_episodes_per_worker": d.get("n_episodes_per_worker"),
            "n_demos": d.get("n_demos"),
            "F_values_used": d.get("F_values_used"),
            "true_type_distribution": d.get("true_type_distribution"),
            "fit_seconds": d.get("fit_seconds"),
            "verdict": d.get("verdict"),
            "mmicrl_results": {
                "n_demonstrations":  mmi.get("n_demonstrations"),
                "n_types_discovered": mmi.get("n_types_discovered"),
                "type_proportions":  mmi.get("type_proportions"),
                "mutual_information": mmi.get("mutual_information"),
                "mi_collapsed":      mmi.get("mi_collapsed"),
                "objective_value":   mmi.get("objective_value"),
                "theta_per_type":    mmi.get("theta_per_type"),
                "lambda1":           mmi.get("lambda1"),
                "lambda2":           mmi.get("lambda2"),
                "true_K":            mmi.get("true_K"),
                "K_discovered":      mmi.get("K_discovered"),
                "K_match":           mmi.get("K_match"),
                "k_selection_method": mmi.get("k_selection_method"),
                "k_selection_n_seeds": mmi.get("k_selection_n_seeds"),
                "k_selection_per_k_seed_scores": mmi.get("k_selection_per_k_seed_scores"),
                "k_selection_median_per_k": mmi.get("k_selection_median_per_k"),
                "k_selection_seconds": mmi.get("k_selection_seconds"),
            },
            "ground_truth_metrics": {
                "predicted_assignments": gtm.get("predicted_assignments"),
                "true_types": gtm.get("true_types"),
                "ARI": gtm.get("ARI"),
                "NMI": gtm.get("NMI"),
                "confusion_matrix": gtm.get("confusion_matrix"),
            },
        }
    return out


# ------------------------------------------------------------------
# Main entry points
# ------------------------------------------------------------------
def run_exp1():
    print("=" * 78); print("Result 1 (EXP1)"); print("=" * 78)
    grid = ROOT / "Results 1" / "logs"
    # baseline lineup: ippo dropped from headline; happo + shielded_mappo added.
    # Only include methods whose log directory actually exists on disk (graceful skip).
    candidate_methods = ["hcmarl", "mappo", "mappo_lag", "happo", "shielded_mappo"]
    methods = {m: grid / m for m in candidate_methods if (grid / m).exists()}
    print(f"  Methods present on disk: {sorted(methods.keys())}")
    quant = analyze_method_grid(grid, methods, is_synthetic=False)
    out_dir = RESULTS_4 / "Result 1 Analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_learning_curves(quant, out_dir / "_learning_curves.json")
    with open(out_dir / "_quant_analysis.json", "w", encoding="utf-8") as f:
        json.dump(strip_learning_curves(quant), f, indent=2)
    write_metrics_table(quant, out_dir / "_metrics_table.csv")
    with open(out_dir / "_pairwise_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(quant["pairwise"], f, indent=2)
    print(f"  Wrote -> {out_dir}/")
    return quant


def run_exp2():
    print("=" * 78); print("Result 2 (EXP2 v5 ablations + per-chip anchor)"); print("=" * 78)
    grid = ROOT / "Results 2" / "logs"
    # EXP2 v5: 5 ablation rungs + per-chip hcmarl_anchor.
    # The v5 design adds (a) ablation_no_mmicrl as the 5th rung, and (b) the
    # per-chip hcmarl_anchor as the PRIMARY reference for ablation deltas
    # (preserves bit-level chip-determinism vs comparing against EXP1's hcmarl
    # which ran on a different L4 chip).
    candidate_rungs = [
        "hcmarl_anchor",
        "ablation_no_ecbf",
        "ablation_no_nswf",
        "ablation_no_divergent",
        "ablation_no_reperfusion",
        "ablation_no_mmicrl",
    ]
    rungs = {r: grid / r for r in candidate_rungs if (grid / r).exists()}
    print(f"  Rungs present on disk: {sorted(rungs.keys())}")
    quant = analyze_method_grid(grid, rungs, is_synthetic=False)
    # Primary anchor for ablation deltas: per-chip EXP2 v5 hcmarl_anchor (when present).
    if "hcmarl_anchor" in quant.get("per_method", {}):
        quant["primary_anchor"] = "hcmarl_anchor"
        quant["primary_anchor_source"] = "EXP2 v5 per-chip (Results 2/logs/hcmarl_anchor)"
    # Cross-chip reference: EXP1 hcmarl (kept for chip-determinism comparison).
    try:
        exp1_quant = json.load(open(RESULTS_4 / "Result 1 Analysis" / "_quant_analysis.json"))
        hcmarl_exp1 = exp1_quant["per_method"]["hcmarl"]["per_metric_training_log"]["cumulative_reward"]["final_window_mean_xseed"]
        quant["hcmarl_full_anchor_from_exp1"] = hcmarl_exp1
    except Exception as e:
        quant["hcmarl_full_anchor_from_exp1"] = {"error": str(e)}
    out_dir = RESULTS_4 / "Result 2 Analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_learning_curves(quant, out_dir / "_learning_curves.json")
    with open(out_dir / "_quant_analysis.json", "w", encoding="utf-8") as f:
        json.dump(strip_learning_curves(quant), f, indent=2)
    write_metrics_table(quant, out_dir / "_metrics_table.csv")
    with open(out_dir / "_pairwise_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(quant["pairwise"], f, indent=2)
    print(f"  Wrote -> {out_dir}/")
    return quant


def run_exp3():
    print("=" * 78); print("Result 3 Part 1 (EXP3 K-discovery)"); print("=" * 78)
    out_dir = RESULTS_4 / "Result 3 Analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    part1 = analyze_exp3_part1(ROOT / "Results 3" / "Part 1")
    with open(out_dir / "_quant_part1.json", "w", encoding="utf-8") as f:
        json.dump(part1, f, indent=2)
    print(f"  Part 1: {list(part1['regimes'].keys())}")
    for regime, info in part1["regimes"].items():
        m = info["mmicrl_results"]; g = info["ground_truth_metrics"]
        print(f"    {regime}: K_discovered={m.get('K_discovered')}  MI={m.get('mutual_information')}  "
              f"ARI={g.get('ARI')}  NMI={g.get('NMI')}  verdict={info.get('verdict')[:60]}...")

    print("=" * 78); print("Result 3 Part 2 (EXP3 SYNTHETIC HCMARL)"); print("=" * 78)
    grid = ROOT / "Results 3" / "Part 2"
    arms = {
        "hcmarl_with_mmicrl": grid / "part2_synthetic_hcmarl_k3",
        "hcmarl_no_mmicrl":   grid / "part2_synthetic_no_mmicrl_k3",
    }
    if not all(p.exists() for p in arms.values()):
        print("  Part 2 dirs missing -- skipping.")
        return part1
    part2 = analyze_method_grid(grid, arms, is_synthetic=True)
    write_learning_curves(part2, out_dir / "_learning_curves_part2.json")
    with open(out_dir / "_quant_part2.json", "w", encoding="utf-8") as f:
        json.dump(strip_learning_curves(part2), f, indent=2)
    with open(out_dir / "_pairwise_part2.json", "w", encoding="utf-8") as f:
        json.dump(part2["pairwise"], f, indent=2)
    write_metrics_table(part2, out_dir / "_metrics_table_part2.csv")

    # combined index
    with open(out_dir / "_quant_index.json", "w", encoding="utf-8") as f:
        json.dump({
            "part1_file": "_quant_part1.json",
            "part2_file": "_quant_part2.json",
            "is_synthetic_part1": False,
            "is_synthetic_part2": True,
            "n_regimes_part1": len(part1["regimes"]),
            "n_arms_part2": part2.get("n_methods", 0),
        }, f, indent=2)
    print(f"  Wrote -> {out_dir}/")
    return part1, part2


def main():
    run_exp1()
    run_exp2()
    run_exp3()
    print("\n" + "=" * 78); print("STAGE 1 COMPLETE — quantitative analysis written"); print("=" * 78)


if __name__ == "__main__":
    main()
