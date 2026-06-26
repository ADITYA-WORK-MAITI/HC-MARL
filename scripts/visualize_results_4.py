"""EXP4 stage 2 — visualizations from stage-1 quant JSONs.

Reads:
  Results 4/Result 1 Analysis/_quant_analysis.json + _learning_curves.json + _pairwise_comparisons.json
  Results 4/Result 2 Analysis/_quant_analysis.json + _learning_curves.json + _pairwise_comparisons.json
  Results 4/Result 3 Analysis/_quant_part1.json + _quant_part2.json + _learning_curves_part2.json + _pairwise_part2.json

Writes PNGs (300 dpi) to the same Result {1,2,3} Analysis/ folders, plus a
combined/ folder under Results 4/ for cross-experiment comparisons.

Synthetic-data panels (Result 3 Part 2) carry a visible "SYNTHETIC PLACEHOLDER"
watermark on every figure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_4 = ROOT / "Results 4"

# Paper-grade defaults
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.frameon": False,
})

METHOD_COLORS = {
    "hcmarl":               "#2A6F97",   # deep blue
    "mappo":                "#D62828",   # red
    "ippo":                 "#F77F00",   # orange
    "mappo_lag":            "#7209B7",   # purple
    "happo":                "#3A9D5D",   # green
    "shielded_mappo":       "#9B5DE5",   # purple-pink
    "hcmarl_anchor":        "#2A6F97",
    "ablation_no_ecbf":         "#D62828",
    "ablation_no_nswf":         "#F77F00",
    "ablation_no_divergent":    "#7209B7",
    "ablation_no_reperfusion":  "#8D5524",
    "ablation_no_mmicrl":       "#E63946",
    "hcmarl_with_mmicrl":   "#2A6F97",
    "hcmarl_no_mmicrl":     "#E63946",
}
METHOD_PRETTY = {
    "hcmarl": "HCMARL", "mappo": "MAPPO", "ippo": "PS-IPPO", "mappo_lag": "MAPPO-Lag",
    "happo": "HAPPO", "shielded_mappo": "Shielded-MAPPO",
    "hcmarl_anchor": "HCMARL anchor",
    "ablation_no_ecbf": "no ECBF",
    "ablation_no_nswf": "no NSWF",
    "ablation_no_divergent": "no divergent",
    "ablation_no_reperfusion": "no reperfusion",
    "ablation_no_mmicrl": "no MMICRL",
    "hcmarl_with_mmicrl": "HCMARL + MMICRL",
    "hcmarl_no_mmicrl":   "HCMARL  no MMICRL",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def watermark(ax, text="SYNTHETIC PLACEHOLDER", color="#cc0000", alpha=0.12):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=32, color=color,
            alpha=alpha, rotation=18, weight="bold", zorder=0)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur


def plot_learning_curve(ax, methods: List[str], curves: Dict, metric: str,
                        ylabel: str, log_y: bool = False):
    """Overlay IQM + 95% CI ribbon for `metric` across methods."""
    plotted = False
    for m in methods:
        if m not in curves or metric not in curves[m]:
            continue
        c = curves[m][metric]
        x = np.asarray(c["bin_centers_episode"])
        y = np.asarray(c["iqm"])
        lo = np.asarray(c["ci_lo"])
        hi = np.asarray(c["ci_hi"])
        color = METHOD_COLORS.get(m, "gray")
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.6,
                label=f"{METHOD_PRETTY.get(m, m)}")
        plotted = True
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if log_y:
        try:
            ax.set_yscale("symlog")
        except Exception:
            pass
    if plotted:
        ax.legend(loc="best", fontsize=8)


def plot_iqm_bars(ax, methods: List[str], quant: Dict, metric: str,
                  use_best: bool = False, ylabel: str = "", title: str = ""):
    """Horizontal bar of IQM with 95% CI error bars, sorted by IQM."""
    rows = []
    for m in methods:
        info = safe_get(quant, "per_method", m, "per_metric_training_log", metric)
        if info is None:
            continue
        key = "best_xseed" if use_best else "final_window_mean_xseed"
        s = info.get(key)
        if s is None:
            continue
        rows.append({
            "method": m,
            "iqm": s["iqm"],
            "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
            "n": s["n_seeds"],
        })
    if not rows:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center")
        return
    # Sort: higher_is_better -> descending; else ascending
    hib = HIGHER_IS_BETTER.get(metric, True)
    rows.sort(key=lambda r: r["iqm"], reverse=bool(hib))
    labels = [METHOD_PRETTY.get(r["method"], r["method"]) for r in rows]
    iqms = [r["iqm"] for r in rows]
    err_lo = [r["iqm"] - r["ci_lo"] for r in rows]
    err_hi = [r["ci_hi"] - r["iqm"] for r in rows]
    colors = [METHOD_COLORS.get(r["method"], "gray") for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, iqms, xerr=[err_lo, err_hi], color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.6, capsize=4)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    # annotate values
    for yi, r in zip(y, rows):
        ax.text(r["iqm"], yi, f"  {r['iqm']:.2f}", va="center", fontsize=8)


HIGHER_IS_BETTER = {
    "actor_loss": False, "cost_ema": False, "critic_loss": False,
    "cumulative_cost": False, "cumulative_reward": True,
    "ecbf_interventions": False, "forced_rest_rate": None,
    "jain_fairness": True, "lazy_agent_flag": False, "peak_fatigue": False,
    "per_agent_entropy_mean": None, "per_agent_entropy_min": None,
    "safety_autonomy_index": None, "safety_rate": True,
    "tasks_completed": True, "value_loss": False, "violation_rate": False,
    "wall_time": None, "lambda": None, "cost_critic_loss": False,
    "policy_loss": False, "constraint_recovery_time": None,
    "entropy": None,
}


# ----------------------------------------------------------------------
# Result 1 (EXP1) figures
# ----------------------------------------------------------------------
def viz_result1():
    out_dir = RESULTS_4 / "Result 1 Analysis"
    quant_path = out_dir / "_quant_analysis.json"
    curves_path = out_dir / "_learning_curves.json"
    if not quant_path.exists() or not curves_path.exists():
        print(f"  [Result 1] missing {quant_path.name} or {curves_path.name}, skipping")
        return
    quant = json.load(open(quant_path))
    curves = json.load(open(curves_path))
    # 5-method headline lineup. Filter to methods actually present in the JSON
    # so the figures gracefully skip any rung whose data is missing rather
    # than silently producing a mis-sized chart.
    candidate_methods = ["hcmarl", "mappo", "mappo_lag", "happo", "shielded_mappo"]
    methods = [m for m in candidate_methods if m in quant.get("per_method", {})]

    # --- 1. 2x2 learning-curve grid (reward, safety_rate, violation_rate, peak_fatigue) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_learning_curve(axes[0, 0], methods, curves, "cumulative_reward",
                        "Episode reward (cumulative)", log_y=False)
    axes[0, 0].set_title("Learning curve  reward (IQM, 95% CI)", loc="left", fontweight="bold")
    plot_learning_curve(axes[0, 1], methods, curves, "safety_rate",
                        "Safety rate")
    axes[0, 1].set_title("Safety rate", loc="left", fontweight="bold")
    plot_learning_curve(axes[1, 0], methods, curves, "violation_rate",
                        "Violation rate")
    axes[1, 0].set_title("Violation rate", loc="left", fontweight="bold")
    plot_learning_curve(axes[1, 1], methods, curves, "peak_fatigue",
                        "Peak fatigue (MF)")
    axes[1, 1].set_title("Peak fatigue", loc="left", fontweight="bold")
    fig.suptitle("Result 1 (EXP1)  HCMARL vs baselines, 10 seeds, 2M steps",
                 fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_01_learning_curves.png", bbox_inches="tight")
    plt.close(fig)

    # --- 2. IQM bar chart: final-window reward + best reward + safety + violation ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    plot_iqm_bars(axes[0, 0], methods, quant, "cumulative_reward",
                  use_best=False, ylabel="Reward (final 500 ep, IQM)",
                  title="Final reward (IQM + 95% CI)")
    plot_iqm_bars(axes[0, 1], methods, quant, "cumulative_reward",
                  use_best=True, ylabel="Best reward (IQM)",
                  title="Best reward")
    plot_iqm_bars(axes[1, 0], methods, quant, "safety_rate",
                  use_best=False, ylabel="Safety rate (final, IQM)",
                  title="Safety rate")
    plot_iqm_bars(axes[1, 1], methods, quant, "violation_rate",
                  use_best=False, ylabel="Violation rate (final, IQM)",
                  title="Violation rate")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_02_iqm_bars.png", bbox_inches="tight")
    plt.close(fig)

    # --- 3. Per-seed strip plot for cumulative_reward and best_reward ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in [
        (axes[0], "per_seed_final_window_mean", "Final 500-ep mean reward"),
        (axes[1], "per_seed_best",               "Best reward across run"),
    ]:
        ys = []
        for i, m in enumerate(methods):
            info = safe_get(quant, "per_method", m, "per_metric_training_log",
                            "cumulative_reward")
            if info is None: continue
            vals = np.asarray(info[key])
            ys.append(vals)
            ax.scatter(np.full(vals.size, i) + np.random.uniform(-0.08, 0.08, vals.size),
                       vals, s=42, color=METHOD_COLORS[m], alpha=0.85,
                       edgecolor="black", linewidth=0.5, zorder=3)
            iqm_pt = info["final_window_mean_xseed"]["iqm"] if "final" in key else info["best_xseed"]["iqm"]
            ax.scatter(i, iqm_pt, s=180, marker="D", color=METHOD_COLORS[m],
                       edgecolor="black", linewidth=1.2, zorder=4, label="_nolegend_")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_PRETTY[m] for m in methods], rotation=15)
        ax.set_ylabel(label)
        ax.set_title(label, loc="left", fontweight="bold")
    fig.suptitle("Per-seed rewards (n=10) with IQM diamond",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_03_per_seed_strip.png", bbox_inches="tight")
    plt.close(fig)

    # --- 4. Pairwise PoI matrix on cumulative_reward final-window ---
    pairwise = quant.get("pairwise", {})
    n = len(methods)
    poi_mat = np.full((n, n), np.nan)
    diff_mat = np.full((n, n), np.nan)
    for i, ma in enumerate(methods):
        for j, mb in enumerate(methods):
            if i == j:
                poi_mat[i, j] = 0.5; diff_mat[i, j] = 0.0; continue
            key = f"{ma}__vs__{mb}" if f"{ma}__vs__{mb}" in pairwise else f"{mb}__vs__{ma}"
            entry = pairwise.get(key, {}).get("cumulative_reward__final_window")
            if entry is None: continue
            if key.startswith(ma):
                poi_mat[i, j] = entry["poi"]
                diff_mat[i, j] = entry["mean_diff"]
            else:
                poi_mat[i, j] = 1.0 - entry["poi"]
                diff_mat[i, j] = -entry["mean_diff"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title, cmap, vmin, vmax, fmt in [
        (axes[0], poi_mat, "P(row > column)  reward final-window", "RdBu_r", 0.0, 1.0, "{:.2f}"),
        (axes[1], diff_mat, "row mean - column mean (reward, final)", "RdBu_r", None, None, "{:+.0f}"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([METHOD_PRETTY[m] for m in methods], rotation=30, ha="right")
        ax.set_yticklabels([METHOD_PRETTY[m] for m in methods])
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, fmt.format(mat[i, j]),
                            ha="center", va="center", fontsize=8,
                            color="black" if 0.3 < (mat[i, j] - (vmin or mat.min())) / (((vmax or mat.max()) - (vmin or mat.min())) or 1) < 0.7 else "white")
        ax.set_title(title, fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_04_pairwise_poi.png", bbox_inches="tight")
    plt.close(fig)

    # --- 5. Multi-metric IQM table heatmap ---
    metrics = ["cumulative_reward", "safety_rate", "violation_rate",
               "peak_fatigue", "ecbf_interventions", "tasks_completed",
               "jain_fairness"]
    iqm_grid = np.full((len(methods), len(metrics)), np.nan)
    cell_text = [["" for _ in metrics] for _ in methods]
    for i, m in enumerate(methods):
        for j, met in enumerate(metrics):
            info = safe_get(quant, "per_method", m, "per_metric_training_log", met)
            if info is None: continue
            s = info["final_window_mean_xseed"]
            iqm_grid[i, j] = s["iqm"]
            cell_text[i][j] = f"{s['iqm']:.2f}\n[{s['ci_lo']:.2f},{s['ci_hi']:.2f}]"
    # Per-column z-normalize so heatmap is comparable
    z = np.zeros_like(iqm_grid)
    for j in range(iqm_grid.shape[1]):
        col = iqm_grid[:, j]
        finite = col[np.isfinite(col)]
        if finite.size > 1 and finite.std() > 0:
            z[:, j] = (col - finite.mean()) / finite.std()
            if HIGHER_IS_BETTER.get(metrics[j]) is False:
                z[:, j] = -z[:, j]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(z, cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticklabels([METHOD_PRETTY[m] for m in methods])
    for i in range(len(methods)):
        for j in range(len(metrics)):
            ax.text(j, i, cell_text[i][j], ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("All metrics: IQM [95% CI]  green=better",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_05_metric_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  [Result 1] 5 PNGs written to {out_dir}")


# ----------------------------------------------------------------------
# Result 2 (EXP2 ablations)
# ----------------------------------------------------------------------
def viz_result2():
    out_dir = RESULTS_4 / "Result 2 Analysis"
    quant_path = out_dir / "_quant_analysis.json"
    curves_path = out_dir / "_learning_curves.json"
    if not quant_path.exists():
        print(f"  [Result 2] missing {quant_path.name}, skipping"); return
    quant = json.load(open(quant_path))
    curves = json.load(open(curves_path))
    # EXP2 v5: 5 ablation rungs + per-chip hcmarl_anchor.
    # Filter to rungs actually present (graceful skip if any rung is missing).
    candidate_rungs = ["hcmarl_anchor",
                       "ablation_no_ecbf", "ablation_no_nswf",
                       "ablation_no_divergent", "ablation_no_reperfusion",
                       "ablation_no_mmicrl"]
    rungs = [r for r in candidate_rungs if r in quant.get("per_method", {})]
    # Ablation-only list (excludes anchor) for delta plots and heatmap rows.
    ablation_rungs = [r for r in rungs if r != "hcmarl_anchor"]

    # --- 1. Learning curves ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_learning_curve(axes[0, 0], rungs, curves, "cumulative_reward", "Episode reward")
    axes[0, 0].set_title("Reward (ablations + anchor)", loc="left", fontweight="bold")
    plot_learning_curve(axes[0, 1], rungs, curves, "safety_rate", "Safety rate")
    axes[0, 1].set_title("Safety rate", loc="left", fontweight="bold")
    plot_learning_curve(axes[1, 0], rungs, curves, "violation_rate", "Violation rate")
    axes[1, 0].set_title("Violation rate", loc="left", fontweight="bold")
    plot_learning_curve(axes[1, 1], rungs, curves, "peak_fatigue", "Peak fatigue")
    axes[1, 1].set_title("Peak fatigue", loc="left", fontweight="bold")
    # Primary anchor: v5 per-chip hcmarl_anchor IQM (if present).
    v5_anchor = safe_get(quant, "per_method", "hcmarl_anchor",
                         "per_metric_training_log", "cumulative_reward",
                         "final_window_mean_xseed")
    if v5_anchor:
        axes[0, 0].axhline(v5_anchor["iqm"], linestyle="--", color="black",
                           linewidth=1.4,
                           label=f"v5 anchor (per-chip) {v5_anchor['iqm']:.0f}")
        axes[0, 0].fill_between(axes[0, 0].get_xlim(),
                                v5_anchor["ci_lo"], v5_anchor["ci_hi"],
                                color="black", alpha=0.07)
        axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Result 2 (EXP2 v5)  per-chip anchor + 5 remove-one ablations, 10 seeds each, 2M steps",
                 fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_01_learning_curves.png", bbox_inches="tight")
    plt.close(fig)

    # --- 2. Ablation reward bar (final IQM) with v5 per-chip anchor ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    rows = []
    for r in rungs:  # includes hcmarl_anchor when present
        info = safe_get(quant, "per_method", r, "per_metric_training_log",
                        "cumulative_reward")
        if info is None: continue
        s = info["final_window_mean_xseed"]
        label = METHOD_PRETTY.get(r, r)
        if r == "hcmarl_anchor":
            label = "HCMARL anchor (per-chip)"
        rows.append({"label": label,
                     "iqm": s["iqm"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                     "color": METHOD_COLORS.get(r, "#2A6F97" if r == "hcmarl_anchor" else "gray")})
    # Cross-chip EXP1 reference (kept for chip-determinism comparison).
    exp1_ref = quant.get("hcmarl_full_anchor_from_exp1")
    if isinstance(exp1_ref, dict) and "iqm" in exp1_ref:
        rows.append({"label": "HCMARL EXP1 (other chip ref.)",
                     "iqm": exp1_ref["iqm"], "ci_lo": exp1_ref["ci_lo"], "ci_hi": exp1_ref["ci_hi"],
                     "color": "#7F7F7F"})
    rows.sort(key=lambda r_: r_["iqm"], reverse=True)
    y = np.arange(len(rows))
    ax.barh(y, [r["iqm"] for r in rows],
            xerr=[[r["iqm"] - r["ci_lo"] for r in rows],
                  [r["ci_hi"] - r["iqm"] for r in rows]],
            color=[r["color"] for r in rows], edgecolor="black", capsize=4)
    ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Final 500-ep reward (IQM, 95% CI)")
    ax.set_title("Ablation impact on final reward",
                 loc="left", fontweight="bold")
    for yi, r in zip(y, rows):
        ax.text(r["iqm"], yi, f"  {r['iqm']:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_02_ablation_reward_bar.png", bbox_inches="tight")
    plt.close(fig)

    # --- 3. Multi-metric heatmap (anchor + ablations) ---
    metrics = ["cumulative_reward", "safety_rate", "violation_rate",
               "peak_fatigue", "ecbf_interventions", "tasks_completed", "jain_fairness"]
    heat_rungs = rungs  # anchor first (if present), then ablations
    iqm_grid = np.full((len(heat_rungs), len(metrics)), np.nan)
    cell_text = [["" for _ in metrics] for _ in heat_rungs]
    for i, m in enumerate(heat_rungs):
        for j, met in enumerate(metrics):
            info = safe_get(quant, "per_method", m, "per_metric_training_log", met)
            if info is None: continue
            s = info["final_window_mean_xseed"]
            iqm_grid[i, j] = s["iqm"]
            cell_text[i][j] = f"{s['iqm']:.2f}\n[{s['ci_lo']:.2f},{s['ci_hi']:.2f}]"
    z = np.zeros_like(iqm_grid)
    for j in range(iqm_grid.shape[1]):
        col = iqm_grid[:, j]; finite = col[np.isfinite(col)]
        if finite.size > 1 and finite.std() > 0:
            z[:, j] = (col - finite.mean()) / finite.std()
            if HIGHER_IS_BETTER.get(metrics[j]) is False:
                z[:, j] = -z[:, j]
    fig, ax = plt.subplots(figsize=(13, max(4.8, 0.95 * len(heat_rungs))))
    im = ax.imshow(z, cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_yticks(range(len(heat_rungs)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    yticklabels = []
    for r in heat_rungs:
        yticklabels.append("HCMARL anchor (per-chip)" if r == "hcmarl_anchor"
                           else METHOD_PRETTY.get(r, r))
    ax.set_yticklabels(yticklabels)
    for i in range(len(heat_rungs)):
        for j in range(len(metrics)):
            ax.text(j, i, cell_text[i][j], ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("EXP2 v5  anchor + ablations all metrics  IQM [CI]  green=better",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_03_metric_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # --- 4. Per-component attribution Delta_X bar chart (paired mean_diff with bootstrap CI) ---
    # Use paired mean_diff from the pairwise JSON (not IQM-IQM) so the bar height
    # and CI come from the same statistic. With outliers in the anchor (seeds 2, 8 from
    # MMICRL escape), the IQM-IQM and mean-diff numbers diverge -- mean-diff captures
    # the outlier-driven story honestly (IQM trims them out).
    pw_path = out_dir / "_pairwise_comparisons.json"
    if v5_anchor and ablation_rungs and pw_path.exists():
        pw = json.load(open(pw_path))
        deltas = []
        for r in ablation_rungs:
            pair_key = f"hcmarl_anchor__vs__{r}"
            pair = pw.get(pair_key, {}).get("cumulative_reward__final_window")
            if pair is None: continue
            # mean_diff = mean(anchor[i] - rung[i]) -- positive means anchor is higher
            deltas.append({
                "rung": r,
                "label": METHOD_PRETTY.get(r, r).replace("ablation_", "no_"),
                "delta": pair["mean_diff"],
                "ci_lo": pair["diff_ci_lo"],
                "ci_hi": pair["diff_ci_hi"],
            })
        # sort: largest impact first
        deltas.sort(key=lambda x: x["delta"], reverse=True)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        y = np.arange(len(deltas))
        vals = [dx["delta"] for dx in deltas]
        errs_lo = [(dx["delta"] - dx["ci_lo"]) if dx["ci_lo"] is not None else 0 for dx in deltas]
        errs_hi = [(dx["ci_hi"] - dx["delta"]) if dx["ci_hi"] is not None else 0 for dx in deltas]
        bars = ax.barh(y, vals, xerr=[errs_lo, errs_hi],
                       color=["#C0392B" if v > 0 else "#2980B9" for v in vals],
                       edgecolor="black", capsize=4)
        ax.set_yticks(y); ax.set_yticklabels([dx["label"] for dx in deltas])
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Delta_X = paired mean(anchor[i] - no_X[i]);  +ve = component helps; -ve = component hurts")
        ax.set_title("EXP2 v5 per-component reward attribution (paired mean diff, bootstrap 95% CI)",
                     loc="left", fontweight="bold")
        for yi, dx in zip(y, deltas):
            ax.text(dx["delta"], yi, f"  {dx['delta']:+.0f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_04_per_component_attribution.png", bbox_inches="tight")
        plt.close(fig)

    print(f"  [Result 2] PNGs written to {out_dir}")


# ----------------------------------------------------------------------
# Result 3 Part 1 + Part 2
# ----------------------------------------------------------------------
def viz_result3():
    out_dir = RESULTS_4 / "Result 3 Analysis"
    p1 = json.load(open(out_dir / "_quant_part1.json"))

    # ---- Part 1 figure: K=3 confusion matrix only ----
    # Earlier versions stacked a K-selection bar-chart row and a K=1 collapse
    # panel alongside. The k_selection_median_per_k dict is empty in the
    # current artefact and the K=1 single-cell matrix renders as a near-blank
    # square, so both were dropped. The K=1 verdict is captured in the
    # caption and discussed in the prose; the visual evidence that matters
    # is the diagonal K=3 confusion matrix.
    info = p1["regimes"]["K3"]
    gtm = info["ground_truth_metrics"]
    cm = np.asarray(gtm.get("confusion_matrix") or [[0]])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(cm.shape[1])); ax.set_yticks(range(cm.shape[0]))
    ax.set_xlabel("Predicted type"); ax.set_ylabel("True type")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white",
                    fontsize=14, fontweight="bold")
    m = info["mmicrl_results"]
    ax.set_title(f"K3: K_disc={m.get('K_discovered')}  ARI={gtm.get('ARI'):.3f}  "
                 f"NMI={gtm.get('NMI'):.3f}  MI={m.get('mutual_information'):.3f}",
                 fontsize=10, fontweight="bold")
    fig.suptitle("Result 3 Part 1  MMICRL K-discovery on synthetic data",
                 fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_part1_kselection.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Part 2: synthetic HCMARL learning curves with watermark ----
    p2_path = out_dir / "_quant_part2.json"
    p2c_path = out_dir / "_learning_curves_part2.json"
    if not p2_path.exists() or not p2c_path.exists():
        print(f"  [Result 3] Part 2 missing, skipping Part 2 figs"); return
    p2 = json.load(open(p2_path))
    curves = json.load(open(p2c_path))
    arms = list(p2["per_method"].keys())

    # 2x2 learning curves with WATERMARK
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_learning_curve(axes[0, 0], arms, curves, "cumulative_reward",
                        "Episode reward")
    axes[0, 0].set_title("Reward  HCMARL on synthetic K=3 (5 seeds each)", loc="left", fontweight="bold")
    plot_learning_curve(axes[0, 1], arms, curves, "safety_rate", "Safety rate")
    axes[0, 1].set_title("Safety rate", loc="left", fontweight="bold")
    plot_learning_curve(axes[1, 0], arms, curves, "violation_rate", "Violation rate")
    axes[1, 0].set_title("Violation rate", loc="left", fontweight="bold")
    plot_learning_curve(axes[1, 1], arms, curves, "peak_fatigue", "Peak fatigue")
    axes[1, 1].set_title("Peak fatigue", loc="left", fontweight="bold")
    for ax in axes.ravel():
        watermark(ax)
    fig.suptitle("Result 3 Part 2  SYNTHETIC paired learning curves (placeholders)",
                 fontsize=12, fontweight="bold", y=1.00, color="#cc0000")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_part2_learning_curves_SYNTHETIC.png", bbox_inches="tight")
    plt.close(fig)

    # IQM bar  paired
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    plot_iqm_bars(axes[0], arms, p2, "cumulative_reward",
                  use_best=False, ylabel="Final-window reward (IQM)",
                  title="Reward (final 500 ep) SYNTHETIC")
    plot_iqm_bars(axes[1], arms, p2, "safety_rate",
                  use_best=False, ylabel="Safety rate (IQM)",
                  title="Safety rate (final) SYNTHETIC")
    for ax in axes.ravel():
        watermark(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_part2_iqm_bars_SYNTHETIC.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  [Result 3] 3 PNGs written to {out_dir}")


# ----------------------------------------------------------------------
# Combined cross-experiment figure
# ----------------------------------------------------------------------
def viz_combined():
    out_dir = RESULTS_4 / "Combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        q1 = json.load(open(RESULTS_4 / "Result 1 Analysis" / "_quant_analysis.json"))
        q2 = json.load(open(RESULTS_4 / "Result 2 Analysis" / "_quant_analysis.json"))
    except Exception as e:
        print(f"  [Combined] missing inputs: {e}"); return

    # ---- Figure 1: cross-experiment reward ladder (EXP1 5 methods + EXP2 v5 6 rungs) ----
    rows = []
    for m in ["hcmarl", "mappo", "mappo_lag", "happo", "shielded_mappo"]:
        info = safe_get(q1, "per_method", m, "per_metric_training_log", "cumulative_reward")
        if info: rows.append({
            "label": METHOD_PRETTY.get(m, m) + " (EXP1)",
            "iqm": info["final_window_mean_xseed"]["iqm"],
            "ci_lo": info["final_window_mean_xseed"]["ci_lo"],
            "ci_hi": info["final_window_mean_xseed"]["ci_hi"],
            "color": METHOD_COLORS.get(m, "#666666"),
        })
    for r in ["hcmarl_anchor", "ablation_no_ecbf", "ablation_no_nswf",
              "ablation_no_divergent", "ablation_no_reperfusion", "ablation_no_mmicrl"]:
        info = safe_get(q2, "per_method", r, "per_metric_training_log", "cumulative_reward")
        if info:
            label = "HCMARL anchor (EXP2 v5, per-chip)" if r == "hcmarl_anchor" \
                    else METHOD_PRETTY.get(r, r) + " (EXP2 v5)"
            rows.append({
                "label": label,
                "iqm": info["final_window_mean_xseed"]["iqm"],
                "ci_lo": info["final_window_mean_xseed"]["ci_lo"],
                "ci_hi": info["final_window_mean_xseed"]["ci_hi"],
                "color": "#2A6F97" if r == "hcmarl_anchor" else METHOD_COLORS.get(r, "#666666"),
            })
    rows.sort(key=lambda r_: r_["iqm"], reverse=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(rows))
    ax.barh(y, [r["iqm"] for r in rows],
            xerr=[[r["iqm"] - r["ci_lo"] for r in rows],
                  [r["ci_hi"] - r["iqm"] for r in rows]],
            color=[r["color"] for r in rows], edgecolor="black", capsize=4)
    ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Final-window cumulative reward (IQM, 95% CI)")
    ax.set_title("Cross-experiment reward ladder  EXP1 5 methods + EXP2 v5 anchor + 5 ablations",
                 loc="left", fontweight="bold")
    for yi, r in zip(y, rows):
        ax.text(r["iqm"], yi, f"  {r['iqm']:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_01_cross_experiment_reward_ladder.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: Reward x Safety Pareto frontier ----
    fig, ax = plt.subplots(figsize=(10, 6.5))
    points = []
    for m in ["hcmarl", "mappo", "mappo_lag", "happo", "shielded_mappo"]:
        rinfo = safe_get(q1, "per_method", m, "per_metric_training_log", "cumulative_reward")
        sinfo = safe_get(q1, "per_method", m, "per_metric_training_log", "safety_rate")
        if rinfo and sinfo:
            points.append({
                "label": METHOD_PRETTY.get(m, m) + " (EXP1)",
                "r": rinfo["final_window_mean_xseed"]["iqm"],
                "s": sinfo["final_window_mean_xseed"]["iqm"],
                "color": METHOD_COLORS.get(m, "#666"), "marker": "o", "size": 110,
            })
    for r in ["hcmarl_anchor", "ablation_no_ecbf", "ablation_no_nswf",
              "ablation_no_divergent", "ablation_no_reperfusion", "ablation_no_mmicrl"]:
        rinfo = safe_get(q2, "per_method", r, "per_metric_training_log", "cumulative_reward")
        sinfo = safe_get(q2, "per_method", r, "per_metric_training_log", "safety_rate")
        if rinfo and sinfo:
            label = "HCMARL anchor (EXP2 v5)" if r == "hcmarl_anchor" \
                    else METHOD_PRETTY.get(r, r) + " (EXP2 v5)"
            points.append({
                "label": label,
                "r": rinfo["final_window_mean_xseed"]["iqm"],
                "s": sinfo["final_window_mean_xseed"]["iqm"],
                "color": "#2A6F97" if r == "hcmarl_anchor" else METHOD_COLORS.get(r, "#666"),
                "marker": "s" if r == "hcmarl_anchor" else "^",
                "size": 130 if r == "hcmarl_anchor" else 90,
            })
    # Plot points with `label=` so each one shows up in the legend.  In-plot
    # text labels were unreadable: 9 of 11 methods cluster near (R=-1400, S=0.97)
    # and (R=-9000, S=0.97), so per-point bbox annotations stacked into an
    # illegible blob.  Side legend with marker glyphs handles disambiguation.
    for p in points:
        ax.scatter(p["r"], p["s"], color=p["color"], marker=p["marker"],
                   s=p["size"], edgecolor="black", linewidth=1.0, zorder=3,
                   label=p["label"])
    ax.set_xlabel("Final-window cumulative reward (IQM)  higher is better")
    ax.set_ylabel("Final-window safety rate (IQM)  higher is better")
    ax.set_title("Reward x Safety  EXP1 + EXP2 v5  (top-right is Pareto-best)",
                 loc="left", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    # Pareto frontier highlight: non-dominated points
    pareto = []
    for p in sorted(points, key=lambda x: -x["r"]):
        if not any(q["r"] >= p["r"] and q["s"] > p["s"] for q in points):
            pareto.append(p)
    pareto.sort(key=lambda x: x["r"])
    ax.plot([p["r"] for p in pareto], [p["s"] for p in pareto],
            color="#cc0000", linestyle="--", linewidth=1.5, alpha=0.6,
            label="Pareto frontier", zorder=2)
    # Legend OUTSIDE the plot on the right; nothing overlaps the markers.
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8.5, frameon=True, framealpha=0.95,
              edgecolor="gray", title="Method", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_02_pareto_frontier.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: per-component attribution (paired mean_diff, paper money figure) ----
    pw2 = json.load(open(RESULTS_4 / "Result 2 Analysis" / "_pairwise_comparisons.json"))
    deltas = []
    for r in ["ablation_no_reperfusion", "ablation_no_ecbf",
              "ablation_no_nswf", "ablation_no_mmicrl", "ablation_no_divergent"]:
        pair = pw2.get(f"hcmarl_anchor__vs__{r}", {}).get("cumulative_reward__final_window")
        if pair is None: continue
        deltas.append({
            "rung": r, "label": r.replace("ablation_", ""),
            "delta": pair["mean_diff"],
            "ci_lo": pair["diff_ci_lo"],
            "ci_hi": pair["diff_ci_hi"],
        })
    if deltas:
        deltas.sort(key=lambda x: x["delta"], reverse=True)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        y = np.arange(len(deltas))
        vals = [dx["delta"] for dx in deltas]
        errs_lo = [dx["delta"] - dx["ci_lo"] for dx in deltas]
        errs_hi = [dx["ci_hi"] - dx["delta"] for dx in deltas]
        ax.barh(y, vals, xerr=[errs_lo, errs_hi],
                color=["#C0392B" if v > 100 else "#2980B9" if v < -100 else "#7F8C8D" for v in vals],
                edgecolor="black", capsize=4)
        ax.set_yticks(y); ax.set_yticklabels([dx["label"] for dx in deltas])
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Delta_X = paired mean(anchor[i] - no_X[i]) ;  +ve = component helps; -ve = component hurts")
        ax.set_title("Per-component reward attribution (EXP2 v5 paired mean diff, bootstrap 95% CI)",
                     loc="left", fontweight="bold")
        for yi, dx in zip(y, deltas):
            color = "white" if abs(dx["delta"]) > 5000 else "black"
            ax.text(dx["delta"] / 2 if abs(dx["delta"]) > 5000 else dx["delta"],
                    yi, f"  {dx['delta']:+.0f}", va="center",
                    ha="left" if abs(dx["delta"]) <= 5000 else "center",
                    color=color, fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_03_per_component_attribution.png", bbox_inches="tight")
        plt.close(fig)

    # ---- Figure 4: MMICRL chip-determinism comparison (EXP1 vs EXP2 v5) ----
    # Read per-seed MMICRL outputs from raw data
    fig, ax = plt.subplots(figsize=(11, 4.5))
    seeds = list(range(10))
    mi_exp1 = []
    mi_exp2 = []
    exp1_logs = ROOT / "Results 1" / "logs" / "hcmarl"
    exp2_logs = ROOT / "Results 2" / "logs" / "hcmarl_anchor"
    for s in seeds:
        p1 = exp1_logs / f"seed_{s}" / "mmicrl" / "mmicrl_results.json"
        p2 = exp2_logs / f"seed_{s}" / "mmicrl" / "mmicrl_results.json"
        if p1.exists():
            mi_exp1.append(json.load(open(p1)).get("mutual_information", 0.0))
        else:
            mi_exp1.append(np.nan)
        if p2.exists():
            mi_exp2.append(json.load(open(p2)).get("mutual_information", 0.0))
        else:
            mi_exp2.append(np.nan)
    x = np.arange(len(seeds)); w = 0.38
    ax.bar(x - w/2, mi_exp1, w, color="#A23B72", edgecolor="black",
           label="EXP1 (VM1 chip)  10/10 collapsed (max MI 5.96e-08)")
    ax.bar(x + w/2, mi_exp2, w, color="#2A6F97", edgecolor="black",
           label="EXP2 v5 (this chip)  2/10 escape (seeds 2 + 8)")
    ax.axhline(0.01, color="red", linestyle=":", linewidth=1.0,
               label="MI collapse threshold = 0.01")
    ax.set_xticks(x); ax.set_xticklabels([f"seed_{s}" for s in seeds])
    ax.set_xlabel("Seed (paired across chips)")
    ax.set_ylabel("MMICRL mutual information I(τ; z)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title("MMICRL chip-determinism  same Path G profile bytes; same seed; different L4 chips",
                 loc="left", fontweight="bold")
    # Legend BELOW the plot (outside data area) — the seed_2 + seed_8 bars
    # spike to ~1+ at the top, so any in-plot legend overlaps them.
    ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
              fontsize=9, frameon=True, framealpha=0.95,
              edgecolor="gray", ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_04_mmicrl_chip_determinism.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 5: compute cost ledger (info-table) ----
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.axis("off")
    cost_table = [
        ["Experiment", "Hardware", "Runs", "Wall-clock", "Spend (units)"],
        ["EXP0 (local CPU smoke)", "CPU only", "9 stages", "8 min", "~0"],
        ["EXP1 (VM1, headline)", "L4 spot", "5 methods x 10 seeds", "41h 50m", "~4,500"],
        ["EXP2 v5 (this VM, ablations)", "L4 spot", "6 rungs x 10 seeds", "~17.6h x 6 / 6-way parallel", "~1,036"],
        ["EXP3 Part A (local CPU)", "CPU only", "1 K=1 + 1 K=3 synthetic", "~10 min", "~0"],
        ["Appendix F probe", "L4 spot", "3 seeds + 1 smoke", "~22 min", "~38"],
        ["Total VM cost", "", "", "", "~5574 units"],
    ]
    tb = ax.table(cellText=cost_table[1:], colLabels=cost_table[0],
                  loc="center", cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1.0, 1.6)
    for i in range(len(cost_table[0])):
        tb[(0, i)].set_facecolor("#2A6F97")
        tb[(0, i)].set_text_props(color="white", fontweight="bold")
    for j in range(1, len(cost_table)):
        for i in range(len(cost_table[0])):
            if j == len(cost_table) - 1:  # totals row
                tb[(j, i)].set_facecolor("#FBE9D7")
                tb[(j, i)].set_text_props(fontweight="bold")
    ax.set_title("EXP4 compute ledger (all 4 experiments + Appendix F)",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_05_compute_cost_ledger.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  [Combined] 5 PNGs written to {out_dir}")


def main():
    print("=" * 78); print("STAGE 2  visualization"); print("=" * 78)
    viz_result1()
    viz_result2()
    viz_result3()
    viz_combined()
    print("\n" + "=" * 78); print("STAGE 2 COMPLETE  PNGs written"); print("=" * 78)


if __name__ == "__main__":
    main()
