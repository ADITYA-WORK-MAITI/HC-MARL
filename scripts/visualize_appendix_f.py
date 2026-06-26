"""EXP4 Stage 2 -- Appendix F probe figures.

Reads `Results 4/Result Appendix F Analysis/_quant_appendix_f.json` and produces:

  fig_01_log_std_evolution.png   Gaussian policy log_std per episode (3 seeds + smoke)
  fig_02_allocator_activity.png  per-episode NSWF allocator calls (proves wired)
  fig_03_reward_trajectory.png   per-seed cumulative_reward over training
  fig_04_mmicrl_pretrain_grid.png  per-seed MMICRL K + MI + props
  fig_05_diagnostics_panel.png   safety + peak_fatigue + ECBF interventions

Headline claim each figure supports:
  fig_01: Gaussian actor explores (no collapse, no divergence) -- continuous policy alive
  fig_02: NSWF allocator IS called every episode -- wired in continuous mode
  fig_03: reward improves (50K is short, not converged, but trajectory is real)
  fig_04: MMICRL pretrains successfully on Path G profiles regardless of action_mode
  fig_05: no NaN, ECBF active, fatigue under control
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APPENDIX_F = ROOT / "Results Appendix F"
OUT_DIR = ROOT / "Results 4" / "Result Appendix F Analysis"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

SEED_COLORS = {0: "#2A6F97", 1: "#A23B72", 2: "#F18F01", "smoke": "#6B6B6B"}


def read_csv_dict(path: Path) -> Dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path, newline="")))
    cols: Dict[str, np.ndarray] = {}
    for k in rows[0].keys():
        try:
            cols[k] = np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan
                                for r in rows])
        except ValueError:
            cols[k] = np.array([r[k] for r in rows], dtype=object)
    return cols


def load_csvs():
    csvs = {}
    for sid in [0, 1, 2]:
        p = APPENDIX_F / "logs" / "continuous_probe" / f"seed_{sid}" / "training_log.csv"
        if p.exists():
            csvs[sid] = read_csv_dict(p)
    smoke = APPENDIX_F / "logs" / "continuous_probe_smoke" / "seed_0" / "training_log.csv"
    if smoke.exists():
        csvs["smoke"] = read_csv_dict(smoke)
    return csvs


def fig_log_std(csvs):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for sid in [0, 1, 2]:
        if sid not in csvs: continue
        ep = csvs[sid].get("episode")
        ls = csvs[sid].get("gaussian_log_std_mean")
        if ep is None or ls is None: continue
        ax.plot(ep, ls, color=SEED_COLORS[sid], label=f"seed_{sid}",
                linewidth=1.4, alpha=0.9)
    if "smoke" in csvs:
        ep = csvs["smoke"].get("episode")
        ls = csvs["smoke"].get("gaussian_log_std_mean")
        if ep is not None and ls is not None:
            ax.plot(ep, ls, color=SEED_COLORS["smoke"], label="Gate-0 smoke (5K, seed_0)",
                    linewidth=1.0, alpha=0.7, linestyle="--")
    ax.axhline(2.0, color="#cc0000", linestyle=":", linewidth=1.0,
               label="LOG_STD_MAX = +2.0 (clamp)")
    ax.axhline(-5.0, color="#cc0000", linestyle=":", linewidth=1.0,
               label="LOG_STD_MIN = −5.0 (clamp)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean log σ (Gaussian actor)")
    ax.set_title("Appendix F  Gaussian policy log σ evolves freely within clamp [−5, +2]",
                 loc="left", fontweight="bold")
    ax.set_ylim(-1.0, 2.5)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_01_log_std_evolution.png", bbox_inches="tight")
    plt.close(fig)


def fig_allocator(csvs):
    fig, ax = plt.subplots(figsize=(9, 4.0))
    for sid in [0, 1, 2]:
        if sid not in csvs: continue
        ep = csvs[sid].get("episode")
        ac = csvs[sid].get("allocator_calls")
        if ep is None or ac is None: continue
        ax.plot(ep, ac, color=SEED_COLORS[sid], label=f"seed_{sid}",
                linewidth=1.2, marker="o", markersize=2, alpha=0.9)
    ax.axhline(16, color="black", linestyle="--", linewidth=1.0,
               label="Expected = max_steps / allocation_interval = 480 / 30 = 16")
    ax.set_ylim(0, 20)
    ax.set_xlabel("Episode")
    ax.set_ylabel("NSWF allocator calls per episode")
    ax.set_title("Appendix F  NSWF allocator fires 16/episode (wired and active in continuous mode)",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_02_allocator_activity.png", bbox_inches="tight")
    plt.close(fig)


def fig_reward_trajectory(csvs):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for sid in [0, 1, 2]:
        if sid not in csvs: continue
        ep = csvs[sid].get("episode")
        R = csvs[sid].get("cumulative_reward")
        if ep is None or R is None: continue
        ax.plot(ep, R, color=SEED_COLORS[sid], label=f"seed_{sid}",
                linewidth=1.4, alpha=0.9)
        # 10-ep moving average overlay
        if R.size >= 10:
            ma = np.convolve(R, np.ones(10)/10, mode="valid")
            ax.plot(ep[9:], ma, color=SEED_COLORS[sid], linestyle="-",
                    linewidth=2.5, alpha=0.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward per episode")
    ax.set_title("Appendix F  reward trajectory  3 seeds × 50K steps  (faded line = 10-ep MA)",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_03_reward_trajectory.png", bbox_inches="tight")
    plt.close(fig)


def fig_mmicrl_grid(quant):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    seeds = sorted(quant["production_runs"].keys())
    Ks = [quant["production_runs"][s].get("mmicrl", {}).get("n_types_discovered", np.nan) for s in seeds]
    MIs = [quant["production_runs"][s].get("mmicrl", {}).get("mutual_information", np.nan) for s in seeds]
    if quant.get("smoke_run"):
        seeds_disp = [str(s) for s in seeds] + ["smoke"]
        Ks.append(quant["smoke_run"].get("mmicrl", {}).get("n_types_discovered", np.nan))
        MIs.append(quant["smoke_run"].get("mmicrl", {}).get("mutual_information", np.nan))
    else:
        seeds_disp = [str(s) for s in seeds]
    x = np.arange(len(seeds_disp))
    bars1 = axes[0].bar(x, Ks, color=[SEED_COLORS.get(int(s) if s.isdigit() else "smoke",
                                                      SEED_COLORS["smoke"])
                                       for s in seeds_disp], edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(seeds_disp)
    axes[0].set_ylabel("K_discovered (MMICRL)")
    axes[0].set_title("MMICRL K_discovered per seed", loc="left", fontweight="bold")
    axes[0].set_ylim(0, max(Ks) + 0.5 if Ks else 5)
    for xi, k in zip(x, Ks):
        if not np.isnan(k):
            axes[0].text(xi, k, f" K={int(k)}", ha="center", va="bottom", fontsize=9)

    bars2 = axes[1].bar(x, MIs, color=[SEED_COLORS.get(int(s) if s.isdigit() else "smoke",
                                                      SEED_COLORS["smoke"])
                                       for s in seeds_disp], edgecolor="black")
    axes[1].axhline(0.01, color="red", linestyle=":", linewidth=1.0,
                    label="MI collapse threshold = 0.01")
    axes[1].set_xticks(x); axes[1].set_xticklabels(seeds_disp)
    axes[1].set_ylabel("Mutual information I(τ; z)")
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set_title("MMICRL mutual information per seed (log scale, threshold marked)",
                      loc="left", fontweight="bold")
    # Add headroom so the seed_2 bar tip + its value label do not collide
    # with the legend or the panel title.  Then place the legend in the
    # upper-LEFT (away from the seed_2 spike at x=2).
    if MIs and any(not np.isnan(mi) and mi > 0 for mi in MIs):
        max_mi = max(mi for mi in MIs if not np.isnan(mi))
        axes[1].set_ylim(top=max_mi * 4.0)
    axes[1].legend(loc="upper left", fontsize=8, frameon=True,
                   framealpha=0.95, edgecolor="gray")
    for xi, mi in zip(x, MIs):
        if not np.isnan(mi):
            axes[1].text(xi, mi, f" {mi:.2g}" if mi > 0 else " 0", ha="center",
                         va="bottom", fontsize=8)
    fig.suptitle("Appendix F  MMICRL pretrain on Path G profiles (continuous-mode runs use the same shared profiles)",
                 fontsize=11, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_04_mmicrl_pretrain_grid.png", bbox_inches="tight")
    plt.close(fig)


def fig_diagnostics(csvs):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # safety_rate
    for sid in [0, 1, 2]:
        if sid not in csvs: continue
        ep = csvs[sid].get("episode"); sr = csvs[sid].get("safety_rate")
        if ep is None or sr is None: continue
        axes[0].plot(ep, sr, color=SEED_COLORS[sid], label=f"seed_{sid}",
                     linewidth=1.4, alpha=0.9)
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Safety rate per episode")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Safety rate", loc="left", fontweight="bold")
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    axes[0].grid(alpha=0.3)

    # peak_fatigue
    for sid in [0, 1, 2]:
        if sid not in csvs: continue
        ep = csvs[sid].get("episode"); pf = csvs[sid].get("peak_fatigue")
        if ep is None or pf is None: continue
        axes[1].plot(ep, pf, color=SEED_COLORS[sid], label=f"seed_{sid}",
                     linewidth=1.4, alpha=0.9)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Peak fatigue (per-episode max)")
    axes[1].set_title("Peak muscle fatigue", loc="left", fontweight="bold")
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    axes[1].grid(alpha=0.3)

    # ECBF interventions cumulative
    for sid in [0, 1, 2]:
        if sid not in csvs: continue
        ep = csvs[sid].get("episode"); ec = csvs[sid].get("ecbf_interventions")
        if ep is None or ec is None: continue
        axes[2].plot(ep, np.nancumsum(ec), color=SEED_COLORS[sid], label=f"seed_{sid}",
                     linewidth=1.4, alpha=0.9)
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Cumulative ECBF interventions")
    axes[2].set_title("ECBF safety filter activity", loc="left", fontweight="bold")
    axes[2].legend(loc="best", fontsize=8, frameon=False)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Appendix F  diagnostics panel (3 seeds × 50K, continuous mode)",
                 fontsize=11, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_05_diagnostics_panel.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quant_path = OUT_DIR / "_quant_appendix_f.json"
    if not quant_path.exists():
        print(f"Missing {quant_path}; run analyze_appendix_f.py first.")
        sys.exit(1)
    quant = json.load(open(quant_path))
    csvs = load_csvs()
    print(f"Loaded {len(csvs)} CSVs ({sorted(csvs.keys(), key=str)})", flush=True)
    fig_log_std(csvs); print("  fig_01_log_std_evolution.png", flush=True)
    fig_allocator(csvs); print("  fig_02_allocator_activity.png", flush=True)
    fig_reward_trajectory(csvs); print("  fig_03_reward_trajectory.png", flush=True)
    fig_mmicrl_grid(quant); print("  fig_04_mmicrl_pretrain_grid.png", flush=True)
    fig_diagnostics(csvs); print("  fig_05_diagnostics_panel.png", flush=True)
    print("Stage 2 (Appendix F) complete.", flush=True)


if __name__ == "__main__":
    main()
