"""
visualize_results_0.py — Stage 2 of Results 4 / Result 0 Analysis.

Reads "_quant_analysis.json" produced by analyze_results_0.py and emits
seven 300-dpi PNGs under "Results 4/Result 0 Analysis/".

Plots:
  fig_01_test_suite_breakdown.png   - horizontal bars per test file (pass/skip)
  fig_02_smoke_reward_comparison.png - 9-config mean reward/step bar chart
  fig_03_smoke_peakmf_comparison.png - 9-config mean and max peak_MF + theta_max line
  fig_04_mmicrl_kselection.png       - heldout_nll vs K (log y), winner + project-K marked
  fig_05_ecbf_state_sweep.png        - C_nominal heatmap on MF x MA grid
  fig_06_three_cc_r_smoke_curves.png - peak_MF trajectories for HCMARL vs ablation_no_reperfusion
  fig_07_constants_provenance.png    - constants by module (PRIMARY vs DESIGN stacked)

Stage 2 wall clock: ~5 seconds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Results 4" / "Result 0 Analysis"
RESULTS_0 = ROOT / "Results 0"
DPI = 300

quant = json.loads((OUT_DIR / "_quant_analysis.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# fig_01 — test suite breakdown
# ---------------------------------------------------------------------------

def fig_test_suite() -> None:
    files = quant["test_suite"]["per_file"]
    files = sorted(files, key=lambda f: f["passed"], reverse=False)
    names = [f["file"].replace("test_", "").replace(".py", "") for f in files]
    passed = [f["passed"] for f in files]
    skipped = [f["skipped"] for f in files]

    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(names))
    ax.barh(y, passed, color="#2e7d32", label="passed", height=0.7)
    ax.barh(y, skipped, left=passed, color="#fbc02d", label="skipped", height=0.7)
    for i, (p, s) in enumerate(zip(passed, skipped)):
        label = f"{p}" if s == 0 else f"{p} +{s} skip"
        ax.text(p + s + 0.5, i, label, va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("number of tests")
    ax.set_title(
        f"Experiment 0 — full pytest suite\n"
        f"{quant['test_suite']['total_passed']} passed / "
        f"{quant['test_suite']['total_skipped']} skipped / "
        f"{quant['test_suite']['total_failed']} failed across "
        f"{quant['test_suite']['n_files']} files "
        f"({quant['test_suite']['wall_seconds']:.0f}s wall)",
        fontsize=11,
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_01_test_suite_breakdown.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig_02 — smoke reward bar chart
# ---------------------------------------------------------------------------

def _smoke_label(rec: dict) -> str:
    av = rec["ablation_variant"]
    if av in (None, "(headline)"):
        return rec["method"]
    return av.replace("ablation_", "")


def fig_smoke_reward() -> None:
    recs = quant["smoke_runs"]
    labels = [_smoke_label(r) for r in recs]
    rewards = [r["smoke_metrics"]["mean_reward_per_step"] for r in recs]
    cats = [r["category"] for r in recs]
    colors = ["#1565c0" if c == "method" else "#c62828" for c in cats]

    order = sorted(range(len(rewards)), key=lambda i: rewards[i])
    labels = [labels[i] for i in order]
    rewards = [rewards[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(np.arange(len(labels)), rewards, color=colors, edgecolor="black", linewidth=0.5)
    for i, (b, r) in enumerate(zip(bars, rewards)):
        ax.text(r - 0.3, i, f"{r:.3f}", va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean reward / step  (100-step uniform-random smoke)")
    ax.set_title(
        "Experiment 0 — 100-step smoke forward-pass per config\n"
        "blue = headline method, red = ablation rung; lower = harder env"
    )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    # Legend in upper-left empty space (top bars are SHORT, only reach
    # ~-7; upper-left is far from the bottom bar's tail at -17).
    ax.legend(handles=[
        Patch(color="#1565c0", label="headline method (4)"),
        Patch(color="#c62828", label="ablation rung (5)"),
    ], loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_02_smoke_reward_comparison.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig_03 — peak_MF comparison
# ---------------------------------------------------------------------------

def fig_smoke_peakmf() -> None:
    recs = quant["smoke_runs"]
    labels = [_smoke_label(r) for r in recs]
    mean_mf = [r["smoke_metrics"]["mean_peak_MF"] for r in recs]
    max_mf = [r["smoke_metrics"]["max_peak_MF"] for r in recs]
    cats = [r["category"] for r in recs]

    order = sorted(range(len(mean_mf)), key=lambda i: mean_mf[i])
    labels = [labels[i] for i in order]
    mean_mf = [mean_mf[i] for i in order]
    max_mf = [max_mf[i] for i in order]
    cats = [cats[i] for i in order]

    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width/2, mean_mf, width, label="mean peak_MF (100 steps)", color="#1565c0")
    ax.bar(x + width/2, max_mf, width, label="max peak_MF (100 steps)", color="#ef6c00")

    # shoulder theta_max ceiling at 0.7
    ax.axhline(0.7, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="shoulder theta_max = 0.7")
    ax.axhline(0.45, color="purple", linestyle=":", linewidth=1.0, alpha=0.5,
               label="grip/elbow theta_max = 0.45")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("peak fatigue MF (global, across 6 muscles x 6 workers)")
    ax.set_title(
        "Experiment 0 — fatigue ceiling check on uniform-random smoke\n"
        "ablation_no_reperfusion drives MF higher (r=1 instead of r=15/30)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max_mf) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_03_smoke_peakmf_comparison.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig_04 — MMICRL K-selection
# ---------------------------------------------------------------------------

def fig_mmicrl_k() -> None:
    sel = quant["mmicrl_k_selection"]
    ks = sorted(sel["k_selection_values"].keys())
    vals = [sel["k_selection_values"][k] for k in ks]
    project_K = sel["project_chosen_K"]
    snapshot_K = sel["k_selection_winner"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(ks, vals, color="#546e7a", edgecolor="black")
    ax.set_yscale("log")
    ax.set_xlabel("K (number of latent worker types)")
    ax.set_ylabel("heldout NLL  (log scale, lower = better)")
    ax.set_title(
        f"MMICRL K-selection on Path G (n=102 demonstrations)\n"
        f"snapshot winner K={snapshot_K} (lowest NLL)  |  "
        f"project narrative uses K={project_K}"
    )
    for k, v in zip(ks, vals):
        ax.text(k, v * 1.15, f"{v:,.1f}", ha="center", fontsize=9)

    # mark winner and project-K
    ax.axvline(snapshot_K, color="green", linestyle="--", alpha=0.6,
               label=f"snapshot winner K={snapshot_K}")
    ax.axvline(project_K, color="red", linestyle=":", linewidth=2,
               label=f"project K={project_K}")
    ax.set_xticks(ks)
    # Place legend BELOW the axes so it cannot cover the K=1 bar
    # (which reaches ~91% of axes height; an in-plot legend at any
    # corner partially occludes either the bar top label or a vline).
    ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center", ncol=2,
              frameon=True, framealpha=0.95, edgecolor="gray", fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")

    # MI-collapse note: place ABOVE the short K=2..K=4 bars (in the empty
    # log-space between ~5e3 bar tips and the y-max ~1e7).  Anchored at
    # axes (0.55, 0.55), well clear of every bar and both vertical markers.
    note = (f"MI(tau; z) = {sel['mutual_information']:.3f}  ->  "
            f"MI-collapse guard fires; rescale-to-floor activates.\n"
            f"K is downstream of the floor — narrative independent of "
            f"snapshot K.")
    ax.text(0.55, 0.55, note, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, style="italic",
            bbox=dict(boxstyle="round", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.95))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_04_mmicrl_kselection.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig_05 — ECBF state sweep
# ---------------------------------------------------------------------------

def fig_ecbf_sweep() -> None:
    rows = []
    with (RESULTS_0 / "Result ecbf_filter.py.csv").open() as f:
        rows = list(csv.DictReader(f))
    mf = np.array([float(r["MF"]) for r in rows])
    ma = np.array([float(r["MA"]) for r in rows])
    cnom = np.array([float(r["C_nominal"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8.5, 6))
    sc = ax.scatter(mf, ma, c=cnom, cmap="RdYlGn_r", s=120, edgecolors="black",
                    linewidths=0.4, vmin=0, vmax=cnom.max())
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("C_nominal (PPO commanded effort, before ECBF filter)")
    ax.set_xlabel("MF — fatigued fraction")
    ax.set_ylabel("MA — active fraction")
    ax.set_title(
        "Experiment 0 — ECBF QP feasibility sweep (TL=0.55)\n"
        f"{quant['ecbf_state_sweep']['n_states_tested']} states tested, "
        f"all infeasible at this static load (slack-augmented QP would absorb)"
    )
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.6)
    ax.grid(alpha=0.3)
    # Annotate the dashed line in the inter-row gap (MA=0.55, between
    # the MA=0.5 and MA=0.6 marker rows).  Previously placed near the
    # top, where the box overlapped the rightmost MA=0.6 markers.
    ax.text(0.715, 0.55, "shoulder θ_max = 0.7",
            color="#a52a00", fontsize=9, ha="left", va="center", rotation=0,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#a52a00", alpha=0.95))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_05_ecbf_state_sweep.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig_06 — 3CC-r smoke curves: HCMARL vs no_reperfusion (the only diverging ablation)
# ---------------------------------------------------------------------------

def fig_smoke_curves() -> None:
    def _read_csv(p: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with p.open() as f:
            rd = list(csv.DictReader(f))
        steps = np.array([int(r["step"]) for r in rd])
        peak = np.array([float(r["peak_MF_global"]) for r in rd])
        mean = np.array([float(r["mean_MF_global"]) for r in rd])
        return steps, peak, mean

    steps_h, pk_h, mn_h = _read_csv(RESULTS_0 / "Result hcmarl_config.yaml.csv")
    steps_a, pk_a, mn_a = _read_csv(
        RESULTS_0 / "Result ablation_no_reperfusion_config.yaml.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(steps_h, pk_h, color="#1565c0", linewidth=2, label="hcmarl (r=15/30)")
    ax.plot(steps_a, pk_a, color="#c62828", linewidth=2, label="no_reperfusion (r=1)")
    ax.fill_between(steps_h, mn_h, pk_h, color="#1565c0", alpha=0.15)
    ax.fill_between(steps_a, mn_a, pk_a, color="#c62828", alpha=0.15)
    ax.axhline(0.7, color="red", linestyle="--", alpha=0.6, label="shoulder theta_max")
    ax.set_xlabel("env step")
    ax.set_ylabel("peak_MF (global)")
    ax.set_title("Smoke fatigue trajectory — HCMARL vs no_reperfusion")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    delta = pk_a - pk_h
    ax.fill_between(steps_h, 0, delta, color="#c62828", alpha=0.4)
    ax.plot(steps_h, delta, color="#c62828", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("env step")
    ax.set_ylabel("peak_MF (no_reperfusion)  -  peak_MF (hcmarl)")
    ax.set_title("Excess fatigue from removing reperfusion")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_06_three_cc_r_smoke_curves.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig_07 — constants provenance
# ---------------------------------------------------------------------------

def fig_constants() -> None:
    rows = quant["constants_ledger"]["rows"]
    by_mod_primary: dict[str, int] = {}
    by_mod_design: dict[str, int] = {}
    for r in rows:
        cls = "DESIGN" if "DESIGN" in r["source"] else "PRIMARY"
        if cls == "PRIMARY":
            by_mod_primary[r["module"]] = by_mod_primary.get(r["module"], 0) + 1
        else:
            by_mod_design[r["module"]] = by_mod_design.get(r["module"], 0) + 1

    modules = sorted(set(list(by_mod_primary) + list(by_mod_design)))
    primary = [by_mod_primary.get(m, 0) for m in modules]
    design = [by_mod_design.get(m, 0) for m in modules]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(modules))
    ax.bar(x, primary, color="#2e7d32", label="primary-source (PDF-verified)", edgecolor="black")
    ax.bar(x, design, bottom=primary, color="#fbc02d", label="DESIGN choice", edgecolor="black")
    for i, (p, d) in enumerate(zip(primary, design)):
        if p:
            ax.text(i, p / 2, str(p), ha="center", va="center", fontweight="bold")
        if d:
            ax.text(i, p + d / 2, str(d), ha="center", va="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modules, rotation=15, ha="right")
    ax.set_ylabel("number of constants")
    ax.set_title(
        f"Experiment 0 — constants provenance ledger\n"
        f"{quant['constants_ledger']['n_primary_source']} primary-source / "
        f"{quant['constants_ledger']['n_design_choice']} design-choice  "
        f"(total {quant['constants_ledger']['total_constants']})"
    )
    # Tallest bar (three_cc_r) is at the rightmost x-position — upper-right
    # legend would cover it.  Move to upper-left where the short bars
    # (ecbf_filter at 5, niosh_calibration at 2, nswf_allocator at 1) leave
    # plenty of empty headroom.
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_07_constants_provenance.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[visualize_results_0] writing PNGs to {OUT_DIR}")
    for name, fn in [
        ("fig_01_test_suite_breakdown",       fig_test_suite),
        ("fig_02_smoke_reward_comparison",    fig_smoke_reward),
        ("fig_03_smoke_peakmf_comparison",    fig_smoke_peakmf),
        ("fig_04_mmicrl_kselection",          fig_mmicrl_k),
        ("fig_05_ecbf_state_sweep",           fig_ecbf_sweep),
        ("fig_06_three_cc_r_smoke_curves",    fig_smoke_curves),
        ("fig_07_constants_provenance",       fig_constants),
    ]:
        fn()
        print(f"  wrote {name}.png")
    print("[visualize_results_0] OK")


if __name__ == "__main__":
    main()
