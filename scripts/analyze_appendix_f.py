"""EXP4 Stage 1 -- Appendix F probe analysis.

Reads `Results Appendix F/` (3 production seeds + 1 smoke) and produces:
  _quant_appendix_f.json   per-seed summary numbers (R, log_std, allocator, NaN, MMICRL)
  _metrics_table.csv        flat per-seed table for spreadsheet review
  _provenance.txt           commit + hardware origin

The probe is short (50K env-steps, 105 episodes per seed) so cross-seed bootstrap CIs
are NOT meaningful (n=3 production seeds). Reporting per-seed numbers + diagnostics is
sufficient for the Appendix F sanity-check claim:

  (1) continuous-mode pipeline runs end-to-end without NaN/Inf
  (2) Gaussian actor explores (log_std evolves in [-5, 2] without collapse)
  (3) NSWF allocator is wired (allocator_calls > 0 every episode)
  (4) MMICRL pretrain runs and produces type-discovery output
  (5) reward improves over training (negative direction is fine on 50K)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APPENDIX_F = ROOT / "Results Appendix F"
OUT_DIR = ROOT / "Results 4" / "Result Appendix F Analysis"


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


def per_seed_block(seed_dir: Path) -> Dict:
    """Extract every meaningful number from one probe seed directory."""
    log_path = seed_dir / "training_log.csv"
    sum_path = seed_dir / "summary.json"
    cfg_path = seed_dir / "config.yaml"
    mmi_path = seed_dir / "mmicrl" / "mmicrl_results.json"

    out: Dict = {"seed_dir": str(seed_dir.relative_to(ROOT))}
    if sum_path.exists():
        out["summary"] = json.load(open(sum_path))
    if mmi_path.exists():
        out["mmicrl"] = json.load(open(mmi_path))
    if not log_path.exists():
        out["training_log_missing"] = True
        return out

    cols = read_csv_dict(log_path)
    n_eps = len(cols.get("episode", []))
    out["n_episodes"] = n_eps

    # Reward trajectory
    R = cols.get("cumulative_reward")
    if R is not None and R.size:
        out["reward_first_ep"] = float(R[0])
        out["reward_last_ep"] = float(R[-1])
        out["reward_best"] = float(np.nanmax(R))
        out["reward_worst"] = float(np.nanmin(R))
        out["reward_mean"] = float(np.nanmean(R))
        out["reward_last10_mean"] = float(np.nanmean(R[-10:]))

    # Continuous-mode diagnostics
    log_std = cols.get("gaussian_log_std_mean")
    if log_std is not None and log_std.size:
        valid = log_std[~np.isnan(log_std)]
        if valid.size:
            out["log_std_first"] = float(valid[0])
            out["log_std_last"] = float(valid[-1])
            out["log_std_min"] = float(np.min(valid))
            out["log_std_max"] = float(np.max(valid))
            out["log_std_mean"] = float(np.mean(valid))
            # Within valid clamp [-5, 2]?
            out["log_std_within_clamp"] = bool(valid.min() >= -5.0 and valid.max() <= 2.0)
            # Saturated near boundary?
            out["log_std_saturated_low"] = bool(valid.min() < -4.5)
            out["log_std_saturated_high"] = bool(valid.max() > 1.5)
            out["log_std_n_valid"] = int(valid.size)
            out["log_std_trajectory"] = valid.tolist()

    nan_inf = cols.get("nan_inf_events")
    if nan_inf is not None and nan_inf.size:
        valid = nan_inf[~np.isnan(nan_inf)]
        out["nan_inf_total"] = int(np.nansum(valid))
        out["nan_inf_any"] = bool(np.any(valid > 0))

    alloc = cols.get("allocator_calls")
    if alloc is not None and alloc.size:
        valid = alloc[~np.isnan(alloc)]
        out["allocator_calls_total"] = int(np.nansum(valid))
        out["allocator_calls_per_episode_mean"] = float(np.nanmean(valid))
        out["allocator_calls_per_episode_min"] = int(np.nanmin(valid))
        out["allocator_calls_per_episode_max"] = int(np.nanmax(valid))
        # Expected = max_steps / allocation_interval = 480 / 30 = 16
        out["allocator_calls_per_episode_expected"] = 16
        out["allocator_calls_match_expected"] = bool(np.all(valid[~np.isnan(valid)] == 16))

    # Safety + fatigue
    sr = cols.get("safety_rate")
    if sr is not None:
        valid = sr[~np.isnan(sr)]
        if valid.size:
            out["safety_rate_first"] = float(valid[0])
            out["safety_rate_last"] = float(valid[-1])
            out["safety_rate_mean"] = float(np.nanmean(valid))
    pf = cols.get("peak_fatigue")
    if pf is not None:
        valid = pf[~np.isnan(pf)]
        if valid.size:
            out["peak_fatigue_first"] = float(valid[0])
            out["peak_fatigue_last"] = float(valid[-1])
            out["peak_fatigue_max"] = float(np.nanmax(valid))
    ec = cols.get("ecbf_interventions")
    if ec is not None:
        out["ecbf_interventions_total"] = float(np.nansum(ec))

    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing Appendix F probe artifacts in: {APPENDIX_F}", flush=True)

    quant: Dict = {
        "is_continuous_mode": True,
        "experiment": "Appendix F continuous-mode probe",
        "production_runs": {},
        "smoke_run": None,
    }

    # Production: 3 seeds x 50K
    prod_dir = APPENDIX_F / "logs" / "continuous_probe"
    if prod_dir.exists():
        for sd in sorted(prod_dir.glob("seed_*")):
            sid = int(sd.name.split("_")[1])
            block = per_seed_block(sd)
            quant["production_runs"][sid] = block
            r = block.get("reward_last10_mean", float("nan"))
            ls = block.get("log_std_mean", float("nan"))
            mi = block.get("mmicrl", {}).get("mutual_information") if "mmicrl" in block else None
            print(f"  prod seed {sid}: R_last10={r:.0f}  "
                  f"log_std_mean={ls:.3f}  MMICRL_MI={mi}", flush=True)

    # Smoke: 1 seed x 5K (Gate 0)
    smoke_dir = APPENDIX_F / "logs" / "continuous_probe_smoke" / "seed_0"
    if smoke_dir.exists():
        quant["smoke_run"] = per_seed_block(smoke_dir)
        sb = quant["smoke_run"]
        print(f"  smoke seed_0: R_last={sb.get('reward_last_ep', float('nan')):.0f}  "
              f"n_eps={sb.get('n_episodes')}  nan_inf_any={sb.get('nan_inf_any')}",
              flush=True)

    # Save
    out_json = OUT_DIR / "_quant_appendix_f.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(quant, f, indent=2, default=str)
    print(f"  Wrote {out_json} ({out_json.stat().st_size} bytes)", flush=True)

    # Flat CSV
    out_csv = OUT_DIR / "_metrics_table.csv"
    fields = [
        "run_kind", "seed", "n_episodes",
        "reward_first_ep", "reward_last10_mean", "reward_best",
        "log_std_first", "log_std_last", "log_std_mean", "log_std_within_clamp",
        "nan_inf_total", "nan_inf_any",
        "allocator_calls_total", "allocator_calls_per_episode_mean",
        "allocator_calls_match_expected",
        "safety_rate_first", "safety_rate_last", "safety_rate_mean",
        "peak_fatigue_first", "peak_fatigue_last", "peak_fatigue_max",
        "ecbf_interventions_total",
        "mmicrl_K", "mmicrl_MI", "mmicrl_collapsed", "mmicrl_n_demos",
    ]
    rows = []
    for sid, block in quant["production_runs"].items():
        m = block.get("mmicrl", {})
        rows.append({
            "run_kind": "production",
            "seed": sid,
            "n_episodes": block.get("n_episodes"),
            "reward_first_ep": block.get("reward_first_ep"),
            "reward_last10_mean": block.get("reward_last10_mean"),
            "reward_best": block.get("reward_best"),
            "log_std_first": block.get("log_std_first"),
            "log_std_last": block.get("log_std_last"),
            "log_std_mean": block.get("log_std_mean"),
            "log_std_within_clamp": block.get("log_std_within_clamp"),
            "nan_inf_total": block.get("nan_inf_total"),
            "nan_inf_any": block.get("nan_inf_any"),
            "allocator_calls_total": block.get("allocator_calls_total"),
            "allocator_calls_per_episode_mean": block.get("allocator_calls_per_episode_mean"),
            "allocator_calls_match_expected": block.get("allocator_calls_match_expected"),
            "safety_rate_first": block.get("safety_rate_first"),
            "safety_rate_last": block.get("safety_rate_last"),
            "safety_rate_mean": block.get("safety_rate_mean"),
            "peak_fatigue_first": block.get("peak_fatigue_first"),
            "peak_fatigue_last": block.get("peak_fatigue_last"),
            "peak_fatigue_max": block.get("peak_fatigue_max"),
            "ecbf_interventions_total": block.get("ecbf_interventions_total"),
            "mmicrl_K": m.get("n_types_discovered"),
            "mmicrl_MI": m.get("mutual_information"),
            "mmicrl_collapsed": m.get("mi_collapsed"),
            "mmicrl_n_demos": m.get("n_demonstrations"),
        })
    if quant["smoke_run"]:
        sb = quant["smoke_run"]; m = sb.get("mmicrl", {})
        rows.append({
            "run_kind": "smoke",
            "seed": 0,
            "n_episodes": sb.get("n_episodes"),
            "reward_first_ep": sb.get("reward_first_ep"),
            "reward_last10_mean": sb.get("reward_last10_mean"),
            "reward_best": sb.get("reward_best"),
            "log_std_first": sb.get("log_std_first"),
            "log_std_last": sb.get("log_std_last"),
            "log_std_mean": sb.get("log_std_mean"),
            "log_std_within_clamp": sb.get("log_std_within_clamp"),
            "nan_inf_total": sb.get("nan_inf_total"),
            "nan_inf_any": sb.get("nan_inf_any"),
            "allocator_calls_total": sb.get("allocator_calls_total"),
            "allocator_calls_per_episode_mean": sb.get("allocator_calls_per_episode_mean"),
            "allocator_calls_match_expected": sb.get("allocator_calls_match_expected"),
            "safety_rate_first": sb.get("safety_rate_first"),
            "safety_rate_last": sb.get("safety_rate_last"),
            "safety_rate_mean": sb.get("safety_rate_mean"),
            "peak_fatigue_first": sb.get("peak_fatigue_first"),
            "peak_fatigue_last": sb.get("peak_fatigue_last"),
            "peak_fatigue_max": sb.get("peak_fatigue_max"),
            "ecbf_interventions_total": sb.get("ecbf_interventions_total"),
            "mmicrl_K": m.get("n_types_discovered"),
            "mmicrl_MI": m.get("mutual_information"),
            "mmicrl_collapsed": m.get("mi_collapsed"),
            "mmicrl_n_demos": m.get("n_demonstrations"),
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  Wrote {out_csv} ({out_csv.stat().st_size} bytes)", flush=True)
    print("Stage 1 (Appendix F) complete.", flush=True)


if __name__ == "__main__":
    main()
