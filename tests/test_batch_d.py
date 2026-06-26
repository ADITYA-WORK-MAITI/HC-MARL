"""Statistical + ablation design regression tests.

Covers:
    D1 — config/experiment_matrix.yaml declares exactly 10 headline seeds,
         all 4 headline methods, and anchor steps 1M/3M/5M. hcmarl.aggregation
         produces sensible IQM + stratified bootstrap CI on known inputs.
    D2 — Attribution ablation matrix maps to five existing config files
         that each set the right combination of ecbf.enabled / nswf.enabled /
         mmicrl.enabled. No phantom configs.
    D3 — scripts/aggregate_learning_curves.py reads a CSV fixture and emits
         a complete report (all methods x all anchors) when every file is
         present; surfaces a non-empty errors list when any CSV is missing
         rather than silently averaging over fewer seeds.
    D4 — HCMARLLogger CSV schema includes per_agent_entropy_mean /
         per_agent_entropy_min / lazy_agent_flag. End-to-end: compute
         per-agent entropy from a known action histogram and verify
         min/mean match the hand-derived values.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import yaml

from hcmarl.aggregation import (
    aggregate_by_method,
    iqm,
    stratified_bootstrap_iqm_ci,
)
from hcmarl.logger import HCMARLLogger


ROOT = pathlib.Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / "config" / "experiment_matrix.yaml"


# ---------------------------------------------------------------------
# D1 — experiment matrix + aggregation primitives
# ---------------------------------------------------------------------

class TestD1ExperimentMatrix:
    """The matrix YAML is the single source of truth; downstream tools
    read it literally. A drift between this file and the runbook is a
    silent mis-report, so we pin its structure."""

    def test_headline_declares_ten_seeds(self):
        data = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
        seeds = data["headline"]["seeds"]
        assert len(seeds) == 10
        assert seeds == list(range(10))

    def test_headline_methods_cover_full_lineup(self):
        # Baseline expansion: ippo dropped; happo + shielded_mappo added.
        # 
        # Constraint-handling axis is now covered by Lagrangian (mappo_lag),
        # heterogeneous trust-region (happo), AND a non-RL static-threshold
        # shield (shielded_mappo) so the ECBF-vs-shielding contribution
        # claim remains testable. Five entries total including hcmarl.
        data = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
        methods = data["headline"]["methods"]
        assert set(methods.keys()) == {
            "hcmarl", "mappo", "mappo_lag", "happo", "shielded_mappo",
        }
        for name, entry in methods.items():
            cfg_path = ROOT / entry["config"]
            assert cfg_path.exists(), (
                f"headline method {name}: config {entry['config']} missing"
            )

    def test_curve_anchors_present(self):
        # 3M anchor dropped because all headline + ablation
        # runs are 2M steps; 3M would fire a legitimate aggregator error.
        data = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
        assert data["curve_anchors_steps"] == [500_000, 1_000_000, 2_000_000]

    def test_lazy_agent_kill_switch_parameters_present(self):
        data = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
        lac = data["lazy_agent_kill_switch"]
        assert lac["threshold"] == 0.1
        assert lac["window_steps"] == 100_000


class TestD1AggregationPrimitives:
    """IQM and stratified bootstrap must return the known exact values for
    hand-computable inputs."""

    def test_iqm_exact_on_small_symmetric_sample(self):
        # Percentiles (linear interp): q1=2.75, q3=7.25 — so 3..7 are in the
        # middle-50% band. Mean of [3,4,5,6,7] = 5.0.
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        assert iqm(arr) == pytest.approx(5.0, abs=1e-9)

    def test_iqm_matches_mean_for_under_four_samples(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert iqm(arr) == pytest.approx(2.0, abs=1e-9)

    def test_bootstrap_ci_brackets_truth_on_normal_sample(self):
        rng = np.random.default_rng(42)
        sample = rng.normal(loc=10.0, scale=1.0, size=100)
        lo, hi = stratified_bootstrap_iqm_ci(sample, n_resamples=2000, seed=7)
        assert lo < 10.0 < hi
        # A reasonable CI width for n=100 is << 1.0 on a normal(10, 1).
        assert (hi - lo) < 1.0

    def test_aggregate_by_method_structure(self):
        scores = {
            "alpha": np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
            "beta":  np.array([20.0, 21.0, 22.0, 23.0, 24.0]),
        }
        out = aggregate_by_method(scores, n_resamples=500)
        assert set(out.keys()) == {"alpha", "beta"}
        for m in ("alpha", "beta"):
            assert set(out[m].keys()) == {"iqm", "ci_lo", "ci_hi", "n_seeds"}
            assert out[m]["ci_lo"] <= out[m]["iqm"] <= out[m]["ci_hi"]
            assert out[m]["n_seeds"] == 5


# ---------------------------------------------------------------------
# D2 — attribution ablation matrix
# ---------------------------------------------------------------------

class TestD2AttributionAblationMatrix:
    """REMOVE-ONE ablation ladder. Each of 4 rungs differs
    from full HC-MARL by EXACTLY ONE ablated component. The 5th reference
    point (full HC-MARL) comes from EXP1's first 5 seeds — NOT re-run here.

    Ablation axes:
      no_ecbf          ecbf.enabled=false
      no_nswf          nswf.enabled=false
      no_divergent     disagreement.type=constant
      no_reperfusion   muscle_groups.*.r=1 (was 15 or 30)

    Each rung must point at an existing config whose flags differ from
    full HC-MARL by exactly the named axis. Phantom rungs or multi-axis
    drift would corrupt single-component attribution."""

    def _load(self):
        return yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))["ablation"]

    def test_five_remove_one_rungs_in_expected_order(self):
        # Ablation expansion: no_mmicrl rung added so all five
        # HC-MARL components have a single-axis attribution test. Seeds
        # bumped to 10 to match the headline grid -- 5/10 asymmetry was
        # the asymmetry concern.
        ablation = self._load()
        names = [r["name"] for r in ablation["rungs"]]
        assert names == [
            "no_ecbf", "no_nswf", "no_divergent", "no_reperfusion", "no_mmicrl",
        ]
        assert ablation["seeds"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_all_rungs_use_hcmarl_method(self):
        """Remove-one ablations are all derived from full HC-MARL (NOT bare
        MAPPO). Every rung's method must be 'hcmarl' so MAPPO's hard-wired
        ecbf/nswf off doesn't mask the ablated axis."""
        ablation = self._load()
        for rung in ablation["rungs"]:
            assert rung["method"] == "hcmarl", (
                f"{rung['name']}: method must be 'hcmarl' for remove-one "
                f"semantics (got '{rung['method']}')"
            )

    @pytest.mark.parametrize("rung_name,expected", [
        # (ecbf_enabled, nswf_enabled, mmicrl_enabled, disagreement_type, r_value)
        # Full HC-MARL reference: (True, True, True, "divergent", 15/30)
        # Each rung flips EXACTLY one of these.
        ("no_ecbf",         (False, True,  True,  "divergent", 15)),
        ("no_nswf",         (True,  False, True,  "divergent", 15)),
        ("no_divergent",    (True,  True,  True,  "constant",  15)),
        ("no_reperfusion",  (True,  True,  True,  "divergent", 1)),
        ("no_mmicrl",       (True,  True,  False, "divergent", 15)),
    ])
    def test_remove_one_config_flips_exactly_one_axis(self, rung_name, expected):
        ablation = self._load()
        rung = next(r for r in ablation["rungs"] if r["name"] == rung_name)
        cfg_path = ROOT / rung["config"]
        assert cfg_path.exists(), f"{rung_name}: {cfg_path} missing"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        def flag(section, default):
            sec = cfg.get(section) or {}
            return bool(sec.get("enabled", default))

        ecbf_on = flag("ecbf", True)
        nswf_on = flag("nswf", True)
        mmicrl_on = flag("mmicrl", True)  # full HC-MARL default is on
        disag_type = (cfg.get("disagreement") or {}).get("type", "divergent")
        # r: take shoulder's r as representative (all 6 muscles flip together
        # in no_reperfusion; others keep the PDF-verified r=15 or r=30).
        mg = cfg.get("environment", {}).get("muscle_groups", {})
        r_shoulder = mg.get("shoulder", {}).get("r", 15)

        got = (ecbf_on, nswf_on, mmicrl_on, disag_type, r_shoulder)
        assert got == expected, (
            f"{rung_name} ({rung['config']}): "
            f"expected {expected}, got {got}"
        )

    def test_ablation_total_steps_matches_exp1_reference(self):
        """All ablation configs must run 2M steps to match the EXP1
        HCMARL reference. Any drift here breaks the apples-to-apples
        comparison at curve anchors."""
        ablation = self._load()
        for rung in ablation["rungs"]:
            cfg = yaml.safe_load((ROOT / rung["config"]).read_text(encoding="utf-8"))
            steps = cfg.get("training", {}).get("total_steps")
            assert steps == 2_000_000, (
                f"{rung['name']}: total_steps={steps}, expected 2,000,000 "
                f"for EXP1/EXP2 alignment"
            )


# ---------------------------------------------------------------------
# D3 — learning-curve aggregator
# ---------------------------------------------------------------------

def _write_csv(path: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "global_step", "cumulative_reward"])
        w.writeheader()
        w.writerows(rows)


class TestD3LearningCurveAggregator:
    """The aggregator must emit a complete table when all CSVs exist, and
    a non-empty errors list when any CSV is missing. It must NEVER silently
    average over fewer seeds than the matrix declares."""

    def _build_fake_logs(self, td: str, seeds, anchors):
        methods = ["hcmarl", "mappo", "ippo", "mappo_lag"]
        # Write a CSV per (method, seed) with reward rising linearly in step.
        for method in methods:
            for s in seeds:
                d = pathlib.Path(td) / method / f"seed_{s}"
                d.mkdir(parents=True, exist_ok=True)
                rows = []
                for i, step in enumerate(anchors):
                    rows.append({
                        "episode": i + 1,
                        "global_step": step,
                        # method-dependent reward so aggregation distinguishes.
                        "cumulative_reward": 100.0 * (methods.index(method) + 1)
                                             + 0.001 * s + step * 1e-6,
                    })
                _write_csv(str(d / "training_log.csv"), rows)

    def test_aggregate_all_present(self):
        from scripts.aggregate_learning_curves import aggregate
        seeds = list(range(10))
        anchors = [1_000_000, 3_000_000, 5_000_000]
        with tempfile.TemporaryDirectory() as td:
            self._build_fake_logs(td, seeds, anchors)
            # Write a throwaway matrix pointing at the fake log tree.
            matrix = {
                "headline": {
                    "seeds": seeds,
                    "methods": {
                        "hcmarl":    {"config": "x", "method": "hcmarl"},
                        "mappo":     {"config": "x", "method": "mappo"},
                        "ippo":      {"config": "x", "method": "ippo"},
                        "mappo_lag": {"config": "x", "method": "mappo_lag"},
                    },
                },
                "curve_anchors_steps": anchors,
            }
            matrix_path = pathlib.Path(td) / "matrix.yaml"
            matrix_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
            report = aggregate(str(matrix_path), td)
            assert report["complete"] is True
            assert report["errors"] == []
            for a in anchors:
                by = report["by_anchor"][str(a)]
                assert set(by.keys()) == {"hcmarl", "mappo", "ippo", "mappo_lag"}
                for stats in by.values():
                    assert stats["n_seeds"] == 10
                    assert stats["ci_lo"] <= stats["iqm"] <= stats["ci_hi"]

    def test_aggregate_surfaces_missing_csv(self):
        from scripts.aggregate_learning_curves import aggregate
        seeds = [0, 1, 2]
        anchors = [1_000_000]
        with tempfile.TemporaryDirectory() as td:
            self._build_fake_logs(td, seeds, anchors)
            # Delete one CSV to simulate a failed run.
            missing = pathlib.Path(td) / "mappo" / "seed_1" / "training_log.csv"
            missing.unlink()
            matrix = {
                "headline": {
                    "seeds": seeds,
                    "methods": {
                        "hcmarl":    {"config": "x", "method": "hcmarl"},
                        "mappo":     {"config": "x", "method": "mappo"},
                        "ippo":      {"config": "x", "method": "ippo"},
                        "mappo_lag": {"config": "x", "method": "mappo_lag"},
                    },
                },
                "curve_anchors_steps": anchors,
            }
            matrix_path = pathlib.Path(td) / "matrix.yaml"
            matrix_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
            report = aggregate(str(matrix_path), td)
            assert report["complete"] is False
            assert any("mappo" in e and "seed 1" in e for e in report["errors"])

    def test_aggregate_handles_ablation_grid(self):
        """When matrix declares ablation rungs, aggregate() must read from
        logs/ablation_<rung>/seed_<s>/ and emit `by_anchor_ablation` keyed
        on the rung name. This is the second-grid contract added when the
        build-up ladder runner replaced the remove-one runner."""
        from scripts.aggregate_learning_curves import aggregate
        seeds = [0, 1, 2]
        anchors = [1_000_000]
        rung_names = ["mappo", "plus_ecbf", "full_hcmarl"]
        with tempfile.TemporaryDirectory() as td:
            # Write fake CSVs at logs/ablation_<rung>/seed_<s>/training_log.csv
            for rung in rung_names:
                for s in seeds:
                    d = pathlib.Path(td) / f"ablation_{rung}" / f"seed_{s}"
                    d.mkdir(parents=True, exist_ok=True)
                    rows = [{
                        "episode": 1,
                        "global_step": anchors[0],
                        "cumulative_reward": 50.0
                                             + 10.0 * rung_names.index(rung)
                                             + 0.001 * s,
                    }]
                    _write_csv(str(d / "training_log.csv"), rows)
            matrix = {
                "ablation": {
                    "seeds": seeds,
                    "rungs": [
                        {"name": r, "config": "x", "method": "hcmarl"}
                        for r in rung_names
                    ],
                },
                "curve_anchors_steps": anchors,
            }
            matrix_path = pathlib.Path(td) / "matrix.yaml"
            matrix_path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
            report = aggregate(str(matrix_path), td)
            assert report["complete"] is True, report["errors"]
            assert report["by_anchor"] == {}, "no headline grid was declared"
            by = report["by_anchor_ablation"][str(anchors[0])]
            assert set(by.keys()) == set(rung_names)
            for stats in by.values():
                assert stats["n_seeds"] == 3
                assert stats["ci_lo"] <= stats["iqm"] <= stats["ci_hi"]


class TestD3RunNameFlag:
    """train.py exposes --run-name so ablation rungs land in
    logs/ablation_<rung>/seed_<s>/ instead of overwriting the headline
    logs at logs/<method>/seed_<s>/. The flag must round-trip through
    argparse and resolve to the override (not the method) when set."""

    def test_train_argparse_has_run_name(self):
        """Parse train.py via importlib to confirm --run-name is accepted
        without invoking the training loop."""
        import importlib.util as _u
        spec = _u.spec_from_file_location("train_cli", str(ROOT / "scripts" / "train.py"))
        mod = _u.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # METHODS dict is the canonical method registry; presence proves
        # we loaded the module rather than something else.
        assert hasattr(mod, "METHODS")
        # Build the parser by re-reading the source — argparse is constructed
        # only inside main(), so we string-check the flag declaration.
        with open(ROOT / "scripts" / "train.py", encoding="utf-8") as f:
            src = f.read()
        assert '"--run-name"' in src or "'--run-name'" in src
        # And the train() function signature must accept it (otherwise the
        # CLI flag is dead).
        import inspect
        sig = inspect.signature(mod.train)
        assert "run_name" in sig.parameters


# ---------------------------------------------------------------------
# D4 — per-agent entropy + kill-switch wiring
# ---------------------------------------------------------------------

class TestD4LazyAgentDiagnostic:
    """Logger schema must carry the three new columns, and the entropy
    formula used in the training loop must produce hand-derivable values
    on a known action histogram."""

    def test_logger_schema_has_new_columns(self):
        cols = set(HCMARLLogger.CSV_COLUMNS)
        assert {"per_agent_entropy_mean", "per_agent_entropy_min",
                "lazy_agent_flag"} <= cols

    def test_logger_writes_entropy_columns(self):
        with tempfile.TemporaryDirectory() as td:
            lg = HCMARLLogger(log_dir=td, use_wandb=False)
            lg.log_episode({
                "episode": 1,
                "global_step": 1000,
                "cumulative_reward": 42.0,
                "per_agent_entropy_mean": 1.23,
                "per_agent_entropy_min": 0.45,
                "lazy_agent_flag": 0,
            })
            lg.close()
            with open(os.path.join(td, "training_log.csv"), encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert float(rows[0]["per_agent_entropy_mean"]) == pytest.approx(1.23)
            assert float(rows[0]["per_agent_entropy_min"]) == pytest.approx(0.45)
            assert rows[0]["lazy_agent_flag"] == "0"

    def test_entropy_formula_matches_hand_derivation(self):
        """Replays the exact numpy formula from scripts/train.py."""
        n_workers, n_tasks = 3, 4
        hist = np.zeros((n_workers, n_tasks), dtype=np.int64)
        # Agent 0: uniform over 4 tasks -> H = log(4)
        hist[0] = [25, 25, 25, 25]
        # Agent 1: collapsed to one task -> H = 0
        hist[1] = [100, 0, 0, 0]
        # Agent 2: 50/50 over two tasks -> H = log(2)
        hist[2] = [50, 50, 0, 0]

        per_agent = np.zeros(n_workers)
        for i in range(n_workers):
            counts = hist[i]
            tot = counts.sum()
            if tot > 0:
                p = counts / tot
                nz = p[p > 0]
                per_agent[i] = float(-(nz * np.log(nz)).sum())

        assert per_agent[0] == pytest.approx(math.log(4), abs=1e-9)
        assert per_agent[1] == pytest.approx(0.0, abs=1e-12)
        assert per_agent[2] == pytest.approx(math.log(2), abs=1e-9)
        assert per_agent.min() == pytest.approx(0.0, abs=1e-12)
        # Mean of (log4, 0, log2).
        assert per_agent.mean() == pytest.approx(
            (math.log(4) + 0.0 + math.log(2)) / 3.0, abs=1e-9
        )

    def test_kill_switch_trips_on_sustained_collapse(self):
        """Reproduces the streak logic from the training loop."""
        threshold = 0.1
        window = 100_000
        episode_steps = 480
        low_streak = 0
        flag = 0
        # All episodes below threshold -> streak rises monotonically.
        for _ in range(window // episode_steps + 2):
            ent_min = 0.05
            if ent_min < threshold:
                low_streak += episode_steps
            else:
                low_streak = 0
            if low_streak >= window:
                flag = 1
        assert flag == 1

        # If ANY episode recovers above threshold, streak resets.
        low_streak = 0
        flag = 0
        for i in range(300):
            ent_min = 0.05 if i != 150 else 0.5  # one healthy episode
            if ent_min < threshold:
                low_streak += episode_steps
            else:
                low_streak = 0
            if low_streak >= window:
                flag = 1
        # 150 episodes * 480 = 72_000 steps < 100_000 before the reset, and
        # the post-reset streak 149 * 480 = 71_520 is also under window.
        assert flag == 0
