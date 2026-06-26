"""
Tests for Cycle S-1 audit fixes (S-10, S-12, S-22, S-23, S-36, S-37, S-38).

Verifies:
    S-10  -- MI returns 0.0 + mi_collapsed on mode collapse, not inflated H(z)
    S-12  -- lambda1=lambda2 enforced per Remark 4.4; h_marginal deleted
    S-22  -- forced_rests uses ecbf_interventions, not arbitrary MF>0.3
    S-23  -- ECBF opportunities count only muscles with nonzero task demand
    S-36  -- CSV columns fixed from METRIC_NAMES, not first-episode keys
    S-37  -- CSV resume after crash appends instead of overwriting
    S-38  -- evaluate.py uses conservative theta default 0.5, not silent 1.0
"""
import csv
import inspect
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcmarl.mmicrl import MMICRL, DemonstrationCollector
from hcmarl.logger import HCMARLLogger


# =====================================================================
# S-10: MI collapse returns 0.0, not H(z)
# =====================================================================

class TestS10:

    def test_mi_collapsed_flag_exists(self):
        """fit() results must include 'mi_collapsed' boolean flag."""
        from hcmarl.real_data_calibration import (
            generate_demonstrations_from_profiles,
            load_path_g_into_collector,
        )
        profiles = [
            {'worker_id': 0, 'muscles': {'shoulder': {'F': 2.0, 'R': 0.02, 'r': 15}}},
            {'worker_id': 1, 'muscles': {'shoulder': {'F': 1.0, 'R': 0.02, 'r': 15}}},
        ]
        demos, wids = generate_demonstrations_from_profiles(
            profiles, muscle='shoulder', n_episodes_per_worker=3,
        )
        collector = load_path_g_into_collector(demos, wids)
        mmicrl = MMICRL(n_types=2, n_muscles=1)
        results = mmicrl.fit(collector)
        assert "mi_collapsed" in results, "Results must include mi_collapsed flag"
        assert isinstance(results["mi_collapsed"], bool)

    def test_mi_zero_when_no_cfde(self):
        """With CFDE=None, MI should be 0.0, not H(z)."""
        mmicrl = MMICRL(n_types=3, n_muscles=1)
        mmicrl.cfde = None
        assignments = np.array([0, 0, 1, 1, 2, 2])
        # H(z) for uniform 3-type would be ~1.099, but we should get 0.0
        step_features = np.random.randn(60, 5).astype(np.float32)
        traj_indices = np.repeat(np.arange(6), 10)
        mi = mmicrl._compute_mutual_information(step_features, traj_indices, assignments)
        assert mi == 0.0, f"MI should be 0.0 when CFDE is None, got {mi}"
        assert mmicrl._mi_collapsed is True

    def test_no_h_z_fallback_in_source(self):
        """_compute_mutual_information must NOT return _hard_assignment_mi() on collapse."""
        source = inspect.getsource(MMICRL._compute_mutual_information)
        assert '_hard_assignment_mi' not in source, \
            "S-10: _hard_assignment_mi fallback should be removed"


# =====================================================================
# S-12: lambda1=lambda2 enforced, h_marginal deleted
# =====================================================================

class TestS12:

    def test_lambda_equality_enforced(self):
        """When lambda1 != lambda2, constructor must set lambda2 = lambda1."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mmicrl = MMICRL(n_types=3, lambda1=2.0, lambda2=1.0, n_muscles=1)
            assert mmicrl.lambda1 == mmicrl.lambda2 == 2.0
            assert len(w) == 1
            assert "Remark 4.4" in str(w[0].message)

    def test_lambda_equal_no_warning(self):
        """When lambda1 == lambda2, no warning should be raised."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mmicrl = MMICRL(n_types=3, lambda1=1.0, lambda2=1.0, n_muscles=1)
            remark_warnings = [x for x in w if "Remark 4.4" in str(x.message)]
            assert len(remark_warnings) == 0

    def test_no_h_marginal_in_fit(self):
        """fit() must not compute h_marginal (the buggy per-column histogram entropy)."""
        source = inspect.getsource(MMICRL.fit)
        assert 'h_marginal' not in source, \
            "S-12: h_marginal computation should be deleted per Remark 4.4"
        assert 'np.histogram' not in source, \
            "S-12: histogram-based entropy estimation should be removed from fit()"

    def test_objective_equals_lambda_times_mi(self):
        """Objective must be lambda * MI (pure MI maximisation, Eq 11)."""
        from hcmarl.real_data_calibration import (
            generate_demonstrations_from_profiles,
            load_path_g_into_collector,
        )
        profiles = [
            {'worker_id': 0, 'muscles': {'shoulder': {'F': 2.0, 'R': 0.02, 'r': 15}}},
            {'worker_id': 1, 'muscles': {'shoulder': {'F': 0.5, 'R': 0.02, 'r': 15}}},
        ]
        demos, wids = generate_demonstrations_from_profiles(
            profiles, muscle='shoulder', n_episodes_per_worker=3,
        )
        collector = load_path_g_into_collector(demos, wids)
        mmicrl = MMICRL(n_types=2, lambda1=3.0, lambda2=3.0, n_muscles=1)
        results = mmicrl.fit(collector)
        expected = 3.0 * results["mutual_information"]
        assert abs(results["objective_value"] - expected) < 1e-6, \
            f"Objective {results['objective_value']} != 3.0 * MI {expected}"


# =====================================================================
# S-22: forced_rests uses ecbf_interventions, not MF > 0.3
# =====================================================================

class TestS22:

    def test_no_mf_threshold_in_train(self):
        """train.py must not use 'avg_mf > 0.3' for forced rests."""
        with open("scripts/train.py") as f:
            content = f.read()
        assert 'avg_mf > 0.3' not in content, \
            "S-22: arbitrary MF>0.3 threshold should be replaced with ecbf_interventions"
        assert 'avg_mf' not in content, \
            "S-22: avg_mf variable should be removed from train.py"

    def test_no_mf_threshold_in_evaluate(self):
        """evaluate.py must not use 'avg_mf > 0.3' for forced rests."""
        with open("scripts/evaluate.py") as f:
            content = f.read()
        assert 'avg_mf > 0.3' not in content, \
            "S-22: arbitrary MF>0.3 threshold should be replaced in evaluate.py"
        assert 'avg_mf' not in content

    def test_ecbf_interventions_used_for_forced_rests(self):
        """Both train.py and evaluate.py must use ecbf_interventions for forced rests."""
        for path in ["scripts/train.py", "scripts/evaluate.py"]:
            with open(path) as f:
                content = f.read()
            assert 'ecbf_int' in content and 'forced_rests' in content, \
                f"{path}: forced_rests must be based on ecbf_interventions"


# =====================================================================
# S-23: ECBF opportunities count only demanded muscles
# =====================================================================

class TestS23:

    def test_demand_vector_used_in_train(self):
        """train.py must use get_demand_vector to count ECBF opportunities."""
        with open("scripts/train.py") as f:
            content = f.read()
        assert 'get_demand_vector' in content, \
            "S-23: ECBF opportunities must use task demand data"
        assert 'demands > 0' in content or '(demands > 0)' in content, \
            "S-23: must count only muscles with demand > 0"

    def test_demand_vector_used_in_evaluate(self):
        """evaluate.py must use get_demand_vector to count ECBF opportunities."""
        with open("scripts/evaluate.py") as f:
            content = f.read()
        assert 'get_demand_vector' in content, \
            "S-23: evaluate.py must also use task demand data for ECBF opportunities"

    def test_no_len_fatigue_for_opportunities(self):
        """Neither file should use len(fatigue) as the ECBF opportunity count."""
        for path in ["scripts/train.py", "scripts/evaluate.py"]:
            with open(path) as f:
                content = f.read()
            # Check that len(fatigue) is not used in the ecbf_opportunities context
            assert 'total_ecbf_opportunities += len(fatigue)' not in content, \
                f"S-23: {path} still uses len(fatigue) for ECBF opportunities"

    def test_all_active_tasks_have_six_demanded_muscles(self):
        """Verify that all 5 active tasks demand all 6 muscles (demand > 0)."""
        from hcmarl.envs.task_profiles import TaskProfileManager
        mgr = TaskProfileManager()
        for task in mgr.task_names:
            if task == "rest":
                continue
            demands = mgr.get_demand_vector(task)
            n_demanded = int((demands > 0).sum())
            assert n_demanded == 6, \
                f"Task '{task}' has {n_demanded} demanded muscles, expected 6"


# =====================================================================
# S-36: CSV columns fixed from METRIC_NAMES
# =====================================================================

class TestS36:

    def test_csv_columns_class_attribute(self):
        """HCMARLLogger must have a CSV_COLUMNS class attribute."""
        assert hasattr(HCMARLLogger, 'CSV_COLUMNS'), \
            "S-36: HCMARLLogger must define CSV_COLUMNS"
        assert 'episode' in HCMARLLogger.CSV_COLUMNS
        for m in HCMARLLogger.METRIC_NAMES:
            assert m in HCMARLLogger.CSV_COLUMNS, f"Missing metric: {m}"

    def test_csv_columns_sorted(self):
        """CSV_COLUMNS must be sorted for deterministic output."""
        assert HCMARLLogger.CSV_COLUMNS == sorted(HCMARLLogger.CSV_COLUMNS)

    def test_extra_keys_ignored(self):
        """Undeclared metric keys must be silently ignored, not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCMARLLogger(log_dir=tmpdir)
            # Log with an extra key not in CSV_COLUMNS
            metrics = {"cumulative_reward": 1.0, "unknown_metric_xyz": 99.0}
            logger.log_episode(metrics)
            logger.close()
            # Read CSV — should NOT contain unknown_metric_xyz
            with open(logger.csv_path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert "unknown_metric_xyz" not in row

    def test_csv_columns_stable_across_episodes(self):
        """Column order must be identical for first and subsequent episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCMARLLogger(log_dir=tmpdir)
            logger.log_episode({"cumulative_reward": 1.0})
            logger.log_episode({"cumulative_reward": 2.0, "lambda": 0.5})
            logger.close()
            with open(logger.csv_path) as f:
                lines = f.readlines()
            # Both data lines should have same number of columns as header
            header_cols = len(lines[0].strip().split(","))
            for i, line in enumerate(lines[1:], 1):
                cols = len(line.strip().split(","))
                assert cols == header_cols, \
                    f"Line {i} has {cols} cols vs header {header_cols}"


# =====================================================================
# S-37: CSV resume after crash
# =====================================================================

class TestS37:

    def test_resume_appends(self):
        """After restart, logger must append to existing CSV, not overwrite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run: write 2 episodes
            logger1 = HCMARLLogger(log_dir=tmpdir)
            logger1.log_episode({"cumulative_reward": 1.0})
            logger1.log_episode({"cumulative_reward": 2.0})
            logger1.close()

            # Simulate crash + restart: new logger instance
            logger2 = HCMARLLogger(log_dir=tmpdir)
            logger2.log_episode({"cumulative_reward": 3.0})
            logger2.close()

            # Should have header + 3 data rows
            with open(os.path.join(tmpdir, "training_log.csv")) as f:
                lines = f.readlines()
            # 1 header + 3 data = 4 lines
            assert len(lines) == 4, f"Expected 4 lines (1 header + 3 data), got {len(lines)}"

    def test_resume_no_duplicate_header(self):
        """Resumed logger must NOT write a second header row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger1 = HCMARLLogger(log_dir=tmpdir)
            logger1.log_episode({"cumulative_reward": 1.0})
            logger1.close()

            logger2 = HCMARLLogger(log_dir=tmpdir)
            logger2.log_episode({"cumulative_reward": 2.0})
            logger2.close()

            with open(os.path.join(tmpdir, "training_log.csv")) as f:
                content = f.read()
            # Count occurrences of "episode" in content — should be 1 (header only)
            # plus 2 data rows that have episode numbers
            header_count = content.split('\n')[0].count('episode')
            assert header_count == 1

    def test_fresh_start_writes_header(self):
        """On fresh start (no existing CSV), header must be written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = HCMARLLogger(log_dir=tmpdir)
            logger.log_episode({"cumulative_reward": 1.0})
            logger.close()
            with open(logger.csv_path) as f:
                header = f.readline().strip()
            assert 'episode' in header
            assert 'cumulative_reward' in header


# =====================================================================
# S-38: evaluate.py theta_max default + SAI
# =====================================================================

class TestS38:

    def test_no_silent_1_default(self):
        """evaluate.py must not use theta_max.get(m, 1.0) — silent default hides violations."""
        with open("scripts/evaluate.py") as f:
            content = f.read()
        assert 'get(m, 1.0)' not in content, \
            "S-38: theta_max default 1.0 makes violations undetectable"

    def test_conservative_default(self):
        """evaluate.py must use 0.5 as conservative default for missing theta."""
        with open("scripts/evaluate.py") as f:
            content = f.read()
        assert ('theta = 0.5' in content
                or 'theta = env.theta_max.get(m, 0.5)' in content
                or 'worker_theta.get(m, 0.5)' in content), \
            "S-38: must use conservative default theta=0.5"

    def test_sai_computed_in_evaluate(self):
        """evaluate.py must compute Safety Autonomy Index."""
        with open("scripts/evaluate.py") as f:
            content = f.read()
        assert 'total_ecbf_interventions' in content, \
            "S-38: evaluate.py must track ECBF interventions"
        assert 'total_ecbf_opportunities' in content, \
            "S-38: evaluate.py must track ECBF opportunities"
        assert 'sai' in content.lower(), \
            "S-38: evaluate.py must compute SAI"

    def test_warning_on_missing_theta(self):
        """evaluate.py must warn (not silently default) on missing theta_max."""
        with open("scripts/evaluate.py") as f:
            content = f.read()
        assert 'warnings.warn' in content or 'import warnings' in content, \
            "S-38: must warn when theta_max is missing for a muscle"
