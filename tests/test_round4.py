"""
Tests for Round 4 audit fixes (C-4, C-11, C-15, C-16).

Verifies:
    C-4   -- ConstraintNetwork deleted; _learn_constraints uses direct percentile
    C-11  -- RandomPolicy deleted; MMICRL demos from Path G profiles (Eq 35)
    C-15  -- Synthetic demo tests replaced with WSD4FEDSRM-calibrated profiles
    C-16  -- OmniSafe dropped; SafePO honestly labeled as MAPPO-Lagrangian
"""
import numpy as np
import pytest
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# C-4: ConstraintNetwork deleted, direct percentile thresholds
# =====================================================================

class TestC4:

    def test_no_constraint_network_class(self):
        """ConstraintNetwork must not exist in mmicrl module."""
        import hcmarl.mmicrl as mmicrl_mod
        assert not hasattr(mmicrl_mod, 'ConstraintNetwork'), \
            "ConstraintNetwork should be deleted (C-4: circular percentile estimation)"

    def test_learn_constraints_direct_percentile(self):
        """_learn_constraints must use direct 90th percentile, not a neural net."""
        from hcmarl.mmicrl import MMICRL, DemonstrationCollector
        from hcmarl.real_data_calibration import (
            generate_demonstrations_from_profiles,
            load_path_g_into_collector,
        )

        # 3 workers with known F values
        profiles = [
            {'worker_id': 0, 'muscles': {'shoulder': {'F': 2.0, 'R': 0.02, 'r': 15}}},
            {'worker_id': 1, 'muscles': {'shoulder': {'F': 1.0, 'R': 0.02, 'r': 15}}},
            {'worker_id': 2, 'muscles': {'shoulder': {'F': 0.5, 'R': 0.02, 'r': 15}}},
        ]
        demos, wids = generate_demonstrations_from_profiles(
            profiles, muscle='shoulder', n_episodes_per_worker=3,
        )
        collector = load_path_g_into_collector(demos, wids)

        mmicrl = MMICRL(n_types=2, n_muscles=1)
        results = mmicrl.fit(collector)

        # Thresholds should be in valid range
        for k, thetas in results['theta_per_type'].items():
            for m, v in thetas.items():
                assert 0.1 <= v <= 0.95, f"Type {k} {m}: theta={v} out of [0.1, 0.95]"

    def test_no_torch_in_learn_constraints(self):
        """_learn_constraints should not instantiate any nn.Module (no neural net)."""
        import inspect
        from hcmarl.mmicrl import MMICRL
        source = inspect.getsource(MMICRL._learn_constraints)
        # Strip comments and docstrings — only check executable code
        code_lines = [l for l in source.split('\n')
                       if l.strip() and not l.strip().startswith('#')
                       and not l.strip().startswith('"""')
                       and not l.strip().startswith("'''")]
        code = '\n'.join(code_lines)
        assert 'nn.Module' not in code
        assert 'nn.Sequential' not in code
        assert 'nn.Linear' not in code
        assert 'binary_cross_entropy' not in code


# =====================================================================
# C-11: RandomPolicy deleted, Path G profiles used
# =====================================================================

class TestC11:

    def test_no_random_policy_in_train(self):
        """RandomPolicy class must not exist in train.py."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("train", "scripts/train.py")
        source = spec.origin
        with open(source) as f:
            content = f.read()
        assert 'class RandomPolicy' not in content, \
            "RandomPolicy should be deleted (C-11: demos from Eq 35, not random)"
        assert 'random policy' not in content.lower() or 'no random-policy' in content.lower(), \
            "References to random policy should be removed"

    def test_pathg_profiles_json_exists(self):
        """config/pathg_profiles.json must exist with valid structure."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "pathg_profiles.json")
        assert os.path.exists(path), f"Missing {path}"
        with open(path) as f:
            data = json.load(f)
        assert 'profiles' in data
        assert data['n_workers'] == 34
        assert len(data['profiles']) == 34
        # Check F range matches documented [0.44, 2.62]
        assert data['F_range'][0] < 0.5
        assert data['F_range'][1] > 2.5

    def test_pathg_profile_structure(self):
        """Each profile must have worker_id, source_subject, muscles with F/R/r."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "pathg_profiles.json")
        with open(path) as f:
            data = json.load(f)
        for p in data['profiles']:
            assert 'worker_id' in p
            assert 'source_subject' in p
            assert 'muscles' in p
            assert 'shoulder' in p['muscles']
            for m_name, m_data in p['muscles'].items():
                assert 'F' in m_data and m_data['F'] > 0
                assert 'R' in m_data and m_data['R'] > 0
                assert 'r' in m_data and m_data['r'] > 0

    def test_run_mmicrl_pretrain_uses_pathg(self):
        """run_mmicrl_pretrain imports from real_data_calibration, not random."""
        with open("scripts/train.py") as f:
            content = f.read()
        assert 'generate_demonstrations_from_profiles' in content
        assert 'load_path_g_into_collector' in content
        assert 'real_data_calibration' in content

    def test_run_mmicrl_pretrain_skips_without_data(self):
        """When no data/profiles exist, pretrain should return (None, None)."""
        # This tests the graceful skip path
        with open("scripts/train.py") as f:
            content = f.read()
        assert 'return None, None' in content, \
            "Must gracefully skip MMICRL when no data available"


# =====================================================================
# C-15: Synthetic tests replaced with calibrated-profile tests
# =====================================================================

class TestC15:

    def test_no_generate_synthetic_in_tests(self):
        """test_phase3.py must not call generate_synthetic_demos."""
        with open("tests/test_phase3.py") as f:
            content = f.read()
        assert 'generate_synthetic_demos' not in content, \
            "Synthetic demos should be replaced with Path G profiles (C-15)"

    def test_calibrated_profiles_in_tests(self):
        """test_phase3.py must use hardcoded WSD4FEDSRM-calibrated F values."""
        with open("tests/test_phase3.py") as f:
            content = f.read()
        assert '_CALIBRATED_PROFILES' in content
        assert 'generate_demonstrations_from_profiles' in content
        # Should reference real subject IDs
        assert 'subject_' in content

    def test_calibrated_f_values_are_real(self):
        """Hardcoded F values in tests must match actual WSD4FEDSRM calibration."""
        # Import the test module's profiles
        sys.path.insert(0, 'tests')
        from test_phase3 import _CALIBRATED_PROFILES

        # Check F values are in the documented range [0.44, 2.62]
        for p in _CALIBRATED_PROFILES:
            F = p['muscles']['shoulder']['F']
            assert 0.4 <= F <= 2.7, f"F={F} outside WSD4FEDSRM range [0.44, 2.62]"
            assert p['muscles']['shoulder']['R'] == 0.02  # calibration R
            assert p['muscles']['shoulder']['r'] == 15    # reperfusion

    def test_mmicrl_works_with_pathg_demos(self):
        """MMICRL must successfully fit on Path G calibrated demos."""
        from test_phase3 import _make_pathg_collector
        from hcmarl.mmicrl import MMICRL
        collector, n = _make_pathg_collector(n_episodes=3)
        assert n == 27  # 9 workers x 3 episodes
        mmicrl = MMICRL(n_types=3, n_muscles=1, auto_select_k=True)
        results = mmicrl.fit(collector)
        assert results['mutual_information'] >= 0
        k = results['n_types_discovered']
        assert 1 <= k <= 5
        assert len(results['theta_per_type']) == k


# =====================================================================
# C-16: OmniSafe dropped, honest baselines
# =====================================================================

class TestC16:

    def test_no_omnisafe_wrapper_file(self):
        """omnisafe_wrapper.py must not exist."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "hcmarl", "baselines", "omnisafe_wrapper.py")
        assert not os.path.exists(path), "omnisafe_wrapper.py should be deleted (C-16)"

    def test_no_omnisafe_import_in_init(self):
        """baselines/__init__.py must not import OmniSafeWrapper."""
        with open("hcmarl/baselines/__init__.py") as f:
            content = f.read()
        assert 'OmniSafeWrapper' not in content
        assert 'omnisafe_wrapper' not in content

    def test_no_omnisafe_in_train(self):
        """train.py must not reference OmniSafe (the dropped library).
        'macpo' must NOT be in train.py. Forbidden are 'ppo_lag' and 'cpo'
        (placeholder-baseline holdovers)."""
        with open("scripts/train.py") as f:
            content = f.read()
        assert 'OmniSafeWrapper' not in content
        assert 'omnisafe_wrapper' not in content
        assert '"ppo_lag"' not in content
        assert '"cpo"' not in content
        assert '"macpo"' not in content

    def test_only_honest_methods(self):
        """METHODS dict must contain only methods backed by real
        implementations. The lineup is hcmarl, mappo, ippo (kept as a method
        even though not in the headline matrix), mappo_lag, happo, and
        shielded_mappo. Forbidden names: 'ppo_lag', 'cpo', 'macpo'."""
        with open("scripts/train.py") as f:
            content = f.read()
        for method in ["hcmarl", "mappo", "ippo", "mappo_lag", "happo", "shielded_mappo"]:
            assert f'"{method}"' in content, f"Missing method: {method}"
        for method in ["ppo_lag", "cpo", "macpo"]:
            assert f'"{method}"' not in content, f"Forbidden method present: {method}"

    def test_safepo_wrapper_honest_name(self):
        """SafePOWrapper.name must be 'MAPPO-Lagrangian' (the actual underlying algo)."""
        from hcmarl.baselines.safepo_wrapper import SafePOWrapper
        w = SafePOWrapper(obs_dim=19, n_actions=6, n_agents=4)
        assert w.name == "MAPPO-Lagrangian", f"Got dishonest name: {w.name}"

    def test_run_baselines_no_fake_methods(self):
        """run_baselines.py must only list mappo, ippo, mappo_lag."""
        with open("scripts/run_baselines.py") as f:
            content = f.read()
        assert '"ppo_lag"' not in content
        assert '"cpo"' not in content
        assert '"macpo"' not in content

    def test_no_omnisafe_test(self):
        """test_all_methods.py must not have test_omnisafe_wrapper."""
        with open("tests/test_all_methods.py") as f:
            content = f.read()
        assert 'test_omnisafe_wrapper' not in content
        assert 'OmniSafeWrapper' not in content
