"""
Tests for Minor audit items M-1, M-2, M-3.

M-1: seed_everything logs deterministic mode (utils.py)
M-2: _BatchNormFlow batch_mean/batch_var initialized in __init__ (mmicrl.py)
M-3: safety_cost is binary with documented rationale (reward_functions.py)
"""

import inspect
import logging
import unittest

import numpy as np
import torch

from hcmarl.envs.reward_functions import safety_cost
from hcmarl.utils import seed_everything


# ── M-1: seed_everything logs deterministic mode ──

class TestM1(unittest.TestCase):
    """M-1: cudnn deterministic mode is logged."""

    def test_seed_everything_runs(self):
        """seed_everything completes without error."""
        seed_everything(42)

    def test_deterministic_mode_logged(self):
        """When CUDA is available, a log message about deterministic mode is emitted."""
        if not torch.cuda.is_available():
            self.skipTest("No CUDA — deterministic log only emitted on GPU")
        with self.assertLogs("hcmarl.utils", level="INFO") as cm:
            seed_everything(123)
        messages = " ".join(cm.output)
        self.assertIn("deterministic", messages.lower())

    def test_deterministic_comment_present(self):
        """M-1 rationale comment exists in source."""
        source = inspect.getsource(seed_everything)
        self.assertIn("M-1", source)
        self.assertIn("reproducibility", source.lower())


# ── M-2: _BatchNormFlow initialization ──

class TestM2(unittest.TestCase):
    """M-2: _BatchNormFlow batch stats exist from __init__."""

    def _make_bnflow(self, dim=4):
        from hcmarl.mmicrl import _BatchNormFlow
        return _BatchNormFlow(dim)

    def test_batch_mean_exists_at_init(self):
        """batch_mean is accessible immediately after construction."""
        bn = self._make_bnflow()
        self.assertTrue(hasattr(bn, "batch_mean"))
        self.assertEqual(bn.batch_mean.shape, (4,))

    def test_batch_var_exists_at_init(self):
        """batch_var is accessible immediately after construction."""
        bn = self._make_bnflow()
        self.assertTrue(hasattr(bn, "batch_var"))
        self.assertEqual(bn.batch_var.shape, (4,))

    def test_inverse_before_direct_no_error(self):
        """Calling inverse mode in training before direct doesn't crash."""
        bn = self._make_bnflow(dim=3)
        bn.train()
        x = torch.randn(8, 3)
        # This would AttributeError before M-2 fix
        y, logdet = bn(x, mode='inverse')
        self.assertEqual(y.shape, (8, 3))

    def test_direct_forward_updates_batch_stats(self):
        """After a direct forward pass, batch_mean reflects input statistics."""
        bn = self._make_bnflow(dim=2)
        bn.train()
        x = torch.ones(16, 2) * 5.0
        bn(x, mode='direct')
        self.assertTrue(torch.allclose(bn.batch_mean, torch.tensor([5.0, 5.0])))

    def test_state_dict_excludes_batch_stats(self):
        """batch_mean/batch_var are NOT in state_dict (ephemeral)."""
        bn = self._make_bnflow(dim=2)
        sd = bn.state_dict()
        self.assertNotIn("batch_mean", sd)
        self.assertNotIn("batch_var", sd)

    def test_running_stats_in_state_dict(self):
        """running_mean/running_var ARE in state_dict (persistent buffers)."""
        bn = self._make_bnflow(dim=2)
        sd = bn.state_dict()
        self.assertIn("running_mean", sd)
        self.assertIn("running_var", sd)


# ── M-3: safety_cost is dense (continuous) with documentation ──

class TestM3(unittest.TestCase):
    """M-3: safety_cost returns dense continuous cost proportional to violation magnitude."""

    def test_no_violation_returns_zero(self):
        fatigue = {"shoulder": 0.3, "ankle": 0.2}
        theta = {"shoulder": 0.7, "ankle": 0.8}
        self.assertEqual(safety_cost(fatigue, theta), 0.0)

    def test_single_violation_returns_excess(self):
        fatigue = {"shoulder": 0.8, "ankle": 0.2}
        theta = {"shoulder": 0.7, "ankle": 0.8}
        cost = safety_cost(fatigue, theta)
        self.assertAlmostEqual(cost, 0.1, places=5)

    def test_multiple_violations_sum_excess(self):
        """Dense: cost = sum of max(0, MF_m - theta_m) across muscles."""
        fatigue = {"shoulder": 0.8, "ankle": 0.9, "knee": 0.7}
        theta = {"shoulder": 0.7, "ankle": 0.8, "knee": 0.6}
        cost = safety_cost(fatigue, theta)
        self.assertAlmostEqual(cost, 0.1 + 0.1 + 0.1, places=5)

    def test_m3_rationale_documented(self):
        """M-3 design rationale is present in the docstring."""
        source = inspect.getsource(safety_cost)
        self.assertIn("Dense cost", source)


if __name__ == "__main__":
    unittest.main()
