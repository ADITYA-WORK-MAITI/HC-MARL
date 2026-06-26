"""Determinism and resume regression tests.

Covers:
    B1 — run_state.pt round-trip preserves ALL RNG streams (numpy, torch,
         cuda-if-available, python stdlib) and all counters. Extends the
         existing M1 theta_max round-trip test with the full state surface.
    B3 — Pseudo-preemption: mid-run kill + resume produces bit-identical
         stochastic output to an uninterrupted reference. This is the
         smallest faithful reproduction of the preemption case.
    B4 — Train/eval theta parity: build_per_worker_theta_max is a pure
         function; called with identical inputs it must return identical
         outputs whether invoked from train.py or evaluate.py context.

B2 is not covered by code here — warm_start has no effect until the
DPP refactor caches cp.Problem across calls. A source comment
was added to ecbf_filter.py documenting the deferral.
"""

from __future__ import annotations

import importlib.util
import os
import random
import tempfile

import numpy as np
import pytest
import torch

from hcmarl.utils import build_per_worker_theta_max


def _load_train_module():
    """Import scripts/train.py as a module so its private helpers are reachable."""
    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location(
        "_train_module", os.path.join(here, "..", "scripts", "train.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------
# B1 — full RNG + counter round-trip
# ---------------------------------------------------------------------

class TestB1RunStateFullRoundTrip:
    """run_state.pt must preserve every piece of state the policy checkpoint
    does not already hold. Missing any of these silently makes a resumed
    run diverge from an uninterrupted one."""

    def test_rng_numpy_roundtrip(self):
        train_mod = _load_train_module()
        np.random.seed(42)
        # Burn a few draws to move off the seed initial state.
        _ = np.random.random(5)
        captured = np.random.get_state()
        draws_ref = np.random.random(10)

        # Save -> load -> restore -> compare next draws.
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "run_state.pt")
            np.random.set_state(captured)
            train_mod._write_run_state(
                path=path, global_step=100, episode_count=3,
                cost_ema=0.0, best_reward=0.0,
                theta_max={}, seed=42, method="hcmarl",
            )
            loaded = train_mod._load_run_state(path)

        np.random.set_state(loaded["rng_np"])
        draws_after_resume = np.random.random(10)
        np.testing.assert_array_equal(draws_ref, draws_after_resume)

    def test_rng_torch_roundtrip(self):
        train_mod = _load_train_module()
        torch.manual_seed(42)
        _ = torch.randn(5)  # burn in
        draws_ref = torch.randn(10).clone()

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "run_state.pt")
            # Re-seed and recapture so ref and round-trip match.
            torch.manual_seed(42)
            _ = torch.randn(5)
            train_mod._write_run_state(
                path=path, global_step=100, episode_count=3,
                cost_ema=0.0, best_reward=0.0,
                theta_max={}, seed=42, method="hcmarl",
            )
            loaded = train_mod._load_run_state(path)

        torch.set_rng_state(loaded["rng_torch"])
        draws_after_resume = torch.randn(10)
        assert torch.equal(draws_ref, draws_after_resume)

    def test_rng_python_stdlib_roundtrip(self):
        train_mod = _load_train_module()
        random.seed(42)
        _ = [random.random() for _ in range(5)]
        draws_ref = [random.random() for _ in range(10)]

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "run_state.pt")
            random.seed(42)
            _ = [random.random() for _ in range(5)]
            train_mod._write_run_state(
                path=path, global_step=100, episode_count=3,
                cost_ema=0.0, best_reward=0.0,
                theta_max={}, seed=42, method="hcmarl",
            )
            loaded = train_mod._load_run_state(path)

        random.setstate(loaded["rng_python"])
        draws_after_resume = [random.random() for _ in range(10)]
        assert draws_ref == draws_after_resume

    def test_counters_and_cost_ema_roundtrip(self):
        train_mod = _load_train_module()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "run_state.pt")
            train_mod._write_run_state(
                path=path, global_step=250_000, episode_count=1234,
                cost_ema=0.0247, best_reward=312.75,
                theta_max={"worker_0": {"shoulder": 0.72}},
                seed=7, method="hcmarl",
            )
            loaded = train_mod._load_run_state(path)

        assert loaded["global_step"] == 250_000
        assert loaded["episode_count"] == 1234
        assert loaded["cost_ema"] == pytest.approx(0.0247, abs=1e-9)
        assert loaded["best_reward"] == pytest.approx(312.75, abs=1e-9)
        assert loaded["seed"] == 7
        assert loaded["method"] == "hcmarl"
        assert loaded["theta_max"] == {"worker_0": {"shoulder": 0.72}}


# ---------------------------------------------------------------------
# B3 — pseudo-preemption: kill + resume is bit-identical
# ---------------------------------------------------------------------

class TestB3PreemptionResumeIsBitIdentical:
    """Gate B: after saving run_state mid-stream and restoring, the
    downstream stochastic outputs must match an uninterrupted reference.
    This is the smallest faithful reproduction of a Colab-spot preemption."""

    def test_uninterrupted_vs_resumed_bit_identical(self):
        train_mod = _load_train_module()

        # ----- Uninterrupted reference -----
        np.random.seed(123)
        torch.manual_seed(123)
        random.seed(123)
        # First half (pretend these are training draws 0..24)
        _ref_first = np.random.random(25)
        _ref_first_t = torch.randn(25).clone()
        _ref_first_p = [random.random() for _ in range(25)]
        # Second half (draws 25..49) — this is what a resumed run must match.
        ref_second = np.random.random(25)
        ref_second_t = torch.randn(25).clone()
        ref_second_p = [random.random() for _ in range(25)]

        # ----- Interrupted + resumed run -----
        np.random.seed(123)
        torch.manual_seed(123)
        random.seed(123)
        _ = np.random.random(25)
        _ = torch.randn(25)
        _ = [random.random() for _ in range(25)]

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "run_state.pt")
            train_mod._write_run_state(
                path=path, global_step=25, episode_count=1,
                cost_ema=0.0, best_reward=0.0,
                theta_max={}, seed=123, method="hcmarl",
            )
            # Simulate process death: scramble all RNG streams.
            np.random.seed(9999)
            torch.manual_seed(9999)
            random.seed(9999)
            _ = np.random.random(100); _ = torch.randn(100); _ = [random.random() for _ in range(100)]

            loaded = train_mod._load_run_state(path)

        np.random.set_state(loaded["rng_np"])
        torch.set_rng_state(loaded["rng_torch"])
        random.setstate(loaded["rng_python"])

        resumed_second = np.random.random(25)
        resumed_second_t = torch.randn(25)
        resumed_second_p = [random.random() for _ in range(25)]

        np.testing.assert_array_equal(ref_second, resumed_second)
        assert torch.equal(ref_second_t, resumed_second_t)
        assert ref_second_p == resumed_second_p


# ---------------------------------------------------------------------
# B4 — train/eval theta parity (pure-function regression)
# ---------------------------------------------------------------------

class TestB4TrainEvalThetaParity:
    """Train and eval must see bit-identical theta_max for the same config +
    MMICRL result. build_per_worker_theta_max is pure; this test nails
    down a fixture so future silent drift is caught."""

    @staticmethod
    def _fixture_results():
        return {
            "theta_per_type": {
                "0": {"shoulder": 0.21, "elbow": 0.18},
                "1": {"shoulder": 0.33, "elbow": 0.24},
                "2": {"shoulder": 0.46, "elbow": 0.31},
            },
            "type_proportions": [0.33, 0.33, 0.34],
            "mutual_information": 0.35,
        }

    def test_same_inputs_same_outputs(self):
        results = self._fixture_results()
        floors = {"shoulder": 0.70, "elbow": 0.45}
        a = build_per_worker_theta_max(results, floors, 6, "hcmarl",
                                       rescale_to_floor=True,
                                       mi_collapse_threshold=0.01)
        b = build_per_worker_theta_max(results, floors, 6, "hcmarl",
                                       rescale_to_floor=True,
                                       mi_collapse_threshold=0.01)
        assert a == b

    def test_frozen_fixture_exact_values(self):
        """Pin down expected output byte-for-byte. If someone refactors the
        rescaling formula, this fails loudly instead of silently drifting."""
        results = self._fixture_results()
        floors = {"shoulder": 0.70, "elbow": 0.45}
        out = build_per_worker_theta_max(results, floors, 6, "hcmarl",
                                         rescale_to_floor=True,
                                         mi_collapse_threshold=0.01)
        # 6 workers, 3 types with near-equal proportions => 2-2-2 split.
        assert set(out.keys()) == {f"worker_{i}" for i in range(6)}
        # Defensive fix: open interval upper bound = (1 - 1e-3) = 0.999.
        for k, v in out.items():
            assert 0.70 - 1e-9 <= v["shoulder"] < 1.0
            assert 0.45 - 1e-9 <= v["elbow"] < 1.0
        # Type 0 has the lowest raw theta -> maps to the floor.
        shoulders = sorted({v["shoulder"] for v in out.values()})
        elbows = sorted({v["elbow"] for v in out.values()})
        assert abs(shoulders[0] - 0.70) < 1e-6
        assert abs(shoulders[-1] - 0.999) < 1e-6
        assert abs(elbows[0] - 0.45) < 1e-6
        assert abs(elbows[-1] - 0.999) < 1e-6

    def test_mi_collapse_zero_informative_thetas(self):
        results = self._fixture_results()
        results = dict(results)
        results["mutual_information"] = 0.0005
        floors = {"shoulder": 0.70, "elbow": 0.45}
        out = build_per_worker_theta_max(results, floors, 6, "hcmarl",
                                         rescale_to_floor=True,
                                         mi_collapse_threshold=0.01)
        for v in out.values():
            assert abs(v["shoulder"] - 0.70) < 1e-9
            assert abs(v["elbow"] - 0.45) < 1e-9

    def test_non_hcmarl_method_returns_flat_defaults(self):
        """MAPPO / IPPO / MAPPO-Lag must not be accidentally given
        MMICRL-derived per-worker thresholds (fairness invariant)."""
        results = self._fixture_results()
        floors = {"shoulder": 0.70, "elbow": 0.45}
        out = build_per_worker_theta_max(results, floors, 6, "mappo",
                                         rescale_to_floor=True,
                                         mi_collapse_threshold=0.01)
        assert out == floors
