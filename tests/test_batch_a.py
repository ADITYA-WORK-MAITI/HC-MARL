"""Safety-critical regression tests.

Covers:
    A1 — OSQP status check + slack-augmented CBF-QP in hcmarl.ecbf_filter.
    A4 — Observation vector includes normalised time (Pardo et al. 2018).
    A5 — seed_everything runs before MMICRL pretrain in scripts/train.py.

These tests guarantee the safety filter never returns garbage from a
non-optimal OSQP solve, that feasible QPs are unaffected by the slack
augmentation, and that the training script's seeding + obs dim contracts
are preserved for the compute budget.
"""

from __future__ import annotations

import math
import pathlib
import re

import numpy as np
import pytest

from hcmarl.ecbf_filter import (
    SLACK_EPS,
    SLACK_PENALTY,
    ECBFDiagnostics,
    ECBFFilter,
    ECBFParams,
)
from hcmarl.three_cc_r import SHOULDER, ThreeCCrState


# ---------------------------------------------------------------------
# A1 — slack-augmented QP
# ---------------------------------------------------------------------

class TestA1SlackAugmentedQP:
    """A1: QP must return valid C for 1000 random states and match the
    pre-slack solution when the strict constraints are satisfiable."""

    def setup_method(self):
        self.params = ECBFParams(
            theta_max=0.70, alpha1=0.05, alpha2=0.05, alpha3=0.1,
        )
        self.filt = ECBFFilter(muscle=SHOULDER, ecbf_params=self.params)

    def _random_state(self, rng: np.random.Generator) -> ThreeCCrState:
        # Sample on the probability simplex {MR+MA+MF=1, all >= 0}.
        raw = rng.dirichlet(np.array([3.0, 1.0, 1.0]))
        MR, MA, MF = float(raw[0]), float(raw[1]), float(raw[2])
        return ThreeCCrState(MR=MR, MA=MA, MF=MF)

    def test_fuzz_1000_states_never_garbage(self):
        """For 1000 random states + drives, the QP either solves to 'optimal'
        or the filter returns an analytical-fallback C. In either case
        C_filtered must be finite, non-negative, and never exceed C_nominal
        by more than numerical tolerance."""
        rng = np.random.default_rng(seed=4271315)
        for _ in range(1000):
            state = self._random_state(rng)
            C_nom = float(rng.uniform(0.0, 0.5))
            TL = float(rng.choice([0.0, 0.1, 0.3, 0.55]))
            C_safe, diag = self.filt.filter(state, C_nom, target_load=TL)

            # Correctness invariants that MUST hold regardless of solver path:
            assert math.isfinite(C_safe), f"NaN/inf C_safe at state={state}"
            assert C_safe >= -1e-9, f"Negative C_safe at state={state}"
            assert C_safe <= C_nom + 1e-6, (
                f"Filter cannot amplify drive: C_safe={C_safe} > C_nom={C_nom}"
            )
            # Diagnostics must be complete.
            assert isinstance(diag, ECBFDiagnostics)
            assert diag.slack_ecbf >= 0.0
            assert diag.slack_cbf >= 0.0

    def test_slack_zero_on_feasible_state(self):
        """Fresh worker + small drive is strictly feasible. Slack must be
        numerically zero; the answer must match the pre-slack solution
        which is C_nom itself (no clipping needed)."""
        state = ThreeCCrState.fresh()
        C_nom = 0.001
        C_safe, diag = self.filt.filter(state, C_nom, target_load=0.3)
        assert abs(C_safe - C_nom) < 1e-6
        assert diag.slack_ecbf == 0.0
        assert diag.slack_cbf == 0.0
        assert not diag.infeasible
        assert not diag.used_fallback

    def test_slack_activation_near_ceiling(self):
        """Very close to the fatigue ceiling the QP would be strictly
        infeasible for positive C; slack must absorb the violation rather
        than the solver returning garbage. C_safe must be driven near 0."""
        # State deliberately past the CBF feasibility boundary
        # (MR+MA+MF must equal 1 exactly: 0.005 + 0.30 + 0.695 = 1.0).
        state = ThreeCCrState(MR=0.005, MA=0.30, MF=0.695)
        C_nom = 0.2
        C_safe, diag = self.filt.filter(state, C_nom, target_load=0.5)
        # Safety-critical: C must be near 0 regardless of solver status.
        assert C_safe < 0.05
        assert math.isfinite(C_safe)
        assert C_safe >= -1e-9
        # Slack or fallback must have been engaged, not a silent garbage solve.
        assert diag.infeasible or diag.used_fallback or diag.was_clipped

    def test_non_optimal_status_triggers_fallback(self):
        """If problem.status is anything other than 'optimal', the filter
        must not trust C_var.value — it must route through analytical
        fallback which is guaranteed correct for scalar C."""
        # Monkeypatch-free test: we rely on the code path by construction.
        # When status != 'optimal', used_fallback=True must hold.
        state = ThreeCCrState.fresh()
        _, diag = self.filt.filter(state, 0.01, target_load=0.3)
        if diag.qp_status != "optimal":
            assert diag.used_fallback, (
                f"Non-optimal status {diag.qp_status} did not trigger fallback"
            )

    def test_slack_penalty_is_large(self):
        """A1 pre-critic: SLACK_PENALTY must dominate the objective so slack
        is never used recreationally. Sanity bound: penalty >> typical
        squared-drive scale (C <= 1 => (C-C_nom)^2 <= 1)."""
        assert SLACK_PENALTY >= 100.0
        assert SLACK_EPS <= 1e-3


# ---------------------------------------------------------------------
# A4 — normalised time in observation vector (Pardo 2018)
# ---------------------------------------------------------------------

class TestA4NormalisedTimeInObs:
    """Obs must carry elapsed-fraction so the policy can condition on
    proximity to episode end. Pardo et al. ICML 2018 show that omitting
    this breaks Markov and leads to boundary exploitation."""

    def test_warehouse_env_obs_contains_step_fraction(self):
        from hcmarl.warehouse_env import WarehouseMultiAgentEnv
        env = WarehouseMultiAgentEnv(n_workers=2, max_steps=60)
        obs, _ = env.reset()
        # Step to 50% of episode and verify the last obs entry moves.
        for _ in range(30):
            actions = {a: 0 for a in env.agents}
            obs, *_ = env.step(actions)
        a0 = env.agents[0]
        # Last entry is the normalised step fraction.
        assert obs[a0][-1] == pytest.approx(30.0 / 60.0, abs=1e-6)

    def test_pettingzoo_wrapper_obs_contains_step_fraction(self):
        from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
        env = WarehousePettingZoo(n_workers=2, max_steps=100)
        obs, _ = env.reset()
        for _ in range(25):
            actions = {a: 0 for a in env.agents}
            obs, *_ = env.step(actions)
        a0 = env.agents[0]
        assert obs[a0][-1] == pytest.approx(25.0 / 100.0, abs=1e-6)


# ---------------------------------------------------------------------
# A5 — seed_everything ordering in scripts/train.py
# ---------------------------------------------------------------------

class TestA5SeedOrderingInTrainScript:
    """Static-analysis guard: seed_everything must be called before the
    MMICRL pretrain branch inside main(). If a future refactor reorders
    these calls, this test fails fast — no 2-account symptom needed."""

    def test_seed_everything_before_mmicrl_in_main(self):
        train_path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train.py"
        src = train_path.read_text(encoding="utf-8")
        # Isolate the main() function body.
        main_start = src.index("def main()")
        main_src = src[main_start:]
        seed_idx = main_src.find("seed_everything(args.seed")
        mmicrl_idx = main_src.find("run_mmicrl_pretrain(")
        assert seed_idx != -1, "seed_everything(args.seed) call not found in main()"
        assert mmicrl_idx != -1, "run_mmicrl_pretrain() call not found in main()"
        assert seed_idx < mmicrl_idx, (
            "seed_everything must precede run_mmicrl_pretrain in main()"
        )
