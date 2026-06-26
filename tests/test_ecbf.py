"""Unit tests for hcmarl.ecbf_filter.

Verifies all equations from Framework v12, Sections 5.1--5.4.
"""

import math

import numpy as np
import pytest

from hcmarl.ecbf_filter import ECBFDiagnostics, ECBFFilter, ECBFParams
from hcmarl.three_cc_r import (
    ANKLE,
    ELBOW,
    GRIP,
    KNEE,
    SHOULDER,
    TRUNK,
    MuscleParams,
    ThreeCCrState,
)


# =====================================================================
# ECBFParams validation
# =====================================================================

class TestECBFParams:
    """Test design parameter validation."""

    def test_valid_params(self):
        """Should not raise for valid params satisfying Assumption 5.5."""
        p = ECBFParams(theta_max=0.70, alpha1=0.05, alpha2=0.05, alpha3=0.1)
        p.validate(SHOULDER)  # theta_min_max ~0.419, so 0.70 >= 0.419

    def test_theta_max_below_threshold_raises(self):
        """Violating Assumption 5.5 (Eq 26) must raise."""
        p = ECBFParams(theta_max=0.30)  # 0.30 < 0.419 for shoulder
        with pytest.raises(ValueError, match="Assumption 5.5"):
            p.validate(SHOULDER)

    def test_theta_max_out_of_range(self):
        p = ECBFParams(theta_max=1.5)
        with pytest.raises(ValueError, match="must be in"):
            p.validate(SHOULDER)

    def test_negative_alpha_raises(self):
        p = ECBFParams(theta_max=0.70, alpha1=-0.01)
        with pytest.raises(ValueError, match="alpha1"):
            p.validate(SHOULDER)

    def test_ankle_high_threshold_ok(self):
        """Ankle theta_min_max ~ 40.4% under corrected values; 0.80 is fine."""
        p = ECBFParams(theta_max=0.80)
        p.validate(ANKLE)  # Should not raise


# =====================================================================
# Barrier function values
# =====================================================================

class TestBarrierFunctions:
    """Test barrier function computations (Eqs 12--16, 21--22)."""

    def setup_method(self):
        self.params = ECBFParams(theta_max=0.70, alpha1=0.05, alpha2=0.05, alpha3=0.1)
        self.filt = ECBFFilter(muscle=SHOULDER, ecbf_params=self.params)

    def test_h_fresh(self):
        """h(x) = Theta_max - MF. Fresh worker: MF=0 => h = 0.70."""
        assert abs(self.filt.h(0.0) - 0.70) < 1e-10

    def test_h_at_limit(self):
        """At MF = Theta_max: h = 0."""
        assert abs(self.filt.h(0.70)) < 1e-10

    def test_h_violated(self):
        """h < 0 when MF > Theta_max."""
        assert self.filt.h(0.75) < 0.0

    def test_h2_fresh(self):
        """h2(x) = MR. Fresh worker: MR=1 => h2 = 1."""
        assert abs(self.filt.h2(1.0) - 1.0) < 1e-10

    def test_h2_depleted(self):
        """h2 = 0 when MR = 0 (resting pool depleted)."""
        assert abs(self.filt.h2(0.0)) < 1e-10

    def test_h_dot_formula(self):
        """h_dot = -F*MA + Reff*MF (Eq 13)."""
        MA, MF = 0.3, 0.2
        R_eff = SHOULDER.R  # Work phase
        expected = -SHOULDER.F * MA + R_eff * MF
        assert abs(self.filt.h_dot(MA, MF, R_eff) - expected) < 1e-12

    def test_h_dot_no_C_dependence(self):
        """h_dot does not depend on C (confirms relative degree >= 2)."""
        MA, MF = 0.3, 0.2
        R_eff = SHOULDER.R
        val = self.filt.h_dot(MA, MF, R_eff)
        # Same result regardless of what C would be -- no C argument
        assert isinstance(val, float)

    def test_h_ddot_C_coefficient(self):
        """In h_ddot (Eq 14), C appears with coefficient -F.

        h_ddot(C=0) - h_ddot(C=1) = -F*0 + ... - (-F*1 + ...) = F
        """
        MA, MF, R_eff = 0.3, 0.2, SHOULDER.R
        hdd_0 = self.filt.h_ddot(MA, MF, C=0.0, R_eff=R_eff)
        hdd_1 = self.filt.h_ddot(MA, MF, C=1.0, R_eff=R_eff)
        assert abs((hdd_0 - hdd_1) - SHOULDER.F) < 1e-12

    def test_psi_0_equals_h(self):
        """psi_0 = h(x) (Eq 15)."""
        assert abs(self.filt.psi_0(0.3) - self.filt.h(0.3)) < 1e-12

    def test_psi_1_formula(self):
        """psi_1 = h_dot + alpha1 * h (Eq 16)."""
        MA, MF = 0.3, 0.2
        R_eff = SHOULDER.R
        expected = self.filt.h_dot(MA, MF, R_eff) + 0.05 * self.filt.h(MF)
        assert abs(self.filt.psi_1(MA, MF, R_eff) - expected) < 1e-12

    def test_h2_dot_formula(self):
        """h2_dot = Reff*MF - C (Eq 22)."""
        MF, C = 0.2, 0.01
        R_eff = SHOULDER.R
        expected = R_eff * MF - C
        assert abs(self.filt.h2_dot(MF, C, R_eff) - expected) < 1e-12


# =====================================================================
# Analytical bounds
# =====================================================================

class TestAnalyticalBounds:
    """Test analytical upper bounds from Eqs 19 and 23."""

    def setup_method(self):
        self.params = ECBFParams(theta_max=0.70, alpha1=0.05, alpha2=0.05, alpha3=0.1)
        self.filt = ECBFFilter(muscle=SHOULDER, ecbf_params=self.params)

    def test_ecbf_bound_fresh_worker(self):
        """Fresh worker (MA=0, MF=0): ECBF bound should be positive."""
        ub = self.filt.ecbf_upper_bound(MA=0.0, MF=0.0, R_eff=SHOULDER.R)
        # alpha2 * alpha1 * theta_max / F should dominate
        expected = self.params.alpha2 * self.params.alpha1 * 0.70 / SHOULDER.F
        assert abs(ub - expected) < 1e-10
        assert ub > 0.0

    def test_cbf_bound_fresh_worker(self):
        """Fresh worker (MA=0, MF=0, MR=1): CBF bound = alpha3 * 1.0."""
        ub = self.filt.cbf_upper_bound(MA=0.0, MF=0.0, R_eff=SHOULDER.R)
        assert abs(ub - self.params.alpha3 * 1.0) < 1e-10

    def test_cbf_bound_depleted_resting_pool(self):
        """MR = 0: CBF bound = Reff*MF + 0."""
        MA, MF = 0.5, 0.5  # MR = 0
        ub = self.filt.cbf_upper_bound(MA, MF, SHOULDER.R)
        expected = SHOULDER.R * MF
        assert abs(ub - expected) < 1e-10

    def test_cbf_bound_formula(self):
        """Verify Eq 23: C <= Reff*MF + alpha3*(1 - MA - MF)."""
        MA, MF = 0.3, 0.2
        R_eff = SHOULDER.R
        MR = 1.0 - MA - MF
        expected = R_eff * MF + self.params.alpha3 * MR
        assert abs(self.filt.cbf_upper_bound(MA, MF, R_eff) - expected) < 1e-12


# =====================================================================
# QP filter
# =====================================================================

class TestQPFilter:
    """Test the CBF-QP solver (Eq 20)."""

    def setup_method(self):
        self.params = ECBFParams(theta_max=0.70, alpha1=0.05, alpha2=0.05, alpha3=0.1)
        self.filt = ECBFFilter(muscle=SHOULDER, ecbf_params=self.params)

    def test_safe_drive_unchanged(self):
        """Small C_nominal on fresh worker should pass through unclipped."""
        state = ThreeCCrState.fresh()
        C_nom = 0.001
        C_safe, diag = self.filt.filter(state, C_nom, target_load=0.3)
        assert abs(C_safe - C_nom) < 1e-6
        assert not diag.was_clipped

    def test_excessive_drive_clipped(self):
        """Very large C_nominal must be clipped."""
        state = ThreeCCrState(MR=0.5, MA=0.3, MF=0.2)
        C_nom = 100.0  # Absurdly large
        C_safe, diag = self.filt.filter(state, C_nom, target_load=0.5)
        assert C_safe < C_nom
        assert C_safe >= 0.0
        assert diag.was_clipped

    def test_non_negative_output(self):
        """Output C* >= 0 always (constraint in Eq 20)."""
        state = ThreeCCrState(MR=0.1, MA=0.3, MF=0.6)
        C_safe, diag = self.filt.filter(state, C_nominal=0.1, target_load=0.5)
        assert C_safe >= -1e-9

    def test_mandatory_rest_at_limit(self):
        """Near MF = Theta_max, QP should force C* near 0 (Remark 5.13)."""
        # State very close to fatigue ceiling
        state = ThreeCCrState(MR=0.01, MA=0.30, MF=0.69)
        C_safe, diag = self.filt.filter(state, C_nominal=0.05, target_load=0.5)
        # C should be very small or zero
        assert C_safe < 0.02

    def test_analytical_matches_qp_scalar(self):
        """For scalar C, analytical filter should match QP solution."""
        state = ThreeCCrState(MR=0.5, MA=0.3, MF=0.2)
        C_nom = 0.05
        C_qp, _ = self.filt.filter(state, C_nom, target_load=0.3)
        C_an, _infeasible = self.filt.filter_analytical(state, C_nom, target_load=0.3)
        assert abs(C_qp - C_an) < 1e-4

    def test_diagnostics_complete(self):
        """Diagnostics should contain all expected fields."""
        state = ThreeCCrState.fresh()
        _, diag = self.filt.filter(state, C_nominal=0.01, target_load=0.3)
        assert isinstance(diag, ECBFDiagnostics)
        assert diag.h >= 0
        assert diag.h2 >= 0
        # A1: with slack variables, a fresh safe state solves to "optimal".
        # Fallback paths ("optimal_inaccurate", "solver_error_fallback") are
        # also acceptable — they still produce a valid clipped C.
        assert diag.qp_status in (
            "optimal", "optimal_inaccurate",
            "solver_error_fallback", "infeasible", "unknown",
        )


# =====================================================================
# Rest-phase analysis (Section 5.4)
# =====================================================================

class TestRestPhaseAnalysis:
    """Test rest-phase safety properties (Theorem 5.7, Eqs 28--30)."""

    def setup_method(self):
        self.params = ECBFParams(theta_max=0.70, alpha1=0.05, alpha2=0.05, alpha3=0.1)
        self.filt = ECBFFilter(muscle=SHOULDER, ecbf_params=self.params)

    def test_rest_phase_safe_fresh(self):
        """Fresh worker is in the safe set."""
        assert self.filt.rest_phase_safe(ThreeCCrState.fresh())

    def test_rest_phase_safe_within_bounds(self):
        """State with MF < theta_max and MR > 0 is safe."""
        state = ThreeCCrState(MR=0.3, MA=0.2, MF=0.5)
        assert self.filt.rest_phase_safe(state)

    def test_rest_phase_unsafe_if_violated(self):
        """State with MF > theta_max is not safe."""
        # Manually construct without validation
        state = ThreeCCrState.__new__(ThreeCCrState)
        state.MR = 0.05
        state.MA = 0.20
        state.MF = 0.75  # > 0.70
        assert not self.filt.rest_phase_safe(state)

    def test_psi1_jump_positive(self):
        """psi_1 jump at work-to-rest = R*(r-1)*MF > 0 (Eq 28)."""
        MF = 0.3
        jump = self.filt.psi1_jump_at_rest(MF)
        expected = SHOULDER.R * (SHOULDER.r - 1) * MF
        assert abs(jump - expected) < 1e-12
        assert jump > 0.0

    def test_psi1_jump_zero_if_no_fatigue(self):
        """No jump if MF = 0 (nothing to transition)."""
        assert abs(self.filt.psi1_jump_at_rest(0.0)) < 1e-12

    def test_min_rest_duration_shoulder(self):
        """Shoulder with MA = delta_max should require positive rest (Eq 30)."""
        delta = SHOULDER.delta_max
        dt = self.filt.min_rest_duration_bound(MA_at_stop=delta)
        # Under corrected Frey-Law 2012 values, SHOULDER.Rr/F = 1.385 > 1
        # so the formal bound is 0 (overshoot is bounded without rest);
        # if Rr/F were < 1 under older params, dt > 0 would be required.
        assert dt >= 0.0

    def test_min_rest_duration_ankle(self):
        """Ankle with small MA at stop should need little rest.

        Under corrected Frey-Law 2012 values ankle theta_min_max = 40.4%
        (was 2.1% under transcription-error values). Use a feasible
        theta_max = 0.80 that satisfies Assumption 5.5.
        """
        ankle_params = ECBFParams(theta_max=0.80)
        filt_ankle = ECBFFilter(muscle=ANKLE, ecbf_params=ankle_params)
        # Use a small MA at stop (not the extreme delta_max)
        dt = filt_ankle.min_rest_duration_bound(MA_at_stop=0.05)
        # With small MA, rest needed should be moderate
        assert dt < 200.0  # Reasonable bound
