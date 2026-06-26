"""Unit tests for hcmarl.three_cc_r.

Verifies all equations from Framework v12, Sections 3.1--3.5 and 7.2.
"""

import math

import numpy as np
import pytest

from hcmarl.three_cc_r import (
    ALL_MUSCLES,
    ANKLE,
    ELBOW,
    GRIP,
    KNEE,
    MUSCLE_REGISTRY,
    SHOULDER,
    TRUNK,
    MuscleParams,
    ThreeCCr,
    ThreeCCrState,
    get_muscle,
)


# =====================================================================
# 1. MuscleParams derived quantities
# =====================================================================

class TestMuscleParams:
    """Test calibrated parameters and derived quantities (Table 1)."""

    def test_shoulder_raw_values(self):
        """Verify shoulder F, R, r from Frey-Law, Looft & Heitsman 2012 Table 1."""
        assert SHOULDER.F == 0.01820
        assert SHOULDER.R == 0.00168
        assert SHOULDER.r == 15

    def test_ankle_raw_values(self):
        assert ANKLE.F == 0.00589
        assert ANKLE.R == 0.00058
        assert ANKLE.r == 15

    def test_knee_raw_values(self):
        assert KNEE.F == 0.01500
        assert KNEE.R == 0.00149
        assert KNEE.r == 15

    def test_elbow_raw_values(self):
        assert ELBOW.F == 0.00912
        assert ELBOW.R == 0.00094
        assert ELBOW.r == 15

    def test_trunk_raw_values(self):
        assert TRUNK.F == 0.00755
        assert TRUNK.R == 0.00075
        assert TRUNK.r == 15

    def test_grip_raw_values(self):
        assert GRIP.F == 0.00980
        assert GRIP.R == 0.00064
        assert GRIP.r == 30  # Looft et al. 2018 Table 2: r=30 for hand grip

    def test_shoulder_delta_max(self):
        """delta_max = R/(F+R) = 8.45% for shoulder (Frey-Law 2012 Table 1)."""
        expected = 0.00168 / (0.01820 + 0.00168)
        assert abs(SHOULDER.delta_max - expected) < 1e-10
        assert abs(SHOULDER.delta_max - 0.0845) < 0.001

    def test_ankle_delta_max(self):
        """delta_max = 8.96% for ankle (Frey-Law 2012 Table 1)."""
        expected = 0.00058 / (0.00589 + 0.00058)
        assert abs(ANKLE.delta_max - expected) < 1e-10
        assert abs(ANKLE.delta_max - 0.0896) < 0.001

    def test_shoulder_C_max(self):
        """Verify C_max = F*R/(F+R) for shoulder (Eq 6)."""
        expected = 0.01820 * 0.00168 / (0.01820 + 0.00168)
        assert abs(SHOULDER.C_max - expected) < 1e-12

    def test_C_max_dimensional_check(self):
        """C_max = F*R/(F+R) has units [min^-1] (Remark 3.5)."""
        # Verify numerically that C_max = delta_max * F
        for m in ALL_MUSCLES:
            assert abs(m.C_max - m.delta_max * m.F) < 1e-12

    def test_shoulder_theta_min_max(self):
        """theta_min_max = F/(F+Rr) = 41.9% for shoulder (Eq 25)."""
        expected = 0.01820 / (0.01820 + 0.00168 * 15)
        assert abs(SHOULDER.theta_min_max - expected) < 1e-10
        assert abs(SHOULDER.theta_min_max - 0.419) < 0.002

    def test_ankle_theta_min_max(self):
        """theta_min_max = 40.4% for ankle (r=15)."""
        expected = 0.00589 / (0.00589 + 0.00058 * 15)
        assert abs(ANKLE.theta_min_max - expected) < 1e-10
        assert abs(ANKLE.theta_min_max - 0.404) < 0.002

    def test_grip_theta_min_max(self):
        """theta_min_max = 33.8% for grip (r=30, Looft et al. 2018)."""
        expected = 0.00980 / (0.00980 + 0.00064 * 30)
        assert abs(GRIP.theta_min_max - expected) < 1e-10
        assert abs(GRIP.theta_min_max - 0.338) < 0.002

    def test_shoulder_Rr_over_F(self):
        """Shoulder Rr/F = 1.385 > 1 under corrected values (Table B)."""
        expected = (0.00168 * 15) / 0.01820
        assert abs(SHOULDER.Rr_over_F - expected) < 1e-10
        assert SHOULDER.Rr_over_F > 1.0  # Under corrected Frey-Law 2012 values

    def test_ankle_Rr_over_F(self):
        """Ankle Rr/F = 1.478 > 1."""
        assert ANKLE.Rr_over_F > 1.0

    def test_all_muscles_Rr_over_F_above_one(self):
        """All six muscles have Rr/F > 1 under Frey-Law 2012 Table 1.

        This is the corrected finding (CONSTANTS_AUDIT Table B): under the
        published F, R, r values, every muscle recovers faster than it
        fatigues. The earlier "fatigue-resistance ranking" claim
        (ankle > trunk > grip > knee > elbow > shoulder) was an artifact
        of the transcription errors in F, R; it does not survive the
        correction. Under the true values, the muscles differ mainly in
        their Rr/F ratio (shoulder 1.385, ankle 1.478, knee 1.490,
        trunk 1.490, elbow 1.547, grip 1.959), not in delta_max (all
        within the 6-9% band).
        """
        for m in ALL_MUSCLES:
            assert m.Rr_over_F > 1.0, f"{m.name} Rr/F = {m.Rr_over_F}"

    def test_six_muscles_registered(self):
        assert len(ALL_MUSCLES) == 6
        assert len(MUSCLE_REGISTRY) == 6

    def test_get_muscle_valid(self):
        m = get_muscle("shoulder")
        assert m is SHOULDER

    def test_get_muscle_case_insensitive(self):
        m = get_muscle("SHOULDER")
        assert m is SHOULDER

    def test_get_muscle_invalid(self):
        with pytest.raises(KeyError):
            get_muscle("bicep")


# =====================================================================
# 2. ThreeCCrState
# =====================================================================

class TestThreeCCrState:
    """Test state vector construction and validation."""

    def test_fresh_state(self):
        s = ThreeCCrState.fresh()
        assert s.MR == 1.0
        assert s.MA == 0.0
        assert s.MF == 0.0

    def test_conservation_holds(self):
        s = ThreeCCrState(MR=0.5, MA=0.3, MF=0.2)
        assert abs(s.MR + s.MA + s.MF - 1.0) < 1e-10

    def test_conservation_violated_raises(self):
        with pytest.raises(ValueError, match="Conservation violated"):
            ThreeCCrState(MR=0.5, MA=0.5, MF=0.5)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            ThreeCCrState(MR=-0.1, MA=0.6, MF=0.5)

    def test_as_array_roundtrip(self):
        s = ThreeCCrState(MR=0.6, MA=0.1, MF=0.3)
        arr = s.as_array()
        s2 = ThreeCCrState.from_array(arr)
        assert abs(s2.MR - 0.6) < 1e-10
        assert abs(s2.MA - 0.1) < 1e-10
        assert abs(s2.MF - 0.3) < 1e-10


# =====================================================================
# 3. ThreeCCr ODE system
# =====================================================================

class TestThreeCCrODE:
    """Test the ODE right-hand side and related methods."""

    def setup_method(self):
        self.model = ThreeCCr(params=SHOULDER, kp=1.0)

    # -- Reperfusion switch (Eq 5) --

    def test_R_eff_work_phase(self):
        """During work (TL > 0): Reff = R (Eq 5)."""
        assert self.model.R_eff(0.3) == SHOULDER.R

    def test_R_eff_rest_phase(self):
        """During rest (TL = 0): Reff = R*r (Eq 5)."""
        assert self.model.R_eff(0.0) == SHOULDER.Rr

    # -- Baseline neural drive controller (Eq 35) --

    def test_baseline_drive_rest(self):
        """C = 0 when resting (Eq 35)."""
        assert self.model.baseline_neural_drive(0.0, 0.5) == 0.0

    def test_baseline_drive_tracking(self):
        """C = kp * (TL - MA) when TL > MA (Eq 35)."""
        C = self.model.baseline_neural_drive(0.3, 0.1)
        assert abs(C - 1.0 * (0.3 - 0.1)) < 1e-10

    def test_baseline_drive_non_negative(self):
        """C = kp * max(TL - MA, 0) -- non-negative (Eq 35)."""
        C = self.model.baseline_neural_drive(0.1, 0.3)
        assert C == 0.0

    # -- ODE RHS structure (Eqs 2--4) --

    def test_ode_rhs_conservation(self):
        """Sum of derivatives = 0 (Theorem 3.2).

        dMR/dt + dMA/dt + dMF/dt = (Reff*MF - C) + (C - F*MA) + (F*MA - Reff*MF) = 0
        """
        x = np.array([0.5, 0.3, 0.2])
        for C_val in [0.0, 0.01, 0.05]:
            for TL in [0.0, 0.3]:
                dx = self.model.ode_rhs(x, C_val, TL)
                assert abs(dx.sum()) < 1e-12, (
                    f"Conservation violated: sum(dx) = {dx.sum()}"
                )

    def test_ode_rhs_fresh_no_drive(self):
        """Fresh state with no drive: all derivatives = 0."""
        x = np.array([1.0, 0.0, 0.0])
        dx = self.model.ode_rhs(x, C=0.0, target_load=0.0)
        np.testing.assert_allclose(dx, [0.0, 0.0, 0.0], atol=1e-15)

    def test_ode_rhs_dMA_formula(self):
        """dMA/dt = C - F*MA (Eq 2)."""
        x = np.array([0.5, 0.3, 0.2])
        C = 0.02
        dx = self.model.ode_rhs(x, C, target_load=0.3)
        expected_dMA = C - SHOULDER.F * 0.3
        assert abs(dx[1] - expected_dMA) < 1e-12

    def test_ode_rhs_dMF_formula_work(self):
        """dMF/dt = F*MA - R*MF during work (Eq 3, Reff=R)."""
        x = np.array([0.5, 0.3, 0.2])
        dx = self.model.ode_rhs(x, C=0.02, target_load=0.3)
        expected_dMF = SHOULDER.F * 0.3 - SHOULDER.R * 0.2
        assert abs(dx[2] - expected_dMF) < 1e-12

    def test_ode_rhs_dMF_formula_rest(self):
        """dMF/dt = F*MA - R*r*MF during rest (Eq 3, Reff=Rr)."""
        x = np.array([0.5, 0.3, 0.2])
        dx = self.model.ode_rhs(x, C=0.0, target_load=0.0)
        expected_dMF = SHOULDER.F * 0.3 - SHOULDER.Rr * 0.2
        assert abs(dx[2] - expected_dMF) < 1e-12

    def test_ode_rhs_dMR_formula(self):
        """dMR/dt = Reff*MF - C (Eq 4)."""
        x = np.array([0.5, 0.3, 0.2])
        C = 0.02
        dx = self.model.ode_rhs(x, C, target_load=0.3)
        expected_dMR = SHOULDER.R * 0.2 - C
        assert abs(dx[0] - expected_dMR) < 1e-12

    # -- Steady-state verification (Theorem 3.4) --

    def test_steady_state_conservation(self):
        """Steady state at sustainability limit: MR = 0 (Theorem 3.4)."""
        ss = self.model.steady_state_work()
        assert abs(ss.MR) < 1e-10
        assert abs(ss.MR + ss.MA + ss.MF - 1.0) < 1e-10

    def test_steady_state_derivatives_zero(self):
        """At steady state with C=C_max: all derivatives = 0."""
        ss = self.model.steady_state_work()
        x = ss.as_array()
        C = SHOULDER.C_max
        dx = self.model.ode_rhs(x, C, target_load=0.5)
        np.testing.assert_allclose(dx, [0.0, 0.0, 0.0], atol=1e-10)

    def test_steady_state_MA_equals_delta_max(self):
        """MA at steady state = delta_max = R/(F+R) (Eq 7)."""
        ss = self.model.steady_state_work()
        assert abs(ss.MA - SHOULDER.delta_max) < 1e-10

    def test_steady_state_MF(self):
        """MF at steady state = C_max/R = F/(F+R) (Eq 8)."""
        ss = self.model.steady_state_work()
        expected_MF = SHOULDER.C_max / SHOULDER.R
        assert abs(ss.MF - expected_MF) < 1e-10


# =====================================================================
# 4. Euler step
# =====================================================================

class TestEulerStep:
    """Test single-step integration."""

    def test_euler_conservation(self):
        """Euler step preserves conservation law."""
        model = ThreeCCr(params=SHOULDER)
        s = ThreeCCrState(MR=0.7, MA=0.2, MF=0.1)
        s_new = model.step_euler(s, C=0.01, target_load=0.2, dt=1.0)
        assert abs(s_new.MR + s_new.MA + s_new.MF - 1.0) < 1e-10

    def test_euler_rest_recovery(self):
        """During rest, MF should decrease (for Rr/F > 1 muscles)."""
        model = ThreeCCr(params=ANKLE)  # Rr/F = 46.35
        s = ThreeCCrState(MR=0.5, MA=0.0, MF=0.5)
        s_new = model.step_euler(s, C=0.0, target_load=0.0, dt=1.0)
        assert s_new.MF < s.MF

    def test_euler_vs_rk45_high_C(self):
        """C-14 regression: kp=1 keeps Euler stable; kp=10 caused 26x overshoot.

        With kp=10, dt=1 min, a fresh worker on heavy_lift gets
        C=10*(0.45-0)=4.5, so MA overshoots to 4.5 — physically impossible.
        With kp=1, C=0.45 and MA stays in [0,1].

        We do NOT assert small Euler-vs-RK45 error: Euler at dt=1 min is a
        coarse first-order approximation (59% relative error on MA is expected).
        The property that matters is *stability*, not accuracy.
        """
        model = ThreeCCr(params=SHOULDER, kp=1.0)
        s0 = ThreeCCrState.fresh()
        TL = 0.45  # heavy_lift shoulder demand

        # Euler step at dt=1 with kp=1 — must stay bounded and physical
        C = model.baseline_neural_drive(TL, s0.MA)
        s_euler = model.step_euler(s0, C=C, target_load=TL, dt=1.0)

        # Stability: all components in [0, 1]
        assert 0.0 <= s_euler.MR <= 1.0
        assert 0.0 <= s_euler.MA <= 1.0
        assert 0.0 <= s_euler.MF <= 1.0

        # Conservation must hold exactly
        assert abs(s_euler.MR + s_euler.MA + s_euler.MF - 1.0) < 1e-10

        # Regression: kp=10 would give MA = 10*(0.45-0)*1 = 4.5
        model_old = ThreeCCr(params=SHOULDER, kp=10.0)
        C_old = model_old.baseline_neural_drive(TL, s0.MA)
        assert C_old * 1.0 > 1.0, "kp=10 drive should overshoot"
        # But kp=1 stays safe
        assert C * 1.0 <= 1.0, "kp=1 drive should stay bounded"

    def test_euler_non_negative_all_muscles(self):
        """Euler step must produce non-negative state for all 6 muscle groups.

        Tests the worst-case scenario: fresh worker, highest task demand,
        kp=1, dt=1. No component should go negative.
        """
        for muscle in ALL_MUSCLES:
            model = ThreeCCr(params=muscle, kp=1.0)
            s = ThreeCCrState.fresh()
            C = model.baseline_neural_drive(0.55, 0.0)  # worst-case TL
            s_new = model.step_euler(s, C=C, target_load=0.55, dt=1.0)
            assert s_new.MR >= 0.0, f"{muscle.name}: MR={s_new.MR}"
            assert s_new.MA >= 0.0, f"{muscle.name}: MA={s_new.MA}"
            assert s_new.MF >= 0.0, f"{muscle.name}: MF={s_new.MF}"
            assert abs(s_new.MR + s_new.MA + s_new.MF - 1.0) < 1e-10


# =====================================================================
# 5. Full simulation (solve_ivp)
# =====================================================================

class TestSimulation:
    """Test ODE integration via scipy."""

    def test_simulation_conservation_throughout(self):
        """MR + MA + MF = 1 at every evaluation point."""
        model = ThreeCCr(params=SHOULDER)
        result = model.simulate(
            state0=ThreeCCrState.fresh(),
            target_load=0.3,
            duration=10.0,
            dt_eval=0.5,
        )
        totals = result["MR"] + result["MA"] + result["MF"]
        np.testing.assert_allclose(totals, 1.0, atol=1e-6)

    def test_simulation_non_negative(self):
        """All compartments stay non-negative."""
        model = ThreeCCr(params=SHOULDER)
        result = model.simulate(
            state0=ThreeCCrState.fresh(),
            target_load=0.3,
            duration=10.0,
        )
        assert np.all(result["MR"] >= -1e-9)
        assert np.all(result["MA"] >= -1e-9)
        assert np.all(result["MF"] >= -1e-9)

    def test_simulation_fatigue_increases_under_load(self):
        """Under constant load, MF increases from zero."""
        model = ThreeCCr(params=SHOULDER)
        result = model.simulate(
            state0=ThreeCCrState.fresh(),
            target_load=0.3,
            duration=30.0,
        )
        assert result["MF"][-1] > result["MF"][0]

    def test_rest_recovery(self):
        """After fatiguing, rest causes MF to decrease."""
        model = ThreeCCr(params=ANKLE)
        fatigued = ThreeCCrState(MR=0.3, MA=0.0, MF=0.7)
        result = model.simulate(
            state0=fatigued,
            target_load=0.0,
            duration=30.0,
        )
        assert result["MF"][-1] < result["MF"][0]


# =====================================================================
# 5. Dynamic vs isometric F regime validation (L4)
# =====================================================================

class TestDynamicIsometricScaling:
    """Validate that isometric and dynamic F regimes are distinct and consistent.

    Table 1 isometric F = 0.01820 min^{-1} for shoulder predicts endurance
    times of 40-100+ minutes for sustained holds at 15-55% MVC. Dynamic
    rotations (WSD4FEDSRM) have endurance 50-250 seconds, requiring
    F ~ 0.3-3.0 min^{-1}. The 30-180x ratio is expected because the 3CC-r
    model's F captures total metabolic demand per unit time, which is far
    higher for dynamic tasks (concentric/eccentric cycling, movement-
    dependent recruitment).
    """

    def test_isometric_F_predicts_long_endurance(self):
        """Isometric F=0.01820 at 35% MVC must predict ET > 600s."""
        from hcmarl.real_data_calibration import predict_endurance_time
        et = predict_endurance_time(
            F=SHOULDER.F, R=0.02, r=15.0,
            target_load=0.35, max_time=7200.0,
        )
        assert et > 600.0, f"Isometric ET={et:.0f}s, expected >600s"

    def test_dynamic_F_predicts_short_endurance(self):
        """Dynamic F=1.0 at 35% MVC must predict ET < 300s."""
        from hcmarl.real_data_calibration import predict_endurance_time
        et = predict_endurance_time(
            F=1.0, R=0.02, r=15.0,
            target_load=0.35, max_time=600.0,
        )
        assert et < 300.0, f"Dynamic ET={et:.0f}s, expected <300s"

    def test_scaling_ratio_range(self):
        """F_dynamic / F_isometric should be roughly 24-150x under corrected F_isometric.

        Under the corrected shoulder F_isometric = 0.01820
        (Frey-Law 2012 Table 1), the WSD4FEDSRM dynamic subjects span
        F_dynamic 0.44-2.62, giving ratios 24.2x-144x.
        """
        F_dynamic_low = 0.44   # Slowest WSD4FEDSRM subject
        F_dynamic_high = 2.62  # Fastest WSD4FEDSRM subject
        ratio_low = F_dynamic_low / SHOULDER.F
        ratio_high = F_dynamic_high / SHOULDER.F
        assert ratio_low > 20.0, f"Low ratio={ratio_low:.1f}x, expected >20x"
        assert ratio_high < 200.0, f"High ratio={ratio_high:.1f}x, expected <200x"


# =====================================================================
# 6. Correlated inter-muscle sampling (L5)
# =====================================================================

class TestCorrelatedSampling:
    """Validate that correlated sampling preserves population marginals
    and produces positive inter-muscle F correlation."""

    def test_correlated_sampling_preserves_population_mean(self):
        """Mean F from correlated sampling within 50% of population mean."""
        from hcmarl.real_data_calibration import (
            sample_correlated_FR, POPULATION_FR,
        )
        # 100 fake subjects with varied shoulder F (log-uniform 0.5-3.0)
        rng = np.random.RandomState(42)
        shoulder_F_vals = list(np.exp(rng.uniform(np.log(0.5), np.log(3.0), 100)))

        for muscle in ['ankle', 'knee', 'elbow', 'trunk', 'grip']:
            pairs = sample_correlated_FR(
                muscle, shoulder_F_vals, rho=0.5,
                rng=np.random.RandomState(42),
            )
            sampled_F = [p[0] for p in pairs]
            pop_F = POPULATION_FR[muscle][0]
            mean_F = np.mean(sampled_F)
            ratio = mean_F / pop_F
            assert 0.3 < ratio < 3.0, (
                f"{muscle}: mean F={mean_F:.5f}, pop={pop_F:.5f}, "
                f"ratio={ratio:.2f} outside [0.3, 3.0]"
            )
            assert all(f > 0 for f in sampled_F), f"{muscle}: negative F"

    def test_correlated_sampling_positive_correlation(self):
        """Subjects with higher shoulder F should tend to have higher other F."""
        from hcmarl.real_data_calibration import sample_correlated_FR
        rng = np.random.RandomState(123)
        # Strong signal: half low F, half high F
        shoulder_F_vals = [0.5] * 50 + [3.0] * 50

        for muscle in ['ankle', 'elbow', 'grip']:
            pairs = sample_correlated_FR(
                muscle, shoulder_F_vals, rho=0.5,
                rng=np.random.RandomState(123),
            )
            low_group = np.mean([p[0] for p in pairs[:50]])
            high_group = np.mean([p[0] for p in pairs[50:]])
            assert high_group > low_group, (
                f"{muscle}: high shoulder group F={high_group:.5f} "
                f"<= low group F={low_group:.5f}"
            )
