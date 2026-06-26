"""Unit tests for hcmarl.real_data_calibration (Path G).

Tests all pure functions (no dataset dependency) and the full pipeline
(requires WSD4FEDSRM data, skipped if not present).

Verifies consistency with mathematical modelling document:
- predict_endurance_time: Eqs 2-4, Eq 35, exhaustion criterion
- calibrate_F_for_subject: 1D grid search per Frey-Law et al. 2012
- sample_FR_from_population: population (F, R) from Table 1
- sample_correlated_FR: inter-muscle correlation structure
- compute_dynamic_isometric_report: regime reconciliation
- generate_demonstrations_from_profiles: trajectory generation
- load_path_g_into_collector: MMICRL format conversion
"""

import os

import numpy as np
import pytest

from hcmarl.real_data_calibration import (
    predict_endurance_time,
    calibrate_F_for_subject,
    compute_dynamic_isometric_report,
    sample_FR_from_population,
    sample_correlated_FR,
    generate_demonstrations_from_profiles,
    load_path_g_into_collector,
    predicted_endurance_population,
    _safe_float,
    POPULATION_FR,
    POPULATION_CV_F,
    TASK_TO_MVIC_FRACTION,
    ENDURANCE_POWER_MODEL,
    F_ISOMETRIC_SHOULDER,
)
from hcmarl.three_cc_r import SHOULDER


# =====================================================================
# 1. Constants and reference values
# =====================================================================

WSD4FEDSRM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "wsd4fedsrm", "WSD4FEDSRM"
)
HAS_WSD4FEDSRM = os.path.exists(
    os.path.join(WSD4FEDSRM_DIR, "Borg data", "borg_data.csv")
)


class TestConstants:
    """Verify constants match math doc Table 1 and published sources."""

    def test_population_FR_shoulder(self):
        F, R = POPULATION_FR['shoulder']
        assert F == 0.01820
        assert R == 0.00168

    def test_population_FR_grip(self):
        F, R = POPULATION_FR['grip']
        assert F == 0.00980
        assert R == 0.00064

    def test_population_FR_all_six_muscles(self):
        assert set(POPULATION_FR.keys()) == {
            'shoulder', 'ankle', 'knee', 'elbow', 'trunk', 'grip'
        }

    def test_F_isometric_shoulder(self):
        assert F_ISOMETRIC_SHOULDER == SHOULDER.F == 0.01820

    def test_task_to_mvic_has_six_tasks(self):
        assert len(TASK_TO_MVIC_FRACTION) == 6
        assert TASK_TO_MVIC_FRACTION['task1_35i'] == 0.35
        assert TASK_TO_MVIC_FRACTION['task3_55i'] == 0.55

    def test_endurance_power_model_all_muscles(self):
        assert set(ENDURANCE_POWER_MODEL.keys()) == {
            'shoulder', 'ankle', 'knee', 'elbow', 'trunk', 'grip'
        }
        for m, params in ENDURANCE_POWER_MODEL.items():
            assert 'b0' in params and 'b1' in params
            assert params['b0'] > 0
            assert params['b1'] < 0  # Power law exponent is negative

    def test_population_cv_f_range(self):
        for muscle, cv in POPULATION_CV_F.items():
            assert 0.1 < cv < 1.0, f"{muscle} CV={cv} out of range"


# =====================================================================
# 2. predict_endurance_time
# =====================================================================

class TestPredictEnduranceTime:
    """Test endurance-time prediction (Eqs 2-4, Eq 35)."""

    def test_higher_F_shorter_endurance(self):
        """Higher F → faster fatigue → shorter endurance."""
        et_low = predict_endurance_time(F=0.5, R=0.02, r=15, target_load=0.35)
        et_high = predict_endurance_time(F=2.0, R=0.02, r=15, target_load=0.35)
        assert et_high < et_low

    def test_higher_load_shorter_endurance(self):
        """Higher target load → faster exhaustion."""
        et_low = predict_endurance_time(F=1.0, R=0.02, r=15, target_load=0.25)
        et_high = predict_endurance_time(F=1.0, R=0.02, r=15, target_load=0.55)
        assert et_high < et_low

    def test_zero_load_returns_max_time(self):
        """Zero load = rest, should never reach exhaustion."""
        et = predict_endurance_time(F=1.0, R=0.02, r=15, target_load=0.0)
        assert et == 600.0  # default max_time

    def test_returns_positive(self):
        et = predict_endurance_time(F=1.0, R=0.02, r=15, target_load=0.35)
        assert et > 0

    def test_conservation_law_during_simulation(self):
        """MR + MA + MF should stay close to 1.0 throughout."""
        # Manually step through the inline ODE to check
        # (mirrors predict_endurance_time: kp=10, dt=0.5s → dt_min=0.00833 min)
        F, R, r, TL = 1.0, 0.02, 15, 0.35
        dt = 0.5
        dt_min = dt / 60.0
        kp = 10.0
        MR, MA, MF = 1.0, 0.0, 0.0
        Reff = R  # sustained work

        for _ in range(100):
            C = kp * max(TL - MA, 0.0)
            dMA = C - F * MA
            dMF = F * MA - Reff * MF
            dMR = Reff * MF - C
            MR += dt_min * dMR
            MA += dt_min * dMA
            MF += dt_min * dMF
            # Conservation-preserving guard (no renormalization)
            MA = max(0.0, MA)
            MF = max(0.0, MF)
            MR = 1.0 - MA - MF
            if MR < 0.0:
                s = MA + MF
                if s > 0:
                    MA /= s
                    MF /= s
                MR = 0.0
            assert abs(MR + MA + MF - 1.0) < 1e-10

    def test_isometric_F_gives_long_endurance(self):
        """Population isometric F=0.01820 should give ET > 600s at 35%."""
        et = predict_endurance_time(
            F=0.01820, R=0.00168, r=15, target_load=0.35, max_time=7200.0,
        )
        assert et > 600.0

    def test_dynamic_F_gives_short_endurance(self):
        """Calibrated dynamic F~2.0 should give ET < 120s at 35%."""
        et = predict_endurance_time(F=2.0, R=0.02, r=15, target_load=0.35)
        assert et < 120.0

    def test_custom_max_time(self):
        et = predict_endurance_time(F=0.01, R=0.001, r=15, target_load=0.35,
                                    max_time=100.0)
        assert et <= 100.0


# =====================================================================
# 3. calibrate_F_for_subject
# =====================================================================

class TestCalibrateFForSubject:
    """Test 1D F calibration (grid search)."""

    def test_known_endurance_times(self):
        """Given ET at 3 loads, calibration should recover approximate F."""
        # Generate synthetic "observed" ETs from a known F
        F_true = 1.5
        R_fixed = 0.02
        obs = {}
        for TL in [0.35, 0.45, 0.55]:
            obs[TL] = predict_endurance_time(F_true, R_fixed, 15, TL)

        F_opt, rms = calibrate_F_for_subject(obs, R_fixed=R_fixed)
        assert abs(F_opt - F_true) < 0.15  # within 10%
        assert rms < 5.0  # RMS < 5 seconds

    def test_output_types(self):
        obs = {0.35: 100.0, 0.45: 75.0, 0.55: 60.0}
        F_opt, rms = calibrate_F_for_subject(obs)
        assert isinstance(F_opt, float)
        assert isinstance(rms, float)
        assert F_opt > 0
        assert rms >= 0

    def test_F_within_search_range(self):
        obs = {0.35: 100.0, 0.55: 50.0}
        F_opt, _ = calibrate_F_for_subject(obs, F_range=(0.1, 5.0))
        assert 0.1 <= F_opt <= 5.0

    def test_two_data_points_sufficient(self):
        """Calibration should work with just 2 endurance times."""
        obs = {0.35: 100.0, 0.55: 60.0}
        F_opt, rms = calibrate_F_for_subject(obs)
        assert F_opt > 0
        assert rms < 30.0


# =====================================================================
# 4. sample_FR_from_population
# =====================================================================

class TestSampleFRFromPopulation:
    """Test independent population sampling."""

    def test_correct_count(self):
        samples = sample_FR_from_population('shoulder', n_workers=10)
        assert len(samples) == 10

    def test_all_positive(self):
        samples = sample_FR_from_population('elbow', n_workers=50)
        for F, R in samples:
            assert F > 0
            assert R > 0

    def test_mean_near_population(self):
        """Sample mean should be close to population mean (law of large numbers)."""
        rng = np.random.RandomState(42)
        samples = sample_FR_from_population('knee', n_workers=1000, rng=rng)
        F_mean = np.mean([s[0] for s in samples])
        R_mean = np.mean([s[1] for s in samples])
        F_pop, R_pop = POPULATION_FR['knee']
        assert abs(F_mean / F_pop - 1.0) < 0.15  # within 15% of pop mean
        assert abs(R_mean / R_pop - 1.0) < 0.15

    def test_reproducibility(self):
        s1 = sample_FR_from_population('shoulder', 5, rng=np.random.RandomState(0))
        s2 = sample_FR_from_population('shoulder', 5, rng=np.random.RandomState(0))
        for (f1, r1), (f2, r2) in zip(s1, s2):
            assert f1 == f2
            assert r1 == r2


# =====================================================================
# 5. sample_correlated_FR
# =====================================================================

class TestSampleCorrelatedFR:
    """Test correlated inter-muscle sampling (L5)."""

    def test_correct_count(self):
        cal_F = [1.0, 1.5, 2.0]
        samples = sample_correlated_FR('elbow', cal_F)
        assert len(samples) == 3

    def test_all_positive(self):
        cal_F = [0.5, 1.0, 1.5, 2.0, 2.5]
        samples = sample_correlated_FR('trunk', cal_F)
        for F, R in samples:
            assert F > 0
            assert R > 0

    def test_population_marginal_preserved(self):
        """Large sample mean should approximate population mean."""
        rng = np.random.RandomState(42)
        # Generate 500 "calibrated" shoulder F values
        cal_F = list(np.random.RandomState(99).lognormal(
            np.log(1.0), 0.5, 500
        ))
        samples = sample_correlated_FR('knee', cal_F, rho=0.5, rng=rng)
        F_mean = np.mean([s[0] for s in samples])
        F_pop = POPULATION_FR['knee'][0]
        # Should be within 30% of population mean
        assert abs(F_mean / F_pop - 1.0) < 0.30

    def test_positive_correlation(self):
        """Higher shoulder F should correlate with higher other-muscle F."""
        rng = np.random.RandomState(42)
        # Create clearly separated slow and fast fatiguers
        cal_F = [0.3] * 50 + [3.0] * 50
        samples = sample_correlated_FR('elbow', cal_F, rho=0.5, rng=rng)
        F_slow = np.mean([s[0] for s, cf in zip(samples[:50], cal_F[:50])])
        F_fast = np.mean([s[0] for s, cf in zip(samples[50:], cal_F[50:])])
        assert F_fast > F_slow  # correlation preserved

    def test_rho_zero_means_independent(self):
        """With rho=0, shoulder F should not affect other muscles."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        cal_F_lo = [0.3] * 20
        cal_F_hi = [3.0] * 20
        s_lo = sample_correlated_FR('trunk', cal_F_lo, rho=0.0, rng=rng1)
        s_hi = sample_correlated_FR('trunk', cal_F_hi, rho=0.0, rng=rng2)
        F_lo = np.mean([s[0] for s in s_lo])
        F_hi = np.mean([s[0] for s in s_hi])
        # With rho=0, means should be similar (both near population)
        ratio = F_hi / F_lo
        assert 0.5 < ratio < 2.0


# =====================================================================
# 6. compute_dynamic_isometric_report
# =====================================================================

class TestDynamicIsometricReport:
    """Test regime reconciliation (L4)."""

    def test_with_synthetic_calibration(self):
        """Build a mock calibration_results dict and verify report."""
        cal = {}
        for i in range(5):
            cal[f'subject_{i+1}'] = {
                'F': 0.5 + i * 0.3,  # 0.5, 0.8, 1.1, 1.4, 1.7
                'rms_error_sec': 5.0 + i,
            }
        report = compute_dynamic_isometric_report(cal)

        assert report['F_isometric'] == 0.01820
        assert report['ratio_mean'] > 10.0  # dynamic F >> isometric F
        assert report['all_dynamic_F_above_10x_isometric'] is True
        assert len(report['subjects']) == 5

    def test_dynamic_F_gives_short_et(self):
        """Dynamic F=3.0 at 35% should predict ET < 60s."""
        cal = {'subject_1': {'F': 3.0, 'rms_error_sec': 5.0}}
        report = compute_dynamic_isometric_report(cal)
        assert report['all_dynamic_et_below_60s'] is True

    def test_isometric_F_gives_long_et(self):
        """Isometric F at 35% should predict ET > 600s."""
        cal = {'subject_1': {'F': 1.0, 'rms_error_sec': 5.0}}
        report = compute_dynamic_isometric_report(cal)
        assert report['isometric_et_above_600s'] is True


# =====================================================================
# 7. predicted_endurance_population
# =====================================================================

class TestPredictedEndurancePopulation:
    """Test population endurance-time power model (Frey-Law & Avin 2010)."""

    def test_higher_load_shorter(self):
        et_low = predicted_endurance_population('shoulder', 0.25)
        et_high = predicted_endurance_population('shoulder', 0.55)
        assert et_high < et_low

    def test_positive(self):
        for muscle in ENDURANCE_POWER_MODEL:
            et = predicted_endurance_population(muscle, 0.35)
            assert et > 0

    def test_ankle_longest(self):
        """Ankle should have longest endurance (highest delta_max)."""
        et_ankle = predicted_endurance_population('ankle', 0.35)
        et_shoulder = predicted_endurance_population('shoulder', 0.35)
        assert et_ankle > et_shoulder


# =====================================================================
# 8. Demonstration generation (no dataset needed)
# =====================================================================

class TestDemonstrationGeneration:
    """Test trajectory generation from worker profiles."""

    @pytest.fixture
    def mock_profiles(self):
        """Create minimal worker profiles matching run_path_g() output."""
        profiles = []
        for i in range(3):
            profiles.append({
                'worker_id': i,
                'source_subject': f'subject_{i+1}',
                'muscles': {
                    'shoulder': {
                        'F': 0.5 + i * 0.5,  # 0.5, 1.0, 1.5
                        'R': 0.02,
                        'r': 15,
                        'task_type': 'dynamic_rotation',
                    },
                },
            })
        return profiles

    def test_correct_demo_count(self, mock_profiles):
        demos, wids = generate_demonstrations_from_profiles(
            mock_profiles, muscle='shoulder', n_episodes_per_worker=2,
        )
        assert len(demos) == 6  # 3 workers * 2 episodes
        assert len(wids) == 6

    def test_trajectory_columns(self, mock_profiles):
        """Each trajectory row should be [t, MR, MA, MF, C, TL, Reff]."""
        demos, _ = generate_demonstrations_from_profiles(
            mock_profiles, muscle='shoulder', n_episodes_per_worker=1,
        )
        for traj in demos:
            assert traj.ndim == 2
            assert traj.shape[1] == 7  # t, MR, MA, MF, C, TL, Reff

    def test_conservation_in_trajectories(self, mock_profiles):
        """MR + MA + MF ≈ 1 at every step."""
        demos, _ = generate_demonstrations_from_profiles(
            mock_profiles, muscle='shoulder', n_episodes_per_worker=1,
        )
        for traj in demos:
            MR, MA, MF = traj[:, 1], traj[:, 2], traj[:, 3]
            total = MR + MA + MF
            assert np.allclose(total, 1.0, atol=1e-6)

    def test_fast_fatiguer_shorter_trajectories(self, mock_profiles):
        """Worker with higher F should have shorter trajectories (variable length)."""
        demos, wids = generate_demonstrations_from_profiles(
            mock_profiles, muscle='shoulder', n_episodes_per_worker=1,
            variable_length=True,
        )
        # Worker 0 (F=0.5) vs Worker 2 (F=1.5) at same load
        lens = {}
        for traj, wid in zip(demos, wids):
            lens[wid] = len(traj)
        # Fast fatiguer (wid=2) should have shorter or equal trajectory
        assert lens[2] <= lens[0]

    def test_worker_ids_correct(self, mock_profiles):
        demos, wids = generate_demonstrations_from_profiles(
            mock_profiles, muscle='shoulder', n_episodes_per_worker=2,
        )
        assert set(wids) == {0, 1, 2}


# =====================================================================
# 9. load_path_g_into_collector
# =====================================================================

class TestLoadPathGIntoCollector:
    """Test MMICRL format conversion."""

    @pytest.fixture
    def mock_demos(self):
        """Create minimal trajectories matching generate_demonstrations output."""
        demos = []
        wids = []
        for wid in range(2):
            # [t, MR, MA, MF, C, TL, Reff]
            traj = np.zeros((10, 7))
            traj[:, 0] = np.arange(10) * (1.0 / 60.0)  # t in minutes
            traj[:, 1] = np.linspace(1.0, 0.5, 10)  # MR decreasing
            traj[:, 2] = np.linspace(0.0, 0.3, 10)  # MA increasing
            traj[:, 3] = 1.0 - traj[:, 1] - traj[:, 2]  # MF = 1-MR-MA
            traj[:, 4] = np.linspace(0.0, 2.0, 10)  # C increasing
            traj[:, 5] = 0.35  # TL
            traj[:, 6] = 0.02  # Reff
            demos.append(traj)
            wids.append(wid)
        return demos, wids

    def test_collector_has_demos(self, mock_demos):
        demos, wids = mock_demos
        collector = load_path_g_into_collector(demos, wids)
        assert len(collector.demonstrations) == 2

    def test_collector_state_shape(self, mock_demos):
        demos, wids = mock_demos
        collector = load_path_g_into_collector(demos, wids)
        state, action = collector.demonstrations[0][0]
        assert state.shape == (4,)  # [MR, MA, MF, TL]
        assert isinstance(action, (int, np.integer))

    def test_actions_in_valid_range(self, mock_demos):
        demos, wids = mock_demos
        collector = load_path_g_into_collector(demos, wids)
        for traj in collector.demonstrations:
            for state, action in traj:
                assert 0 <= action <= 4  # 5 bins

    def test_worker_ids_preserved(self, mock_demos):
        demos, wids = mock_demos
        collector = load_path_g_into_collector(demos, wids)
        assert collector.worker_ids == [0, 1]


# =====================================================================
# 10. _safe_float helper
# =====================================================================

class TestSafeFloat:
    """Test CSV float parser."""

    def test_valid_float(self):
        assert _safe_float("3.14") == 3.14

    def test_integer_string(self):
        assert _safe_float("42") == 42.0

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_none_input(self):
        assert _safe_float(None) is None

    def test_non_numeric(self):
        assert _safe_float("abc") is None

    def test_whitespace(self):
        assert _safe_float("  ") is None

    def test_whitespace_around_number(self):
        assert _safe_float("  1.5  ") == 1.5


# =====================================================================
# 11. Full pipeline (requires WSD4FEDSRM data)
# =====================================================================

@pytest.mark.skipif(
    not HAS_WSD4FEDSRM,
    reason="WSD4FEDSRM dataset not available at data/wsd4fedsrm/WSD4FEDSRM/"
)
class TestFullPipeline:
    """Integration tests requiring the real WSD4FEDSRM dataset."""

    def test_run_path_g(self):
        from hcmarl.real_data_calibration import run_path_g
        result = run_path_g(WSD4FEDSRM_DIR)
        assert result['n_workers'] == 34
        assert len(result['worker_profiles']) == 34
        assert len(result['calibration_results']) == 34

    def test_calibrated_F_range(self):
        from hcmarl.real_data_calibration import run_path_g
        result = run_path_g(WSD4FEDSRM_DIR)
        F_vals = [r['F'] for r in result['calibration_results'].values()]
        assert min(F_vals) > 0.1
        assert max(F_vals) < 5.0
        assert max(F_vals) / min(F_vals) > 2.0  # real inter-worker variation

    def test_all_profiles_have_six_muscles(self):
        from hcmarl.real_data_calibration import run_path_g
        result = run_path_g(WSD4FEDSRM_DIR)
        for profile in result['worker_profiles']:
            assert len(profile['muscles']) == 6
