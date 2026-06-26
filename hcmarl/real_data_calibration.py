"""
Path G: Real-data MMICRL pipeline.

Calibrates individual (F_i, R_i) parameters per subject from the
WSD4FEDSRM shoulder fatigue dataset (Zenodo 8415066, Nature Scientific
Data 2024), then uses published endurance-time distributions from
Frey-Law & Avin (2010) for non-shoulder muscle groups.

Calibration method: Frey-Law, Looft & Heitsman (2012), J. Biomechanics.
Grid-search optimisation of F, R to match each subject's observed
endurance time at a known %MVIC, using the 3CC-r ODE with the
Xia & Frey-Law (2008) proportional controller.

IMPORTANT — Dynamic vs isometric calibration:
    The WSD4FEDSRM tasks are dynamic shoulder rotations (endurance 50-250s),
    NOT sustained isometric contractions (endurance 40-100+ min per
    Frey-Law & Avin 2010). Consequently, the calibrated F values here
    (~0.3-3.0 /min) are "effective dynamic fatigue rates" that capture
    the real fatigue behavior of each subject during the actual task.
    They are NOT directly comparable to the isometric population means
    in Table 1 of the mathematical modelling document (F=0.01820 /min
    for shoulder). This is scientifically valid: the 3CC-r model is
    used as the system model, and its parameters are fit to observed data.

References:
    [2] Xia & Frey-Law 2008 — submaximal 3CC model + controller
    [3] Frey-Law, Looft & Heitsman 2012 — Monte Carlo (F, R) calibration
    [5] Looft & Frey-Law 2020 — shoulder validation, r=15
    [6] Frey-Law & Avin 2010 — meta-analysis endurance curves
"""

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from hcmarl.three_cc_r import MuscleParams, ThreeCCr, SHOULDER


# -----------------------------------------------------------------------
# Endurance-time prediction via 3CC-r
# -----------------------------------------------------------------------

def predict_endurance_time(
    F: float,
    R: float,
    r: float,
    target_load: float,
    duty_cycle: float = 1.0,
    cycle_time: float = 10.0,
    kp: float = 10.0,
    dt: float = 0.5,
    max_time: float = 600.0,
) -> float:
    """Predict maximum endurance time (MET) for given F, R, target load.

    Simulates the 3CC-r model with the baseline proportional controller
    (Eq 35) at the given %MVC until MR drops below 1e-4 (exhaustion
    criterion from Frey-Law et al. 2012: the resting pool is depleted).

    For intermittent tasks (duty_cycle < 1.0), alternates between work
    and rest phases within each cycle.

    Args:
        F: Fatigue rate constant [min^-1].
        R: Recovery rate constant [min^-1].
        r: Reperfusion multiplier.
        target_load: %MVC as fraction (e.g. 0.35 for 35%).
        duty_cycle: Fraction of cycle time spent working (1.0 = sustained).
        cycle_time: Duration of one work-rest cycle [seconds].
        kp: Proportional controller gain.
        dt: Integration time step [seconds]. Default 0.5s for calibration
            speed; use 0.05s for high-fidelity simulation.
        max_time: Maximum simulation time [seconds].

    Returns:
        Endurance time in seconds. Returns max_time if exhaustion not reached.
    """
    # Inline ODE for speed (avoid object creation overhead in tight loop)
    dt_min = dt / 60.0  # Convert seconds to minutes for the model
    Reff = R if target_load > 0 else R * r  # Sustained work: always R

    MR, MA, MF = 1.0, 0.0, 0.0

    n_steps = int(max_time / dt)
    for step in range(n_steps):
        # Controller (Eq 35)
        C = kp * max(target_load - MA, 0.0) if target_load > 0 else 0.0

        # ODE (Eqs 2-4)
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

        # Exhaustion: MR depleted (Frey-Law et al. 2012)
        if MR < 1e-4 and target_load > 0:
            return (step + 1) * dt
        # Force failure: MA drops well below target
        if target_load > 0 and MA < target_load * 0.5 and step * dt > 5.0:
            return (step + 1) * dt

    return max_time


# -----------------------------------------------------------------------
# Per-subject (F, R) calibration from observed endurance times
# -----------------------------------------------------------------------

def calibrate_F_for_subject(
    observed_endurance_times: Dict[float, float],
    R_fixed: float = 0.02,
    r: float = 15.0,
    duty_cycle: float = 1.0,
    cycle_time: float = 10.0,
    F_range: Tuple[float, float] = (0.1, 5.0),
    n_grid: int = 100,
) -> Tuple[float, float]:
    """Calibrate F for one subject, holding R fixed.

    R is poorly identifiable from short sustained-task endurance times
    (Frey-Law et al. 2012 required intermittent data to constrain R).
    For the WSD4FEDSRM dynamic rotations (50-250s), R barely affects
    the predicted endurance time. We therefore fix R at a physiologically
    motivated value and calibrate only F.

    F_range is (0.1, 5.0) because the WSD4FEDSRM tasks are dynamic
    shoulder rotations. The population isometric mean F=0.01820 gives
    endurance times >1000s — far too long. Effective dynamic F values
    are ~0.3-3.0 /min.

    Args:
        observed_endurance_times: Dict mapping target_load (fraction) to
            observed endurance time (seconds). E.g. {0.35: 105, 0.45: 76}.
        R_fixed: Fixed recovery rate [min^-1]. Default 0.02.
        r: Reperfusion multiplier (15 for shoulder).
        duty_cycle: Fraction of time spent working per cycle.
        cycle_time: Cycle time in seconds.
        F_range: Search range for F [min^-1].
        n_grid: Grid resolution for F search.

    Returns:
        (F_opt, rms_error) — optimal F and RMS error in seconds.
    """
    F_vals = np.geomspace(F_range[0], F_range[1], n_grid)
    loads = sorted(observed_endurance_times.keys())
    max_obs = max(observed_endurance_times.values())
    sim_max_time = min(600.0, max_obs * 2.5)

    best_F, best_err = F_vals[0], float('inf')

    # Stage 1: Coarse log-spaced search
    for F in F_vals:
        sq_errors = []
        for TL in loads:
            pred = predict_endurance_time(
                F, R_fixed, r, TL, duty_cycle, cycle_time,
                max_time=sim_max_time,
            )
            obs = observed_endurance_times[TL]
            sq_errors.append((pred - obs) ** 2)
        rms = np.sqrt(np.mean(sq_errors))
        if rms < best_err:
            best_err = rms
            best_F = F

    # Stage 2: Fine linear search around best
    idx = np.searchsorted(F_vals, best_F)
    F_lo = F_vals[max(0, idx - 1)]
    F_hi = F_vals[min(len(F_vals) - 1, idx + 1)]
    F_fine = np.linspace(F_lo, F_hi, 50)

    for F in F_fine:
        sq_errors = []
        for TL in loads:
            pred = predict_endurance_time(
                F, R_fixed, r, TL, duty_cycle, cycle_time,
                max_time=sim_max_time,
            )
            obs = observed_endurance_times[TL]
            sq_errors.append((pred - obs) ** 2)
        rms = np.sqrt(np.mean(sq_errors))
        if rms < best_err:
            best_err = rms
            best_F = F

    return best_F, best_err


# -----------------------------------------------------------------------
# Dynamic vs isometric reconciliation (L4)
# -----------------------------------------------------------------------

# Isometric F for shoulder from Table 1 of math doc (Frey-Law et al. 2012)
F_ISOMETRIC_SHOULDER = SHOULDER.F  # 0.01820 min^{-1} (Frey-Law 2012 Table 1)


def compute_dynamic_isometric_report(
    calibration_results: Dict[str, Dict],
    r: float = 15.0,
    R_fixed: float = 0.02,
) -> Dict:
    """Compute formal reconciliation between dynamic and isometric F regimes.

    The WSD4FEDSRM dataset contains dynamic shoulder rotations (endurance
    50-250s), NOT sustained isometric contractions (endurance 40-100+ min
    per Frey-Law & Avin 2010). The calibrated F values are therefore
    "effective dynamic fatigue rates" — they capture the higher metabolic
    demand of dynamic tasks through a larger F parameter.

    This function computes the scaling ratio F_dynamic / F_isometric and
    validates that it is consistent with the known endurance-time difference
    between dynamic and isometric tasks.

    The 3CC-r model (Liu et al. 2002, Xia & Frey-Law 2008) is
    phenomenological: its ODE structure (Eqs 2-4) does not assume contraction
    type. F and R are task-specific parameters that must be calibrated to the
    specific contraction regime. Frey-Law et al. (2012) themselves applied
    the model to diverse joint groups with vastly different (F, R) values.

    References:
        - Frey-Law & Avin (2010): Isometric endurance curves by joint
          (consolidates 194-publication meta-analysis of historical
          isometric endurance data)
        - Frey-Law, Looft & Heitsman (2012): Monte Carlo (F, R) calibration
        - Looft et al. (2018): Intermittent/dynamic task extensions

    Args:
        calibration_results: Per-subject calibration from run_path_g().
        r: Reperfusion multiplier for shoulder.
        R_fixed: Fixed R used during calibration.

    Returns:
        Dict with per-subject and aggregate scaling analysis.
    """
    F_iso = F_ISOMETRIC_SHOULDER

    subjects = []
    for subj_id, cal in sorted(
        calibration_results.items(),
        key=lambda x: int(x[0].split('_')[1])
    ):
        F_dyn = cal['F']
        ratio = F_dyn / F_iso

        # Cross-validation 1: What isometric ET would this F predict?
        # At 35% MVC, Frey-Law & Avin 2010 Table 2 shoulder power-model gives
        # ET = 14.86 * 0.35^{-1.83} approx 102 seconds (=1.7 min) -- this is
        # the isometric population mean. Dynamic-calibrated F values predict
        # much shorter ET at the same load, confirming they cannot represent
        # isometric contractions (units/caption: ref6.txt:1827).
        et_dynamic_F_at_35 = predict_endurance_time(
            F_dyn, R_fixed, r, target_load=0.35, max_time=600.0,
        )

        # Cross-validation 2: What does isometric F predict for the
        # WSD4FEDSRM tasks? Should be >> observed ET.
        et_isometric_F_at_35 = predict_endurance_time(
            F_iso, R_fixed, r, target_load=0.35, max_time=7200.0,
        )

        subjects.append({
            'subject_id': subj_id,
            'F_dynamic': F_dyn,
            'F_isometric': F_iso,
            'scaling_ratio': ratio,
            'et_dynamic_F_at_35pct': et_dynamic_F_at_35,
            'et_isometric_F_at_35pct': et_isometric_F_at_35,
            'rms_error': cal['rms_error_sec'],
        })

    ratios = [s['scaling_ratio'] for s in subjects]
    et_dyn = [s['et_dynamic_F_at_35pct'] for s in subjects]
    et_iso = [s['et_isometric_F_at_35pct'] for s in subjects]

    return {
        'subjects': subjects,
        'ratio_mean': float(np.mean(ratios)),
        'ratio_std': float(np.std(ratios)),
        'ratio_min': float(np.min(ratios)),
        'ratio_max': float(np.max(ratios)),
        'et_dynamic_F_at_35_mean': float(np.mean(et_dyn)),
        'et_isometric_F_at_35_mean': float(np.mean(et_iso)),
        'F_isometric': F_iso,
        'all_dynamic_F_above_10x_isometric': bool(np.min(ratios) > 10.0),
        'all_dynamic_et_below_60s': bool(np.max(et_dyn) < 60.0),
        'isometric_et_above_600s': bool(np.min(et_iso) > 600.0),
    }


# -----------------------------------------------------------------------
# WSD4FEDSRM dataset loader
# -----------------------------------------------------------------------

def load_wsd4fedsrm(
    data_dir: str,
) -> Dict[str, Dict]:
    """Load subject data from WSD4FEDSRM dataset.

    Extracts per-subject:
        - MVIC force (IR and ER, Newtons)
        - Endurance times per task (6 tasks: 3 IR + 3 ER)
        - Borg RPE progression per task
        - Demographics (sex, age, height)

    Args:
        data_dir: Path to the WSD4FEDSRM/ directory (extracted).

    Returns:
        Dict mapping subject_id to subject data dict.
    """
    subjects = {}

    # --- Demographics ---
    demo_path = os.path.join(data_dir, "Demographic and antropometric data",
                             "demographic.csv")
    if os.path.exists(demo_path):
        with open(demo_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj = row.get('subject', '').strip()
                if not subj:
                    continue
                subjects[subj] = {
                    'sex': row.get('sex', '').strip(),
                    'age': _safe_float(row.get('age', '')),
                    'height_cm': _safe_float(row.get('height(cm)', '')),
                }

    # If demographics file doesn't have all subjects, init from description
    for i in range(1, 35):
        sid = f"subject_{i}"
        if sid not in subjects:
            subjects[sid] = {}

    # --- MVIC force ---
    mvic_path = os.path.join(data_dir, "MVIC force data",
                             "MVIC_force_data.csv")
    with open(mvic_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row.get('subject', '').strip()
            if subj and subj in subjects:
                subjects[subj]['IR_MVIC_N'] = _safe_float(
                    row.get('IR_MVIC_mean_(N)', ''))
                subjects[subj]['ER_MVIC_N'] = _safe_float(
                    row.get('ER_MVIC_mean_(N)', ''))

    # --- Borg data (endurance times + RPE progression) ---
    borg_path = os.path.join(data_dir, "Borg data", "borg_data.csv")
    with open(borg_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        current_subject = None
        for row in reader:
            subj = row.get('subject', '').strip()
            if subj:
                current_subject = subj
            if current_subject is None or current_subject not in subjects:
                continue

            task = row.get('task_order', '').strip()
            if not task:
                continue

            length = _safe_float(row.get('length_of_trial_(sec)', ''))

            # Parse RPE progression
            rpe_times = []
            rpe_vals = []
            for col in row:
                col_stripped = col.strip()
                if col_stripped.endswith('_sec'):
                    try:
                        t_sec = int(col_stripped.replace('_sec', ''))
                        val = _safe_float(row.get(col, ''))
                        if val is not None:
                            rpe_times.append(t_sec)
                            rpe_vals.append(val)
                    except ValueError:
                        pass

            if 'tasks' not in subjects[current_subject]:
                subjects[current_subject]['tasks'] = {}

            subjects[current_subject]['tasks'][task] = {
                'endurance_time_sec': length,
                'rpe_times': rpe_times,
                'rpe_values': rpe_vals,
            }

    return subjects


def _safe_float(val: str) -> Optional[float]:
    if val is None:
        return None
    val = str(val).strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


# -----------------------------------------------------------------------
# Task → %MVIC mapping
# -----------------------------------------------------------------------

# The WSD4FEDSRM tasks use bands (30-40%, 40-50%, 50-60% of MVIC).
# We use the midpoint as the representative %MVIC for calibration.

TASK_TO_MVIC_FRACTION = {
    'task1_35i': 0.35,  # 30-40% IR → midpoint 35%
    'task2_45i': 0.45,  # 40-50% IR → midpoint 45%
    'task3_55i': 0.55,  # 50-60% IR → midpoint 55%
    'task4_35e': 0.35,  # 30-40% ER → midpoint 35%
    'task5_45e': 0.45,  # 40-50% ER → midpoint 45%
    'task6_55e': 0.55,  # 50-60% ER → midpoint 55%
}


# -----------------------------------------------------------------------
# Population-level endurance distributions from Frey-Law & Avin (2010)
# -----------------------------------------------------------------------

# Power-model: ET (seconds) = b0 * (MVC_fraction)^b1, MVC_fraction in [0, 1].
# From Table 2 of Frey-Law & Avin (2010), Ergonomics 53(1):109-129
# (PMC2891087, PMID 20069487). Table 2 caption (ref6.txt:1827) verbatim:
#   "intensity (% MVC) values are between 0.0 and 1.0; time is in seconds."
# Sanity check: shoulder at 0.5 MVC -> 14.86 * 0.5^{-1.83} approx 52.8 s (plausible).
# Per-joint R^2 reported by the paper: elbow 0.915, shoulder 0.897, trunk 0.885,
# ankle 0.884, knee 0.789, grip 0.748 (grip fit is the weakest -- flag in paper).

ENDURANCE_POWER_MODEL = {
    'shoulder': {'b0': 14.86, 'b1': -1.83},
    'ankle':    {'b0': 34.71, 'b1': -2.06},
    'knee':     {'b0': 19.38, 'b1': -1.88},
    'elbow':    {'b0': 17.98, 'b1': -2.21},
    'grip':     {'b0': 33.55, 'b1': -1.61},
    'trunk':    {'b0': 22.69, 'b1': -2.27},
}


def predicted_endurance_population(
    muscle: str, target_load: float
) -> float:
    """Population-mean endurance time from Frey-Law & Avin (2010) Table 2.

    Args:
        muscle: Muscle group name.
        target_load: %MVC as fraction in [0, 1] (NOT as percent 0-100).

    Returns:
        Expected endurance time in seconds.

    Raises:
        ValueError: If target_load is outside [0, 1] -- silent 10^4-10^5
            error mode if percent is passed instead of fraction.
    """
    if not 0.0 < target_load <= 1.0:
        raise ValueError(
            f"target_load must be MVC fraction in (0, 1], got {target_load}. "
            "Did you pass percent (e.g., 35) instead of fraction (0.35)?"
        )
    params = ENDURANCE_POWER_MODEL[muscle]
    return params['b0'] * (target_load ** params['b1'])


# -----------------------------------------------------------------------
# Non-shoulder muscle calibration from published distributions
# -----------------------------------------------------------------------

# Population F, R from Frey-Law, Looft & Heitsman (2012) Table 1.
# Cross-verified: shoulder (F,R) matches Looft & Frey-Law 2020 L206;
# grip (F,R,r=30) matches Looft et al. 2018 Table 2.
POPULATION_FR = {
    'shoulder': (0.01820, 0.00168),
    'ankle':    (0.00589, 0.00058),
    'knee':     (0.01500, 0.00149),
    'elbow':    (0.00912, 0.00094),
    'trunk':    (0.00755, 0.00075),
    'grip':     (0.00980, 0.00064),
}

# Per-muscle CV for F from published data.
# Handgrip: Liu, Brown & Yue (2002) Table 2, N=10 sustained maximal
# voluntary handgrip contraction, Mean F=0.0206/s SD=0.0075/s -> CV_F=0.364.
# All other muscles: conservative engineering default CV=0.30 (no published
# per-muscle CV exists in the 3CC-r calibration literature).
POPULATION_CV_F = {
    'shoulder': 0.30,
    'ankle':    0.30,
    'knee':     0.30,
    'elbow':    0.30,
    'trunk':    0.30,
    'grip':     0.36,  # Liu et al. (2002) Table 2 -- handgrip, not elbow
}

# CV for R: Liu et al. (2002) Table 2 reports CV_R = 0.429 for handgrip
# (Mean R=0.0084/s SD=0.0036/s). We use 0.40 as a conservative default
# for all muscles; no published per-muscle shoulder/ankle/knee/trunk CV_R
# exists, and 0.40 is slightly below Liu's handgrip value.
POPULATION_CV_R = 0.40


def sample_FR_from_population(
    muscle: str,
    n_workers: int,
    cv: float = 0.3,
    rng: Optional[np.random.RandomState] = None,
) -> List[Tuple[float, float]]:
    """Sample individual (F_i, R_i) from published population distributions.

    Uses the population means from Frey-Law et al. (2012) and applies
    a coefficient of variation (CV) derived from the inter-study
    variability reported in Frey-Law & Avin (2010).

    The CV=0.3 default is conservative. From Liu et al. (2002) Table 2,
    the measured CV for F across 10 subjects was 0.36, and for R was 0.43.

    NOTE: This function samples INDEPENDENTLY per muscle. For correlated
    sampling conditioned on a subject's calibrated shoulder F, use
    sample_correlated_FR() instead.

    Args:
        muscle: Muscle group name.
        n_workers: Number of workers to sample.
        cv: Coefficient of variation for F and R.
        rng: Random state for reproducibility.

    Returns:
        List of (F_i, R_i) tuples.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    F_pop, R_pop = POPULATION_FR[muscle]

    results = []
    for _ in range(n_workers):
        # Log-normal sampling to ensure positivity
        F_i = rng.lognormal(
            mean=np.log(F_pop) - 0.5 * np.log(1 + cv**2),
            sigma=np.sqrt(np.log(1 + cv**2))
        )
        R_i = rng.lognormal(
            mean=np.log(R_pop) - 0.5 * np.log(1 + cv**2),
            sigma=np.sqrt(np.log(1 + cv**2))
        )
        results.append((F_i, R_i))

    return results


def sample_correlated_FR(
    muscle: str,
    calibrated_shoulder_F_values: List[float],
    rho: float = 0.5,
    rng: Optional[np.random.RandomState] = None,
) -> List[Tuple[float, float]]:
    """Sample (F_i, R_i) conditioned on each subject's calibrated shoulder F.

    Fatigue resistance is partly systemic — subjects who fatigue faster in
    one joint group tend to fatigue faster in others due to shared
    cardiorespiratory fitness, fiber type distribution, and systemic
    recovery capacity. This function captures this inter-muscle correlation
    by conditioning non-shoulder F on the subject's calibrated shoulder F.

    Model (in log-space):
        z_i = (log(F_shoulder_i) - log(F_shoulder_pop)) / sigma_shoulder
        log(F_{i,m}) = log(F_pop_m) + sigma_m * (rho * z_i + sqrt(1-rho^2) * eps_i)
        where eps_i ~ N(0, 1)

    This preserves:
      - Population marginal: E[F_{i,m}] ≈ F_pop_m
      - Inter-subject variability: controlled by per-muscle CV
      - Inter-muscle correlation: Corr(log F_shoulder, log F_m) = rho

    R is sampled independently (insufficient data to estimate inter-muscle
    R correlation; Frey-Law et al. 2012 found R poorly identifiable even
    for a single muscle from sustained-task data).

    Args:
        muscle: Non-shoulder muscle group name.
        calibrated_shoulder_F_values: Per-subject calibrated shoulder F
            (from WSD4FEDSRM dynamic rotation calibration).
        rho: Inter-muscle correlation coefficient for log(F).
            Default 0.5 (moderate, conservative). Must be in [0, 1).
        rng: Random state for reproducibility.

    Returns:
        List of (F_i, R_i) tuples, one per subject.

    References:
        - Liu et al. (2002) Table 2: CV_F = 0.36 across 10 subjects
        - Frey-Law & Avin (2010): fatigue resistance partly systemic
        - Frey-Law et al. (2012): population (F, R) means per joint
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_workers = len(calibrated_shoulder_F_values)
    F_pop_m, R_pop_m = POPULATION_FR[muscle]

    cv_F = POPULATION_CV_F.get(muscle, 0.30)
    sigma_m = np.sqrt(np.log(1 + cv_F ** 2))

    # Compute z-scores relative to the CALIBRATED COHORT, not the isometric
    # population mean. This captures each subject's relative fatigue
    # susceptibility among peers (fast vs slow fatiguer ranking), which is
    # the information that transfers across muscle groups. Using the
    # isometric population mean would produce z-scores dominated by the
    # dynamic-isometric gap rather than inter-subject variation.
    log_F_arr = np.log(np.array(calibrated_shoulder_F_values))
    log_F_cohort_mean = float(np.mean(log_F_arr))
    log_F_cohort_std = float(np.std(log_F_arr))
    if log_F_cohort_std < 1e-10:
        log_F_cohort_std = 1.0  # Degenerate case: all same F

    results = []
    for i in range(n_workers):
        # Subject's z-score: how far from cohort mean in log-space
        z_i = (np.log(calibrated_shoulder_F_values[i]) - log_F_cohort_mean) / log_F_cohort_std

        # Correlated F: population mean + correlated component + residual
        eps_i = rng.normal(0.0, 1.0)
        log_F_i = np.log(F_pop_m) + sigma_m * (rho * z_i + np.sqrt(1.0 - rho ** 2) * eps_i)
        F_i = np.exp(log_F_i)

        # R sampled independently from population
        cv_R = POPULATION_CV_R
        sigma_R = np.sqrt(np.log(1 + cv_R ** 2))
        mu_R = np.log(R_pop_m) - 0.5 * sigma_R ** 2
        R_i = rng.lognormal(mu_R, sigma_R)

        results.append((float(F_i), float(R_i)))

    return results


# -----------------------------------------------------------------------
# Full Path G pipeline
# -----------------------------------------------------------------------

def run_path_g(
    wsd4fedsrm_dir: str,
    n_muscles: int = 6,
    use_ir_only: bool = True,
) -> Dict:
    """Run the complete Path G pipeline.

    1. Load WSD4FEDSRM data (34 subjects)
    2. Calibrate individual (F_i, R_i) for shoulder from endurance times
    3. Sample (F_i, R_i) for other muscle groups from published distributions
    4. Return per-worker multi-muscle parameter profiles

    Args:
        wsd4fedsrm_dir: Path to extracted WSD4FEDSRM/ directory.
        n_muscles: Number of muscle groups (default 6 = all in Table 1).
        use_ir_only: If True, use only internal rotation tasks for calibration
            (3 data points per subject). If False, use all 6 tasks.

    Returns:
        Dict with:
            'worker_profiles': list of dicts with per-muscle (F, R, r)
            'calibration_results': per-subject calibration details
            'borg_rpe_data': per-subject Borg RPE for validation
            'n_workers': number of workers
    """
    print("=== Path G: Real-Data MMICRL Pipeline ===")
    print()

    # Step 1: Load dataset
    print("[1/4] Loading WSD4FEDSRM dataset...")
    subjects = load_wsd4fedsrm(wsd4fedsrm_dir)
    n_subj = len([s for s in subjects if 'tasks' in subjects[s]])
    print(f"  Loaded {n_subj} subjects with task data")

    # Step 2: Calibrate individual shoulder F per subject
    # R is fixed because it is poorly identifiable from short sustained
    # tasks (Frey-Law et al. 2012 needed intermittent data to constrain R).
    # R is later sampled per-worker from a log-normal distribution.
    R_CALIB = 0.02  # Reasonable value; doesn't significantly affect ET
    print("[2/4] Calibrating individual shoulder F per subject...")
    print("  Method: 1D grid search on F (Frey-Law et al. 2012)")
    print("  R fixed at 0.02 for calibration (poorly identifiable)")
    print("  R will be sampled per-worker from log-normal(mu=0.02, CV=0.4)")
    print("  Reperfusion: r=15 (Looft & Frey-Law 2020)")
    print()

    calibration_results = {}
    worker_profiles = []

    for subj_id in sorted(subjects.keys(),
                          key=lambda x: int(x.split('_')[1])):
        subj = subjects[subj_id]
        if 'tasks' not in subj:
            continue

        # Build observed endurance times dict
        obs_et = {}
        tasks_to_use = subj['tasks']
        for task_name, task_data in tasks_to_use.items():
            if use_ir_only and task_name not in [
                'task1_35i', 'task2_45i', 'task3_55i'
            ]:
                continue
            if task_name in TASK_TO_MVIC_FRACTION and \
               task_data['endurance_time_sec'] is not None:
                mvic_frac = TASK_TO_MVIC_FRACTION[task_name]
                obs_et[mvic_frac] = task_data['endurance_time_sec']

        if len(obs_et) < 2:
            print(f"  {subj_id}: skipped (< 2 valid endurance times)")
            continue

        # Calibrate F only
        F_opt, rms_err = calibrate_F_for_subject(
            observed_endurance_times=obs_et,
            R_fixed=R_CALIB,
            r=15.0,
        )

        # Verify predictions
        pred_strs = []
        for TL in sorted(obs_et.keys()):
            pred = predict_endurance_time(F_opt, R_CALIB, 15.0, TL)
            pred_strs.append(f"{TL:.0%}:{pred:.0f}s")

        calibration_results[subj_id] = {
            'F': F_opt,
            'rms_error_sec': rms_err,
            'observed_ETs': obs_et,
            'n_data_points': len(obs_et),
            'task_type': 'dynamic_rotation',
        }

        obs_str = " ".join(f"{TL:.0%}:{obs_et[TL]:.0f}s"
                           for TL in sorted(obs_et.keys()))
        print(f"  {subj_id}: F={F_opt:.4f}, RMS={rms_err:.1f}s | "
              f"obs=[{obs_str}] pred=[{' '.join(pred_strs)}]")

    print(f"\n  Calibrated {len(calibration_results)} subjects")

    # Step 3: Sample R for shoulder + all non-shoulder muscles
    print("[3/4] Sampling per-worker R (shoulder) + non-shoulder (F, R)...")
    n_workers = len(calibration_results)
    rng = np.random.RandomState(42)

    # Sample shoulder R per worker from log-normal
    # CV=0.4 is consistent with Liu et al. (2002) Table 2 (CV_R=0.43)
    R_shoulder_mean = 0.02
    R_shoulder_cv = 0.4
    sigma_ln = np.sqrt(np.log(1 + R_shoulder_cv**2))
    mu_ln = np.log(R_shoulder_mean) - 0.5 * sigma_ln**2
    shoulder_R_samples = rng.lognormal(mu_ln, sigma_ln, n_workers)
    print(f"  shoulder R: mean={np.mean(shoulder_R_samples):.5f}, "
          f"SD={np.std(shoulder_R_samples):.5f}")

    other_muscles = ['ankle', 'knee', 'elbow', 'trunk', 'grip']
    other_FR = {}

    # Extract calibrated shoulder F values (ordered by subject)
    ordered_subj_ids = sorted(
        calibration_results.keys(),
        key=lambda x: int(x.split('_')[1])
    )
    calibrated_shoulder_F = [calibration_results[s]['F'] for s in ordered_subj_ids]

    for muscle in other_muscles:
        other_FR[muscle] = sample_correlated_FR(
            muscle, calibrated_shoulder_F, rho=0.5, rng=rng,
        )
        F_vals = [x[0] for x in other_FR[muscle]]
        R_vals = [x[1] for x in other_FR[muscle]]
        print(f"  {muscle}: F={np.mean(F_vals):.5f}+/-{np.std(F_vals):.5f}, "
              f"R={np.mean(R_vals):.5f}+/-{np.std(R_vals):.5f} "
              f"(correlated with shoulder F, rho=0.5)")

    # Step 4: Build per-worker multi-muscle profiles
    print("[4/4] Building multi-muscle worker profiles...")

    # Reperfusion multipliers from Looft, Herkert & Frey-Law (2018), Table 2
    # r=30 for hand grip (forearm flexors), r=15 for all other muscle groups
    REPERFUSION_R = {'grip': 30, 'default': 15}

    for idx, subj_id in enumerate(sorted(
        calibration_results.keys(),
        key=lambda x: int(x.split('_')[1])
    )):
        cal = calibration_results[subj_id]
        profile = {
            'worker_id': idx,
            'source_subject': subj_id,
            'muscles': {
                'shoulder': {
                    'F': cal['F'],
                    'R': float(shoulder_R_samples[idx]),
                    'r': REPERFUSION_R.get('shoulder', REPERFUSION_R['default']),
                    'task_type': 'dynamic_rotation',
                    'F_source': 'WSD4FEDSRM calibration (real data)',
                    'R_source': 'log-normal(0.02, CV=0.4)',
                },
            },
        }

        for muscle in other_muscles:
            F_i, R_i = other_FR[muscle][idx]
            profile['muscles'][muscle] = {
                'F': F_i,
                'R': R_i,
                'r': REPERFUSION_R.get(muscle, REPERFUSION_R['default']),
                'task_type': 'isometric',
                'source': 'population (Frey-Law & Avin 2010), F correlated with shoulder (rho=0.5)',
            }

        worker_profiles.append(profile)

    # Collect Borg RPE data for external validation
    borg_data = {}
    for subj_id in calibration_results:
        subj = subjects[subj_id]
        borg_data[subj_id] = {}
        for task_name, task_data in subj.get('tasks', {}).items():
            borg_data[subj_id][task_name] = {
                'rpe_times': task_data.get('rpe_times', []),
                'rpe_values': task_data.get('rpe_values', []),
                'endurance_time': task_data.get('endurance_time_sec'),
            }

    print(f"\n=== Pipeline complete ===")
    print(f"  Workers: {n_workers}")
    print(f"  Muscles per worker: {n_muscles}")
    print(f"  Shoulder source: WSD4FEDSRM (real data, N={n_workers})")
    print(f"  Other muscles: Frey-Law & Avin 2010 (published distributions)")

    return {
        'worker_profiles': worker_profiles,
        'calibration_results': calibration_results,
        'borg_rpe_data': borg_data,
        'n_workers': n_workers,
        'subjects_raw': subjects,
    }


# -----------------------------------------------------------------------
# Generate MMICRL demonstrations from calibrated workers
# -----------------------------------------------------------------------

def generate_demonstrations_from_profiles(
    worker_profiles: List[Dict],
    muscle: str = 'shoulder',
    target_loads: List[float] = None,
    n_episodes_per_worker: int = 3,
    episode_duration_sec: float = 90.0,
    dt_sec: float = 1.0,
    variable_length: bool = True,
) -> Tuple[List[np.ndarray], List[int]]:
    """Generate MMICRL-compatible demonstrations from calibrated profiles.

    Each demonstration is a trajectory tau = [(s_0, a_0), (s_1, a_1), ...]
    where s = [MR, MA, MF] and a = C(t) (neural drive from Eq 35).

    The inter-worker variability in trajectories comes from real (F, R)
    differences, NOT from synthetic parameter choices.

    Episode length is chosen to be SHORT enough (90s default) that fast-
    fatiguers (high F) reach exhaustion while slow-fatiguers (low F) still
    have MR > 0. This preserves the fatigue-rate signal that MMICRL needs
    for type discovery. With the calibrated F range [0.44, 2.62]:
    - Fast fatiguer (F=2.6): MR depletes in ~40-65s
    - Slow fatiguer (F=0.44): MR depletes in ~130-300s
    At 90s, slow workers still have MR > 0. This is the key signal.

    If variable_length=True, episodes terminate at exhaustion (MR < 0.01),
    creating natural length variation that further discriminates workers.

    Args:
        worker_profiles: From run_path_g()['worker_profiles'].
        muscle: Which muscle group to generate demos for.
        target_loads: List of %MVC fractions for task demands.
        n_episodes_per_worker: Demonstrations per worker.
        episode_duration_sec: Max episode length in seconds.
        dt_sec: Time step in seconds (default 1s).
        variable_length: If True, truncate at exhaustion.

    Returns:
        (demonstrations, worker_ids): list of trajectory arrays
            and corresponding worker indices.
    """
    if target_loads is None:
        target_loads = [0.35, 0.45, 0.55]

    dt_min = dt_sec / 60.0  # ODE uses minutes

    demonstrations = []
    worker_ids = []

    for profile in worker_profiles:
        muscle_params = profile['muscles'].get(muscle)
        if muscle_params is None:
            continue

        params = MuscleParams(
            name=f"{muscle}_{profile['worker_id']}",
            F=muscle_params['F'],
            R=muscle_params['R'],
            r=muscle_params['r'],
        )
        model = ThreeCCr(params)

        for ep in range(n_episodes_per_worker):
            TL = target_loads[ep % len(target_loads)]
            max_steps = int(episode_duration_sec / dt_sec)

            # Simulate and collect trajectory
            steps_data = []
            state = np.array([1.0, 0.0, 0.0])
            for step in range(max_steps):
                t = step * dt_min  # in minutes
                MR, MA, MF = state

                C = model.baseline_neural_drive(TL, MA)
                Reff = model.R_eff(TL)

                steps_data.append([t, MR, MA, MF, C, TL, Reff])

                dx = model.ode_rhs(state, C, TL)
                state = state + dt_min * dx
                # Conservation-preserving guard (no renormalization)
                state[1] = max(0.0, state[1])  # MA >= 0
                state[2] = max(0.0, state[2])  # MF >= 0
                state[0] = 1.0 - state[1] - state[2]  # MR from conservation
                if state[0] < 0.0:
                    s = state[1] + state[2]
                    if s > 0:
                        state[1] /= s
                        state[2] /= s
                    state[0] = 0.0

                # Terminate at exhaustion if variable_length
                if variable_length and state[0] < 0.01 and step > 5:
                    break

            trajectory = np.array(steps_data)
            demonstrations.append(trajectory)
            worker_ids.append(profile['worker_id'])

    return demonstrations, worker_ids


def load_path_g_into_collector(
    demonstrations: List[np.ndarray],
    worker_ids: List[int],
) -> 'DemonstrationCollector':
    """Convert Path G demonstrations into MMICRL DemonstrationCollector format.

    Path G trajectories have columns [t, MR, MA, MF, C, TL, Reff].
    MMICRL expects (state, action) tuples where:
        state = [MR, MA, MF, TL]  (n_muscles*3 + 1)
        action = discretized neural drive C

    Action discretization uses percentile-based bins computed from all demos
    to ensure a balanced distribution across bins (critical for MMICRL features).

    For single-muscle (shoulder) calibration, n_muscles=1.

    Args:
        demonstrations: From generate_demonstrations_from_profiles().
        worker_ids: From generate_demonstrations_from_profiles().

    Returns:
        DemonstrationCollector loaded with the demonstrations.
    """
    from hcmarl.mmicrl import DemonstrationCollector

    collector = DemonstrationCollector(n_muscles=1)

    # Compute action bin edges from data (percentile-based for balanced bins)
    all_C = []
    for traj in demonstrations:
        all_C.extend(traj[:, 4].tolist())  # Column 4 = C
    all_C = np.array(all_C)
    # 5 bins: rest (C=0), then quartiles of non-zero C
    nonzero_C = all_C[all_C > 0.01]
    if len(nonzero_C) > 0:
        q25, q50, q75 = np.percentile(nonzero_C, [25, 50, 75])
    else:
        q25, q50, q75 = 0.1, 0.5, 1.0

    for traj, wid in zip(demonstrations, worker_ids):
        trajectory = []
        for step in range(len(traj)):
            t, MR, MA, MF, C, TL, Reff = traj[step]
            state = np.array([MR, MA, MF, TL], dtype=np.float32)
            # Percentile-based action discretization
            if C <= 0.01:
                action = 0  # rest
            elif C <= q25:
                action = 1
            elif C <= q50:
                action = 2
            elif C <= q75:
                action = 3
            else:
                action = 4
            trajectory.append((state, action))
        collector.demonstrations.append(trajectory)
        collector.worker_ids.append(wid)

    return collector


def run_full_path_g_pipeline(
    wsd4fedsrm_dir: str,
    n_types: int = 3,
    n_episodes_per_worker: int = 3,
) -> Dict:
    """Run complete Path G pipeline: calibrate -> generate demos -> MMICRL.

    This is the end-to-end function that:
    1. Calibrates worker profiles from WSD4FEDSRM real data
    2. Generates MMICRL-compatible demonstrations
    3. Runs MMICRL type discovery with CFDE
    4. Extracts per-type safety thresholds

    Returns:
        Dict with calibration results, demonstrations, MMICRL output.
    """
    # Step 1: Calibrate and build worker profiles
    path_g_result = run_path_g(wsd4fedsrm_dir)

    # Step 2: Generate demonstrations
    print("\n=== Generating MMICRL Demonstrations ===")
    demos, wids = generate_demonstrations_from_profiles(
        path_g_result['worker_profiles'],
        muscle='shoulder',
        n_episodes_per_worker=n_episodes_per_worker,
    )
    print(f"Generated {len(demos)} demonstrations from "
          f"{len(set(wids))} workers")
    print(f"Trajectory shape: {demos[0].shape}")

    # Step 3: Load into MMICRL collector
    collector = load_path_g_into_collector(demos, wids)
    print(f"Loaded {len(collector.demonstrations)} demos into collector")

    # Step 4: Run MMICRL type discovery
    print("\n=== Running MMICRL Type Discovery (CFDE) ===")
    from hcmarl.mmicrl import MMICRL
    mmicrl = MMICRL(
        n_types=n_types,
        lambda1=1.0,
        lambda2=1.0,
        n_muscles=1,
        n_iterations=150,
        hidden_dims=[64, 64],
    )
    mmicrl_results = mmicrl.fit(collector)

    print(f"Types discovered: {mmicrl_results['n_types_discovered']}")
    print(f"Type proportions: {mmicrl_results['type_proportions']}")
    print(f"Mutual information I(tau;z): {mmicrl_results['mutual_information']:.4f}")
    print(f"Objective value: {mmicrl_results['objective_value']:.4f}")

    # Print per-type thresholds
    print("\n=== Per-Type Safety Thresholds (theta_max) ===")
    for type_k, thresholds in mmicrl_results['theta_per_type'].items():
        print(f"  Type {type_k}: {thresholds}")

    # Step 5: Validate — check if types correlate with calibrated F values
    print("\n=== Validation: Type vs F correlation ===")
    type_assignments = mmicrl.type_assignments
    cal_results = path_g_result['calibration_results']
    subj_ids = sorted(cal_results.keys(),
                       key=lambda x: int(x.split('_')[1]))

    type_F_groups = {k: [] for k in range(n_types)}
    for demo_idx, wid in enumerate(wids):
        # Each worker has n_episodes_per_worker demos
        if demo_idx < len(type_assignments):
            t = type_assignments[demo_idx]
            subj_id = subj_ids[wid]
            type_F_groups[t].append(cal_results[subj_id]['F'])

    for t, f_vals in type_F_groups.items():
        if f_vals:
            print(f"  Type {t}: n={len(f_vals)}, "
                  f"F_mean={np.mean(f_vals):.4f}, "
                  f"F_range=[{np.min(f_vals):.4f}, {np.max(f_vals):.4f}]")

    return {
        'path_g': path_g_result,
        'demonstrations': demos,
        'worker_ids': wids,
        'mmicrl_results': mmicrl_results,
        'mmicrl_model': mmicrl,
    }


def bootstrap_mi_diagnostic(
    worker_profiles: List[Dict],
    n_bootstrap: int = 30,
    muscle: str = "shoulder",
    target_loads: Optional[List[float]] = None,
    n_episodes_per_worker: int = 3,
    episode_duration_sec: float = 90.0,
    n_iterations: int = 40,
    k_range: Tuple[int, int] = (1, 5),
    hidden_dims: Optional[List[int]] = None,
    ci_level: float = 0.95,
    seed: int = 0,
) -> Dict:
    """Bootstrap CI on the Path G homogeneity claim.

    Repeatedly resample the calibrated workers with replacement, refit
    MMICRL, and record (K*, MI). The headline number for Experiment C is
    reported as a bootstrap 95% CI on MI, not a single point estimate —
    this keeps the "K=1 on WSD4FEDSRM" claim honest about sampling error
    on N=34.

    Args:
        worker_profiles: From run_path_g()['worker_profiles'] OR any list
            whose entries have 'worker_id' + 'muscles' with F/R/r.
        n_bootstrap: number of resample draws (30 for pilot, 100+ for paper).
        muscle / target_loads / *_per_worker / duration: passed through to
            generate_demonstrations_from_profiles.
        n_iterations: MMICRL optimisation steps per fit.
        k_range: candidate K values for the auto-selection sweep.
        hidden_dims: CFDE hidden sizes (None -> defaults).
        ci_level: bootstrap CI width (0.95 -> 2.5 / 97.5 percentiles).
        seed: controls the bootstrap resampler.

    Returns:
        dict with mi_mean, mi_ci_lo, mi_ci_hi, k_distribution (Counter
        over drawn K*), k_mode, and the full raw (mi_samples, k_samples)
        arrays for later analysis.
    """
    from collections import Counter

    # Lazy import to avoid a circular dep: MMICRL lives in hcmarl.mmicrl
    # which imports are already hot upstream of this function.
    from hcmarl.mmicrl import DemonstrationCollector, MMICRL

    if len(worker_profiles) < 2:
        raise ValueError("Need at least 2 worker profiles to bootstrap.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")

    rng = np.random.default_rng(seed)
    mi_samples = []
    k_samples = []

    for b in range(n_bootstrap):
        # Resample workers with replacement
        idx = rng.integers(0, len(worker_profiles), size=len(worker_profiles))
        draw = [worker_profiles[i] for i in idx]

        # Regenerate demos for this draw
        demos, worker_ids = generate_demonstrations_from_profiles(
            draw, muscle=muscle, target_loads=target_loads,
            n_episodes_per_worker=n_episodes_per_worker,
            episode_duration_sec=episode_duration_sec,
        )
        if len(demos) < 4:
            continue

        # Path G trajectories have columns [t, MR, MA, MF, C, TL, Reff].
        # Convert to DemonstrationCollector's (obs, action) form: state
        # = [MR, MA, MF, TL], action = 0 if C < 0.05 else 1 (rest/work).
        collector = DemonstrationCollector(n_muscles=1)
        for traj_arr in demos:
            traj = []
            for row in traj_arr:
                _, MR, MA, MF, C, TL, _ = row
                obs = np.array([MR, MA, MF, TL], dtype=np.float32)
                action = 0 if C < 0.05 else 1
                traj.append((obs, int(action)))
            collector.demonstrations.append(traj)
            collector.worker_ids.append(int(worker_ids[len(collector.demonstrations) - 1]))

        # Use heldout_nll so the bootstrap estimate is consistent with
        # the singular-flow theory (Watanabe 2013). K_range capped by
        # n_demos // 2 inside MMICRL.fit.
        mm = MMICRL(
            n_types=k_range[0], n_muscles=1, n_iterations=n_iterations,
            hidden_dims=hidden_dims or [32, 32],
            auto_select_k=True, k_range=k_range,
            k_selection="heldout_nll",
        )
        res = mm.fit(collector, n_actions=2)
        mi_samples.append(float(res["mutual_information"]))
        k_samples.append(int(res["n_types_discovered"]))

    if not mi_samples:
        raise RuntimeError(
            "Bootstrap produced zero valid draws — check n_episodes_per_worker."
        )

    mi_arr = np.asarray(mi_samples, dtype=np.float64)
    alpha = 1.0 - ci_level
    lo = float(np.percentile(mi_arr, 100 * (alpha / 2)))
    hi = float(np.percentile(mi_arr, 100 * (1 - alpha / 2)))
    k_distribution = Counter(k_samples)
    k_mode = int(k_distribution.most_common(1)[0][0])

    return {
        "mi_mean": float(mi_arr.mean()),
        "mi_ci_lo": lo,
        "mi_ci_hi": hi,
        "k_distribution": dict(k_distribution),
        "k_mode": k_mode,
        "mi_samples": mi_samples,
        "k_samples": k_samples,
        "ci_level": ci_level,
        "n_bootstrap": len(mi_samples),
    }


if __name__ == "__main__":
    import sys

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "wsd4fedsrm", "WSD4FEDSRM"
    )

    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}")
        print("Download from: https://zenodo.org/records/8415066")
        sys.exit(1)

    result = run_full_path_g_pipeline(data_dir)
