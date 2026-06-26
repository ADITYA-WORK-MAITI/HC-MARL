"""Tests for F-anchored per-worker safety ceilings (Method 1).

These tests pin the behavior of ``build_per_worker_theta_max_from_F``
against the four failure modes of the MMICRL-rescale approach:

  alpha (sign):        high F must yield lower ceiling (stricter).
  beta (ICRL misuse):  function must not consume demo trajectories.
  gamma (noise amp):   small F perturbation must produce small ceiling
                       perturbation (no 30x stretching).
  epsilon (feasibility): ceiling must stay above per-worker Eq 26 floor.

Also covers homogeneous-F degenerate behavior and outlier clipping.
"""

from __future__ import annotations

import numpy as np
import pytest

from hcmarl.utils import build_per_worker_theta_max_from_F


# Helper: build a worker_F_R_r_per_muscle dict from a list of F values.
# R is held fixed at the Frey-Law 2012 population value, r from defaults.
_R_POP = {
    'shoulder': 0.00168, 'ankle': 0.00058, 'knee': 0.00149,
    'elbow':    0.00094, 'trunk': 0.00075, 'grip':  0.00064,
}
_R_LITERAL = {'shoulder': 15, 'ankle': 15, 'knee': 15,
              'elbow': 15, 'trunk': 15, 'grip': 30}
_CONFIG_DEFAULTS = {'shoulder': 0.70, 'ankle': 0.80, 'knee': 0.60,
                    'elbow': 0.45, 'trunk': 0.65, 'grip': 0.45}


def _make_Fs(muscle_Fs):
    """{muscle: [F_w, ...]} -> {muscle: [(F_w, R_pop_m, r_m), ...]}."""
    return {
        m: [(F, _R_POP[m], _R_LITERAL[m]) for F in Fs]
        for m, Fs in muscle_Fs.items()
    }


# -----------------------------------------------------------------------
# alpha — sign convention: higher F -> lower ceiling (stricter safety)
# -----------------------------------------------------------------------

def test_alpha_sign_high_F_gets_lower_ceiling():
    """One-sided design: the worker with the highest F must receive a
    strictly lower theta_max than workers with lower F, and that ceiling
    must be strictly below the config default. Workers at or below mean F
    sit at the config default (never relaxed above it).
    """
    # 3 workers with a visible F spread on shoulder
    Fs = {'shoulder': [0.015, 0.018, 0.022]}  # low, mid, high
    worker_params = _make_Fs(Fs)
    config = {'shoulder': 0.70}

    out = build_per_worker_theta_max_from_F(
        worker_params, config, n_workers=3, alpha=0.05,
    )

    low_worker  = out['worker_0']['shoulder']  # F = 0.015 (below mean)
    mid_worker  = out['worker_1']['shoulder']  # F = 0.018 (at/below mean)
    high_worker = out['worker_2']['shoulder']  # F = 0.022 (above mean)

    # Non-strict monotonic (below-mean workers bunch at config default).
    assert low_worker >= mid_worker >= high_worker, (
        f"Order inverted: low_F={low_worker}, mid_F={mid_worker}, "
        f"high_F={high_worker}. Expected low >= mid >= high."
    )
    # The highest-F worker must be strictly below the config default.
    assert high_worker < 0.70, (
        f"High-F worker ceiling {high_worker} must be < config default 0.70 "
        "(one-sided stricter-never-looser property)."
    )
    # The lowest-F worker must equal the config default (one-sided clamp).
    assert abs(low_worker - 0.70) < 1e-9, (
        f"Low-F worker ceiling {low_worker} should equal config default 0.70 "
        "(one-sided design: never relaxed above literature ceiling)."
    )


# -----------------------------------------------------------------------
# homogeneous F degenerate behavior: every worker -> config default
# -----------------------------------------------------------------------

def test_homogeneous_F_yields_config_default():
    """When every worker has identical F, z == 0 for all and every worker
    must receive exactly the config default theta_max (not amplified, not
    perturbed)."""
    Fs = {'shoulder': [0.01820, 0.01820, 0.01820, 0.01820, 0.01820, 0.01820]}
    worker_params = _make_Fs(Fs)
    config = {'shoulder': 0.70}

    out = build_per_worker_theta_max_from_F(
        worker_params, config, n_workers=6, alpha=0.05,
    )

    for w in range(6):
        assert abs(out[f'worker_{w}']['shoulder'] - 0.70) < 1e-9, (
            f"worker_{w} ceiling = {out[f'worker_{w}']['shoulder']}; "
            "homogeneous F must yield the config default exactly."
        )


# -----------------------------------------------------------------------
# epsilon — Eq 26 feasibility: ceiling >= per-worker floor + epsilon
# -----------------------------------------------------------------------

def test_ceiling_respects_per_worker_eq26_floor():
    """Even an extreme high-F worker cannot be pushed below their Eq 26
    rest-phase floor."""
    # Very high F on one worker; others near population mean.
    Fs = {'shoulder': [0.01820, 0.01820, 0.01820, 0.05, 0.01820, 0.01820]}
    worker_params = _make_Fs(Fs)
    config = {'shoulder': 0.70}
    epsilon = 0.005

    out = build_per_worker_theta_max_from_F(
        worker_params, config, n_workers=6, alpha=0.20, epsilon=epsilon,
    )

    for w in range(6):
        F_w = Fs['shoulder'][w]
        R_w = _R_POP['shoulder']
        r_m = _R_LITERAL['shoulder']
        floor = F_w / (F_w + R_w * r_m)
        assert out[f'worker_{w}']['shoulder'] >= floor + epsilon - 1e-9, (
            f"worker_{w} ceiling={out[f'worker_{w}']['shoulder']:.6f} "
            f"violates Eq 26 floor={floor:.6f} + epsilon={epsilon}."
        )


def test_ceiling_bounded_above_by_config_default():
    """No rescaling can push a ceiling above the literature-anchored default."""
    Fs = {'shoulder': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]}
    worker_params = _make_Fs(Fs)
    config = {'shoulder': 0.70}

    out = build_per_worker_theta_max_from_F(
        worker_params, config, n_workers=6, alpha=0.15,
    )

    for w in range(6):
        assert out[f'worker_{w}']['shoulder'] <= 0.70 + 1e-9, (
            f"worker_{w} ceiling={out[f'worker_{w}']['shoulder']} "
            "exceeded config default 0.70."
        )


# -----------------------------------------------------------------------
# z_clip outlier protection
# -----------------------------------------------------------------------

def test_z_clip_and_physiological_edge_case():
    """Two guards combined:
      (a) z is clamped to [-z_clip, z_clip] so a 10-SD outlier cannot
          push raw_ceiling below (config - alpha*z_clip).
      (b) If a worker's own Eq 26 floor exceeds the config default
          (extreme F), the biomech-feasibility floor wins: ceiling is
          set to eq26_floor + epsilon, overriding the one-sided clamp.
    """
    # Inject one extreme outlier + tight cluster at realistic F.
    Fs = {'shoulder': [0.018, 0.018, 0.018, 0.018, 0.018, 0.100]}
    worker_params = _make_Fs(Fs)
    config = {'shoulder': 0.70}
    alpha, z_clip, epsilon = 0.05, 2.0, 0.005

    out = build_per_worker_theta_max_from_F(
        worker_params, config, n_workers=6,
        alpha=alpha, z_clip=z_clip, epsilon=epsilon,
    )

    # Outlier's Eq 26 floor at F_w = 0.100, R_pop = 0.00168, r = 15:
    #   eq26 = 0.100 / (0.100 + 0.00168 * 15) = 0.100 / 0.1252 ~ 0.799
    # Since 0.799 > 0.70 (config default), the physiological-edge
    # branch fires: ceiling = eq26 + epsilon, not the z-clipped value.
    F_w = 0.100
    R_w = _R_POP['shoulder']
    r_m = _R_LITERAL['shoulder']
    eq26 = F_w / (F_w + R_w * r_m)
    expected_outlier_ceiling = eq26 + epsilon

    outlier_ceiling = out['worker_5']['shoulder']
    assert abs(outlier_ceiling - expected_outlier_ceiling) < 1e-6, (
        f"Outlier ceiling {outlier_ceiling:.6f} != expected "
        f"{expected_outlier_ceiling:.6f} (eq26 floor {eq26:.6f} + eps {epsilon}). "
        "Physiological-edge branch not firing correctly."
    )

    # Non-outlier workers: F=0.018 is below the outlier-inflated pop mean,
    # so their z is negative -> clipped at config default (one-sided).
    for w in range(5):
        assert abs(out[f'worker_{w}']['shoulder'] - 0.70) < 1e-9, (
            f"worker_{w} ceiling {out[f'worker_{w}']['shoulder']} != 0.70."
        )


# -----------------------------------------------------------------------
# gamma — noise non-amplification: small F delta -> small ceiling delta
# -----------------------------------------------------------------------

def test_noise_not_amplified_relative_to_alpha():
    """A 1% perturbation in one worker's F must not produce more than
    alpha/(sd_F/F_w) * 1% in that worker's ceiling. This pins Attack gamma."""
    # Tight F cluster -> small sd -> standardised z is sensitive.
    Fs_base = {'shoulder': [0.01820, 0.01820, 0.01820, 0.01820, 0.01820, 0.01820 * 1.001]}
    Fs_perturb = {'shoulder': [0.01820, 0.01820, 0.01820, 0.01820, 0.01820, 0.01820 * 1.002]}
    config = {'shoulder': 0.70}
    alpha = 0.05

    # Add a touch of baseline spread so sd > 0.
    Fs_base['shoulder'][0] = 0.018
    Fs_base['shoulder'][1] = 0.0185
    Fs_perturb['shoulder'][0] = 0.018
    Fs_perturb['shoulder'][1] = 0.0185

    out_base = build_per_worker_theta_max_from_F(
        _make_Fs(Fs_base), config, n_workers=6, alpha=alpha,
    )
    out_perturb = build_per_worker_theta_max_from_F(
        _make_Fs(Fs_perturb), config, n_workers=6, alpha=alpha,
    )

    # With alpha=0.05 and z clamped to [-2, 2], max |ceiling delta| = 0.10.
    # The actual delta must be bounded by this, nowhere near the 300x
    # amplification the MMICRL rescale exhibited at small span.
    for w in range(6):
        delta = abs(out_base[f'worker_{w}']['shoulder']
                    - out_perturb[f'worker_{w}']['shoulder'])
        assert delta <= alpha * 2.0 + 1e-9, (
            f"worker_{w} ceiling perturbation {delta} exceeds alpha*z_clip "
            f"upper bound {alpha*2.0}. Amplification regression."
        )


# -----------------------------------------------------------------------
# Validation: missing muscle / wrong n_workers fail loudly
# -----------------------------------------------------------------------

def test_missing_muscle_raises():
    worker_params = _make_Fs({'shoulder': [0.018]})
    with pytest.raises(ValueError, match="missing from worker_F_R_r_per_muscle"):
        build_per_worker_theta_max_from_F(
            worker_params, {'shoulder': 0.70, 'grip': 0.45},
            n_workers=1,
        )


def test_worker_count_mismatch_raises():
    worker_params = _make_Fs({'shoulder': [0.018, 0.019]})
    with pytest.raises(ValueError, match="parameter tuples but n_workers"):
        build_per_worker_theta_max_from_F(
            worker_params, {'shoulder': 0.70},
            n_workers=3,  # mismatch
        )


# -----------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------

def test_deterministic_across_runs():
    """Same inputs -> bit-identical outputs (no torch, no random state)."""
    Fs = {'shoulder': [0.015, 0.018, 0.022, 0.016, 0.020, 0.019]}
    worker_params = _make_Fs(Fs)
    config = {'shoulder': 0.70}

    out1 = build_per_worker_theta_max_from_F(
        worker_params, config, n_workers=6, alpha=0.05,
    )
    out2 = build_per_worker_theta_max_from_F(
        worker_params, config, n_workers=6, alpha=0.05,
    )
    for w in range(6):
        assert out1[f'worker_{w}']['shoulder'] == out2[f'worker_{w}']['shoulder']


# -----------------------------------------------------------------------
# Multi-muscle: each muscle standardised independently
# -----------------------------------------------------------------------

def test_multi_muscle_independent_standardisation():
    """Two muscles with different F distributions must each standardise
    against their own population, not be pooled."""
    Fs = {
        'shoulder': [0.016, 0.018, 0.020],   # spread
        'grip':     [0.0098, 0.0098, 0.0098], # homogeneous
    }
    worker_params = _make_Fs(Fs)
    config = {'shoulder': 0.70, 'grip': 0.45}

    out = build_per_worker_theta_max_from_F(
        worker_params, config, n_workers=3, alpha=0.05,
    )

    # Shoulder: one-sided design -> below-mean workers clip to config
    # default; above-mean workers strictly below.
    s = [out[f'worker_{w}']['shoulder'] for w in range(3)]
    assert s[0] >= s[1] >= s[2], \
        f"shoulder order violated (expected non-increasing): {s}"
    assert s[2] < 0.70, \
        f"highest-F worker must be strictly below config default; got {s[2]}"

    # Grip: homogeneous (sd == 0), every worker gets config default.
    for w in range(3):
        assert abs(out[f'worker_{w}']['grip'] - 0.45) < 1e-9, \
            f"grip should be config default when F is homogeneous; got {out[f'worker_{w}']['grip']}"
