"""NIOSH calibration regression tests.

F2 only: the F1 deliverable is a narrative markdown (docs/calibration_notes)
and has no executable surface to test. F2 ships scripts/niosh_calibration.py
with a full NIOSH Revised Lifting Equation implementation; these tests pin:

  * Each of the 6 multipliers against Waters et al. (1993) Table 1/5 values
    at canonical inputs.
  * HEAVY_LIFT_CANONICAL and CARRY_CANONICAL lifting indices match the
    values cited in docs/calibration_notes (LI = 1.38 and LI = 0.79).
  * sensitivity_sweep returns the right structure with both bounds on
    each of the six continuous geometry inputs.
  * The sweep keeps heavy_lift in the "elevated risk" bracket and carry
    in the "acceptable" bracket (so the paper claim survives ±20% on
    every input).
"""

from __future__ import annotations

import math

import pytest

from scripts.niosh_calibration import (
    CARRY_CANONICAL,
    HEAVY_LIFT_CANONICAL,
    NIOSHTask,
    asymmetric_multiplier,
    coupling_multiplier,
    distance_multiplier,
    frequency_multiplier,
    horizontal_multiplier,
    sensitivity_sweep,
    vertical_multiplier,
)


# ---------------------------------------------------------------------------
# Multiplier correctness vs. Waters et al. 1993 Table 1 / 5
# ---------------------------------------------------------------------------


def test_f2_horizontal_multiplier_canonical():
    # HM = 25/H. Identity at H=25; 25/50 = 0.5; zero beyond 63cm.
    assert horizontal_multiplier(25.0) == pytest.approx(1.0)
    assert horizontal_multiplier(50.0) == pytest.approx(0.5)
    assert horizontal_multiplier(64.0) == 0.0
    # H < 25 is clamped to 25 per the 1993 paper.
    assert horizontal_multiplier(10.0) == pytest.approx(1.0)


def test_f2_vertical_multiplier_canonical():
    # VM = 1 - 0.0031|V-75| per Waters et al. 1993 Appendix A. Identity at V=75; zero outside [0,175].
    assert vertical_multiplier(75.0) == pytest.approx(1.0)
    assert vertical_multiplier(0.0) == pytest.approx(1.0 - 0.0031 * 75.0)
    assert vertical_multiplier(175.0) == pytest.approx(1.0 - 0.0031 * 100.0)
    assert vertical_multiplier(200.0) == 0.0
    assert vertical_multiplier(-1.0) == 0.0


def test_f2_distance_multiplier_canonical():
    # DM = 0.82 + 4.5/D for D in [25, 175]. D < 25 returns 1.0; D > 175 = 0.
    assert distance_multiplier(25.0) == pytest.approx(0.82 + 4.5 / 25.0)
    assert distance_multiplier(70.0) == pytest.approx(0.82 + 4.5 / 70.0)
    assert distance_multiplier(10.0) == pytest.approx(1.0)
    assert distance_multiplier(200.0) == 0.0


def test_f2_asymmetric_multiplier_canonical():
    # AM = 1 - 0.0032*A for A in [0, 135]. Identity at 0; zero outside.
    assert asymmetric_multiplier(0.0) == pytest.approx(1.0)
    assert asymmetric_multiplier(30.0) == pytest.approx(1.0 - 0.0032 * 30.0)
    assert asymmetric_multiplier(135.0) == pytest.approx(1.0 - 0.0032 * 135.0)
    assert asymmetric_multiplier(180.0) == 0.0


def test_f2_frequency_multiplier_table_lookups():
    # Spot-check Waters 1993 Table 5: at 1 lift/min, V>=75, ≤1h -> 0.94.
    assert frequency_multiplier(1.0, duration_hr=1.0, v_cm=80.0) == pytest.approx(0.94)
    # 2 lifts/min, ≤2h, V>=75 -> 0.84.
    assert frequency_multiplier(2.0, duration_hr=2.0, v_cm=80.0) == pytest.approx(0.84)
    # 3 lifts/min, ≤2h, V>=75 -> 0.79.
    assert frequency_multiplier(3.0, duration_hr=2.0, v_cm=80.0) == pytest.approx(0.79)
    # Extrapolation clamps to the endpoint value.
    assert frequency_multiplier(100.0, duration_hr=8.0, v_cm=80.0) == pytest.approx(0.00)


def test_f2_frequency_multiplier_linear_interp():
    # Between 2/min (0.84) and 3/min (0.79) at t=0.5 -> 0.815.
    val = frequency_multiplier(2.5, duration_hr=2.0, v_cm=80.0)
    assert val == pytest.approx(0.815, abs=1e-3)


def test_f2_coupling_multiplier_canonical():
    assert coupling_multiplier("good") == pytest.approx(1.0)
    # fair: 1.0 if V>=75 else 0.95.
    assert coupling_multiplier("fair", v_cm=80.0) == pytest.approx(1.0)
    assert coupling_multiplier("fair", v_cm=50.0) == pytest.approx(0.95)
    assert coupling_multiplier("poor") == pytest.approx(0.90)
    with pytest.raises(ValueError):
        coupling_multiplier("tight")


# ---------------------------------------------------------------------------
# Canonical task lifting indices match calibration_notes.md numbers
# ---------------------------------------------------------------------------


def test_f2_heavy_lift_canonical_li_matches_paper():
    """docs/calibration_notes cites LI = 1.38 for heavy_lift."""
    li = HEAVY_LIFT_CANONICAL.lifting_index()
    assert li == pytest.approx(1.38, abs=0.02), (
        f"heavy_lift LI = {li:.3f} no longer matches the 1.38 value cited "
        f"in docs/calibration_notes — update one or the other"
    )
    # Must be in "elevated risk" bracket (1 < LI <= 3).
    assert 1.0 < li <= 3.0


def test_f2_carry_canonical_li_matches_paper():
    """docs/calibration_notes cites LI = 0.79 for carry."""
    li = CARRY_CANONICAL.lifting_index()
    assert li == pytest.approx(0.79, abs=0.02), (
        f"carry LI = {li:.3f} no longer matches the 0.79 value cited "
        f"in docs/calibration_notes — update one or the other"
    )
    # Must be in "acceptable for most workers" bracket (LI <= 1).
    assert li <= 1.0


def test_f2_rwl_heavy_lift_positive_and_finite():
    rwl = HEAVY_LIFT_CANONICAL.rwl_kg()
    assert math.isfinite(rwl)
    assert rwl > 0.0
    # Sanity: RWL = load/LI ~ 15 / 1.38 ~ 10.87.
    assert rwl == pytest.approx(15.0 / 1.38, abs=0.1)


# ---------------------------------------------------------------------------
# Sensitivity sweep structure + robustness claim
# ---------------------------------------------------------------------------


def test_f2_sensitivity_sweep_shape():
    sweep = sensitivity_sweep(HEAVY_LIFT_CANONICAL, pct=0.20)
    expected_keys = {"load_kg", "h_cm", "v_cm", "d_cm", "a_deg", "freq_per_min"}
    assert set(sweep.keys()) == expected_keys
    for k, (lo, hi) in sweep.items():
        assert math.isfinite(lo) and math.isfinite(hi)
        # For most continuous params, the sweep should yield a non-empty
        # interval around the canonical LI.
        assert lo >= 0.0 and hi >= 0.0


def test_f2_sensitivity_heavy_lift_stays_elevated():
    """Paper claim (calibration_notes.md §F2): ±20% sweep keeps heavy_lift in
    the elevated-risk bracket [1.10, 1.66]. If any swept LI leaves
    (1.0, 3.0], the paper's robustness assertion is violated."""
    sweep = sensitivity_sweep(HEAVY_LIFT_CANONICAL, pct=0.20)
    for k, (lo, hi) in sweep.items():
        assert 1.0 < lo <= 3.0, f"heavy_lift sweep {k} lo={lo:.2f} left elevated bracket"
        assert 1.0 < hi <= 3.0, f"heavy_lift sweep {k} hi={hi:.2f} left elevated bracket"


def test_f2_sensitivity_carry_stays_acceptable():
    """Paper claim: carry LI stays in (0, 1] across ±20% sweep."""
    sweep = sensitivity_sweep(CARRY_CANONICAL, pct=0.20)
    for k, (lo, hi) in sweep.items():
        assert 0.0 < lo <= 1.0, f"carry sweep {k} lo={lo:.2f} exceeded acceptable"
        assert 0.0 < hi <= 1.0, f"carry sweep {k} hi={hi:.2f} exceeded acceptable"


# ---------------------------------------------------------------------------
# NIOSHTask dataclass round-trip
# ---------------------------------------------------------------------------


def test_f2_nioshtask_round_trip():
    t = NIOSHTask(
        name="probe", load_kg=12.0,
        h_cm=40.0, v_cm=70.0, d_cm=50.0, a_deg=10.0,
        freq_per_min=1.0, duration_hr=1.0, coupling="good",
    )
    rwl = t.rwl_kg()
    li = t.lifting_index()
    assert math.isfinite(rwl) and rwl > 0.0
    assert li == pytest.approx(12.0 / rwl, abs=1e-6)
