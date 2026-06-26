"""Tests covering the Serious-tier audit fixes (S1, S2, S8).

S1 — MMICRL raw thetas are rescaled into [floor, 1] preserving ordering;
     MI-collapse falls back to floor.
S2 — Unknown muscle raises rather than falling back to a silent 0.5 default.
S8 — MMICRL.fit accepts an explicit n_actions and does not warn.
"""
import warnings

import pytest

from hcmarl.utils import build_per_worker_theta_max, _rescale_into_feasibility


# ---------------------------------------------------------------------------
# S1 — rescale preserves ordering, respects floor, collapses on low MI
# ---------------------------------------------------------------------------


def test_s1_rescale_preserves_ordering_and_floor():
    """Raw thetas < floor must map into [floor, 1] while keeping rank order."""
    theta_per_type = {
        "0": {"shoulder": 0.21},
        "1": {"shoulder": 0.27},
        "2": {"shoulder": 0.36},
    }
    floors = {"shoulder": 0.70}
    out = _rescale_into_feasibility(theta_per_type, floors, mi=0.5)

    vals = [out["0"]["shoulder"], out["1"]["shoulder"], out["2"]["shoulder"]]
    # Rescale upper bound is the OPEN interval (floor, 1)
    # because ECBFFilter validates theta_max in (0, 1) strictly. Allow up to
    # (1 - THETA_MAX_EPSILON = 0.999), strictly below 1.0.
    for v in vals:
        assert 0.70 - 1e-9 <= v < 1.0
    # Ordering from raw is preserved
    assert vals[0] < vals[1] < vals[2]
    # Min maps to floor, max maps to (1 - epsilon) = 0.999.
    assert abs(vals[0] - 0.70) < 1e-6
    assert abs(vals[-1] - 0.999) < 1e-6


def test_s1_mi_collapse_falls_back_to_floor():
    """If MI is below the collapse threshold, every type gets the floor."""
    theta_per_type = {
        "0": {"shoulder": 0.21},
        "1": {"shoulder": 0.36},
    }
    floors = {"shoulder": 0.70}
    out = _rescale_into_feasibility(theta_per_type, floors, mi=0.0005)
    assert all(abs(out[k]["shoulder"] - 0.70) < 1e-9 for k in out)


def test_rescale_caps_strictly_below_one():
    """Regression: ECBFFilter validates theta_max in OPEN interval (0, 1).
    The rescale must never produce exactly 1.0 even when one type is the
    clear MMICRL maximum, otherwise ECBFFilter construction crashes downstream
    (open-interval rescale)."""
    # Wide raw spread + high MI = the worst-case input that previously hit 1.0.
    theta_per_type = {
        "0": {"shoulder": 0.05, "elbow": 0.10},
        "1": {"shoulder": 0.55, "elbow": 0.40},
        "2": {"shoulder": 0.95, "elbow": 0.85},
    }
    floors = {"shoulder": 0.30, "elbow": 0.20}
    out = _rescale_into_feasibility(theta_per_type, floors, mi=0.9)
    for k in out:
        for muscle, v in out[k].items():
            assert v < 1.0, (
                f"theta[{k}][{muscle}] = {v} hit closed interval upper bound; "
                "ECBFFilter requires theta_max in open (0, 1)"
            )
            assert v > 0.0
    # The maximum should land at exactly (1 - THETA_MAX_EPSILON) = 0.999
    assert abs(out["2"]["shoulder"] - 0.999) < 1e-9
    assert abs(out["2"]["elbow"] - 0.999) < 1e-9


def test_s1_rescale_identical_types_also_collapses():
    """When all types share the same raw theta (zero span), fall back to floor."""
    theta_per_type = {
        "0": {"shoulder": 0.3},
        "1": {"shoulder": 0.3},
    }
    floors = {"shoulder": 0.70}
    out = _rescale_into_feasibility(theta_per_type, floors, mi=0.5)
    assert all(abs(out[k]["shoulder"] - 0.70) < 1e-9 for k in out)


def test_s1_build_per_worker_uses_rescale_and_logs_distinct_thetas():
    """build_per_worker_theta_max with rescale_to_floor=True should produce
    at least two distinct worker theta values when MMICRL found a real spread."""
    mmicrl_results = {
        "theta_per_type": {
            "0": {"shoulder": 0.21},
            "1": {"shoulder": 0.36},
        },
        "type_proportions": [0.5, 0.5],
        "mutual_information": 0.5,
    }
    floors = {"shoulder": 0.70}
    per_worker = build_per_worker_theta_max(
        mmicrl_results, floors, n_workers=4, method="hcmarl",
        rescale_to_floor=True,
    )
    worker_thetas = {w: per_worker[w]["shoulder"] for w in per_worker}
    # Two distinct clusters
    assert len(set(round(v, 6) for v in worker_thetas.values())) == 2
    assert all(v >= 0.70 - 1e-9 for v in worker_thetas.values())


# ---------------------------------------------------------------------------
# S2 — unknown muscle must raise, not silently substitute 0.5
# ---------------------------------------------------------------------------


def test_s2_unknown_muscle_raises():
    mmicrl_results = {
        "theta_per_type": {"0": {"bogus_muscle": 0.4}},
        "type_proportions": [1.0],
        "mutual_information": 0.5,
    }
    with pytest.raises(ValueError, match="missing from config"):
        build_per_worker_theta_max(
            mmicrl_results, {"shoulder": 0.7}, n_workers=2, method="hcmarl",
        )


# ---------------------------------------------------------------------------
# S8 — passing n_actions explicitly silences the auto-detect warning
# ---------------------------------------------------------------------------


def test_s8_mmicrl_fit_accepts_explicit_n_actions():
    """Regression test: MMICRL.fit emits no n_actions warning when n_actions
    is passed explicitly."""
    import numpy as np
    from hcmarl.mmicrl import MMICRL, DemonstrationCollector

    rng = np.random.default_rng(0)
    collector = DemonstrationCollector(n_muscles=1)
    for w in range(4):
        traj = []
        for _ in range(12):
            s = rng.uniform(0, 1, size=3).astype(np.float32)
            a = int(rng.integers(0, 6))
            traj.append((s, a))
        collector.demonstrations.append(traj)
        collector.worker_ids.append(w)

    mmicrl = MMICRL(n_types=2, n_muscles=1, n_iterations=2,
                    auto_select_k=False, hidden_dims=[16, 16])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mmicrl.fit(collector, n_actions=6)
    assert not any("n_actions auto-detected" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# S1 — legacy hard-clamp path (rescale_to_floor=False) still works
# ---------------------------------------------------------------------------


def test_s1_legacy_hardclamp_still_available():
    mmicrl_results = {
        "theta_per_type": {
            "0": {"shoulder": 0.21},
            "1": {"shoulder": 0.36},
        },
        "type_proportions": [0.5, 0.5],
        "mutual_information": 0.5,
    }
    floors = {"shoulder": 0.70}
    per_worker = build_per_worker_theta_max(
        mmicrl_results, floors, n_workers=4, method="hcmarl",
        rescale_to_floor=False,
    )
    # Hard-clamp path: every worker gets exactly the floor (raw < floor)
    assert all(abs(per_worker[w]["shoulder"] - 0.70) < 1e-9 for w in per_worker)
