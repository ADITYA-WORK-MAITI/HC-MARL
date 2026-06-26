"""Tests for the two-axis (worker, seed) stratified bootstrap.

Worker-level variability is the dominant uncertainty axis when n_workers is
comparable to n_seeds; bootstrapping only over the seed axis underestimates
the CI in that regime."""
import numpy as np
import pytest

from hcmarl.aggregation import (
    iqm,
    worker_seed_stratified_bootstrap_iqm_ci,
    aggregate_by_method_two_axis,
)


def test_two_axis_bootstrap_brackets_truth_on_normal_matrix():
    """Bootstrap CI must contain the true mean on a Gaussian draw."""
    rng = np.random.default_rng(13)
    n_workers, n_seeds = 34, 10
    mu = 100.0
    mat = rng.normal(loc=mu, scale=5.0, size=(n_workers, n_seeds))
    lo, hi = worker_seed_stratified_bootstrap_iqm_ci(mat, n_resamples=2000, seed=7)
    assert lo < mu < hi
    # Reasonable CI width for the parameters above.
    assert (hi - lo) < 5.0


def test_two_axis_bootstrap_wider_than_seed_only_when_worker_variance_dominates():
    """When worker-level variance >> seed-level variance, the two-axis
    bootstrap CI must be at least as wide as a seed-only bootstrap. This
    is the structural reason for the two-axis version."""
    rng = np.random.default_rng(21)
    n_workers, n_seeds = 34, 10
    # Worker means drawn from N(100, sigma_w=10); per-cell N(worker_mean, sigma_s=0.1).
    worker_means = rng.normal(loc=100.0, scale=10.0, size=n_workers)
    mat = worker_means[:, None] + rng.normal(loc=0.0, scale=0.1,
                                               size=(n_workers, n_seeds))

    # Seed-only bootstrap collapses to per-seed mean over workers.
    seed_vector = mat.mean(axis=0)
    from hcmarl.aggregation import stratified_bootstrap_iqm_ci
    lo_s, hi_s = stratified_bootstrap_iqm_ci(seed_vector, n_resamples=2000, seed=7)

    # Two-axis bootstrap.
    lo_2, hi_2 = worker_seed_stratified_bootstrap_iqm_ci(mat, n_resamples=2000, seed=7)

    # The two-axis interval should be MUCH wider when worker variance dominates.
    assert (hi_2 - lo_2) > 5.0 * (hi_s - lo_s)


def test_two_axis_rejects_non_2d_input():
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        worker_seed_stratified_bootstrap_iqm_ci(arr)


def test_aggregate_two_axis_structure():
    rng = np.random.default_rng(42)
    n_workers, n_seeds = 34, 10
    matrices = {
        "hcmarl": rng.normal(loc=120, scale=2, size=(n_workers, n_seeds)),
        "mappo":  rng.normal(loc=100, scale=2, size=(n_workers, n_seeds)),
    }
    out = aggregate_by_method_two_axis(matrices, n_resamples=500)
    assert set(out.keys()) == {"hcmarl", "mappo"}
    for method, stats in out.items():
        assert set(stats.keys()) == {"iqm", "ci_lo", "ci_hi", "n_workers", "n_seeds"}
        assert stats["n_workers"] == n_workers
        assert stats["n_seeds"] == n_seeds
        assert stats["ci_lo"] <= stats["iqm"] <= stats["ci_hi"]
    # IQM ordering should preserve the loc gap (120 > 100) almost surely.
    assert out["hcmarl"]["iqm"] > out["mappo"]["iqm"]


def test_iqm_unaffected_by_new_function():
    """Sanity: the existing iqm function still works on a 1-D vector
    after adding two-axis helpers."""
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    assert iqm(arr) == pytest.approx(5.0, abs=1e-9)
