"""Statistical aggregation helpers for headline + ablation reporting.

Implements the rliable recipe (Agarwal et al. NeurIPS 2021) without taking a
hard dependency on the `rliable` package:
    - Interquartile Mean (IQM): mean of the middle 50% of scores, a robust
      point estimate that ignores the top/bottom quartile noise floor.
    - Stratified bootstrap 95% CI: resample within-method, compute IQM
      on each resample, take 2.5th and 97.5th percentiles.

The function signatures match rliable.library.get_interval_estimates so a
future migration is a one-line swap. Numpy is the only dependency.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def iqm(scores: np.ndarray) -> float:
    """Interquartile mean: arithmetic mean of values between the 25th and
    75th percentiles. For fewer than 4 samples, falls back to the plain
    mean (no middle-50% to take)."""
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        return float(arr.mean())
    q1, q3 = np.percentile(arr, [25.0, 75.0])
    mask = (arr >= q1) & (arr <= q3)
    if not mask.any():
        return float(arr.mean())
    return float(arr[mask].mean())


def stratified_bootstrap_iqm_ci(
    scores: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int | None = 4271316,
) -> Tuple[float, float]:
    """Stratified bootstrap 95% CI for the IQM.

    Stratification: samples are resampled WITH replacement from the same
    method's score vector. With a single method this is ordinary bootstrap;
    with multiple methods, call this per-method and report each CI.

    Args:
        scores: 1-D array of per-seed scores for a single method.
        n_resamples: number of bootstrap resamples.
        ci: confidence level (default 0.95).
        seed: rng seed for reproducibility.

    Returns:
        (lo, hi) tuple — the CI bounds on IQM.
    """
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = arr.size
    estimates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        estimates[i] = iqm(arr[idx])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(estimates, 100.0 * alpha))
    hi = float(np.percentile(estimates, 100.0 * (1.0 - alpha)))
    return (lo, hi)


def aggregate_by_method(
    scores_by_method: Dict[str, np.ndarray],
    n_resamples: int = 10_000,
) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper: {method: 1-D scores} -> {method: {iqm, ci_lo, ci_hi}}."""
    out: Dict[str, Dict[str, float]] = {}
    for method, scores in scores_by_method.items():
        point = iqm(scores)
        lo, hi = stratified_bootstrap_iqm_ci(scores, n_resamples=n_resamples)
        out[method] = {"iqm": point, "ci_lo": lo, "ci_hi": hi,
                       "n_seeds": int(np.asarray(scores).size)}
    return out


# ---------------------------------------------------------------------
# Worker-level uncertainty quantification.
# ---------------------------------------------------------------------

def worker_seed_stratified_bootstrap_iqm_ci(
    score_matrix: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int | None = 4271502,
) -> Tuple[float, float]:
    """Two-axis stratified bootstrap of IQM over (worker, seed) pairs.

    The default `stratified_bootstrap_iqm_ci` resamples only the seed
    dimension, so the resulting CI captures training-RNG variability but
    NOT worker-level variability. For ergonomic data with n_workers
    comparable to n_seeds, this is the dominant axis -- a review-level
    reader will note that bootstrapping only over seeds
    underestimates uncertainty.

    This function resamples BOTH axes independently with replacement,
    flattens the resampled matrix, and computes IQM. The resulting CI
    captures combined seed + worker variability.

    Args:
        score_matrix: (n_workers, n_seeds) array of per-worker per-seed
            scores. Each entry score_matrix[w, s] is the metric value
            (e.g. cumulative reward) for worker w under seed s.
        n_resamples: number of bootstrap resamples.
        ci: confidence level (default 0.95).
        seed: rng seed for reproducibility.

    Returns:
        (lo, hi) tuple -- the 95% CI bounds on IQM under combined
        worker + seed resampling.

    Notes
    -----
    For a single method, a 1-D vector of scores is recovered by
    aggregating the score_matrix along one axis (e.g. mean over workers
    per seed). The default `stratified_bootstrap_iqm_ci` operates on that
    aggregated vector and ONLY captures seed variability. The function
    here keeps both axes alive.
    """
    arr = np.asarray(score_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"score_matrix must be 2-D (n_workers, n_seeds); got shape {arr.shape}"
        )
    n_workers, n_seeds = arr.shape
    if n_workers == 0 or n_seeds == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        w_idx = rng.integers(0, n_workers, size=n_workers)
        s_idx = rng.integers(0, n_seeds, size=n_seeds)
        # Outer-product index: pick the resampled (worker, seed) cells.
        resample = arr[np.ix_(w_idx, s_idx)].ravel()
        estimates[i] = iqm(resample)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(estimates, 100.0 * alpha))
    hi = float(np.percentile(estimates, 100.0 * (1.0 - alpha)))
    return (lo, hi)


def aggregate_by_method_two_axis(
    score_matrix_by_method: Dict[str, np.ndarray],
    n_resamples: int = 10_000,
) -> Dict[str, Dict[str, float]]:
    """Two-axis-bootstrap analogue of aggregate_by_method.

    Args:
        score_matrix_by_method: {method: (n_workers, n_seeds) array}.

    Returns:
        {method: {iqm, ci_lo, ci_hi, n_workers, n_seeds}} where IQM is
        computed over the flattened matrix and CI uses the two-axis
        bootstrap above.
    """
    out: Dict[str, Dict[str, float]] = {}
    for method, mat in score_matrix_by_method.items():
        arr = np.asarray(mat, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"method '{method}' score matrix must be 2-D; got shape {arr.shape}"
            )
        point = iqm(arr.ravel())
        lo, hi = worker_seed_stratified_bootstrap_iqm_ci(arr, n_resamples=n_resamples)
        out[method] = {
            "iqm": point, "ci_lo": lo, "ci_hi": hi,
            "n_workers": int(arr.shape[0]),
            "n_seeds": int(arr.shape[1]),
        }
    return out
