"""
Shared utility functions for HC-MARL.

Provides configuration loading, logging helpers, and numerical utilities
used across all modules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml


# =========================================================================
# Logging
# =========================================================================

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently formatted logger.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# =========================================================================
# Configuration
# =========================================================================

def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def resolve_project_root() -> Path:
    """Find the project root directory (contains setup.py or .git).

    Walks up from this file's location until it finds a marker file.

    Returns:
        Path to the project root.

    Raises:
        RuntimeError: If the root cannot be found.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "setup.py").exists() or (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Cannot locate project root (no setup.py or .git found).")


# =========================================================================
# Numerical utilities
# =========================================================================

def clip_and_normalise(x: np.ndarray) -> np.ndarray:
    """DEPRECATED — do not use for physiological state.

    This function clips to [0,1] then divides by sum, which corrupts
    the physics of the 3CC-r ODE (see audit C-1/C-2/C-5).  Use the
    conservation-preserving guard instead:
        MA = max(0, MA); MF = max(0, MF); MR = 1 - MA - MF

    Kept only for backward compatibility with non-ODE callers.
    """
    import warnings
    warnings.warn(
        "clip_and_normalise is deprecated — use conservation-preserving guard",
        DeprecationWarning,
        stacklevel=2,
    )
    x = np.clip(x, 0.0, 1.0)
    total = x.sum()
    if total > 0.0:
        x = x / total
    return x


def safe_log(x: float, floor: float = 1e-20) -> float:
    """Compute log(x) with a floor to avoid log(0).

    Used in the NSWF objective: ln(U(i,j) - Di).

    Args:
        x: Value to take log of. Must be positive for meaningful result.
        floor: Minimum value to clamp to before taking log.

    Returns:
        log(max(x, floor))
    """
    return float(np.log(max(x, floor)))


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Check if a square matrix is positive definite.

    Used for QP feasibility verification in ECBF.

    Args:
        matrix: Square numpy array.

    Returns:
        True if all eigenvalues are positive.
    """
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues > 0))
    except np.linalg.LinAlgError:
        return False


def _get_floor(config_theta_defaults, muscle):
    """Look up biomech-feasibility floor for a muscle. Raise if unknown.

    Silent 0.3/0.5 fallbacks are unsafe — a config that omits a muscle
    should fail loudly rather than get a threshold below theta_min_max.
    """
    if muscle not in config_theta_defaults:
        raise ValueError(
            f"muscle '{muscle}' missing from config.environment.theta_max. "
            f"Known muscles: {sorted(config_theta_defaults.keys())}. "
            f"Add the muscle's biomech-feasible floor (> theta_min_max, see "
            f"math doc Eq 26) before building per-worker thresholds."
        )
    return float(config_theta_defaults[muscle])


def build_per_worker_theta_max_from_F(
    worker_F_R_r_per_muscle,
    config_theta_defaults,
    n_workers,
    alpha=0.05,
    z_clip=2.0,
    epsilon=0.005,
):
    """F-anchored per-worker safety ceilings.

    Computes theta_max[worker][muscle] from the worker's calibrated F
    relative to the population F distribution for that muscle. This is an
    alternative to MMICRL-rescale-based ceilings (see
    ``build_per_worker_theta_max``) that:

      1. Has the correct safety sign: higher F -> lower ceiling (stricter).
      2. Does not depend on demonstration trajectory percentiles (no ICRL
         category-error exposure).
      3. Degenerates cleanly when F is homogeneous (sd == 0 -> z == 0 ->
         every worker gets the config default).
      4. Is bounded by Eq 26 per-worker floor and the config default, so
         no rescaling can produce infeasible ceilings.

    Formula per worker w, per muscle m (mapping from calibrated F to ceiling):
        z_{w,m}       = clip((F_{w,m} - mean(F_m)) / sd(F_m), -z_clip, z_clip)
        raw_{w,m}     = config_default[m] - alpha * z_{w,m}
        floor_{w,m}   = F_{w,m} / (F_{w,m} + R_{w,m} * r_m)   [Eq 26, per worker]
        theta[w,m]    = clip(raw_{w,m}, floor_{w,m} + epsilon, config_default[m])

    Args:
        worker_F_R_r_per_muscle: dict {muscle_name: [(F_w, R_w, r_m) for w in 0..n_workers-1]}
            Per-worker physiological parameters for each muscle. Length of
            each muscle's list must equal n_workers.
        config_theta_defaults: {muscle: float} from
            config.environment.theta_max (biomech-literature ceilings).
        n_workers: Number of workers in the env.
        alpha: Standardised-F-to-ceiling slope. Default 0.05 (5pp ceiling
            delta per 1 SD of F). Tuned by the synthetic K=3 validation.
        z_clip: Outlier clamp on standardised F. Default 2.0.
        epsilon: Margin above the per-worker Eq 26 floor. Default 0.005.

    Returns:
        {'worker_0': {muscle: theta, ...}, 'worker_1': {...}, ...}

    Raises:
        ValueError: if a muscle in the config is missing from
            worker_F_R_r_per_muscle, or if list lengths do not equal n_workers.
    """
    # Validate inputs up front — fail loudly, not silently.
    for m in config_theta_defaults:
        if m not in worker_F_R_r_per_muscle:
            raise ValueError(
                f"muscle '{m}' missing from worker_F_R_r_per_muscle. "
                f"Every config muscle must have per-worker (F, R, r) tuples. "
                f"Got: {sorted(worker_F_R_r_per_muscle.keys())}."
            )
        if len(worker_F_R_r_per_muscle[m]) != n_workers:
            raise ValueError(
                f"muscle '{m}' has {len(worker_F_R_r_per_muscle[m])} worker "
                f"parameter tuples but n_workers={n_workers}."
            )

    theta_max = {}
    for w in range(n_workers):
        theta_max[f"worker_{w}"] = {}
        for m, fr_list in worker_F_R_r_per_muscle.items():
            F_w, R_w, r_m = fr_list[w]
            floor = _get_floor(config_theta_defaults, m)

            # Population stats for this muscle across workers.
            F_all = np.array([x[0] for x in fr_list], dtype=float)
            mu = float(F_all.mean())
            sd = float(F_all.std(ddof=0))

            # Standardise; if no spread (homogeneous F), z == 0 and the
            # worker inherits the config default.
            if sd < 1e-12:
                z = 0.0
            else:
                z = float((F_w - mu) / sd)
            z = float(np.clip(z, -z_clip, z_clip))

            # Sign convention: higher F -> lower ceiling (stricter safety).
            raw_ceiling = floor - alpha * z

            # Per-worker Eq 26 floor (biomech feasibility on this worker's F).
            denom = F_w + R_w * r_m
            eq26_floor = (F_w / denom) if denom > 0 else floor

            # One-sided design: never looser than the ergonomics-literature
            # default. EXCEPT when the worker's own Eq 26 floor already exceeds
            # that default (physiological edge case) -- feasibility wins.
            upper = max(float(floor), eq26_floor + epsilon)

            theta_max[f"worker_{w}"][m] = float(np.clip(
                raw_ceiling,
                eq26_floor + epsilon,
                upper,
            ))
    return theta_max


def _rescale_into_feasibility(theta_per_type, config_theta_defaults, mi=None,
                              mi_collapse_threshold=0.01):
    """Map raw per-type, per-muscle MMICRL thetas into [floor_m, 1.0]
    preserving discovered ordering.

    Raw MMICRL thetas come from the 90th-percentile MF per type (see
    hcmarl.mmicrl:_fit_thresholds). On single-muscle Path G demos these
    typically sit below the biomech-feasibility floor (theta_min_max,
    Eq 26 of the math doc), making the hard-clamp degenerate into the
    config default for every type — wiping out MMICRL's contribution.

    Rescaling is a monotonic, per-muscle linear map from the observed
    [v_min, v_max] to [floor_m, 1.0]. This preserves the rank-ordering
    MMICRL discovered across types while satisfying feasibility. Both
    raw and rescaled values are logged so nothing is hidden.

    If MMICRL's mutual information collapsed (MI < threshold) or all
    types produced identical theta on a muscle, there is no discovered
    heterogeneity on that muscle: rescaled thetas fall back to the
    floor (identity) and the caller is expected to emit a warning.
    """
    collapsed = mi is not None and float(mi) < mi_collapse_threshold
    type_keys = sorted(theta_per_type.keys(), key=lambda k: int(k))
    muscles = set()
    for k in type_keys:
        muscles.update(theta_per_type[k].keys())

    # Defensive fix:
    # the rescale's mathematical contract is the closed interval [floor, 1.0],
    # but ecbf_filter.py:66 enforces theta_max in the OPEN interval (0, 1) per
    # Assumption 5.5 / Eq 26. When MMICRL discovers structure (MI > threshold),
    # the highest-MI type previously rescaled to exactly 1.0, crashing
    # ECBFFilter construction. Cap the upper bound strictly below 1.0 by
    # shrinking head_room by a numerical epsilon. Preserves ordering, fixes
    # the open/closed mismatch uniformly across all 4 MMICRL-on rungs (and
    # implicitly no_ecbf, where ECBFFilter is still constructed with
    # validation even though ecbf_mode="off" later short-circuits its use).
    THETA_MAX_EPSILON = 1e-3
    rescaled = {k: {} for k in type_keys}
    for muscle in muscles:
        floor = _get_floor(config_theta_defaults, muscle)
        vals = [float(theta_per_type[k].get(muscle, floor)) for k in type_keys]
        v_min, v_max = min(vals), max(vals)
        span = v_max - v_min
        # Strict open-interval rescale: head_room shrunk by epsilon so the
        # highest-value type maps to (1 - epsilon), not 1.0.
        head_room = max(0.0, 1.0 - floor - THETA_MAX_EPSILON)

        if collapsed or span < 1e-6:
            for k in type_keys:
                rescaled[k][muscle] = floor
        else:
            for k, v in zip(type_keys, vals):
                rescaled[k][muscle] = floor + head_room * (v - v_min) / span
    return rescaled


def build_per_worker_theta_max(mmicrl_results, config_theta_defaults, n_workers,
                               method, rescale_to_floor=True,
                               mi_collapse_threshold=0.01):
    """Construct per-worker theta_max dict from MMICRL results.

    Single source of truth shared by train.py and evaluate.py so the env
    built at training time and the env built at evaluation time impose
    identical thresholds on the policy.

    Two strategies for mapping raw MMICRL theta into the biomech-feasible
    interval [floor_m, 1.0] (floor_m = theta_min_max, math doc Eq 26):

    - rescale_to_floor=True (default): monotonic per-muscle rescale of
      raw MMICRL theta into [floor_m, 1.0] preserving the discovered
      cross-type ordering. MI-collapse detection: if the MMICRL
      mutual_information is below mi_collapse_threshold, we treat the
      output as non-informative and fall back to floor for every type.
    - rescale_to_floor=False: hard-clamp each raw theta up to the floor
      (legacy behaviour, collapses to config default when raw < floor).

    Args:
        mmicrl_results: Dict from MMICRL.fit() with 'theta_per_type',
            'type_proportions', and 'mutual_information', or None.
        config_theta_defaults: Dict {muscle: float} from
            config.environment.theta_max (biomech-feasibility floors).
        n_workers: Number of workers in the env.
        method: Only 'hcmarl' uses MMICRL thresholds; other methods get
            the flat config dict.
        rescale_to_floor: If True, monotonically rescale into
            [floor_m, 1.0]; else hard-clamp.
        mi_collapse_threshold: If MMICRL MI < this, treat output as
            non-informative (all types -> floor). 0.01 is conservative.

    Returns:
        Either a per-worker dict {'worker_0': {muscle: theta}, ...}
        or the flat config_theta_defaults for non-hcmarl methods / when
        MMICRL produced nothing.
    """
    if method != "hcmarl" or not mmicrl_results:
        return config_theta_defaults

    theta_per_type = mmicrl_results.get("theta_per_type", {})
    type_proportions = mmicrl_results.get("type_proportions", [])
    mi = mmicrl_results.get("mutual_information", None)
    if not theta_per_type:
        return config_theta_defaults

    if rescale_to_floor:
        effective_theta = _rescale_into_feasibility(
            theta_per_type, config_theta_defaults, mi=mi,
            mi_collapse_threshold=mi_collapse_threshold,
        )
    else:
        effective_theta = {}
        for k, per_muscle in theta_per_type.items():
            effective_theta[k] = {
                m: max(float(v), _get_floor(config_theta_defaults, m))
                for m, v in per_muscle.items()
            }

    type_keys = sorted(effective_theta.keys(), key=lambda k: int(k))
    n_types = len(type_keys)
    theta_max = {}

    if type_proportions and len(type_proportions) == n_types:
        counts = np.round(np.array(type_proportions) * n_workers).astype(int)
        diff = n_workers - counts.sum()
        counts[np.argmax(counts)] += diff
        worker_type_map = []
        for t_idx, count in enumerate(counts):
            worker_type_map.extend([t_idx] * count)
        for w in range(n_workers):
            type_k = type_keys[worker_type_map[w]]
            theta_max[f"worker_{w}"] = dict(effective_theta[type_k])
    else:
        conservative = {}
        for type_k in type_keys:
            for muscle, val in effective_theta[type_k].items():
                if muscle not in conservative or val < conservative[muscle]:
                    conservative[muscle] = val
        theta_max = {f"worker_{w}": dict(conservative) for w in range(n_workers)}

    return theta_max


def seed_everything(seed: int, deterministic: bool = True,
                     high_determinism: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True (default), enable cudnn.deterministic — bit-exact
            reproducibility at ~1.5x slowdown (M6). Turn off only for throughput
            experiments where per-seed variance is acceptable; paper runs must
            keep it True.
        high_determinism: If True (default, B2 patch), additionally
            enable torch.use_deterministic_algorithms(True), force highest
            matmul precision (no TF32), and set CUBLAS_WORKSPACE_CONFIG so
            cuBLAS workspaces are sized deterministically. This is what EXP3
            Part 1 ran under (ARI=1.0, MI=1.099 on synthetic K=3) and is
            required for EXP1/EXP2 reruns to be apples-to-apples with EXP3.
            When False, matmul precision drops to "high" (TF32 enabled on
            Ampere/Ada) and non-deterministic CUDA ops are silently allowed.
            Some non-deterministic CUDA ops (e.g. scatter_add) raise under
            high_determinism=True; the EXP0 50K probe must run cleanly
            before launching the full grid.
    """
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if high_determinism:
        # Sized cuBLAS workspaces -> deterministic matmul on CUDA. Required
        # by torch.use_deterministic_algorithms when CUDA matmul is invoked.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # PyTorch seeding (deferred import, only available in Phase 3+)
    try:
        import torch
        import random as _python_random
        _python_random.seed(seed)
        torch.manual_seed(seed)
        if high_determinism:
            # Highest matmul precision = no TF32, full FP32. ~1.5x slower on
            # Ampere/Ada but bit-identical across runs.
            torch.set_float32_matmul_precision("highest")
            # Raise on first non-deterministic CUDA op rather than silently
            # using a non-deterministic kernel. warn_only=False is intentional:
            # we want a hard fail so we catch the offending op on the EXP0
            # probe rather than seeing untraceable variance in production.
            torch.use_deterministic_algorithms(True, warn_only=False)
        else:
            # TF32 matmul: ~1.5-2x throughput on Ampere/Ada at ~1e-3 relative
            # precision loss. Deterministic within fixed hardware + seed.
            torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # M-1 rationale: deterministic=True ensures bit-exact reproducibility
            # across runs at ~1.5x slowdown. benchmark=False prevents algorithm
            # re-selection between invocations.
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = not bool(deterministic)
            logger = get_logger(__name__)
            if high_determinism:
                logger.info(
                    "HIGH determinism: cudnn.deterministic=True, "
                    "torch.use_deterministic_algorithms(True), "
                    "matmul=highest, CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                    "(EXP3-Part-1-equivalent reproducibility)"
                )
            elif deterministic:
                logger.info("CUDA seed set; cudnn.deterministic=True (reproducible but ~1.5x slower)")
            else:
                logger.info("CUDA seed set; cudnn.deterministic=False (faster, not bit-exact reproducible)")
    except ImportError:
        pass
