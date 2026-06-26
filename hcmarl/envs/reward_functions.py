"""
HC-MARL Canonical Reward and Cost Functions.

Single source of truth for all reward/cost computation across environments.
Matches the mathematical modelling document exactly:

  Reward (Eq 33 + Eq 32):
    r_i(t) = ln(max(U(i,j) - D_i(MF_i), epsilon)) - lambda_safety * violations

  where:
    U(i,j) = sum of task demands T_L^(j) across muscle groups  (Def 7.1)
    D_i(MF) = kappa * MF^2 / (1 - MF)                         (Eq 32)
    epsilon > 0 small (Eq 31: rest surplus)
    violations = number of muscles where MF_m > theta_max_m

  Cost (for MAPPO-Lagrangian):
    c_i(t) = 1 if any muscle exceeds theta_max, else 0         (binary)

Note on the log-transform: The NSWF objective (Eq 33) maximises
sum_i ln(surplus_i). We use ln(surplus) as the per-step reward so that
the RL objective (maximise cumulative reward) directly corresponds to
the NSWF objective (maximise product of surpluses via log-sum).

When surplus <= 0 (worker too fatigued for the task), reward = ln(epsilon),
a large negative value that strongly discourages overwork. This matches
the math doc: when D_i exceeds U(i,j), the worker should rest.
"""

import math
from typing import Dict

from hcmarl.nswf_allocator import NSWF_EPSILON


def disagreement_utility(avg_mf: float, kappa: float = 1.0,
                         disagreement_type: str = "divergent") -> float:
    """Eq 32: D_i(MF) = kappa * MF^2 / (1 - MF).

    Properties (Proposition 6.3):
      D(0) = 0, D'(MF) > 0 for MF in (0,1), D(MF) -> inf as MF -> 1.

    Args:
        avg_mf: Average (or max) fatigued fraction.
        kappa: Scaling constant (Eq 32).
        disagreement_type: "divergent" (Eq 32, default) or "constant" (D_i = kappa).
            The "constant" variant is used in the no_divergent ablation to test
            whether fatigue-dependent disagreement utility matters.
    """
    if disagreement_type == "constant":
        return kappa
    # S-5: Symmetric boundary clamping — both MF<0 (float noise) and
    # MF>=1 (Euler overshoot) are handled by clamping, not by raising
    # ValueError. Asymmetric handling was flagged in audit.
    mf = max(0.0, min(avg_mf, 0.999))
    return kappa * (mf ** 2) / (1.0 - mf)


def nswf_reward(
    productivity: float,
    fatigue_values: Dict[str, float],
    theta_max: Dict[str, float],
    kappa: float = 1.0,
    safety_weight: float = 5.0,
    epsilon: float = NSWF_EPSILON,
    disagreement_type: str = "divergent",
) -> float:
    """Canonical HC-MARL reward matching Eq 33 + Eq 32.

    Args:
        productivity: U(i,j) = sum of task demands across muscles.
        fatigue_values: {muscle_name: MF_value} for the worker.
        theta_max: {muscle_name: theta_max_value} safety thresholds.
        kappa: Disagreement utility scaling (Eq 32).
        safety_weight: Penalty weight per violated muscle.
        epsilon: Rest surplus (Eq 31), must be > 0.
        disagreement_type: "divergent" (Eq 32) or "constant" (D_i = kappa).

    Returns:
        Scalar reward for this worker at this timestep.
    """
    # S-15: Aggregate multi-muscle fatigue via max(MF) — conservative worst-case
    # muscle determines disagreement utility. Consistent with
    # pipeline.WorkerState.fatigue_for_allocation() which also uses max(MF).
    # Math doc Eq 32 uses scalar MF; max is the natural choice since the
    # most-fatigued muscle limits the worker's capacity for the task.
    max_mf = max(fatigue_values.values()) if fatigue_values else 0.0
    di = disagreement_utility(max_mf, kappa, disagreement_type)
    surplus = productivity - di

    # Log-transform matches NSWF objective (Eq 33)
    # When surplus <= 0, worker should rest -> reward = ln(epsilon) < 0
    log_surplus = math.log(max(surplus, epsilon))

    # Safety violation penalty
    violations = sum(
        1 for m in fatigue_values
        if m in theta_max and fatigue_values[m] > theta_max[m]
    )

    return log_surplus - safety_weight * violations


def safety_cost(
    fatigue_values: Dict[str, float],
    theta_max: Dict[str, float],
) -> float:
    """Dense cost signal for constrained RL (MAPPO-Lagrangian and reward shaping).

    Returns sum of max(0, MF_m - theta_max_m) across all muscles,
    providing gradient signal proportional to how far INTO the
    unsafe region each muscle is. Zero when all muscles are safe.

    This replaces the previous binary cost (0 or 1), which gave
    no marginal incentive when violations were deterministic.
    """
    cost = 0.0
    for m in fatigue_values:
        if m in theta_max:
            excess = fatigue_values[m] - theta_max[m]
            if excess > 0:
                cost += excess
    return cost
