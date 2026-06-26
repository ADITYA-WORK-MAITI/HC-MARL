"""
3CC-r: Three-Compartment Controller with Reperfusion.

Implements the physiological fatigue model from:
    - Liu, Brown & Yue (2002): Original 3CC model
    - Xia & Frey-Law (2008): Submaximal extension, MF->MR recovery route
    - Looft, Herkert & Frey-Law (2018): Reperfusion multiplier
    - Frey-Law, Looft & Heitsman (2012): Monte Carlo parameter calibration
    - Frey-Law & Avin (2010): Meta-analysis endurance data

Mathematical reference: HC-MARL Framework v12, Sections 3.1--3.5, 7.2.
All equation numbers below refer to that document.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp


# =========================================================================
# Muscle parameters (Table 1 of framework v12)
# =========================================================================

@dataclasses.dataclass(frozen=True)
class MuscleParams:
    """Calibrated parameters for a single muscle group.

    Sources:
        F, R  -- Frey-Law, Looft & Heitsman (2012), Table 2.
        r     -- Looft, Herkert & Frey-Law (2018); shoulder validated
                  by Looft & Frey-Law (2020).
    """

    name: str
    F: float       # Fatigue rate constant [min^{-1}]
    R: float       # Base recovery rate constant [min^{-1}]
    r: float       # Reperfusion multiplier (dimensionless, r > 1)

    # -- Derived quantities (Eqs 6, 25) --

    @property
    def C_max(self) -> float:
        """Maximum sustainable neural drive [min^{-1}] (Eq 6)."""
        return self.F * self.R / (self.F + self.R)

    @property
    def delta_max(self) -> float:
        """Maximum sustainable duty cycle (dimensionless) (Eq 6)."""
        return self.R / (self.F + self.R)

    @property
    def Rr(self) -> float:
        """Reperfusion-enhanced recovery rate R*r [min^{-1}]."""
        return self.R * self.r

    @property
    def theta_min_max(self) -> float:
        """Rest-phase safety threshold F/(F + R*r) (Eq 25)."""
        return self.F / (self.F + self.R * self.r)

    @property
    def Rr_over_F(self) -> float:
        """Ratio Rr/F -- determines rest-phase overshoot potential."""
        return self.Rr / self.F


# -- Calibrated parameter sets from Table 1 (isometric regime) --
# These are ISOMETRIC fatigue/recovery rates from Frey-Law, Looft &
# Heitsman (2012), calibrated against sustained isometric endurance data
# (Frey-Law & Avin 2010 meta-analysis of 194 publications).
# For dynamic tasks (e.g. WSD4FEDSRM rotations), the effective F is
# 30-180x larger because dynamic contractions impose higher metabolic
# demand per unit time. See real_data_calibration.py for dynamic regime.

# Values match Frey-Law, Looft & Heitsman (2012) Table 1 verbatim.
# Cross-verified: shoulder (F,R) in Looft & Frey-Law 2020 L206; grip (F,R,r)
# in Looft et al. 2018 Table 2. r=15 for all joints except grip (r=30).
SHOULDER = MuscleParams(name="shoulder", F=0.01820, R=0.00168, r=15)
ANKLE    = MuscleParams(name="ankle",    F=0.00589, R=0.00058, r=15)
KNEE     = MuscleParams(name="knee",     F=0.01500, R=0.00149, r=15)
ELBOW    = MuscleParams(name="elbow",    F=0.00912, R=0.00094, r=15)
TRUNK    = MuscleParams(name="trunk",    F=0.00755, R=0.00075, r=15)
GRIP     = MuscleParams(name="grip",     F=0.00980, R=0.00064, r=30)

ALL_MUSCLES = [SHOULDER, ANKLE, KNEE, ELBOW, TRUNK, GRIP]

MUSCLE_REGISTRY: dict[str, MuscleParams] = {m.name: m for m in ALL_MUSCLES}


def get_muscle(name: str) -> MuscleParams:
    """Look up calibrated muscle parameters by name.

    Args:
        name: One of 'shoulder', 'ankle', 'knee', 'elbow', 'trunk', 'grip'.

    Returns:
        MuscleParams for the requested muscle group.

    Raises:
        KeyError: If the muscle name is not in the registry.
    """
    key = name.lower().strip()
    if key not in MUSCLE_REGISTRY:
        valid = ", ".join(sorted(MUSCLE_REGISTRY.keys()))
        raise KeyError(
            f"Unknown muscle group '{name}'. Valid options: {valid}"
        )
    return MUSCLE_REGISTRY[key]


# =========================================================================
# 3CC-r ODE system
# =========================================================================

@dataclasses.dataclass
class ThreeCCrState:
    """Physiological state vector x(t) = [MR, MA, MF] (Def 3.1).

    Each component is the fraction of motor units in that compartment.
    Conservation law (Eq 1): MR + MA + MF = 1.
    """

    MR: float  # Resting / Recovered
    MA: float  # Active / Generating force
    MF: float  # Fatigued / Metabolically refractory

    def __post_init__(self) -> None:
        """Validate physical constraints."""
        for attr in ("MR", "MA", "MF"):
            val = getattr(self, attr)
            if val < -1e-9 or val > 1.0 + 1e-9:
                raise ValueError(
                    f"{attr} = {val} is outside [0, 1]."
                )
        total = self.MR + self.MA + self.MF
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Conservation violated: MR + MA + MF = {total} != 1."
            )

    def as_array(self) -> np.ndarray:
        """Return state as numpy array [MR, MA, MF]."""
        return np.array([self.MR, self.MA, self.MF], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> ThreeCCrState:
        """Construct from numpy array [MR, MA, MF]."""
        return cls(MR=float(arr[0]), MA=float(arr[1]), MF=float(arr[2]))

    @classmethod
    def fresh(cls) -> ThreeCCrState:
        """Fully rested initial state: MR=1, MA=0, MF=0."""
        return cls(MR=1.0, MA=0.0, MF=0.0)


class ThreeCCr:
    """Three-Compartment Controller with Reperfusion (3CC-r).

    Implements the ODE system (Eqs 2--4) with the reperfusion switch (Eq 5)
    and the proportional neural drive controller (Eq 35).

    Args:
        params: Calibrated MuscleParams for the muscle group.
        kp: Proportional gain for baseline neural drive controller (Eq 35).
            Default 1.0 matches the production env (pettingzoo_wrapper,
            warehouse_env, every config ecbf.kp). Xia & Frey-Law 2008
            report LD=LR=10 but their unit convention differs from ours;
            our 1.0 is a design choice verified against endurance data.
    """

    def __init__(self, params: MuscleParams, kp: float = 1.0) -> None:
        self.params = params
        self.kp = kp

    # -----------------------------------------------------------------
    # Effective recovery rate (Eq 5)
    # -----------------------------------------------------------------

    def R_eff(self, target_load: float) -> float:
        """Compute effective recovery rate (Eq 5).

        Args:
            target_load: TL(t), fraction of MVC demanded. TL > 0 means work.

        Returns:
            R during work (TL > 0), R*r during rest (TL = 0).
        """
        if target_load > 0.0:
            return self.params.R
        else:
            return self.params.Rr

    # -----------------------------------------------------------------
    # Baseline neural drive controller (Eq 35)
    # -----------------------------------------------------------------

    def baseline_neural_drive(
        self, target_load: float, MA: float
    ) -> float:
        """Proportional feedback controller for neural drive (Eq 35).

        C(t) = kp * max(TL(t) - MA(t), 0)  if assigned (work phase)
        C(t) = 0                             if resting

        This is the behavioural prior from which MMICRL demonstrations
        are generated. The RL agent replaces this with a learned policy
        in Phase 3.

        Args:
            target_load: TL(t) in [0, 1].
            MA: Current active fraction.

        Returns:
            Neural drive C(t) in [0, kp].
        """
        if target_load <= 0.0:
            return 0.0
        return self.kp * max(target_load - MA, 0.0)

    # -----------------------------------------------------------------
    # ODE right-hand side (Eqs 2--4)
    # -----------------------------------------------------------------

    def ode_rhs(
        self, state: np.ndarray, C: float, target_load: float
    ) -> np.ndarray:
        """Compute dx/dt = f(x, C) for the 3CC-r system.

        State ordering: x = [MR, MA, MF].

        Equations:
            dMA/dt = C(t) - F * MA(t)                     (Eq 2)
            dMF/dt = F * MA(t) - Reff(t) * MF(t)          (Eq 3)
            dMR/dt = Reff(t) * MF(t) - C(t)               (Eq 4)

        Args:
            state: Array [MR, MA, MF].
            C: Neural drive [min^{-1}], already safety-filtered.
            target_load: TL(t), used only to determine Reff.

        Returns:
            Array [dMR/dt, dMA/dt, dMF/dt].
        """
        MR, MA, MF = state[0], state[1], state[2]
        F = self.params.F
        Reff = self.R_eff(target_load)

        dMA_dt = C - F * MA            # Eq 2
        dMF_dt = F * MA - Reff * MF    # Eq 3
        dMR_dt = Reff * MF - C         # Eq 4

        return np.array([dMR_dt, dMA_dt, dMF_dt], dtype=np.float64)

    # -----------------------------------------------------------------
    # Single Euler step (for RL environment integration)
    # -----------------------------------------------------------------

    def step_euler(
        self,
        state: ThreeCCrState,
        C: float,
        target_load: float,
        dt: float = 1.0,
    ) -> ThreeCCrState:
        """Advance state by one Euler step of size dt minutes.

        Args:
            state: Current state.
            C: Neural drive [min^{-1}].
            target_load: TL(t) in [0, 1].
            dt: Time step in minutes.

        Returns:
            New ThreeCCrState after dt minutes.
        """
        x = state.as_array()
        dx = self.ode_rhs(x, C, target_load)
        x_new = x + dt * dx

        # Guard: if any component drifts slightly negative due to
        # floating-point, clamp to zero and adjust MR to preserve
        # conservation exactly (no renormalization — that corrupts state).
        for i in range(3):
            if x_new[i] < 0.0:
                x_new[i] = 0.0
        # Re-derive MR from conservation: MR = 1 - MA - MF
        x_new[0] = 1.0 - x_new[1] - x_new[2]
        if x_new[0] < 0.0:
            # MA + MF > 1: proportionally scale MA and MF to sum to 1
            s = x_new[1] + x_new[2]
            if s > 0:
                x_new[1] /= s
                x_new[2] /= s
            x_new[0] = 0.0

        return ThreeCCrState.from_array(x_new)

    # -----------------------------------------------------------------
    # High-accuracy integration via scipy.integrate.solve_ivp
    # -----------------------------------------------------------------

    def simulate(
        self,
        state0: ThreeCCrState,
        target_load: float,
        duration: float,
        dt_eval: float = 0.1,
        C_override: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Simulate the 3CC-r system over a continuous time interval.

        Uses RK45 (Dormand-Prince) with adaptive step size for accuracy.

        Args:
            state0: Initial state.
            target_load: Constant TL over the interval.
            duration: Simulation duration in minutes.
            dt_eval: Evaluation time step for output [minutes].
            C_override: If provided, use this constant C instead of the
                baseline controller. Useful for testing.

        Returns:
            Dictionary with keys:
                't': time array [minutes]
                'MR': resting fraction trajectory
                'MA': active fraction trajectory
                'MF': fatigued fraction trajectory
                'C': neural drive trajectory
        """
        t_span = (0.0, duration)
        t_eval = np.arange(0.0, duration + dt_eval * 0.5, dt_eval)
        x0 = state0.as_array()

        # S-1: Removed dead C_values list that was appended inside the RHS
        # callback at every internal RK45 stage (non-monotonic, non-uniform
        # times) but never returned. C is recomputed at eval points below.

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            MR, MA, MF = x[0], x[1], x[2]
            if C_override is not None:
                C = C_override
            else:
                C = self.baseline_neural_drive(target_load, MA)
            return self.ode_rhs(x, C, target_load)

        sol = solve_ivp(
            rhs,
            t_span,
            x0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.5,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        # Recompute C at evaluation points for output
        C_out = np.zeros(len(sol.t))
        for i, ti in enumerate(sol.t):
            MA_i = sol.y[1, i]
            if C_override is not None:
                C_out[i] = C_override
            else:
                C_out[i] = self.baseline_neural_drive(target_load, MA_i)

        return {
            "t": sol.t,
            "MR": sol.y[0],
            "MA": sol.y[1],
            "MF": sol.y[2],
            "C": C_out,
        }

    # -----------------------------------------------------------------
    # Analytical checks
    # -----------------------------------------------------------------

    def verify_conservation(self, state: ThreeCCrState, tol: float = 1e-6) -> bool:
        """Check MR + MA + MF = 1 (Eq 1)."""
        total = state.MR + state.MA + state.MF
        return abs(total - 1.0) < tol

    def steady_state_work(self) -> ThreeCCrState:
        """Compute the maximum sustainable steady state (Eqs 7, 8).

        At the sustainability limit (Theorem 3.4):
            MR = 0
            MA = delta_max = R / (F + R)
            MF = C_max / R = F / (F + R)
        """
        dm = self.params.delta_max
        return ThreeCCrState(
            MR=0.0,
            MA=dm,
            MF=1.0 - dm,
        )
