"""Unit tests for hcmarl.nswf_allocator.

Verifies all equations from Framework v12, Section 6.
"""

import math

import numpy as np
import pytest

from hcmarl.nswf_allocator import AllocationResult, NSWFAllocator, NSWFParams


# =====================================================================
# NSWFParams validation
# =====================================================================

class TestNSWFParams:
    """Test parameter validation."""

    def test_valid_params(self):
        p = NSWFParams(kappa=1.0, epsilon=1e-3)
        assert p.kappa == 1.0

    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError, match="kappa"):
            NSWFParams(kappa=-1.0)

    def test_zero_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            NSWFParams(epsilon=0.0)


# =====================================================================
# Disagreement utility (Eq 32)
# =====================================================================

class TestDisagreementUtility:
    """Test D_i(MF) = kappa * MF^2 / (1 - MF) and Proposition 6.3."""

    def setup_method(self):
        self.alloc = NSWFAllocator(NSWFParams(kappa=1.0))

    def test_P1_zero_when_fresh(self):
        """(P1) D_i(0) = 0 -- no premium when fresh."""
        assert self.alloc.disagreement_utility(0.0) == 0.0

    def test_P2_monotonically_increasing(self):
        """(P2) D_i'(MF) > 0 for MF in (0, 1)."""
        mf_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        d_values = [self.alloc.disagreement_utility(mf) for mf in mf_values]
        for i in range(len(d_values) - 1):
            assert d_values[i + 1] > d_values[i], (
                f"D({mf_values[i+1]}) = {d_values[i+1]} <= "
                f"D({mf_values[i]}) = {d_values[i]}"
            )

    def test_P3_diverges_at_burnout(self):
        """(P3) D_i(MF) -> +inf as MF -> 1."""
        assert self.alloc.disagreement_utility(0.999) > 100.0
        assert self.alloc.disagreement_utility(0.9999) > 1000.0

    def test_P3_returns_inf_at_one(self):
        """D_i(1.0) = inf."""
        assert self.alloc.disagreement_utility(1.0) == float("inf")

    def test_formula_specific_value(self):
        """D_i(0.5) = kappa * 0.25 / 0.5 = 0.5."""
        assert abs(self.alloc.disagreement_utility(0.5) - 0.5) < 1e-10

    def test_formula_another_value(self):
        """D_i(0.3) = kappa * 0.09 / 0.7 = 0.12857..."""
        expected = 0.09 / 0.7
        assert abs(self.alloc.disagreement_utility(0.3) - expected) < 1e-10

    def test_low_fatigue_quadratic(self):
        """Remark 6.4: For MF << 1, D_i ~ kappa * MF^2."""
        mf = 0.01
        D = self.alloc.disagreement_utility(mf)
        approx = 1.0 * mf ** 2  # kappa = 1
        assert abs(D - approx) / approx < 0.02  # Within 2%

    def test_kappa_scaling(self):
        """D_i scales linearly with kappa."""
        alloc2 = NSWFAllocator(NSWFParams(kappa=2.0))
        mf = 0.3
        assert abs(
            alloc2.disagreement_utility(mf) - 2.0 * self.alloc.disagreement_utility(mf)
        ) < 1e-10


# =====================================================================
# Disagreement derivative (Prop 6.3 proof)
# =====================================================================

class TestDisagreementDerivative:
    """Test D_i'(MF) = kappa * MF * (2 - MF) / (1 - MF)^2."""

    def setup_method(self):
        self.alloc = NSWFAllocator(NSWFParams(kappa=1.0))

    def test_derivative_at_zero(self):
        """D'(0) = 0."""
        assert self.alloc.disagreement_derivative(0.0) == 0.0

    def test_derivative_positive(self):
        """D'(MF) > 0 for MF in (0, 1) -- confirms P2."""
        for mf in [0.01, 0.1, 0.5, 0.9]:
            assert self.alloc.disagreement_derivative(mf) > 0.0

    def test_derivative_numerical(self):
        """Numerical derivative should match analytical."""
        mf = 0.4
        h = 1e-7
        numerical = (
            self.alloc.disagreement_utility(mf + h)
            - self.alloc.disagreement_utility(mf - h)
        ) / (2 * h)
        analytical = self.alloc.disagreement_derivative(mf)
        assert abs(numerical - analytical) / analytical < 1e-5


# =====================================================================
# Rest utility (Eq 31)
# =====================================================================

class TestRestUtility:
    """Test U(i, 0) = D_i(MF_i) + epsilon (Eq 31)."""

    def setup_method(self):
        self.alloc = NSWFAllocator(NSWFParams(kappa=1.0, epsilon=1e-3))

    def test_rest_surplus_equals_epsilon(self):
        """Surplus of resting = U(i,0) - D_i = epsilon."""
        for mf in [0.0, 0.1, 0.5, 0.8]:
            u_rest = self.alloc.rest_utility(mf)
            d_i = self.alloc.disagreement_utility(mf)
            surplus = u_rest - d_i
            assert abs(surplus - 1e-3) < 1e-12


# =====================================================================
# Allocation solver (Eq 33)
# =====================================================================

class TestAllocation:
    """Test the NSWF allocation solver."""

    def setup_method(self):
        self.alloc = NSWFAllocator(NSWFParams(kappa=1.0, epsilon=1e-3))

    def test_all_rest_when_no_tasks(self):
        """N workers, 0 productive tasks: everyone rests."""
        N = 3
        utility_matrix = np.zeros((N, 0))
        fatigue = np.array([0.0, 0.1, 0.2])
        result = self.alloc.allocate(utility_matrix, fatigue)
        for i in range(N):
            assert result.assignments[i] == 0

    def test_fresh_workers_assigned_to_tasks(self):
        """Fresh workers with high utility should be assigned to tasks."""
        N, M = 2, 2
        # Worker 0 good at task 1, worker 1 good at task 2
        utility_matrix = np.array([
            [5.0, 1.0],
            [1.0, 5.0],
        ])
        fatigue = np.array([0.0, 0.0])
        result = self.alloc.allocate(utility_matrix, fatigue)

        # Both should be assigned to productive tasks (not rest)
        assert result.assignments[0] != 0
        assert result.assignments[1] != 0
        # Worker 0 should get task 1, worker 1 should get task 2
        assert result.assignments[0] == 1
        assert result.assignments[1] == 2

    def test_fatigued_worker_rests(self):
        """Highly fatigued worker should be sent to rest."""
        N, M = 2, 2
        utility_matrix = np.array([
            [2.0, 2.0],
            [2.0, 2.0],
        ])
        # Worker 1 is extremely fatigued: D_i(0.95) is very large
        fatigue = np.array([0.0, 0.95])
        result = self.alloc.allocate(utility_matrix, fatigue)

        # Worker 1 should rest (D_i too large for any task surplus to be positive)
        assert result.assignments[1] == 0

    def test_one_worker_one_task(self):
        """Simplest case: 1 worker, 1 task."""
        utility_matrix = np.array([[3.0]])
        fatigue = np.array([0.1])
        result = self.alloc.allocate(utility_matrix, fatigue)

        D = self.alloc.disagreement_utility(0.1)
        if 3.0 - D > 1e-3:  # Surplus exceeds rest surplus
            assert result.assignments[0] == 1
        else:
            assert result.assignments[0] == 0

    def test_no_two_workers_share_task(self):
        """Constraint (ii): each productive task assigned to at most one worker."""
        N, M = 4, 2
        utility_matrix = np.array([
            [5.0, 1.0],
            [5.0, 1.0],
            [5.0, 1.0],
            [5.0, 1.0],
        ])
        fatigue = np.zeros(N)
        result = self.alloc.allocate(utility_matrix, fatigue)

        # Count how many workers are assigned to each productive task
        task_counts: dict[int, int] = {}
        for i, j in result.assignments.items():
            if j > 0:
                task_counts[j] = task_counts.get(j, 0) + 1

        for j, count in task_counts.items():
            assert count <= 1, f"Task {j} assigned to {count} workers"

    def test_all_surpluses_positive(self):
        """All surpluses in the result must be positive (for log to work)."""
        N, M = 3, 3
        utility_matrix = np.array([
            [3.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 1.0, 3.0],
        ])
        fatigue = np.array([0.0, 0.1, 0.2])
        result = self.alloc.allocate(utility_matrix, fatigue)

        for i, s in result.surpluses.items():
            assert s > 0, f"Worker {i} has non-positive surplus {s}"

    def test_objective_is_sum_log_surpluses(self):
        """Objective = sum_i ln(surplus_i) (Eq 33)."""
        N, M = 2, 2
        utility_matrix = np.array([
            [5.0, 1.0],
            [1.0, 5.0],
        ])
        fatigue = np.array([0.0, 0.0])
        result = self.alloc.allocate(utility_matrix, fatigue)

        expected_obj = sum(
            math.log(s) for s in result.surpluses.values()
        )
        assert abs(result.objective_value - expected_obj) < 1e-6

    def test_result_has_all_workers(self):
        """Every worker must have an assignment."""
        N, M = 4, 2
        utility_matrix = np.ones((N, M))
        fatigue = np.zeros(N)
        result = self.alloc.allocate(utility_matrix, fatigue)
        assert len(result.assignments) == N
        for i in range(N):
            assert i in result.assignments

    def test_more_workers_than_tasks(self):
        """N > M: some workers must rest."""
        N, M = 5, 2
        utility_matrix = np.ones((N, M)) * 3.0
        fatigue = np.zeros(N)
        result = self.alloc.allocate(utility_matrix, fatigue)

        productive_count = sum(1 for j in result.assignments.values() if j > 0)
        rest_count = sum(1 for j in result.assignments.values() if j == 0)
        assert productive_count <= M
        assert rest_count >= N - M
