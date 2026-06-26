"""Tests for Option X: per-worker type-conditioned observation.

Pins the env's handling of `n_types` and `worker_type_assignments`:
  - n_types=0 (default): legacy behavior, obs_dim unchanged.
  - n_types>0: obs_dim grows by n_types; observation includes one-hot z_i.
  - MI-collapse pattern (n_types=1, all workers z=0): constant one-hot,
    policy learns to ignore it (single-type fallback).
  - K>1 with distinct assignments: each worker gets its own one-hot.

Also pins input validation: out-of-range type assignments raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo


# --- Baseline: no type-conditioning ---

def test_legacy_behavior_unchanged_when_n_types_zero():
    """n_types=0 (default) must leave obs_dim identical to the pre-Option-X
    env. Regression guard against accidental obs_dim drift."""
    env = WarehousePettingZoo(n_workers=4, max_steps=10)
    legacy_obs_dim = env.obs_dim
    env.reset()
    obs = env._get_obs(0)
    assert obs.shape == (legacy_obs_dim,)
    # No type-conditioning attributes should affect obs.
    assert env.n_types == 0


# --- K=1 (MI collapse) ---

def test_mi_collapse_n_types_1_adds_constant_one_hot():
    """n_types=1 with every worker z=0: obs gains a single element = 1.0."""
    env = WarehousePettingZoo(
        n_workers=6, max_steps=10,
        n_types=1,
        worker_type_assignments={i: 0 for i in range(6)},
    )
    env.reset()
    for w in range(6):
        obs = env._get_obs(w)
        # Last element must be 1.0 (one-hot of type 0 with n_types=1).
        assert obs[-1] == 1.0, f"worker_{w} missing one-hot(0) tail"
    # obs_dim grew by exactly 1.
    env_no_tc = WarehousePettingZoo(n_workers=6, max_steps=10)
    assert env.obs_dim == env_no_tc.obs_dim + 1


# --- K=3 distinct assignments ---

def test_k3_distinct_assignments_distinct_one_hot():
    """Workers with different type assignments receive different one-hot tails."""
    env = WarehousePettingZoo(
        n_workers=6, max_steps=10,
        n_types=3,
        worker_type_assignments={0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2},
    )
    env.reset()
    # Check tails.
    tail_by_worker = {}
    for w in range(6):
        obs = env._get_obs(w)
        tail = obs[-3:].tolist()
        tail_by_worker[w] = tail
    assert tail_by_worker[0] == [1.0, 0.0, 0.0]
    assert tail_by_worker[1] == [0.0, 1.0, 0.0]
    assert tail_by_worker[2] == [0.0, 0.0, 1.0]
    assert tail_by_worker[3] == [1.0, 0.0, 0.0]  # same type as worker 0
    assert tail_by_worker[4] == [0.0, 1.0, 0.0]
    assert tail_by_worker[5] == [0.0, 0.0, 1.0]
    # obs_dim grew by exactly 3.
    env_no_tc = WarehousePettingZoo(n_workers=6, max_steps=10)
    assert env.obs_dim == env_no_tc.obs_dim + 3


# --- Validation ---

def test_out_of_range_type_raises():
    """A worker assigned to a type >= n_types must raise at construction."""
    with pytest.raises(ValueError, match="out of range"):
        WarehousePettingZoo(
            n_workers=3, max_steps=10,
            n_types=2,
            worker_type_assignments={0: 0, 1: 1, 2: 2},  # type 2 invalid
        )


def test_missing_worker_defaults_to_type_zero():
    """If worker_type_assignments omits a worker, it defaults to type 0."""
    env = WarehousePettingZoo(
        n_workers=4, max_steps=10,
        n_types=3,
        worker_type_assignments={0: 1, 2: 2},  # workers 1 and 3 missing
    )
    assert env.worker_type_assignments[1] == 0
    assert env.worker_type_assignments[3] == 0


# --- reset/step preserve the augmentation ---

def test_reset_and_step_preserve_type_one_hot():
    """After reset and a step, the type one-hot must still ride along in obs."""
    env = WarehousePettingZoo(
        n_workers=2, max_steps=10,
        n_types=2,
        worker_type_assignments={0: 0, 1: 1},
    )
    obs_dict, _ = env.reset()
    # reset returns dict {'worker_0': obs, ...}
    assert obs_dict['worker_0'][-2:].tolist() == [1.0, 0.0]
    assert obs_dict['worker_1'][-2:].tolist() == [0.0, 1.0]

    # Step with the action format the env expects (task indices, discrete).
    actions = {'worker_0': 0, 'worker_1': 0}
    obs_dict, _, _, _, _ = env.step(actions)
    # Tail is still correct.
    assert obs_dict['worker_0'][-2:].tolist() == [1.0, 0.0]
    assert obs_dict['worker_1'][-2:].tolist() == [0.0, 1.0]


# --- obs_dim accounting matches n_types across action modes ---

def test_obs_dim_accounting_discrete_and_continuous():
    """obs_dim must grow by exactly n_types in both discrete and continuous
    action modes."""
    for action_mode in ("discrete", "continuous"):
        no_tc = WarehousePettingZoo(n_workers=3, max_steps=10, action_mode=action_mode)
        with_tc = WarehousePettingZoo(
            n_workers=3, max_steps=10, action_mode=action_mode,
            n_types=4,
            worker_type_assignments={0: 0, 1: 1, 2: 2},
        )
        assert with_tc.obs_dim == no_tc.obs_dim + 4, (
            f"obs_dim delta wrong in {action_mode}: "
            f"no_tc={no_tc.obs_dim}, with_tc={with_tc.obs_dim}"
        )


# --- Determinism: same inputs -> identical obs ---

def test_deterministic_one_hot_across_resets():
    """Two resets with same seed and same type assignments produce identical
    type-one-hot tails."""
    env1 = WarehousePettingZoo(
        n_workers=3, max_steps=10,
        n_types=2,
        worker_type_assignments={0: 0, 1: 1, 2: 0},
    )
    env2 = WarehousePettingZoo(
        n_workers=3, max_steps=10,
        n_types=2,
        worker_type_assignments={0: 0, 1: 1, 2: 0},
    )
    env1.reset(seed=42)
    env2.reset(seed=42)
    for w in range(3):
        assert (env1._get_obs(w)[-2:] == env2._get_obs(w)[-2:]).all()
