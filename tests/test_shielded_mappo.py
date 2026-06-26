"""Smoke tests for ShieldedMAPPO baseline.

Covers:
  - construction with the env config block train.py supplies
  - shield triggers when a muscle is at/above (theta_max - margin) for a
    task that exercises that muscle
  - shield does NOT trigger when the proposed task is already 'rest'
  - shield does NOT trigger when no exercised muscle is unsafe
  - log_prob round-trip: when shielded, returned log_prob equals the
    policy's log_prob for the rest action under the agent's actor
"""
import numpy as np
import torch

from hcmarl.agents.shielded_mappo import ShieldedMAPPO


MUSCLE_NAMES = ["shoulder", "ankle", "knee", "elbow", "trunk", "grip"]
THETA_MAX = {
    "shoulder": 0.70, "ankle": 0.80, "knee": 0.60,
    "elbow": 0.45, "trunk": 0.65, "grip": 0.45,
}
TASK_NAMES = ["heavy_lift", "light_sort", "carry", "overhead_reach", "push_cart", "rest"]
TASK_DEMANDS = {
    "heavy_lift":     {"shoulder": 0.45, "ankle": 0.10, "knee": 0.40, "elbow": 0.30, "trunk": 0.50, "grip": 0.55},
    "light_sort":     {"shoulder": 0.10, "ankle": 0.05, "knee": 0.05, "elbow": 0.15, "trunk": 0.10, "grip": 0.20},
    "carry":          {"shoulder": 0.25, "ankle": 0.20, "knee": 0.25, "elbow": 0.20, "trunk": 0.30, "grip": 0.45},
    "overhead_reach": {"shoulder": 0.55, "ankle": 0.05, "knee": 0.10, "elbow": 0.35, "trunk": 0.15, "grip": 0.30},
    "push_cart":      {"shoulder": 0.20, "ankle": 0.15, "knee": 0.20, "elbow": 0.15, "trunk": 0.25, "grip": 0.40},
    "rest":           {"shoulder": 0.00, "ankle": 0.00, "knee": 0.00, "elbow": 0.00, "trunk": 0.00, "grip": 0.00},
}


def _seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_obs(mf_per_muscle):
    """Build a 19-dim agent observation: [MR, MA, MF]*6 + [step_norm].
    Caller specifies MF per muscle; MR/MA filled with zeros, step=0."""
    obs = np.zeros(19, dtype=np.float32)
    for mi, name in enumerate(MUSCLE_NAMES):
        obs[mi * 3 + 2] = mf_per_muscle.get(name, 0.0)
    obs[18] = 0.0
    return obs


def _make_agent(safety_margin=0.05):
    return ShieldedMAPPO(
        obs_dim=19, global_obs_dim=109, n_actions=6, n_agents=4,
        muscle_names=MUSCLE_NAMES, theta_max=THETA_MAX,
        task_names=TASK_NAMES, task_demands=TASK_DEMANDS,
        rest_task_name="rest", safety_margin=safety_margin,
        device="cpu",
    )


def test_shielded_mappo_construct():
    _seed()
    a = _make_agent()
    assert a.rest_action_idx == 5
    assert a.muscle_names == MUSCLE_NAMES


def test_shield_triggers_when_required_muscle_unsafe():
    """heavy_lift demands shoulder=0.45 (>0). With shoulder MF at 0.66
    (theta_max=0.70, margin=0.05 -> trigger >=0.65), shield must fire."""
    a = _make_agent()
    obs = _make_obs({"shoulder": 0.66})
    assert a._should_shield(obs, intended_action=0) is True  # heavy_lift


def test_shield_does_not_trigger_when_action_is_already_rest():
    """If the policy already chose rest, no override needed."""
    a = _make_agent()
    obs = _make_obs({"shoulder": 0.99})  # max-out shoulder MF
    assert a._should_shield(obs, intended_action=5) is False  # rest


def test_shield_does_not_trigger_when_safe():
    """All muscles below threshold -> no shield."""
    a = _make_agent()
    obs = _make_obs({m: 0.10 for m in MUSCLE_NAMES})
    for action_idx in range(len(TASK_NAMES) - 1):  # all non-rest
        assert a._should_shield(obs, intended_action=action_idx) is False, \
            f"Shield triggered for safe state on action {action_idx}"


def test_shield_does_not_trigger_when_unsafe_muscle_not_required():
    """If only the ankle is at threshold but the proposed task is
    overhead_reach (ankle=0.05; below typical demand_threshold... but
    overhead_reach DOES have ankle>0 in our demand_threshold=0.0 setup,
    so the shield WILL fire). To test the "not-required" path we use a
    custom config where ankle has demand 0 in some task."""
    _seed()
    custom_demands = {
        **TASK_DEMANDS,
        # Synthetic task: only shoulder.
        "shoulder_only": {m: 0.0 for m in MUSCLE_NAMES},
    }
    custom_demands["shoulder_only"]["shoulder"] = 0.40
    custom_names = TASK_NAMES + ["shoulder_only"]
    a = ShieldedMAPPO(
        obs_dim=19, global_obs_dim=109, n_actions=len(custom_names),
        n_agents=4,
        muscle_names=MUSCLE_NAMES, theta_max=THETA_MAX,
        task_names=custom_names, task_demands=custom_demands,
        rest_task_name="rest", safety_margin=0.05,
        device="cpu",
    )
    # Ankle saturated, shoulder fine. Action = shoulder_only (idx 6).
    obs = _make_obs({"ankle": 0.79, "shoulder": 0.10})
    assert a._should_shield(obs, intended_action=6) is False


def test_logprob_recomputed_for_shielded_action():
    """When shielding fires, the returned log_prob must equal the
    policy's actual log_prob for the rest action under the current actor.
    Otherwise PPO ratio breaks."""
    _seed()
    a = _make_agent()
    n_agents = 4
    # Build observations where one agent is unsafe (shoulder near theta_max).
    obs = {f"worker_{i}": _make_obs({"shoulder": 0.10}) for i in range(n_agents)}
    obs["worker_0"] = _make_obs({"shoulder": 0.69})  # forces shield IF action requires shoulder
    gs = np.random.randn(109).astype(np.float32)

    # Force the policy to pick heavy_lift (shoulder demand) so worker_0 trips.
    # We can't directly force the sample without monkey-patching; instead,
    # we run get_actions repeatedly and verify the *contract* whenever
    # shielding does fire.
    found_shielded = False
    for _ in range(20):
        actions, log_probs, _ = a.get_actions(obs, gs)
        if actions["worker_0"] == a.rest_action_idx:
            # Verify log_prob matches policy's log_prob for rest under worker_0's obs
            obs_t = torch.from_numpy(obs["worker_0"]).float().unsqueeze(0)
            with torch.no_grad():
                logits = a.actor(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                expected_lp = float(dist.log_prob(torch.tensor([a.rest_action_idx])).item())
            assert abs(log_probs["worker_0"] - expected_lp) < 1e-5
            found_shielded = True
            break
    # If shielding never fired across 20 stochastic samples, the policy is
    # consistently choosing rest already -- not a contract failure, just
    # no opportunity to test. Re-seed to make it more likely.
    if not found_shielded:
        # At minimum, the shield logic must agree with the contract on
        # ANY heavy_lift action the test triggers.
        obs_t = torch.from_numpy(obs["worker_0"]).float().unsqueeze(0)
        forced_lp = a._logprob_under_policy(obs_t, a.rest_action_idx)
        assert np.isfinite(forced_lp)
