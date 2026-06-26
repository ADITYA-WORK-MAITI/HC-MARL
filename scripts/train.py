"""
HC-MARL Phase 4 (#54): Main Training Script
=============================================
Load config, init env + agent, train loop, checkpoint, log to W&B/CSV.
Optionally runs MMICRL pre-training to discover worker types from real data.

Usage:
    python scripts/train.py --config config/hcmarl_full_config.yaml --seed 0
    python scripts/train.py --config config/mappo_config.yaml --seed 0 --method mappo
    python scripts/train.py --config config/hcmarl_full_config.yaml --seed 0 --device cuda
    python scripts/train.py --config config/hcmarl_full_config.yaml --seed 0 --mmicrl
"""

import argparse
import os
import sys
import time
import yaml
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
from hcmarl.agents.mappo import MAPPO
from hcmarl.agents.mappo_lag import MAPPOLagrangian
from hcmarl.agents.ippo import IPPO
from hcmarl.agents.happo import HAPPO
from hcmarl.agents.shielded_mappo import ShieldedMAPPO
from hcmarl.agents.hcmarl_agent import HCMARLAgent
from hcmarl.nswf_allocator import NSWFParams, create_allocator
from hcmarl.logger import HCMARLLogger


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    "hcmarl": "HC-MARL (MAPPO + ECBF + NSWF)",
    "mappo": "MAPPO (no safety filter)",
    "ippo": "IPPO (independent, no centralised critic)",
    "mappo_lag": "MAPPO-Lagrangian (cost critic + dual variable)",
    "happo": "HAPPO (heterogeneous trust-region PPO; Kuba 2022)",
    "shielded_mappo": "Shielded-MAPPO (MAPPO + static-threshold task-refusal shield)",
}


# ---------------------------------------------------------------------------
# Run-state persistence (C1: resume correctness)
# ---------------------------------------------------------------------------

def _write_run_state(path, global_step, episode_count, cost_ema, best_reward,
                     theta_max, seed, method,
                     n_types=0, worker_type_assignments=None):
    """Persist everything the policy checkpoint does NOT contain.

    Paired with an agent.save() call so a Colab runtime disconnect can be
    resumed without losing step counters, RNG state, or the theta_max the
    policy was trained against (which may come from MMICRL).

    n_types and worker_type_assignments are added for Option X (type-
    conditioned policy). They default to 0 / None so older checkpoints
    (written before Option X) still round-trip through _load_run_state.
    """
    import random as _random
    state = {
        "global_step": int(global_step),
        "episode_count": int(episode_count),
        "cost_ema": float(cost_ema),
        "best_reward": float(best_reward),
        "theta_max": theta_max,
        "seed": int(seed),
        "method": method,
        "n_types": int(n_types),
        "worker_type_assignments": dict(worker_type_assignments or {}),
        "rng_np": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_python": _random.getstate(),
    }
    torch.save(state, path)


def _load_run_state(path):
    """Load a run_state.pt written by _write_run_state."""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def create_agent(method, obs_dim, global_obs_dim, n_actions, n_agents, cfg, device,
                  action_mode="discrete", n_muscles=None, use_nswf=True):
    """Instantiate the correct agent class based on method name."""
    algo = cfg.get("algorithm", {})
    lr_actor = algo.get("lr_actor", 3e-4)
    lr_critic = algo.get("lr_critic", 1e-3)
    gamma = algo.get("gamma", 0.99)
    gae_lambda = algo.get("gae_lambda", 0.95)
    clip_eps = algo.get("clip_eps", 0.2)
    entropy_coeff = algo.get("entropy_coeff", 0.01)
    max_grad_norm = algo.get("max_grad_norm", 0.5)
    n_epochs = algo.get("n_epochs", 10)
    batch_size = algo.get("batch_size", 256)
    hidden_dim = algo.get("hidden_dim", 64)

    if method == "hcmarl":
        welfare_type = cfg.get("welfare_type", "nswf")
        allocation_interval = cfg.get("allocation_interval", 30)
        nswf_cfg = cfg.get("nswf", {})
        nswf_params = NSWFParams(
            kappa=nswf_cfg.get("kappa", 1.0),
            epsilon=nswf_cfg.get("epsilon", 1e-3),
        )
        return HCMARLAgent(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim,
            n_actions=n_actions, n_agents=n_agents,
            theta_max=cfg.get("environment", {}).get("theta_max", {}),
            ecbf_params=cfg.get("ecbf", {}),
            use_nswf=use_nswf,
            action_mode=action_mode,
            n_muscles=n_muscles,
            welfare_type=welfare_type,
            nswf_params=nswf_params,
            allocation_interval=allocation_interval,
            device=device,
            lr_actor=lr_actor, lr_critic=lr_critic,
            gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
            entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
        )
    elif method == "mappo":
        return MAPPO(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim,
            n_actions=n_actions, n_agents=n_agents,
            lr_actor=lr_actor, lr_critic=lr_critic,
            gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
            entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size, device=device,
        )
    elif method == "ippo":
        return IPPO(
            obs_dim=obs_dim, n_actions=n_actions, n_agents=n_agents,
            lr=lr_actor, gamma=gamma, gae_lambda=gae_lambda,
            clip_eps=clip_eps, entropy_coeff=entropy_coeff,
            max_grad_norm=max_grad_norm, n_epochs=n_epochs,
            batch_size=batch_size, hidden_dim=hidden_dim, device=device,
        )
    elif method == "mappo_lag":
        return MAPPOLagrangian(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim,
            n_actions=n_actions, n_agents=n_agents,
            lr_actor=lr_actor, lr_critic=lr_critic,
            lr_lambda=algo.get("lambda_lr", 5e-3),
            gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
            entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            cost_limit=algo.get("cost_limit", 0.1),
            lambda_init=algo.get("lambda_init", 0.5),
            device=device,
        )
    elif method == "happo":
        return HAPPO(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim,
            n_actions=n_actions, n_agents=n_agents,
            lr_actor=lr_actor, lr_critic=lr_critic,
            gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
            entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            hidden_dim=hidden_dim,
            critic_hidden_dim=algo.get("critic_hidden_dim", 128),
            device=device,
        )
    elif method == "shielded_mappo":
        env_cfg = cfg.get("environment", {}) or {}
        muscle_groups = env_cfg.get("muscle_groups", {}) or {}
        muscle_names = list(muscle_groups.keys())
        theta_max_cfg = env_cfg.get("theta_max", {}) or {}
        tasks_cfg = env_cfg.get("tasks", {}) or {}
        task_names = list(tasks_cfg.keys())
        if not task_names or "rest" not in task_names:
            raise ValueError(
                "shielded_mappo requires environment.tasks with a 'rest' entry; "
                f"got tasks={task_names}"
            )
        shield_cfg = cfg.get("shield", {}) or {}
        return ShieldedMAPPO(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim,
            n_actions=n_actions, n_agents=n_agents,
            muscle_names=muscle_names,
            theta_max=theta_max_cfg,
            task_names=task_names,
            task_demands=tasks_cfg,
            rest_task_name="rest",
            safety_margin=float(shield_cfg.get("safety_margin", 0.05)),
            demand_threshold=float(shield_cfg.get("demand_threshold", 0.0)),
            lr_actor=lr_actor, lr_critic=lr_critic,
            gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
            entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size, device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# MMICRL pre-training
# ---------------------------------------------------------------------------

def run_mmicrl_pretrain(cfg, log_dir="logs"):
    """
    Run MMICRL type discovery from Path G (WSD4FEDSRM calibrated demos).

    Demo source priority (per Remark 7.2 of math doc):
      1. WSD4FEDSRM raw data available -> full calibration pipeline
      2. Pre-computed profiles in config/pathg_profiles.json -> generate demos
      3. Neither -> skip MMICRL with warning (no random-policy fallback)
    """
    from hcmarl.mmicrl import MMICRL
    from hcmarl.real_data_calibration import (
        generate_demonstrations_from_profiles,
        load_path_g_into_collector,
    )

    mmicrl_cfg = cfg.get("mmicrl", {})
    n_types = mmicrl_cfg.get("n_types", 3)

    # --- Source 1: full Path G from raw WSD4FEDSRM data ---
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "wsd4fedsrm", "WSD4FEDSRM",
    )
    profiles_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "pathg_profiles.json",
    )

    worker_profiles = None

    if os.path.isdir(data_dir):
        print("Path G: calibrating from WSD4FEDSRM raw data...")
        from hcmarl.real_data_calibration import run_path_g
        path_g_result = run_path_g(data_dir)
        worker_profiles = path_g_result['worker_profiles']
    elif os.path.exists(profiles_path):
        print(f"Path G: loading pre-computed profiles from {profiles_path}")
        with open(profiles_path) as f:
            profiles_data = json.load(f)
        worker_profiles = profiles_data['profiles']
    else:
        print("WARNING: No WSD4FEDSRM data and no pre-computed profiles found.")
        print("  MMICRL requires real-data demonstrations (Remark 7.2).")
        print("  Skipping MMICRL pre-training.")
        return None, None

    # Generate demonstrations from calibrated profiles (Eq 35 controller)
    print("Generating demonstrations from calibrated profiles (Eq 35 controller)...")
    demos, worker_ids = generate_demonstrations_from_profiles(
        worker_profiles,
        muscle='shoulder',
        n_episodes_per_worker=mmicrl_cfg.get("n_episodes_per_worker", 3),
    )
    print(f"  Generated {len(demos)} demonstrations from {len(set(worker_ids))} workers")

    # Load into MMICRL collector
    collector = load_path_g_into_collector(demos, worker_ids)

    # Fit MMICRL (single-muscle shoulder calibration -> n_muscles=1)
    auto_select_k = mmicrl_cfg.get("auto_select_k", True)
    k_range = tuple(mmicrl_cfg.get("k_range", [1, 5]))
    # E2: default to held-out NLL because BIC is invalid for
    # normalizing flows (Watanabe 2013 singular-model theorem). Users who
    # want to reproduce a pre-Batch-E run can set k_selection: bic.
    k_selection = str(mmicrl_cfg.get("k_selection", "heldout_nll"))
    heldout_frac = float(mmicrl_cfg.get("heldout_frac", 0.2))
    mmicrl = MMICRL(
        n_types=n_types,
        lambda1=mmicrl_cfg.get("lambda1", 1.0),
        lambda2=mmicrl_cfg.get("lambda2", 1.0),
        n_muscles=1,
        n_iterations=mmicrl_cfg.get("n_iterations", 150),
        hidden_dims=mmicrl_cfg.get("hidden_dims", [64, 64]),
        auto_select_k=auto_select_k,
        k_range=k_range,
        k_selection=k_selection,
        heldout_frac=heldout_frac,
    )
    # S8: pass action-space size explicitly (derived from config tasks)
    # so MMICRL does not rely on max(action)+1 auto-detection.
    env_tasks = cfg.get("environment", {}).get("tasks", {})
    n_actions = len(env_tasks) if env_tasks else None
    results = mmicrl.fit(collector, n_actions=n_actions)

    # Print results
    print(f"\n--- MMICRL Type Discovery Results ---")
    print(f"  Demonstrations: {results['n_demonstrations']}")
    print(f"  Types discovered: {results['n_types_discovered']}")
    print(f"  Type proportions: {[f'{p:.2f}' for p in results['type_proportions']]}")
    print(f"  Mutual information I(tau;z): {results['mutual_information']:.4f}")
    print(f"  Objective value: {results['objective_value']:.4f}")
    print(f"  Learned thresholds per type (raw):")
    for k, thetas in results['theta_per_type'].items():
        print(f"    Type {k}: {{{', '.join(f'{m}: {v:.3f}' for m, v in thetas.items())}}}")

    # S1: MI-collapse diagnostic. If MI < 1e-2, latent types are behaviorally
    # indistinguishable from the demos, so rescaled thresholds collapse to the
    # config floor (see hcmarl.utils.build_per_worker_theta_max). Paper must
    # report this honestly; training still proceeds under the config defaults.
    mi_collapse_threshold = float(mmicrl_cfg.get("mi_collapse_threshold", 0.01))
    if float(results.get("mutual_information", 0.0)) < mi_collapse_threshold:
        print(f"  WARNING: MI ({results['mutual_information']:.4f}) < "
              f"{mi_collapse_threshold} -- latent types are behaviorally "
              f"indistinguishable on this demo set. Effective per-worker "
              f"theta will fall back to config floors (no MMICRL "
              f"contribution). Consider multi-muscle demos or lambda1 "
              f"tuning to restore discrimination.")

    # Save MMICRL results
    mmicrl_log_dir = os.path.join(log_dir, "mmicrl")
    os.makedirs(mmicrl_log_dir, exist_ok=True)
    with open(os.path.join(mmicrl_log_dir, "mmicrl_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {mmicrl_log_dir}/mmicrl_results.json")

    return results, mmicrl


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg, method, seed, device, resume_from=None, mmicrl_results=None, mmicrl_model=None,
          ecbf_mode="on", use_nswf=True, disagreement_type="divergent",
          drive_backup_dir=None, resume_state=None, run_name=None,
          budget_cap=0.0, cost_per_hour=49.0, budget_margin=0.95):
    """Full training loop with logging and checkpointing.

    If resume_state is given (loaded from run_state.pt by main()), training
    counters, RNG state, and theta_max are restored from it so a resumed
    run picks up the previous trajectory rather than restarting from zero.

    run_name overrides the per-method log/ckpt subdir. Default is `method`,
    so headline runs land at logs/{method}/seed_{seed}/. Ablation runs pass
    e.g. "ablation_plus_ecbf" so they don't collide with the headline mappo
    logs at logs/mappo/seed_0/.

    BUDGET KILL-SWITCH (the only physical guarantee against overspend):
    If budget_cap > 0, training is hard-aborted when wall-clock spend
    reaches budget_cap * budget_margin. Spend = elapsed_hours * cost_per_hour.
    Defaults: cost_per_hour=49.0 (currency-units per GPU-hour), margin=0.95
    (kill at 95% so checkpointing has time to flush). budget_cap=0 disables
    the gate (default for backward-compat with existing tests).
    """
    env_cfg = cfg.get("environment", {})
    train_cfg = cfg.get("training", {})

    # Full seeding: numpy, torch (CPU+CUDA), cudnn.deterministic, PYTHONHASHSEED.
    # Python stdlib random is seeded too — MMICRL / allocator internals may use it.
    # M6: deterministic flag honours training.deterministic in config (default True).
    import random as _random
    from hcmarl.utils import seed_everything
    seed_everything(
        seed,
        deterministic=bool(train_cfg.get("deterministic", True)),
        # B2: HIGH determinism on by default for the EXP1+EXP2
        # rerun. EXP3 Part 1 ran under HIGH and produced the only clean
        # ARI=1.0 / MI=1.099 result on synthetic K=3, so the rerun must
        # match. Set training.high_determinism=false in a config only when
        # debugging non-deterministic CUDA-op crashes.
        high_determinism=bool(train_cfg.get("high_determinism", True)),
    )
    _random.seed(seed)
    n_workers = env_cfg.get("n_workers", 6)
    max_steps = env_cfg.get("max_steps", 480)

    total_steps = train_cfg.get("total_steps", 5_000_000)
    eval_interval = train_cfg.get("eval_interval", 50_000)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 100_000)
    n_eval_episodes = train_cfg.get("n_eval_episodes", 10)

    # Resume: if run_state is present, its theta_max takes priority over
    # a freshly-run MMICRL. Guarantees the resumed env has the same
    # thresholds the saved policy was trained against.
    from hcmarl.utils import build_per_worker_theta_max, build_per_worker_theta_max_from_F
    config_theta_defaults = env_cfg.get("theta_max", {}) or {}
    if resume_state is not None and resume_state.get("theta_max") is not None:
        theta_max = resume_state["theta_max"]
        print(f"Resume: reusing saved theta_max from run_state (step "
              f"{resume_state.get('global_step', 0):,})")
    else:
        mmicrl_cfg = cfg.get("mmicrl", {})
        # Dispatch on ceiling_source: legacy 'mmicrl_rescale' (default),
        # 'F_anchored' (Method 1, per-worker F-standardized), or
        # 'config_only' (flat config, no personalization).
        ceiling_source = str(mmicrl_cfg.get("ceiling_source", "mmicrl_rescale"))
        if method != "hcmarl":
            # Baselines always take the flat config defaults.
            theta_max = config_theta_defaults
        elif ceiling_source == "config_only":
            theta_max = config_theta_defaults
        elif ceiling_source == "F_anchored":
            # Build per-worker (F, R, r) tuples for each config muscle.
            # Until Commit B wires pathg profiles into the env, all workers
            # share the config muscle_groups values, so F_anchored degenerates
            # to config default (correct fallback).
            muscle_groups = env_cfg.get("muscle_groups", {}) or {}
            worker_F_R_r_per_muscle = {}
            for m in config_theta_defaults:
                mg = muscle_groups.get(m)
                if mg is None:
                    raise ValueError(
                        f"muscle '{m}' listed in environment.theta_max but "
                        f"missing from environment.muscle_groups."
                    )
                F, R, r = float(mg['F']), float(mg['R']), float(mg['r'])
                worker_F_R_r_per_muscle[m] = [(F, R, r)] * n_workers
            alpha = float(mmicrl_cfg.get("F_anchor_alpha", 0.05))
            z_clip = float(mmicrl_cfg.get("F_anchor_z_clip", 2.0))
            epsilon = float(mmicrl_cfg.get("F_anchor_epsilon", 0.005))
            theta_max = build_per_worker_theta_max_from_F(
                worker_F_R_r_per_muscle, config_theta_defaults, n_workers,
                alpha=alpha, z_clip=z_clip, epsilon=epsilon,
            )
            print(f"F-anchored ceilings: alpha={alpha}, z_clip={z_clip}, "
                  f"epsilon={epsilon}. (Homogeneous F across workers -> "
                  f"degenerates to config default; pathg per-worker F wiring "
                  f"in next pass.)")
        else:
            rescale = bool(mmicrl_cfg.get("rescale_to_floor", True))
            mi_thresh = float(mmicrl_cfg.get("mi_collapse_threshold", 0.01))
            theta_max = build_per_worker_theta_max(
                mmicrl_results, config_theta_defaults, n_workers, method,
                rescale_to_floor=rescale, mi_collapse_threshold=mi_thresh,
            )
        if mmicrl_results and method == "hcmarl" and isinstance(theta_max, dict) and any(
            isinstance(v, dict) for v in theta_max.values()
        ):
            n_types = len(mmicrl_results.get("theta_per_type", {}))
            mode = "rescaled into [floor,1]" if rescale else "hard-clamped to floor"
            print(f"Using MMICRL-learned thresholds for {n_workers} workers "
                  f"({n_types} types, {mode}; floors: {dict(config_theta_defaults)})")
            print(f"  Effective per-worker theta_max: {theta_max}")

    # Directories. run_name (if provided by caller) overrides the per-method
    # subdir so ablation rungs land in logs/ablation_<rung>/seed_<s>/ instead
    # of overwriting the headline logs at logs/<method>/seed_<s>/.
    log_subdir = run_name if run_name else method
    wandb_run_name = f"{log_subdir}_seed{seed}"
    ckpt_dir = os.path.join("checkpoints", log_subdir, f"seed_{seed}")
    log_dir = os.path.join("logs", log_subdir, f"seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Action mode: continuous for hcmarl method (math-doc-faithful, Remark 7.2)
    action_mode = cfg.get("action_mode", "discrete")
    if method == "hcmarl" and action_mode == "continuous":
        env_action_mode = "continuous"
    else:
        env_action_mode = "discrete"
    # Appendix F probe: emit a clear startup line so Gate 0
    # / Gate 1 stdout immediately confirms which mode is active. Catches
    # accidental discrete-mode fallback if the launch scripts forget the
    # --action-mode flag (now guarded by CLI > YAML > fallback precedence,
    # but a runtime confirmation is cheap insurance).
    print(f"[action_mode] cfg.action_mode={action_mode!r}, method={method!r}, "
          f"env_action_mode={env_action_mode!r}")

    # Build muscle_params_override from config (C-17: no_reperfusion ablation).
    # M8: only F, R, r are recognised by the 3CC-r ODE. Warn loudly if the
    # config has extra keys so nobody silently drops a parameter they thought
    # was being applied.
    muscle_params_override = None
    muscle_groups_cfg = env_cfg.get("muscle_groups")
    if muscle_groups_cfg:
        muscle_params_override = {}
        _allowed = {"F", "R", "r"}
        for m_name, m_params in muscle_groups_cfg.items():
            extras = set(m_params.keys()) - _allowed
            if extras:
                import warnings as _w
                _w.warn(
                    f"muscle_groups.{m_name} has keys {sorted(extras)} that "
                    f"the 3CC-r ODE does not use; they are being dropped. "
                    f"Only {sorted(_allowed)} are applied.",
                    UserWarning, stacklevel=2,
                )
            muscle_params_override[m_name] = {
                k: v for k, v in m_params.items() if k in _allowed
            }

    # ECBF alphas from config (previously hardcoded, now configurable)
    ecbf_cfg = cfg.get("ecbf", {})
    ecbf_alpha1 = ecbf_cfg.get("alpha1", 0.05)
    ecbf_alpha2 = ecbf_cfg.get("alpha2", 0.05)
    ecbf_alpha3 = ecbf_cfg.get("alpha3", 0.1)

    # ------------------------------------------------------------------
    # Option X: per-worker type-conditioning from MMICRL
    # ------------------------------------------------------------------
    # Policy is conditioned on each worker's MMICRL-discovered type
    # (one-hot appended to the observation) when
    # type_conditioning.enabled=True in config.
    #
    # On MI collapse (MI < mi_collapse_threshold): fall back to n_types=1
    # with every worker assigned type 0. Observation gains a constant
    # one-hot [1.0] and the policy learns to ignore it — graceful
    # degenerate single-type behavior.
    #
    # On resume: restore from run_state if present, else recompute from
    # the fresh MMICRL results. This guarantees resumed env has the same
    # obs_dim the saved policy was trained against.
    # ------------------------------------------------------------------
    n_types_effective = 0
    worker_type_assignments = None
    tc_cfg = cfg.get("type_conditioning", {}) or {}
    tc_enabled = bool(tc_cfg.get("enabled", False))
    if resume_state is not None and resume_state.get("n_types", 0) > 0:
        n_types_effective = int(resume_state["n_types"])
        worker_type_assignments = dict(
            resume_state.get("worker_type_assignments") or {}
        )
        # Keys may have serialised as strings; coerce back to int.
        worker_type_assignments = {int(k): int(v) for k, v in worker_type_assignments.items()}
        print(f"Resume: restored type-conditioning n_types={n_types_effective}, "
              f"assignments from run_state.")
    elif method == "hcmarl" and tc_enabled and mmicrl_results is not None:
        mi = float(mmicrl_results.get("mutual_information", 0.0))
        mi_thresh = float(cfg.get("mmicrl", {}).get("mi_collapse_threshold", 0.01))
        if mi < mi_thresh:
            n_types_effective = 1
            worker_type_assignments = {i: 0 for i in range(n_workers)}
            print(f"Option X: MI={mi:.4f} < {mi_thresh} (collapse) -> n_types=1, "
                  f"every worker z=0. Policy trains as single-type.")
        else:
            proportions = mmicrl_results.get("type_proportions", [])
            n_disc = int(mmicrl_results.get("n_types_discovered", len(proportions) or 1))
            n_types_effective = n_disc
            if proportions and len(proportions) == n_types_effective:
                counts = np.round(np.array(proportions, dtype=float) * n_workers).astype(int)
                diff = n_workers - int(counts.sum())
                if diff != 0:
                    counts[int(np.argmax(counts))] += diff
                worker_type_assignments = {}
                w_idx = 0
                for t_idx, count in enumerate(counts):
                    for _ in range(int(count)):
                        if w_idx < n_workers:
                            worker_type_assignments[w_idx] = t_idx
                            w_idx += 1
                while w_idx < n_workers:
                    worker_type_assignments[w_idx] = 0
                    w_idx += 1
            else:
                worker_type_assignments = {i: i % n_types_effective
                                           for i in range(n_workers)}
            print(f"Option X: MI={mi:.4f} >= {mi_thresh} -> n_types={n_types_effective}, "
                  f"assignments={worker_type_assignments}")

    # Environment
    env = WarehousePettingZoo(
        n_workers=n_workers, max_steps=max_steps,
        theta_max=theta_max, ecbf_mode=ecbf_mode,
        action_mode=env_action_mode,
        disagreement_type=disagreement_type,
        muscle_params_override=muscle_params_override,
        ecbf_alpha1=ecbf_alpha1, ecbf_alpha2=ecbf_alpha2, ecbf_alpha3=ecbf_alpha3,
        n_types=n_types_effective,
        worker_type_assignments=worker_type_assignments,
    )
    obs_dim = env.obs_dim
    global_obs_dim = env.global_obs_dim
    n_actions = env.n_tasks
    n_muscles = env.n_muscles

    # Agent
    agent = create_agent(
        method, obs_dim, global_obs_dim, n_actions, n_workers, cfg, device,
        action_mode=env_action_mode, n_muscles=n_muscles, use_nswf=use_nswf,
    )

    # Resume
    if resume_from and os.path.exists(resume_from):
        agent.load(resume_from)
        print(f"Resumed from {resume_from}")

    # Logger
    logger = HCMARLLogger(
        log_dir=log_dir,
        use_wandb=cfg.get("logging", {}).get("use_wandb", False),
        wandb_project=cfg.get("logging", {}).get("project_name", "hcmarl"),
        run_name=wandb_run_name,
        config=cfg,
    )

    # Save config (with MMICRL results if available)
    save_cfg = dict(cfg)
    if mmicrl_results:
        save_cfg["mmicrl_results"] = mmicrl_results
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(save_cfg, f)

    # Detect agent type for buffer storage. The "lagrangian" branch is
    # the path that uses LagrangianRolloutBuffer.store(... cost=..., cost_values=...);
    # MAPPO-Lag is the only such method in the current lineup.
    is_mappo_lag = isinstance(agent, MAPPOLagrangian)
    is_lagrangian = is_mappo_lag
    is_ippo = isinstance(agent, IPPO)
    is_hcmarl = hasattr(agent, 'mappo')

    # Training
    global_step = 0
    episode_count = 0
    best_reward = -float("inf")
    cost_ema = 0.0  # Exponential moving average of per-step cost rate
    next_checkpoint_step = checkpoint_interval
    start_time = time.time()

    # BUDGET KILL-SWITCH: convert spend ceiling -> wall-clock seconds.
    # When budget_cap <= 0, the gate is disabled (no overhead).
    # Spend model: simple wall-clock * hourly_rate; matches how cloud
    # Networks bills L4 instances. The gate is checked every env step
    # (see budget_seconds comparison below).
    if budget_cap > 0.0 and cost_per_hour > 0.0:
        budget_seconds = (budget_cap * budget_margin / cost_per_hour) * 3600.0
    else:
        budget_seconds = float("inf")
    budget_tripped = False

    # D4: lazy-agent kill-switch state. Parameters live in
    # config/experiment_matrix.yaml; the training config may override them
    # in a `lazy_agent_kill_switch` block.
    lac = cfg.get("lazy_agent_kill_switch", {})
    lazy_threshold = float(lac.get("threshold", 0.1))
    lazy_window = int(lac.get("window_steps", 100_000))
    # Number of consecutive env-steps during which min-agent entropy
    # has stayed below `lazy_threshold`. Reset when any episode comes
    # in above threshold.
    lazy_low_streak = 0
    lazy_agent_flag = 0

    # Restore counters + RNG state from run_state (if resuming).
    # Agent weights + optimizer state were already loaded above via agent.load().
    if resume_state is not None:
        global_step = int(resume_state.get("global_step", 0))
        episode_count = int(resume_state.get("episode_count", 0))
        best_reward = float(resume_state.get("best_reward", -float("inf")))
        cost_ema = float(resume_state.get("cost_ema", 0.0))
        next_checkpoint_step = ((global_step // checkpoint_interval) + 1) * checkpoint_interval
        rng_np = resume_state.get("rng_np")
        if rng_np is not None:
            np.random.set_state(rng_np)
        rng_torch = resume_state.get("rng_torch")
        if rng_torch is not None:
            torch.set_rng_state(rng_torch)
        rng_torch_cuda = resume_state.get("rng_torch_cuda")
        if rng_torch_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_torch_cuda)
        rng_python = resume_state.get("rng_python")
        if rng_python is not None:
            _random.setstate(rng_python)
        print(f"Resume: step={global_step:,} episode={episode_count} "
              f"best_reward={best_reward:.2f} cost_ema={cost_ema:.4f}")

    print(f"{'='*60}")
    print(f"HC-MARL Training: {METHODS.get(method, method)}")
    print(f"Seed: {seed} | Device: {device} | Workers: {n_workers}")
    print(f"Total steps: {total_steps:,} | Eval every: {eval_interval:,}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"{'='*60}")

    while global_step < total_steps:
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0
        tasks_per_worker = np.zeros(n_workers)
        peak_mf = 0.0
        safe_steps = 0
        forced_rests = 0
        total_ecbf_interventions = 0
        total_ecbf_opportunities = 0  # muscles where C_nominal > 0
        # D4: per-agent task-selection histogram for this episode — used to
        # compute per-agent entropy (lazy-agent diagnostic, Liu ICML 2023).
        action_hist = np.zeros((n_workers, max(2, int(env.n_tasks))),
                               dtype=np.int64)
        # Appendix F continuous-mode probe diagnostics. All three are
        # populated only when env_action_mode == "continuous"; in discrete
        # mode they remain at their init values (0 / empty) and are written
        # to CSV as zeros. See logger.py CSV_COLUMNS for column docs.
        nan_inf_events_episode = 0
        allocator_calls_episode = 0
        gaussian_log_std_sum = 0.0
        gaussian_log_std_count = 0

        for step in range(max_steps):
            global_state = env._get_global_obs()

            # C-7.A: Hierarchical allocation — run NSWF allocator every K steps
            if is_hcmarl and hasattr(agent, 'allocator') and agent.allocator is not None:
                if step == 0 or agent.should_reallocate():
                    # Gather per-worker fatigue for allocator
                    fatigue_for_alloc = {}
                    for agent_id in env.agents:
                        idx = int(agent_id.split("_")[1])
                        worker_fatigue = env.states.get(idx, {})
                        max_mf = max(
                            (worker_fatigue.get(m, {}).get("MF", 0.0)
                             for m in env.muscle_names), default=0.0
                        )
                        fatigue_for_alloc[agent_id] = max_mf
                    assignments = agent.allocate_tasks(fatigue_for_alloc)
                    # Appendix F probe: increment per-call counter so the
                    # CSV per episode evidences the allocator firing at
                    # K=30 cadence (continuous mode) — direct evidence of
                    # the math-doc-faithful coupling vs the decoupled
                    # discrete-mode case. Counter is incremented in BOTH
                    # modes (the allocator runs in both); discrete-mode
                    # CSVs will show identical values, which is the
                    # expected control.
                    allocator_calls_episode += 1
                    # Push assignments to env (for continuous mode conditioning)
                    if hasattr(env, 'set_task_assignments'):
                        env.set_task_assignments(assignments)

            # Get actions from agent
            result = agent.get_actions(obs, global_state) if not is_ippo else agent.get_actions(obs)
            if isinstance(result, tuple):
                actions = result[0]
                log_probs = result[1] if len(result) > 1 else {}
                value = result[2] if len(result) > 2 else 0.0
                cost_value = result[3] if len(result) > 3 else 0.0
            else:
                actions = result
                log_probs, value, cost_value = {}, 0.0, 0.0

            # Appendix F probe: NaN/Inf assertion hook on continuous-mode
            # action vectors. Critical because without it a numerical
            # pathology (Gaussian reparameterization × deterministic CUBLAS
            # interaction, log-std clamp escape, etc.) produces silent
            # garbage. Strict halt on first detection so the failure mode
            # is diagnosable from the traceback rather than from post-hoc
            # log forensics. Discrete-mode actions are integer task indices
            # and cannot be NaN/Inf; the check is effectively a no-op there.
            if env_action_mode == "continuous":
                for _agent_id, _act in actions.items():
                    _act_arr = np.asarray(_act)
                    if _act_arr.dtype.kind == "f" and (
                        not np.isfinite(_act_arr).all()
                    ):
                        nan_inf_events_episode += 1
                        raise RuntimeError(
                            f"NaN/Inf detected in continuous action for "
                            f"{_agent_id} at global_step={global_step}, "
                            f"episode={episode_count}, step={step}. "
                            f"Action vector: {_act_arr}. This is the "
                            f"Appendix F probe halt; investigate via "
                            f"traceback before re-running."
                        )
                # Capture Gaussian log_std for episode-level diagnostic.
                # HCMARLAgent._get_continuous_actions sets this attribute
                # on every continuous-mode call.
                _ls = getattr(agent, "last_gaussian_log_std_mean", None)
                if _ls is not None:
                    gaussian_log_std_sum += float(_ls)
                    gaussian_log_std_count += 1

            # D4: tally per-agent task selection for entropy diagnostic.
            # Continuous mode does not produce discrete task picks here —
            # skip the histogram in that regime (the NSWF allocator makes
            # the task decision instead, which is logged elsewhere).
            if env_action_mode == "discrete":
                for agent_id, a in actions.items():
                    idx = int(agent_id.split("_")[1])
                    a_int = int(a) if np.isscalar(a) else int(np.asarray(a).item())
                    if 0 <= idx < n_workers and 0 <= a_int < action_hist.shape[1]:
                        action_hist[idx, a_int] += 1

            # Step environment
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            episode_steps += 1
            global_step += 1

            # BUDGET KILL-SWITCH: physical guarantee against L4 overspend.
            if (time.time() - start_time) >= budget_seconds:
                budget_tripped = True
                break  # exit step loop; outer loop sees budget_tripped

            # Track metrics
            step_violations = 0
            for i, (agent_id, info) in enumerate(sorted(infos.items())):
                fatigue = info.get("fatigue", {})
                task = info.get("task", "rest")
                violations = info.get("violations", 0)
                step_violations += violations
                if task != "rest":
                    tasks_per_worker[i] += 1
                for m, mf in fatigue.items():
                    peak_mf = max(peak_mf, mf)
                # ECBF intervention tracking (S-22: forced rest = ECBF clipped)
                ecbf_int = info.get("ecbf_interventions", 0)
                total_ecbf_interventions += ecbf_int
                if ecbf_int > 0:
                    forced_rests += 1
                # S-23: count only muscles with nonzero task demand as ECBF
                # opportunities (not all 6 muscles unconditionally)
                if task != "rest":
                    demands = env.task_mgr.get_demand_vector(task)
                    total_ecbf_opportunities += int((demands > 0).sum())

            episode_cost += step_violations
            if step_violations == 0:
                safe_steps += 1

            # Dense cost for Lagrangian: sum of per-agent costs from env
            step_cost = sum(infos[a].get("cost", 0.0) for a in sorted(infos.keys())) / max(1, n_workers)

            if is_hcmarl and hasattr(agent.mappo, 'buffer'):
                # HCMARLAgent wraps MAPPO — store obs used to SELECT action
                for agent_id in sorted(actions.keys()):
                    agent.mappo.buffer.store(
                        obs=obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                        values=value,
                    )
            elif is_lagrangian:
                # MAPPO-Lagrangian or SafePO fallback: store with cost
                for agent_id in sorted(actions.keys()):
                    agent.buffer.store(
                        obs=obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        cost=step_cost,
                        done=float(terms[agent_id]),
                        values=value,
                        cost_values=cost_value,
                    )
            elif is_ippo:
                # IPPO: per-agent buffer storage
                for agent_id in sorted(actions.keys()):
                    idx = int(agent_id.split("_")[1])
                    agent.store_transition(
                        agent_idx=idx,
                        obs=obs[agent_id],
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                    )
            elif hasattr(agent, 'buffer'):
                # Generic MAPPO / other agents with buffer
                for agent_id in sorted(actions.keys()):
                    agent.buffer.store(
                        obs=obs[agent_id], global_state=global_state,
                        action=actions[agent_id],
                        log_prob=log_probs.get(agent_id, 0.0),
                        reward=rewards[agent_id],
                        done=float(terms[agent_id]),
                        values=value,
                    )

            obs = next_obs

            if all(terms.values()):
                break

        # M7: linear entropy annealing. If algorithm.entropy_coeff_final is set
        # in the config, anneal entropy_coeff from its initial value down to
        # entropy_coeff_final over total_steps — standard PPO practice, prevents
        # the policy from thrashing with high entropy late in training.
        algo_cfg = cfg.get("algorithm", {})
        entropy_final = algo_cfg.get("entropy_coeff_final", None)
        if entropy_final is not None:
            entropy_start = float(algo_cfg.get("entropy_coeff", 0.05))
            frac = min(1.0, global_step / max(1, total_steps))
            new_ent = entropy_start + (float(entropy_final) - entropy_start) * frac
            if is_hcmarl and hasattr(agent, 'mappo'):
                agent.mappo.entropy_coeff = new_ent
            elif hasattr(agent, 'entropy_coeff'):
                agent.entropy_coeff = new_ent

        # PPO update at end of episode
        update_info = {}
        if hasattr(agent, 'update'):
            update_info = agent.update() or {}
        elif is_hcmarl and hasattr(agent.mappo, 'update'):
            update_info = agent.mappo.update() or {}

        # Lagrangian lambda update with EMA-smoothed cost
        if hasattr(agent, 'update_lambda'):
            mean_cost = episode_cost / max(1, episode_steps)
            ema_alpha = 0.05  # smoothing factor — slow update for stable dual variable
            cost_ema = (1 - ema_alpha) * cost_ema + ema_alpha * mean_cost
            agent.update_lambda(cost_ema)

        # Compute episode metrics
        episode_count += 1
        n = n_workers
        jain = float((tasks_per_worker.sum()**2) / (n * (tasks_per_worker**2).sum() + 1e-8)) if tasks_per_worker.sum() > 0 else 1.0

        # D4: per-agent task-selection entropy (Shannon, natural log).
        # H_i = -sum_a p_i(a) log p_i(a). Uniform over K tasks -> log(K).
        # A collapsed single-task agent has H_i = 0.
        per_agent_entropy = np.zeros(n_workers, dtype=np.float64)
        for i in range(n_workers):
            counts = action_hist[i]
            tot = counts.sum()
            if tot > 0:
                p = counts / tot
                nz = p[p > 0]
                per_agent_entropy[i] = float(-(nz * np.log(nz)).sum())
        ent_mean = float(per_agent_entropy.mean())
        ent_min = float(per_agent_entropy.min())

        # Kill-switch: the MIN-over-agents entropy is the signal. If it
        # stays below threshold for >= window_steps consecutive env steps,
        # we flag the run as lazy-agent-collapsed and stop.
        if ent_min < lazy_threshold:
            lazy_low_streak += episode_steps
        else:
            lazy_low_streak = 0
        if lazy_low_streak >= lazy_window:
            lazy_agent_flag = 1

        # Safety Autonomy Index: fraction of work-muscle slots NOT clipped by ECBF
        sai = 1.0 - (total_ecbf_interventions / max(1, total_ecbf_opportunities))

        ep_metrics = {
            "episode": episode_count,
            "global_step": global_step,
            "cumulative_reward": episode_reward,
            "cumulative_cost": episode_cost,
            "violation_rate": episode_cost / max(1, episode_steps * n_workers),
            "safety_rate": safe_steps / max(1, episode_steps),
            "tasks_completed": float(tasks_per_worker.sum()),
            "jain_fairness": jain,
            "peak_fatigue": peak_mf,
            "forced_rest_rate": forced_rests / max(1, episode_steps * n_workers),
            "safety_autonomy_index": sai,
            "ecbf_interventions": total_ecbf_interventions,
            "wall_time": time.time() - start_time,
        }
        if is_lagrangian:
            ep_metrics["lambda"] = agent.lam
        # M5: always log cost_ema — the Lagrangian state variable drives lambda
        # and is needed to reproduce / debug dual-variable trajectories.
        ep_metrics["cost_ema"] = float(cost_ema)
        # D4: persist per-agent entropy + kill-switch signal in CSV.
        ep_metrics["per_agent_entropy_mean"] = ent_mean
        ep_metrics["per_agent_entropy_min"] = ent_min
        ep_metrics["lazy_agent_flag"] = int(lazy_agent_flag)
        # Appendix F continuous-mode probe diagnostics. In discrete mode
        # nan_inf_events stays 0 (discrete actions are integer indices),
        # gaussian_log_std_mean is reported as NaN-marker 0.0 since the
        # discrete path doesn't populate it, and allocator_calls reflects
        # the deterministic K=30 cadence. In continuous mode all three
        # carry signal — see logger.py CSV_COLUMNS for interpretation.
        ep_metrics["nan_inf_events"] = int(nan_inf_events_episode)
        ep_metrics["allocator_calls"] = int(allocator_calls_episode)
        ep_metrics["gaussian_log_std_mean"] = (
            float(gaussian_log_std_sum / gaussian_log_std_count)
            if gaussian_log_std_count > 0 else 0.0
        )
        ep_metrics.update(update_info)

        logger.log_episode(ep_metrics)

        # D4: kill-switch trip. Stops the training loop with a clear
        # marker in the CSV; the caller can decide what to do (the
        # pilot aggregator flags such runs, the main sweep can retry
        # with a different seed).
        if lazy_agent_flag:
            print(f"[lazy-agent kill-switch] min agent entropy "
                  f"{ent_min:.4f} < {lazy_threshold} for "
                  f">= {lazy_window:,} consecutive steps — halting run.")
            break

        # BUDGET KILL-SWITCH: outer-loop exit with checkpoint flush.
        if budget_tripped:
            elapsed_hr = (time.time() - start_time) / 3600.0
            spent = elapsed_hr * cost_per_hour
            print(f"[budget kill-switch] {spent:.0f}/{budget_cap:.0f} units "
                  f"({100*spent/budget_cap:.1f}%) at step {global_step:,} — halting.")
            if hasattr(agent, 'save'):
                agent.save(os.path.join(ckpt_dir, "checkpoint_budget_halt.pt"))
                _write_run_state(
                    os.path.join(ckpt_dir, "run_state.pt"),
                    global_step, episode_count, cost_ema, best_reward,
                    theta_max, seed, method,
                    n_types=n_types_effective,
                    worker_type_assignments=worker_type_assignments,
                )
            break

        # Print progress
        if episode_count % 50 == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            lam_str = f" lam={agent.lam:.3f}" if is_lagrangian else ""
            print(
                f"[Ep {episode_count:5d} | Step {global_step:>9,}/{total_steps:,} | "
                f"{sps:.0f} sps] "
                f"R={episode_reward:7.1f} C={episode_cost:4.0f} "
                f"Safe={ep_metrics['safety_rate']:.2%} SAI={sai:.3f} "
                f"Jain={jain:.3f} PeakMF={peak_mf:.3f}{lam_str}"
            )

        # Checkpoint at regular intervals
        if global_step >= next_checkpoint_step:
            if hasattr(agent, 'save'):
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{global_step}.pt")
                agent.save(ckpt_path)
                # C1: run_state.pt sits next to the policy ckpt so resume
                # restores counters + theta_max alongside the weights.
                _write_run_state(
                    os.path.join(ckpt_dir, "run_state.pt"),
                    global_step, episode_count, cost_ema, best_reward,
                    theta_max, seed, method,
                    n_types=n_types_effective,
                    worker_type_assignments=worker_type_assignments,
                )
            # Mirror to Drive (survives Colab runtime disconnect).
            # Logger flushes CSV every episode, so copy after each checkpoint.
            if drive_backup_dir:
                import shutil
                try:
                    dst_ckpt = os.path.join(drive_backup_dir, ckpt_dir)
                    dst_log = os.path.join(drive_backup_dir, log_dir)
                    os.makedirs(dst_ckpt, exist_ok=True)
                    os.makedirs(dst_log, exist_ok=True)
                    shutil.copytree(ckpt_dir, dst_ckpt, dirs_exist_ok=True)
                    shutil.copytree(log_dir, dst_log, dirs_exist_ok=True)
                except Exception as e:
                    print(f"[drive-backup] WARN: {e}")
            next_checkpoint_step += checkpoint_interval

        # Best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            if hasattr(agent, 'save'):
                agent.save(os.path.join(ckpt_dir, "checkpoint_best.pt"))

    # Final checkpoint
    if hasattr(agent, 'save'):
        agent.save(os.path.join(ckpt_dir, "checkpoint_final.pt"))
        _write_run_state(
            os.path.join(ckpt_dir, "run_state.pt"),
            global_step, episode_count, cost_ema, best_reward,
            theta_max, seed, method,
            n_types=n_types_effective,
            worker_type_assignments=worker_type_assignments,
        )

    # Final Drive mirror
    if drive_backup_dir:
        import shutil
        try:
            dst_ckpt = os.path.join(drive_backup_dir, ckpt_dir)
            dst_log = os.path.join(drive_backup_dir, log_dir)
            os.makedirs(dst_ckpt, exist_ok=True)
            os.makedirs(dst_log, exist_ok=True)
            shutil.copytree(ckpt_dir, dst_ckpt, dirs_exist_ok=True)
            shutil.copytree(log_dir, dst_log, dirs_exist_ok=True)
            print(f"[drive-backup] final mirror -> {drive_backup_dir}")
        except Exception as e:
            print(f"[drive-backup] final WARN: {e}")

    # Save final metrics summary
    elapsed = time.time() - start_time
    summary = {
        "method": method,
        "seed": seed,
        "total_episodes": episode_count,
        "total_steps": global_step,
        "best_reward": best_reward,
        "wall_time_seconds": elapsed,
        "wall_time_hours": elapsed / 3600,
        "budget_tripped": budget_tripped,
        "budget_cap": budget_cap,
        "spent_estimate": (elapsed / 3600) * cost_per_hour if budget_cap > 0 else None,
    }
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete: {episode_count} episodes, {global_step:,} steps")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Wall time: {elapsed/3600:.2f} hours")
    print(f"Checkpoint: {ckpt_dir}/checkpoint_final.pt")
    print(f"{'='*60}")

    logger.close()
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HC-MARL Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--method", type=str, default="hcmarl", choices=list(METHODS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mmicrl", action="store_true", help="Run MMICRL pre-training before RL")
    parser.add_argument("--ecbf-mode", type=str, default=None, choices=["on", "off"],
                        help="ECBF safety filter mode: 'on' (default) or 'off' (ablation)")
    parser.add_argument("--action-mode", type=str, default=None,
                        choices=["discrete", "continuous"],
                        help="Action mode: 'discrete' (task selection) or 'continuous' "
                             "(neural drive per Remark 7.2). Default None means: "
                             "respect cfg['action_mode'] if set, else fall back to 'discrete'. "
                             "Appendix F probe: default was 'discrete', which "
                             "silently overrode YAML configs that explicitly set "
                             "action_mode='continuous'. Default is now None so CLI > YAML > "
                             "fallback precedence is honored.")
    parser.add_argument("--welfare", type=str, default="nswf",
                        choices=["nswf", "utilitarian", "maxmin", "gini"],
                        help="Welfare function for NSWF allocator ablation (C-7.R)")
    parser.add_argument("--allocation-interval", type=int, default=30,
                        help="Steps between NSWF allocator calls (K in hierarchical two-timescale, C-7.A)")
    parser.add_argument("--no-nswf", action="store_true",
                        help="Disable NSWF allocator (no_nswf ablation)")
    parser.add_argument("--disagreement-type", type=str, default=None,
                        choices=["divergent", "constant"],
                        help="Disagreement utility type: 'divergent' (Eq 32) or 'constant' (D_i=kappa)")
    parser.add_argument("--drive-backup-dir", type=str, default=None,
                        help="If set, mirror checkpoints+logs here every checkpoint interval. "
                             "Use /content/drive/MyDrive/hcmarl_backup on Colab to survive runtime disconnects.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the per-method log/ckpt subdir. Default uses --method, "
                             "so headline runs go to logs/{method}/seed_{seed}/. Ablation runs "
                             "should pass e.g. 'ablation_plus_ecbf' to avoid overwriting the "
                             "headline mappo logs at logs/mappo/seed_0/.")
    parser.add_argument("--budget-cap", type=float, default=0.0,
                        help="Hard kill-switch: halt training when wall-clock spend reaches "
                             "this currency-units amount. Default 0 disables.")
    parser.add_argument("--cost-per-hour", type=float, default=49.0,
                        help="GPU instance hourly rate (currency-units per hour). Default 49.0.")
    parser.add_argument("--budget-margin", type=float, default=0.95,
                        help="Trip kill-switch at budget_cap * margin (default 0.95) so checkpoint can flush.")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Fail-fast: baseline configs MUST carry an environment block with
    # muscle_groups, theta_max, and tasks. Without these, safety_cost()
    # silently returns 0 for every step and the baseline trains in a
    # constraint-free world (root cause of the contaminated
    # baseline runs). Guard applies to every method — hcmarl relies on
    # the same floors when MMICRL is skipped or MI collapses.
    _env = cfg.get("environment", {}) or {}
    _missing = [k for k in ("muscle_groups", "theta_max", "tasks")
                if not (_env.get(k) or {})]
    if _missing:
        raise ValueError(
            f"{args.config}: environment section is missing or empty for "
            f"keys {_missing}. This would make safety_cost() return 0 for "
            f"every step and train the policy in a constraint-free env. "
            f"Populate environment.muscle_groups, environment.theta_max, "
            f"and environment.tasks to match config/hcmarl_full_config.yaml "
            f"before launching."
        )

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Seed EVERYTHING before any stochastic work. MMICRL pretrain uses
    # torch/numpy/random and must be seeded to be reproducible across
    # identical --seed invocations. train() also calls seed_everything;
    # the second call is idempotent and re-seeds before the RL loop.
    import random as _random
    from hcmarl.utils import seed_everything
    seed_everything(args.seed)
    _random.seed(args.seed)

    # ---------------------------------------------------------------
    # C-17: Read ablation flags from config AND CLI overrides
    # ---------------------------------------------------------------

    # ECBF mode: CLI > config > default "on"
    ecbf_cfg = cfg.get("ecbf", {})
    if args.ecbf_mode is not None:
        ecbf_mode = args.ecbf_mode
    elif not ecbf_cfg.get("enabled", True):
        ecbf_mode = "off"
    else:
        ecbf_mode = "on"

    # NSWF: CLI > config > default True
    nswf_cfg = cfg.get("nswf", {})
    if args.no_nswf:
        use_nswf = False
    elif not nswf_cfg.get("enabled", True):
        use_nswf = False
    else:
        use_nswf = True

    # Disagreement type: CLI > config > default "divergent"
    disagree_cfg = cfg.get("disagreement", {})
    if args.disagreement_type is not None:
        disagreement_type = args.disagreement_type
    else:
        disagreement_type = disagree_cfg.get("type", "divergent")

    # MMICRL: CLI --mmicrl > config mmicrl.enabled > default False
    mmicrl_cfg = cfg.get("mmicrl", {})
    run_mmicrl = args.mmicrl or mmicrl_cfg.get("enabled", False)
    # But if use_fixed_theta is True, skip MMICRL even if enabled
    if mmicrl_cfg.get("use_fixed_theta", False):
        run_mmicrl = False

    # Resume detection: if run_state.pt sits next to the checkpoint we are
    # resuming from, skip MMICRL — the saved theta_max is authoritative, and
    # MMICRL is stochastic enough that re-running could produce different
    # thresholds than the policy was trained against.
    resume_state = None
    if args.resume:
        resume_state_path = os.path.join(
            os.path.dirname(os.path.abspath(args.resume)), "run_state.pt",
        )
        resume_state = _load_run_state(resume_state_path)
        if resume_state is not None:
            print(f"Resume: loaded run_state from {resume_state_path}")
            if resume_state.get("theta_max") is not None:
                run_mmicrl = False
                print("Resume: skipping MMICRL pretrain (theta_max restored from run_state)")
        else:
            print(f"Resume: no run_state.pt at {resume_state_path} — counters will start at 0")

    mmicrl_results = None
    mmicrl_model = None
    if run_mmicrl:
        # Write MMICRL artifacts under per-seed log dir so concurrent
        # seeds (5 Colab accounts) don't overwrite each other's results.
        # Use run_name if provided so ablation rungs don't share MMICRL output.
        mmicrl_log_subdir = args.run_name if args.run_name else args.method
        mmicrl_log_dir = os.path.join("logs", mmicrl_log_subdir, f"seed_{args.seed}")
        mmicrl_results, mmicrl_model = run_mmicrl_pretrain(cfg, log_dir=mmicrl_log_dir)

    # Inject CLI overrides into config.
    # Appendix F probe fix: action_mode now follows
    # CLI > YAML > fallback ('discrete') precedence. Previously the argparse
    # default 'discrete' silently overrode YAML configs that explicitly
    # set action_mode='continuous' (Probe Gate 0 fragility).
    if args.action_mode is not None:
        cfg["action_mode"] = args.action_mode
    # else: leave cfg['action_mode'] alone; train.py:456 falls back to 'discrete'
    # via cfg.get('action_mode', 'discrete') if neither CLI nor YAML set it.
    cfg["welfare_type"] = args.welfare
    cfg["allocation_interval"] = args.allocation_interval

    train(cfg, args.method, args.seed, device, args.resume, mmicrl_results, mmicrl_model,
          ecbf_mode=ecbf_mode, use_nswf=use_nswf, disagreement_type=disagreement_type,
          drive_backup_dir=args.drive_backup_dir, resume_state=resume_state,
          run_name=args.run_name,
          budget_cap=args.budget_cap, cost_per_hour=args.cost_per_hour,
          budget_margin=args.budget_margin)


if __name__ == "__main__":
    main()
