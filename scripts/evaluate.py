"""
HC-MARL Phase 4 (#55): Evaluation Script
==========================================
Load trained checkpoint, run N episodes, compute all 9 HC-MARL metrics.

Reconstructs the SAME environment the agent was trained in by reading
the saved config.yaml from the training log directory (which includes
MMICRL results, ablation flags, and muscle overrides).

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/hcmarl/seed_0/checkpoint_final.pt \
        --config config/hcmarl_full_config.yaml --method hcmarl --n-episodes 100
"""

import argparse
import os
import sys
import json
import warnings
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hcmarl.envs.pettingzoo_wrapper import WarehousePettingZoo
from hcmarl.agents.ippo import IPPO
from hcmarl.logger import HCMARLLogger
from hcmarl.utils import seed_everything, build_per_worker_theta_max
from scripts.train import create_agent
import yaml


def _build_eval_env(cfg, mmicrl_results=None, method="hcmarl"):
    """Build evaluation env that matches the training env exactly.

    Reads the same config keys as train.py to ensure:
    - ecbf_mode matches (on/off for no_ecbf ablation)
    - disagreement_type matches (constant for no_divergent ablation)
    - muscle_params_override matches (r=1 for no_reperfusion ablation)
    - theta_max matches (MMICRL-derived per-worker thresholds for hcmarl)
    """
    env_cfg = cfg.get("environment", {})
    n_workers = env_cfg.get("n_workers", 6)
    max_steps = env_cfg.get("max_steps", 480)

    ecbf_cfg = cfg.get("ecbf", {})
    ecbf_mode = "off" if not ecbf_cfg.get("enabled", True) else "on"

    disagree_cfg = cfg.get("disagreement", {})
    disagreement_type = disagree_cfg.get("type", "divergent")

    muscle_params_override = None
    muscle_groups_cfg = env_cfg.get("muscle_groups")
    if muscle_groups_cfg:
        muscle_params_override = {}
        for m_name, m_params in muscle_groups_cfg.items():
            muscle_params_override[m_name] = {
                k: v for k, v in m_params.items() if k in ("F", "R", "r")
            }

    config_theta_defaults = env_cfg.get("theta_max", {}) or {}
    # Shared helper applies the same clamp-to-floor logic train.py uses
    # so the eval env thresholds match what the policy was trained under.
    mmicrl_cfg = cfg.get("mmicrl", {})
    theta_max = build_per_worker_theta_max(
        mmicrl_results, config_theta_defaults, n_workers, method,
        rescale_to_floor=bool(mmicrl_cfg.get("rescale_to_floor", True)),
        mi_collapse_threshold=float(mmicrl_cfg.get("mi_collapse_threshold", 0.01)),
    )
    # Fall back to config flat dict for baselines / no-MMICRL runs
    if theta_max is None or (isinstance(theta_max, dict) and not theta_max):
        theta_max = config_theta_defaults if config_theta_defaults else None

    ecbf_alpha1 = ecbf_cfg.get("alpha1", 0.05)
    ecbf_alpha2 = ecbf_cfg.get("alpha2", 0.05)
    ecbf_alpha3 = ecbf_cfg.get("alpha3", 0.1)

    env = WarehousePettingZoo(
        n_workers=n_workers, max_steps=max_steps,
        theta_max=theta_max, ecbf_mode=ecbf_mode,
        disagreement_type=disagreement_type,
        muscle_params_override=muscle_params_override,
        ecbf_alpha1=ecbf_alpha1, ecbf_alpha2=ecbf_alpha2, ecbf_alpha3=ecbf_alpha3,
    )
    return env


def evaluate(cfg, method, checkpoint_path, n_episodes=100, seed=42, device="cpu"):
    """Run evaluation and compute all 9 metrics."""
    import random as _random
    seed_everything(seed)
    _random.seed(seed)

    mmicrl_results = cfg.get("mmicrl_results", None)
    env = _build_eval_env(cfg, mmicrl_results=mmicrl_results, method=method)

    n_workers = env.n_workers
    max_steps = env.max_steps
    obs_dim = env.obs_dim
    global_obs_dim = env.global_obs_dim
    n_actions = env.n_tasks

    is_ippo = (method == "ippo")
    agent = create_agent(method, obs_dim, global_obs_dim, n_actions, n_workers, cfg, device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")

    all_metrics = {k: [] for k in HCMARLLogger.METRIC_NAMES}
    all_metrics["safety_autonomy_index"] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        # Harmonized with train.py metric definitions:
        #   episode_cost  -- count of worker-steps with ANY muscle violation
        #                    (matches train.py "cumulative_cost")
        #   total_dense_cost -- sum of dense per-step info["cost"] values
        #                    (matches what the Lagrangian critic sees)
        episode_cost = 0
        total_dense_cost = 0.0
        safe_steps = 0
        tasks_per_worker = np.zeros(n_workers)
        peak_mf = 0.0
        forced_rests = 0
        total_ecbf_interventions = 0
        total_ecbf_opportunities = 0
        recovery_times = []
        in_violation = {i: False for i in range(n_workers)}
        violation_start = {i: 0 for i in range(n_workers)}

        for step in range(max_steps):
            global_state = env._get_global_obs()
            if is_ippo:
                result = agent.get_actions(obs)
            else:
                result = agent.get_actions(obs, global_state)
            actions = result[0] if isinstance(result, tuple) else result

            obs, rewards, terms, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values())

            step_any_violation = 0  # count of worker-steps with any muscle violation
            step_dense_cost = 0.0
            for i, (agent_id, info) in enumerate(sorted(infos.items())):
                fatigue = info.get("fatigue", {})
                task = info.get("task", "rest")
                worker_theta = env.theta_max_per_worker[i]
                # info["violations"] is int(cost > 0) from the env — 1 iff
                # this worker violated on this step (matches train.py).
                if info.get("violations", 0) > 0:
                    step_any_violation += 1
                step_dense_cost += float(info.get("cost", 0.0))
                for m, mf in fatigue.items():
                    if m not in worker_theta:
                        warnings.warn(f"theta_max missing for muscle '{m}', using conservative default 0.5")
                    theta = worker_theta.get(m, 0.5)
                    if mf > theta:
                        if not in_violation[i]:
                            in_violation[i] = True
                            violation_start[i] = step
                    else:
                        if in_violation[i]:
                            recovery_times.append(step - violation_start[i])
                            in_violation[i] = False
                    peak_mf = max(peak_mf, mf)
                if task != "rest":
                    tasks_per_worker[i] += 1
                ecbf_int = info.get("ecbf_interventions", 0)
                total_ecbf_interventions += ecbf_int
                if ecbf_int > 0:
                    forced_rests += 1
                if task != "rest":
                    demands = env.task_mgr.get_demand_vector(task)
                    total_ecbf_opportunities += int((demands > 0).sum())

            episode_cost += step_any_violation
            total_dense_cost += step_dense_cost
            if step_any_violation == 0:
                safe_steps += 1

            if all(terms.values()):
                break

        n_steps = step + 1
        n = n_workers
        jain = float((tasks_per_worker.sum()**2) / (n * (tasks_per_worker**2).sum() + 1e-8)) if tasks_per_worker.sum() > 0 else 1.0
        sai = 1.0 - (total_ecbf_interventions / max(1, total_ecbf_opportunities))

        # Denominator matches train.py: per worker-step.
        all_metrics["violation_rate"].append(episode_cost / max(1, n_steps * n_workers))
        all_metrics["cumulative_cost"].append(float(episode_cost))
        all_metrics["safety_rate"].append(safe_steps / max(1, n_steps))
        all_metrics["tasks_completed"].append(float(tasks_per_worker.sum()))
        all_metrics["cumulative_reward"].append(total_reward)
        all_metrics["jain_fairness"].append(jain)
        all_metrics["peak_fatigue"].append(peak_mf)
        all_metrics["forced_rest_rate"].append(forced_rests / max(1, n_steps * n_workers))
        all_metrics["constraint_recovery_time"].append(np.mean(recovery_times) if recovery_times else 0.0)
        all_metrics["safety_autonomy_index"].append(sai)
        all_metrics.setdefault("dense_cost_mean", []).append(total_dense_cost / max(1, n_steps * n_workers))

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: R={total_reward:.1f}, Safe={safe_steps/n_steps:.2%}")

    results = {}
    for k, vals in all_metrics.items():
        results[f"{k}_mean"] = float(np.mean(vals))
        results[f"{k}_std"] = float(np.std(vals))

    results["method"] = method
    results["n_episodes"] = n_episodes
    results["seed"] = seed
    results["checkpoint"] = checkpoint_path

    return results


def main():
    parser = argparse.ArgumentParser(description="HC-MARL Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, default="hcmarl")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # If a training config.yaml was saved alongside the checkpoint (by train.py),
    # load it instead — it contains MMICRL results and the exact env settings.
    ckpt_dir = os.path.dirname(args.checkpoint)
    log_dir = ckpt_dir.replace("checkpoints", "logs")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    if os.path.exists(saved_cfg_path):
        print(f"Loading training config from {saved_cfg_path}")
        with open(saved_cfg_path) as f:
            cfg = yaml.safe_load(f)

    results = evaluate(cfg, args.method, args.checkpoint, args.n_episodes, args.seed, args.device)

    print(f"\n{'='*60}")
    print(f"Evaluation Results: {args.method} (seed {args.seed})")
    print(f"{'='*60}")
    for k in HCMARLLogger.METRIC_NAMES:
        print(f"  {k:30s}: {results[f'{k}_mean']:8.4f} +/- {results[f'{k}_std']:8.4f}")
    if "safety_autonomy_index_mean" in results:
        print(f"  {'safety_autonomy_index':30s}: {results['safety_autonomy_index_mean']:8.4f} +/- {results['safety_autonomy_index_std']:8.4f}")

    out_path = args.output or f"results/{args.method}_seed{args.seed}_eval.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
