"""
HC-MARL Phase 2 (#23): PettingZoo ParallelEnv Wrapper
N workers with parallel observations and actions.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from hcmarl.envs.reward_functions import nswf_reward, safety_cost
from hcmarl.ecbf_filter import ECBFFilter, ECBFParams
from hcmarl.three_cc_r import ThreeCCrState, get_muscle


class WarehousePettingZoo:
    """PettingZoo-style parallel environment for N warehouse workers."""

    metadata = {"name": "WarehouseHCMARL-v0"}

    def __init__(self, n_workers=4, muscle_groups=None, tasks=None, theta_max=None,
                 dt=1.0, max_steps=480, kappa=1.0, ecbf_mode="on",
                 action_mode="discrete", disagreement_type="divergent",
                 muscle_params_override=None,
                 ecbf_alpha1=0.05, ecbf_alpha2=0.05, ecbf_alpha3=0.1,
                 n_types=0, worker_type_assignments=None):
        """
        Args:
            action_mode: "discrete" (default) — agent selects a task index.
                "continuous" — agent outputs C_nom per muscle (Remark 7.2).
                In continuous mode, task assignment comes from the NSWF
                allocator at the outer loop (C-7.A), not from the agent.
            disagreement_type: "divergent" (Eq 32, default) or "constant"
                (D_i = kappa). Used for the no_divergent ablation (C-17).
            muscle_params_override: Optional dict {muscle_name: {"F": ..., "R": ..., "r": ...}}
                to override MUSCLE_REGISTRY values. Used for no_reperfusion
                ablation (C-17) where r=1 for all muscles.
        """
        from hcmarl.envs.task_profiles import TaskProfileManager
        if ecbf_mode not in ("on", "off"):
            raise ValueError(f"ecbf_mode must be 'on' or 'off', got '{ecbf_mode}'")
        if action_mode not in ("discrete", "continuous"):
            raise ValueError(f"action_mode must be 'discrete' or 'continuous', got '{action_mode}'")
        self.ecbf_mode = ecbf_mode
        self.action_mode = action_mode
        self.disagreement_type = disagreement_type
        self.n_workers = n_workers
        self.dt = dt
        self.max_steps = max_steps
        self.kappa = kappa

        self.task_mgr = TaskProfileManager()
        self.muscle_names = self.task_mgr.muscle_names
        self.task_names = self.task_mgr.task_names
        self.n_muscles = len(self.muscle_names)
        self.n_tasks = len(self.task_names)

        # Muscle params: use override if provided (e.g., no_reperfusion sets r=1)
        if muscle_params_override:
            self.muscle_params = {}
            for m in self.muscle_names:
                base = {"F": get_muscle(m).F, "R": get_muscle(m).R, "r": get_muscle(m).r}
                if m in muscle_params_override:
                    base.update(muscle_params_override[m])
                self.muscle_params[m] = base
        else:
            self.muscle_params = {m: {"F": get_muscle(m).F, "R": get_muscle(m).R, "r": get_muscle(m).r} for m in self.muscle_names}

        # Default theta_max matches config/hcmarl_full_config.yaml and all
        # baseline configs. Rest-phase floor theta_min_max = F/(F + R*r) under
        # Frey-Law et al. 2012 Table 1 values:
        #   shoulder 41.9% | ankle 40.4% | knee 40.2% | elbow 39.3% |
        #   trunk 40.2%    | grip 33.8%.
        # Margins vs the defaults below:
        #   shoulder 28.1pp | ankle 39.6pp | knee 19.8pp | elbow  5.7pp |
        #   trunk    24.8pp | grip  11.2pp.
        # Grip raised 0.35->0.45 because 0.35 gave only 1.2pp margin above
        # theta_min_max=33.8% and caused ECBF near-infeasibility on grip-
        # intensive tasks. 0.45 sits below the >50% "forceful" threshold
        # of ISO 11228-3 / ACGIH TLV and gives Frey-Law & Avin 2010 ET of
        # ~2 min. Time-weighted mean grip must remain below 0.17 (Byström
        # & Fransson-Hall 1994 intermittent ceiling) -- enforced by duty cycle.
        # Rr/F (recovery bandwidth per unit fatigue) under corrected values:
        #   shoulder 1.38 | ankle 1.48 | knee 1.49 | elbow 1.55 |
        #   trunk    1.49 | grip 1.96.
        # All six muscles recover faster than they fatigue (Rr/F > 1), so
        # fatigue is self-limiting under sustainable load.  The ECBF is
        # most informative for transient overshoot detection, not steady-
        # state confinement.
        default_theta = {
            "shoulder": 0.70, "ankle": 0.80, "knee": 0.60,
            "elbow": 0.45, "trunk": 0.65, "grip": 0.45,
        }
        if theta_max is None:
            self.theta_max_per_worker = {i: dict(default_theta) for i in range(n_workers)}
        elif any(isinstance(v, dict) for v in theta_max.values()):
            # Per-worker thresholds: {"worker_0": {"shoulder": 0.7, ...}, ...}
            self.theta_max_per_worker = {}
            for i in range(n_workers):
                key = f"worker_{i}"
                if key in theta_max:
                    merged = dict(default_theta)
                    merged.update(theta_max[key])
                    self.theta_max_per_worker[i] = merged
                else:
                    self.theta_max_per_worker[i] = dict(default_theta)
        else:
            # Flat dict: {"shoulder": 0.7, ...} — same for all workers
            self.theta_max_per_worker = {i: dict(theta_max) for i in range(n_workers)}
        # Keep a flat reference for backward compatibility
        self.theta_max = self.theta_max_per_worker[0]

        # C-6.A: Build per-worker, per-muscle ECBFFilter instances
        ecbf_alpha1 = ecbf_alpha1
        ecbf_alpha2 = ecbf_alpha2
        ecbf_alpha3 = ecbf_alpha3
        self.ecbf_filters = {}  # {worker_idx: {muscle_name: ECBFFilter}}
        for i in range(n_workers):
            self.ecbf_filters[i] = {}
            for m in self.muscle_names:
                mp = get_muscle(m)
                theta = self.theta_max_per_worker[i].get(m, default_theta.get(m, 0.7))
                # Ensure theta_max satisfies Assumption 5.5
                theta = max(theta, mp.theta_min_max + 0.01)
                params = ECBFParams(
                    theta_max=theta,
                    alpha1=ecbf_alpha1,
                    alpha2=ecbf_alpha2,
                    alpha3=ecbf_alpha3,
                )
                self.ecbf_filters[i][m] = ECBFFilter(muscle=mp, ecbf_params=params)

        # Pre-cache ECBF scalar parameters for inlined filter_analytical.
        # Eliminates per-step method-call + dataclass overhead in _integrate.
        # Layout: _ecbf_cache[worker_idx][muscle_idx] = (F, alpha1, alpha2, alpha3, theta_max)
        self._ecbf_cache = []
        for i in range(n_workers):
            row = []
            for m in self.muscle_names:
                ef = self.ecbf_filters[i][m]
                row.append((ef._F, ef._alpha1, ef._alpha2, ef._alpha3, ef._theta_max))
            self._ecbf_cache.append(row)

        self.possible_agents = [f"worker_{i}" for i in range(n_workers)]
        self.agents = list(self.possible_agents)

        # Option X: per-worker type conditioning (MMICRL).
        # When n_types > 0, each worker's observation is augmented with a
        # one-hot encoding of its MMICRL-discovered type. This lets the
        # policy condition on worker type when type discovery is informative.
        # When n_types == 0 (default), behavior is identical to legacy env.
        #
        # worker_type_assignments: {worker_idx: type_idx} or None.
        #   Missing entries default to type 0 (safe fallback — policy sees
        #   one-hot(0) = [1, 0, ..., 0], equivalent to single-type behavior).
        self.n_types = int(n_types)
        self.worker_type_assignments = dict(worker_type_assignments or {})
        for i in range(n_workers):
            self.worker_type_assignments.setdefault(i, 0)
        if self.n_types > 0:
            for i, z in self.worker_type_assignments.items():
                if not (0 <= int(z) < self.n_types):
                    raise ValueError(
                        f"worker {i} type {z} out of range [0, {self.n_types})"
                    )

        # C-8.A: In continuous mode, obs includes task assignment one-hot
        type_oh_dim = self.n_types if self.n_types > 0 else 0
        if self.action_mode == "continuous":
            # obs = [MR,MA,MF]*muscles + task_one_hot(n_tasks) + step_norm + [type_oh]
            self.obs_dim = self.n_muscles * 3 + self.n_tasks + 1 + type_oh_dim
        else:
            self.obs_dim = self.n_muscles * 3 + 1 + type_oh_dim
        self.global_obs_dim = n_workers * (self.n_muscles * 3) + 1

        # Per-worker current task assignment (for continuous mode, set by allocator)
        self._task_assignments = {i: 0 for i in range(n_workers)}  # 0 = rest

        self.states = {}
        self.current_step = 0

    def _init_worker(self):
        return {m: {"MR": 1.0, "MA": 0.0, "MF": 0.0} for m in self.muscle_names}

    def _get_obs(self, worker_idx):
        obs = []
        for m in self.muscle_names:
            s = self.states[worker_idx][m]
            obs.extend([s["MR"], s["MA"], s["MF"]])
        if self.action_mode == "continuous":
            # Append task assignment one-hot (conditioning for the actor)
            task_oh = np.zeros(self.n_tasks, dtype=np.float32)
            task_idx = self._task_assignments.get(worker_idx, 0)
            task_oh[min(task_idx, self.n_tasks - 1)] = 1.0
            obs.extend(task_oh.tolist())
        obs.append(self.current_step / self.max_steps)
        # Option X: append per-worker type one-hot when type-conditioning is on.
        # Under MI collapse the caller is expected to pass n_types=1 or set
        # every worker's type to 0, so this one-hot is constant and the
        # policy learns to ignore it (graceful degenerate single-type behavior).
        if self.n_types > 0:
            type_oh = np.zeros(self.n_types, dtype=np.float32)
            z_i = self.worker_type_assignments.get(worker_idx, 0)
            type_oh[int(z_i)] = 1.0
            obs.extend(type_oh.tolist())
        return np.array(obs, dtype=np.float32)

    def _get_global_obs(self):
        obs = []
        for i in range(self.n_workers):
            for m in self.muscle_names:
                s = self.states[i][m]
                obs.extend([s["MR"], s["MA"], s["MF"]])
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def _integrate(self, worker_idx, task_name):
        """Integrate 3CC-r ODEs for one worker, one timestep.

        Returns:
            (ecbf_interventions, ecbf_clip_total, barrier_violations):
            - ecbf_interventions: muscles where ECBF clipped C_nom
            - ecbf_clip_total: total clipped magnitude
            - barrier_violations: muscles where discrete-time step crossed
              the barrier (MF > theta_max or MR < 0) despite ECBF filtering.
              S-4: empirical verification that the continuous-time ECBF
              guarantee holds under Euler discretization.
        """
        demands = self.task_mgr.get_demand_vector(task_name)
        kp = 1.0  # proportional gain (Eq 35)
        ecbf_interventions = 0
        ecbf_clip_total = 0.0
        barrier_violations = 0
        worker_theta = self.theta_max_per_worker[worker_idx]
        ecbf_row = self._ecbf_cache[worker_idx]
        for mi, m in enumerate(self.muscle_names):
            p = self.muscle_params[m]
            F, R_base, r = p["F"], p["R"], p["r"]
            TL = demands[mi]
            s = self.states[worker_idx][m]
            MR, MA, MF = s["MR"], s["MA"], s["MF"]
            C_nominal = kp * max(TL - MA, 0.0) if TL > 0 else 0.0
            R_eff = R_base if TL > 0 else R_base * r
            if self.ecbf_mode == "on":
                # Inlined filter_analytical: eliminates 3 method calls +
                # ThreeCCrState dataclass construction per muscle per step.
                eF, a1, a2, a3, tm = ecbf_row[mi]
                # ecbf_upper_bound (Eq 19)
                state_terms = eF**2 * MA + R_eff * eF * MA - R_eff**2 * MF
                a1_term = a1 * (-eF * MA + R_eff * MF)
                psi1_val = (-eF * MA + R_eff * MF) + a1 * (tm - MF)
                ub_ecbf = (state_terms + a1_term + a2 * psi1_val) / eF
                # cbf_upper_bound (Eq 23)
                ub_cbf = R_eff * MF + a3 * (1.0 - MA - MF)
                C = min(C_nominal, ub_ecbf, ub_cbf)
                C = max(0.0, C)
            else:
                C = max(0.0, C_nominal)
            # Track ECBF interventions
            if C_nominal > 1e-9 and (C_nominal - C) > 1e-9:
                ecbf_interventions += 1
                ecbf_clip_total += C_nominal - C
            dMA = C - F*MA
            dMF = F*MA - R_eff*MF
            dMR = R_eff*MF - C
            MA_n = max(0.0, MA + dMA*self.dt)
            MF_n = max(0.0, MF + dMF*self.dt)
            MR_n = 1.0 - MA_n - MF_n
            if MR_n < 0.0:
                s = MA_n + MF_n
                if s > 0:
                    MA_n /= s; MF_n /= s
                MR_n = 0.0
            # S-4: post-step barrier verification (discrete-time check)
            theta_m = worker_theta.get(m, 0.7)
            if MF_n > theta_m + 1e-6:
                barrier_violations += 1
            if MR_n < -1e-6:
                barrier_violations += 1
            self.states[worker_idx][m] = {"MR": MR_n, "MA": MA_n, "MF": MF_n}
        return ecbf_interventions, float(ecbf_clip_total), barrier_violations

    def _integrate_continuous(self, worker_idx, c_nom_per_muscle):
        """Integrate 3CC-r with agent-provided C_nom per muscle (Remark 7.2).

        In continuous mode, the RL policy outputs C_nom directly and the
        ECBF filter clips it. The task assignment determines R_eff only
        (work vs rest recovery rate).

        Args:
            worker_idx: Worker index.
            c_nom_per_muscle: np.array of shape (n_muscles,) with C_nom in [0, 1].

        Returns:
            (ecbf_interventions, ecbf_clip_total, barrier_violations)
        """
        task_idx = self._task_assignments.get(worker_idx, 0)
        task_name = self.task_names[task_idx]
        demands = self.task_mgr.get_demand_vector(task_name)
        ecbf_interventions = 0
        ecbf_clip_total = 0.0
        barrier_violations = 0
        worker_theta = self.theta_max_per_worker[worker_idx]
        ecbf_row = self._ecbf_cache[worker_idx]
        for mi, m in enumerate(self.muscle_names):
            p = self.muscle_params[m]
            F, R_base, r = p["F"], p["R"], p["r"]
            TL = demands[mi]
            s = self.states[worker_idx][m]
            MR, MA, MF = s["MR"], s["MA"], s["MF"]
            C_nominal = float(c_nom_per_muscle[mi])
            R_eff = R_base if TL > 0 else R_base * r
            if self.ecbf_mode == "on":
                # Inlined filter_analytical (same as _integrate).
                eF, a1, a2, a3, tm = ecbf_row[mi]
                state_terms = eF**2 * MA + R_eff * eF * MA - R_eff**2 * MF
                a1_term = a1 * (-eF * MA + R_eff * MF)
                psi1_val = (-eF * MA + R_eff * MF) + a1 * (tm - MF)
                ub_ecbf = (state_terms + a1_term + a2 * psi1_val) / eF
                ub_cbf = R_eff * MF + a3 * (1.0 - MA - MF)
                C = min(C_nominal, ub_ecbf, ub_cbf)
                C = max(0.0, C)
            else:
                C = max(0.0, C_nominal)
            if C_nominal > 1e-9 and (C_nominal - C) > 1e-9:
                ecbf_interventions += 1
                ecbf_clip_total += C_nominal - C
            dMA = C - F * MA
            dMF = F * MA - R_eff * MF
            dMR = R_eff * MF - C
            MA_n = max(0.0, MA + dMA * self.dt)
            MF_n = max(0.0, MF + dMF * self.dt)
            MR_n = 1.0 - MA_n - MF_n
            if MR_n < 0.0:
                s = MA_n + MF_n
                if s > 0:
                    MA_n /= s; MF_n /= s
                MR_n = 0.0
            # S-4: post-step barrier verification
            theta_m = worker_theta.get(m, 0.7)
            if MF_n > theta_m + 1e-6:
                barrier_violations += 1
            if MR_n < -1e-6:
                barrier_violations += 1
            self.states[worker_idx][m] = {"MR": MR_n, "MA": MA_n, "MF": MF_n}
        return ecbf_interventions, float(ecbf_clip_total), barrier_violations

    def set_task_assignments(self, assignments):
        """Set task assignments from the NSWF allocator (C-7.A).

        Args:
            assignments: dict {worker_idx: task_idx} or {agent_name: task_idx}
        """
        for key, task_idx in assignments.items():
            if isinstance(key, str):
                idx = int(key.split("_")[1])
            else:
                idx = int(key)
            self._task_assignments[idx] = task_idx

    def reset(self, seed=None):
        self.agents = list(self.possible_agents)
        self.current_step = 0
        self.states = {i: self._init_worker() for i in range(self.n_workers)}
        self._task_assignments = {i: 0 for i in range(self.n_workers)}
        obs = {a: self._get_obs(int(a.split("_")[1])) for a in self.agents}
        return obs, {a: {} for a in self.agents}

    def step(self, actions):
        rewards, infos = {}, {}
        for agent in self.agents:
            idx = int(agent.split("_")[1])

            if self.action_mode == "continuous":
                # C-8.A: actions[agent] is np.array of C_nom per muscle
                c_nom = np.asarray(actions[agent], dtype=np.float32)
                ecbf_interventions, ecbf_clip_total, barrier_violations = self._integrate_continuous(idx, c_nom)
                task_idx = self._task_assignments.get(idx, 0)
                task_name = self.task_names[task_idx]
            else:
                # Discrete: actions[agent] is task index
                task_name = self.task_names[actions[agent]]
                ecbf_interventions, ecbf_clip_total, barrier_violations = self._integrate(idx, task_name)

            fatigue = {m: self.states[idx][m]["MF"] for m in self.muscle_names}
            prod = self.task_mgr.task_intensity(task_name)
            worker_theta = self.theta_max_per_worker[idx]
            rewards[agent] = nswf_reward(prod, fatigue, worker_theta, kappa=self.kappa,
                                         disagreement_type=self.disagreement_type)
            cost = safety_cost(fatigue, worker_theta)
            violations = int(cost > 0)
            infos[agent] = {
                "task": task_name, "fatigue": fatigue,
                "violations": violations, "cost": cost,
                "ecbf_interventions": ecbf_interventions,
                "ecbf_clip_total": ecbf_clip_total,
                "barrier_violations": barrier_violations,  # S-4: discrete-time barrier check
            }
        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = {a: self._get_obs(int(a.split("_")[1])) for a in self.agents}
        terms = {a: done for a in self.agents}
        truncs = {a: False for a in self.agents}
        return obs, rewards, terms, truncs, infos
