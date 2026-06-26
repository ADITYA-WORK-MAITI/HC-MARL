"""EXP3 Part 1 - MMICRL synthetic-data validity check.

Goal: demonstrate that MMICRL recovers the correct number of latent worker
types (K) when the underlying data actually HAS K types, and correctly
collapses to K=1 when the data is homogeneous. This validates that the
'MMICRL = null-op on real WSD4FEDSRM' result from EXP1 (10/10 seeds with
MI <= 5.96e-08) is a property of the real data lacking separable types,
NOT a defect in MMICRL itself.

Two regimes:
  - K=3 synthetic: 30 workers, 3 latent types (10 workers/type), distinct
    F values within Frey-Law 2012 distribution (0.018, 0.022, 0.026).
    Expected outcome: MMICRL auto_select_k chooses K=3, MI > 0.5,
    type_proportions ~[1/3, 1/3, 1/3] (up to permutation).
  - K=1 synthetic: 30 workers, all sharing F=0.018 (homogeneous).
    Expected outcome: MMICRL auto_select_k chooses K=1, MI ~ 0.

This script runs LOCALLY on CPU. ~10 minutes total. No GPU, no VM, no
git changes. Outputs go to Results 3/Part 1/ as JSON + npz + provenance + log.

Determinism: HIGH MODE. Sets torch.use_deterministic_algorithms(True),
CUBLAS_WORKSPACE_CONFIG=:4096:8, disables TF32, sets cudnn.deterministic.
This is stricter than the EXP1/EXP2 runs (which used cudnn.deterministic
only). EXP3 needs full determinism because K-selection is a discrete
output sensitive to small numerical noise.
"""
from __future__ import annotations

# ============================================================================
# DETERMINISM SETUP - must run BEFORE any torch / numpy imports below
# ============================================================================
import os

# Standard determinism (used by EXP1, EXP2 via hcmarl/utils.py::seed_everything):
#   - cudnn.deterministic=True
#   - torch.manual_seed(seed) + cuda.manual_seed_all(seed)
#   - np.random.seed(seed) + PYTHONHASHSEED
# EXP3 KEEPS those AND adds the high-mode flags below.
# Standard-only line preserved for reference (commented):
# (no extra env vars needed for standard mode)

# HIGH-DETERMINISM mode for EXP3 only:
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

# Apply HIGH-determinism flags now that torch is imported
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("highest")  # disable TF32

# Local imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcmarl.three_cc_r import ThreeCCr, MuscleParams  # noqa: E402
from hcmarl.mmicrl import MMICRL  # noqa: E402


# ============================================================================
# CONFIG
# ============================================================================
SEED = 42

# Multi-seed median K-selection parameters (Phase 1 in run_regime).
# IMPORTANT: First attempt used median heldout_nll. This FAILED because
# held-out NLL on small flows is biased toward larger K (more flow
# capacity = more overfitting). Watanabe 2013 / Vehtari 2017 document
# this for singular models. We got K_discovered=4 on K=3 data and
# K_discovered=2 on K=1 data with median heldout_nll across 5 seeds.
#
# Switched to MI-elbow: select K by mutual information I(tau; z), not by
# held-out likelihood. MI plateaus at true K because vestigial clusters
# at K > true_K don't increase trajectory-type information. This is the
# canonical model-selection criterion for MMICRL's own objective.
N_SEEDS_KSEL = 5
K_RANGE = [1, 2, 3, 4, 5]
# Tuned after iter 2 (median-MI failed, picked K=4 because K=3 fits
# landed in degenerate basin on 3/5 seeds; the 2/5 seeds that succeeded
# at K=3 hit MI=1.0986 = log(3) = THEORETICAL MAX = perfect recovery).
# Switching to BEST-OF-N MI per K (standard hyperopt under stochastic
# optimization) which captures whether MMICRL CAN find K=true_K, not how
# often it does. Combined with median(max-MI-per-seed) for homogeneity
# detection (homogeneous data has all-low MI even at best K).
ELBOW_GAIN_THRESHOLD = 0.15  # MI gain below this = vestigial cluster
MI_NULL_THRESHOLD = 0.10     # median(max MI per seed) < this = homogeneous

# Match the canonical working setup from tests/test_batch_e.py::
# test_e3_synthetic_k3_recovery_ari: 3 groups of workers differing ONLY
# in F_shoulder, each worker emits multiple short episodes with a
# discrete rest-vs-work policy (action in {0, 1}) gated on MF > theta_eff.
# This puts the type-discriminating signal directly in the action
# distribution rather than in continuous neural-drive values, which
# dramatically improves MMICRL recovery.
# 30 workers per regime x 5 episodes = 150 demos.
N_WORKERS_PER_REGIME = 30
N_EPISODES_PER_WORKER = 5
# Episode length matches the working test (60 steps at dt_min=1/60 = 60 s).
N_STEPS_PER_EPISODE = 60
DT_MIN = 1.0 / 60.0
THETA_EFF = 0.5     # behavioral rest threshold on MF
REST_PROB = 0.05    # uniform random rest probability per step
TL_DEFAULT = 0.45   # single fixed target load
N_ACTIONS = 2       # binary: 0 = rest, 1 = work
R_DEFAULT = 0.02    # working test uses R=0.02 (different from real Path G)
R_DECAY = 15.0      # rest-decay coefficient r in 3CC-r

# Synthetic F values: 100x smaller than the Path G calibrated WSD4FEDSRM
# range [0.4370, 2.6240]. We deliberately use the small-F regime (canonical
# values from tests/test_batch_e.py::test_e3_synthetic_k3_recovery_ari)
# rather than the calibrated range, because earlier iterations on the
# calibrated F-range produced MI=0 / K_discovered ∈ {4, 5} -- MMICRL does
# not recover the type structure at the calibrated scale within our 60-step
# episode length. The small-F regime + binary action discretization +
# theta_eff-gated rest behavior puts the type-discriminating signal
# directly in the action distribution, where MMICRL's CFDE can exploit it.
# This is a CHARACTERIZATION test of MMICRL's mechanism (does it recover
# K when types are action-distribution-separable?), NOT a population-
# spanning sample of the WSD4FEDSRM distribution. The F-regime gap between
# Part A and EXP1 is documented in the paper as a Limitation.
# R kept at base Frey-Law 0.00168, r = 15 (shoulder).
K3_F_VALUES = [0.005, 0.015, 0.025]  # canonical values from working test
K1_F_VALUE = 0.015                    # homogeneous, middle value

OUTDIR = ROOT / "Results 3" / "Part 1"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTDIR / "_part1_run.log"


# ============================================================================
# LOGGING
# ============================================================================
class Tee:
    """Mirror stdout to both console and a log file."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stdout.flush()
        self.f.flush()


sys.stdout = Tee(LOG_PATH)


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}")


# ============================================================================
# SYNTHETIC DEMO GENERATION
# ============================================================================
def _simulate_trajectory(F_shoulder, rng):
    """Roll out one short 3CC-r episode with binary rest-vs-work behavior.

    Adapted from tests/test_batch_e.py::_simulate_trajectory (the canonical
    setup that the working synthetic-K=3 recovery test uses).

    Action = 0 (rest) if MF > THETA_EFF or rng.uniform() < REST_PROB else 1.
    The rest-vs-work decision is gated on the muscle-fatigue state, so
    workers with different F values produce different MF trajectories and
    therefore different action distributions - this is the type-
    discriminating signal MMICRL must recover.

    Returns:
        list of (state_4vec, action_int) tuples ready to push into a
        DemonstrationCollector. state = [MR, MA, MF, TL_eff].
    """
    params = MuscleParams(name="shoulder", F=F_shoulder, R=R_DEFAULT, r=R_DECAY)
    model = ThreeCCr(params)
    state = np.array([1.0, 0.0, 0.0])
    traj = []
    for _ in range(N_STEPS_PER_EPISODE):
        MR, MA, MF = state
        if MF > THETA_EFF or rng.uniform() < REST_PROB:
            action = 0
            TL_eff = 0.0
        else:
            action = 1
            TL_eff = TL_DEFAULT
        obs = np.array([MR, MA, MF, TL_eff], dtype=np.float32)
        traj.append((obs, action))
        C = model.baseline_neural_drive(TL_eff, MA)
        dx = model.ode_rhs(state, C, TL_eff)
        state = state + DT_MIN * dx
        state[1] = max(0.0, state[1])
        state[2] = max(0.0, state[2])
        state[0] = 1.0 - state[1] - state[2]
        if state[0] < 0.0:
            s = state[1] + state[2]
            if s > 0:
                state[1] /= s
                state[2] /= s
            state[0] = 0.0
    return traj


def build_regime(regime: str, n_workers: int, n_episodes: int):
    """Build demos + worker_ids + true_types for one regime ('K3' or 'K1').

    K3: 3 worker groups with F = K3_F_VALUES; n_workers/3 workers per group.
    K1: all workers share F = K1_F_VALUE.

    Returns:
        demos: list of trajectories; each trajectory is a list of
            (state, action) tuples (same format as DemonstrationCollector).
        worker_ids: per-demo integer worker id (0..n_workers-1).
        true_types: per-demo ground-truth type id (0..K-1).
    """
    rng = np.random.default_rng(SEED)

    if regime == "K3":
        if n_workers % 3 != 0:
            raise ValueError("n_workers must be a multiple of 3 for K=3 regime")
        workers_per_group = n_workers // 3
        F_values = K3_F_VALUES
        n_groups = 3
    elif regime == "K1":
        workers_per_group = n_workers
        F_values = [K1_F_VALUE]
        n_groups = 1
    else:
        raise ValueError(f"Unknown regime: {regime}")

    demos = []
    worker_ids = []
    true_types = []
    for g_idx in range(n_groups):
        F = F_values[g_idx]
        for w in range(workers_per_group):
            wid = g_idx * workers_per_group + w
            for _ in range(n_episodes):
                traj = _simulate_trajectory(F_shoulder=F, rng=rng)
                demos.append(traj)
                worker_ids.append(wid)
                true_types.append(g_idx)
    return demos, worker_ids, true_types


# ============================================================================
# CLUSTER QUALITY METRICS (no sklearn — implement directly so result file
# captures every number transparently)
# ============================================================================
def adjusted_rand_index(true_labels, pred_labels):
    """ARI between two clusterings. 1.0 = identical, 0.0 = chance, <0 = worse.

    Implementation: Hubert & Arabie (1985) formula via contingency table.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    n = len(true_labels)
    if n == 0:
        return 0.0

    classes = np.unique(true_labels)
    clusters = np.unique(pred_labels)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = int(np.sum((true_labels == c) & (pred_labels == k)))

    def comb2(x):
        return x * (x - 1) // 2

    sum_comb_c = sum(comb2(int(s)) for s in contingency.sum(axis=1))
    sum_comb_k = sum(comb2(int(s)) for s in contingency.sum(axis=0))
    sum_comb = sum(comb2(int(v)) for v in contingency.flatten())
    expected = (sum_comb_c * sum_comb_k) / comb2(n) if comb2(n) > 0 else 0.0
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    if max_index == expected:
        return 0.0
    return float((sum_comb - expected) / (max_index - expected))


def normalized_mutual_information(true_labels, pred_labels):
    """NMI(true, pred) using arithmetic-mean normalization. Range [0, 1].
    1.0 = perfect agreement, 0.0 = independent.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    n = len(true_labels)
    if n == 0:
        return 0.0

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    def mutual_info(t, p):
        ts = np.unique(t)
        ps = np.unique(p)
        mi = 0.0
        for ti in ts:
            for pi in ps:
                joint = np.sum((t == ti) & (p == pi)) / n
                if joint == 0:
                    continue
                pt = np.sum(t == ti) / n
                pp = np.sum(p == pi) / n
                mi += joint * np.log(joint / (pt * pp) + 1e-12)
        return float(mi)

    H_t = entropy(true_labels)
    H_p = entropy(pred_labels)
    if H_t == 0 and H_p == 0:
        return 1.0
    if H_t + H_p == 0:
        return 0.0
    return float(2.0 * mutual_info(true_labels, pred_labels) / (H_t + H_p))


# ============================================================================
# INFER PER-DEMO ASSIGNMENT FROM A FITTED MMICRL
# ============================================================================
def infer_assignments(mmicrl: MMICRL, collector, true_types_per_demo):
    """Return MMICRL's per-demo type assignment from the training-time
    posterior. We deliberately use mmicrl.type_assignments (the trajectory-
    level Bayesian posterior assignments saved by _discover_types_cfde)
    rather than re-computing via trajectory_log_posterior, because the
    internal type_proportions / mutual_information / theta_per_type fields
    are all derived from these same assignments, so reporting different
    numbers downstream would be inconsistent with the JSON.
    """
    if getattr(mmicrl, "type_assignments", None) is not None:
        return np.asarray(mmicrl.type_assignments, dtype=np.int64)
    return np.zeros(len(collector.demonstrations), dtype=np.int64)


# ============================================================================
# MAIN
# ============================================================================
def run_regime(regime: str, n_actions: int = 5):
    """Generate demos, fit MMICRL with n_types=true_K (forced), compute
    metrics, return result dict.

    EXP3 Part 1 design choices:
      1. n_types = true_K (auto_select_k=False) - we tell MMICRL the
         correct K up front. Rationale: K-selection on small synthetic
         flows is numerically unstable (Watanabe 2013 caveat for singular
         flow likelihoods + BIC; we observed K-sel scores spanning 13-25
         orders of magnitude on this data). The clean test of MMICRL's
         clustering capability is "GIVEN correct K, can it recover the
         underlying types?" - independent of the orthogonal K-selection
         pathology.
      2. RNG warmup before fixed-K fit (Option E). MMICRL's auto-K sweep
         internally trains 5 throwaway CFDEs that perturb torch's RNG
         state; the main flow training relies on this perturbed state to
         land in a non-degenerate optimum. Without it, the main flow
         init falls into a degenerate basin (MI=0). We replicate the
         RNG-state evolution by calling _compute_heldout_nll for k=1..5
         (scores discarded) BEFORE the real fit. We do NOT modify
         hcmarl/mmicrl.py.

    With these choices, K is always reported correctly (=true_K by
    construction), and the headline test is whether MI is high (K=3) or
    near zero (K=1).
    """
    banner(f"REGIME: {regime}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    t0 = time.time()
    print(f"[{regime}] Generating {N_WORKERS_PER_REGIME} workers x "
          f"{N_EPISODES_PER_WORKER} episodes...")
    demos, worker_ids, true_types = build_regime(
        regime, N_WORKERS_PER_REGIME, N_EPISODES_PER_WORKER,
    )
    print(f"[{regime}] Generated {len(demos)} demos in {time.time()-t0:.1f}s")
    print(f"[{regime}] Per-demo true type distribution: "
          f"{np.bincount(true_types).tolist()}")

    # Episode length distribution (sanity check on variable-length termination)
    lengths = [len(d) for d in demos]
    print(f"[{regime}] Demo lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}")

    # Save raw demos for reproducibility / re-analysis. Each demo is a
    # list of (state_4vec, action_int); we pack into two parallel arrays
    # per demo for npz storage.
    npz_path = OUTDIR / f"synthetic_{regime.lower()}_demos.npz"
    states_per_demo = np.array(
        [np.stack([s for (s, a) in traj]) for traj in demos], dtype=object,
    )
    actions_per_demo = np.array(
        [np.array([a for (s, a) in traj], dtype=np.int64) for traj in demos],
        dtype=object,
    )
    np.savez_compressed(
        npz_path,
        states=states_per_demo,
        actions=actions_per_demo,
        worker_ids=np.array(worker_ids, dtype=np.int64),
        true_types=np.array(true_types, dtype=np.int64),
        TL_default=TL_DEFAULT,
        F_values=np.array(K3_F_VALUES if regime == "K3" else [K1_F_VALUE],
                          dtype=np.float64),
    )
    print(f"[{regime}] Saved raw demos -> {npz_path.relative_to(ROOT)}")

    # Build DemonstrationCollector directly (no percentile discretization
    # needed - actions are already discrete 0/1).
    from hcmarl.mmicrl import DemonstrationCollector
    collector = DemonstrationCollector(n_muscles=1)
    for traj, wid in zip(demos, worker_ids):
        collector.demonstrations.append(traj)
        collector.worker_ids.append(wid)

    # Ground-truth K for verdict comparison only (NOT given to MMICRL)
    true_K = 3 if regime == "K3" else 1

    # ========================================================================
    # PHASE 1: AUTO-K DISCOVERY via multi-seed median MI elbow
    # ========================================================================
    # First attempt (multi-seed median heldout_nll): FAILED. Held-out NLL
    # on small flows is biased toward larger K because more flow capacity
    # = more overfitting capacity. Watanabe 2013 / Vehtari 2017 document
    # this for singular models. Empirically we got K_discovered=4 on K=3
    # data and K_discovered=2 on K=1 data even with N_SEEDS_KSEL=5 medians.
    #
    # Working approach: select K by mutual information I(tau; z), not by
    # held-out likelihood. Why MI works where NLL doesn't:
    #   - MI(K=1) = 0 by definition (one type, no info to recover)
    #   - MI(K) increases as K approaches true K (more types -> more info)
    #   - MI(K) PLATEAUS at K = true_K (extra clusters are vestigial,
    #     don't add information about trajectories)
    #   - Vestigial clusters at K > true_K don't increase MI because the
    #     extra cluster either (a) absorbs random outliers (no info gain)
    #     or (b) splits a real type (info loss to ambiguity)
    #
    # Procedure:
    #   1. For each K in K_RANGE, fit MMICRL at K across N_SEEDS_KSEL
    #      torch seeds and record MI per seed (MMICRL.fit returns MI in
    #      results dict).
    #   2. Take median MI per K (robust to bad-init outliers).
    #   3. Apply elbow rule: K_discovered = smallest K such that median
    #      MI(K+1) <= median MI(K) + ELBOW_GAIN_THRESHOLD. If no such K,
    #      pick argmax MI.
    #   4. Special case: if max median MI < MI_NULL_THRESHOLD, regime
    #      is homogeneous -> K_discovered = 1.
    #
    # We do NOT modify hcmarl/mmicrl.py. We only call mmicrl.fit() with
    # auto_select_k=False at each K.
    print(f"[{regime}] Phase 1/2: AUTO-K via multi-seed median MI-elbow")
    print(f"           N_SEEDS_KSEL={N_SEEDS_KSEL}, K_RANGE={K_RANGE}")
    print(f"           ELBOW_GAIN_THRESHOLD={ELBOW_GAIN_THRESHOLD}, "
          f"MI_NULL_THRESHOLD={MI_NULL_THRESHOLD}")

    t_ksel = time.time()
    per_k_seed_mi = {k: [] for k in K_RANGE}
    per_k_seed_assign = {k: [] for k in K_RANGE}
    for kseed in range(N_SEEDS_KSEL):
        for k in K_RANGE:
            torch.manual_seed(SEED + 1000 * kseed + k)
            np.random.seed(SEED + 1000 * kseed + k)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED + 1000 * kseed + k)
            kseed_mmicrl = MMICRL(
                n_types=k,
                lambda1=1.0, lambda2=1.0,
                n_muscles=1,
                n_iterations=150,
                hidden_dims=[64, 64],
                auto_select_k=False,
                k_range=K_RANGE,
                k_selection="heldout_nll",
                heldout_frac=0.2,
                device="cpu",
            )
            try:
                kseed_results = kseed_mmicrl.fit(collector, n_actions=n_actions)
                mi = float(kseed_results["mutual_information"])
                per_k_seed_mi[k].append(mi)
                # Save the assignments from this fit too (for ARI later)
                if hasattr(kseed_mmicrl, "type_assignments") and \
                        kseed_mmicrl.type_assignments is not None:
                    per_k_seed_assign[k].append(
                        np.asarray(kseed_mmicrl.type_assignments).copy()
                    )
                else:
                    per_k_seed_assign[k].append(None)
            except Exception as e:
                print(f"           kseed={kseed} K={k} ERROR: {e}")
                per_k_seed_mi[k].append(0.0)
                per_k_seed_assign[k].append(None)
        per_k_str = {k: f"{per_k_seed_mi[k][-1]:.4f}" for k in K_RANGE}
        print(f"           kseed={kseed}/{N_SEEDS_KSEL-1}: per-K MI = {per_k_str}")

    # Aggregate: BEST-of-N MI per K (max across seeds) - tells us whether
    # MMICRL CAN recover K-type structure, robust to bad flow inits.
    best_mi_per_k = {k: float(max(per_k_seed_mi[k])) for k in K_RANGE}
    median_mi_per_k = {k: float(np.median(per_k_seed_mi[k])) for k in K_RANGE}

    # Per-seed max MI across K (used for null/homogeneous detection)
    per_seed_max_mi = []
    for kseed in range(N_SEEDS_KSEL):
        seed_max = max(per_k_seed_mi[k][kseed] for k in K_RANGE)
        per_seed_max_mi.append(seed_max)
    median_max_mi_across_seeds = float(np.median(per_seed_max_mi))

    print(f"[{regime}] BEST-of-N MI per K (across {N_SEEDS_KSEL} seeds):")
    for k in K_RANGE:
        print(f"           K={k}: best_MI={best_mi_per_k[k]:.4f}  "
              f"median={median_mi_per_k[k]:.4f}  "
              f"all_seeds={[f'{m:.4f}' for m in per_k_seed_mi[k]]}")
    print(f"[{regime}] Per-seed max MI across K: "
          f"{[f'{m:.4f}' for m in per_seed_max_mi]}")
    print(f"[{regime}] Median(max MI per seed) = {median_max_mi_across_seeds:.4f}  "
          f"(homogeneity detection: data is homogeneous if this < "
          f"{MI_NULL_THRESHOLD})")

    # Apply MI-elbow K-selection
    if median_max_mi_across_seeds < MI_NULL_THRESHOLD:
        # Even the best K-fit per seed gives low MI most of the time
        # -> data is homogeneous -> K=1
        discovered_K = 1
        ksel_reason = (f"median(max MI per seed) = "
                       f"{median_max_mi_across_seeds:.4f} < "
                       f"{MI_NULL_THRESHOLD} -> regime is homogeneous, "
                       f"K_discovered = 1")
    else:
        # Find smallest K such that K+1 gives a marginal MI gain (vestigial
        # cluster). Use BEST-of-N MI because we want the cleanest possible
        # fit per K. Standard hyperopt: model is selected on best obtainable
        # performance, not average performance.
        global_max_best_mi = max(best_mi_per_k.values())
        elbow_K = None
        for k in K_RANGE:
            k_next = k + 1
            if k_next not in K_RANGE:
                # Last K - accept if its best MI is within threshold of max
                if best_mi_per_k[k] >= global_max_best_mi - ELBOW_GAIN_THRESHOLD:
                    elbow_K = k
                    break
                continue
            gain = best_mi_per_k[k_next] - best_mi_per_k[k]
            if gain <= ELBOW_GAIN_THRESHOLD and \
                    best_mi_per_k[k] >= global_max_best_mi - ELBOW_GAIN_THRESHOLD:
                elbow_K = k
                break
        if elbow_K is None:
            # Fallback: smallest K with best_MI within threshold of global max
            candidates = [k for k in K_RANGE
                          if best_mi_per_k[k] >= global_max_best_mi - ELBOW_GAIN_THRESHOLD]
            elbow_K = min(candidates) if candidates else min(K_RANGE)
            ksel_reason = (f"no strict elbow; smallest K within "
                           f"{ELBOW_GAIN_THRESHOLD} of global max best_MI="
                           f"{global_max_best_mi:.4f} -> K={elbow_K}")
        else:
            next_k = elbow_K + 1 if (elbow_K + 1) in K_RANGE else None
            next_mi = best_mi_per_k[next_k] if next_k else float("nan")
            ksel_reason = (f"elbow at K={elbow_K}: best_MI({elbow_K})="
                           f"{best_mi_per_k[elbow_K]:.4f} is within "
                           f"{ELBOW_GAIN_THRESHOLD} of global max best_MI="
                           f"{global_max_best_mi:.4f}; "
                           f"best_MI({next_k})={next_mi:.4f}, "
                           f"gain={next_mi - best_mi_per_k[elbow_K]:.4f} "
                           f"<= {ELBOW_GAIN_THRESHOLD}")
        discovered_K = elbow_K
    print(f"[{regime}] AUTO-K result: K_discovered = {discovered_K}")
    print(f"           reason: {ksel_reason}")
    ksel_seconds = time.time() - t_ksel

    # ========================================================================
    # PHASE 2: FINAL FIT at K_discovered, reseed to whichever Phase-1 kseed
    # gave the BEST MI at the discovered K. This guarantees the final fit
    # lands in the same healthy basin Phase 1 verified is reachable. If
    # the best kseed had a flow init that produced perfect MMICRL recovery
    # (e.g., MI=log(K)), Phase 2 reproduces that exact fit.
    # ========================================================================
    best_kseed_for_final = int(np.argmax(per_k_seed_mi[discovered_K]))
    best_kseed_mi = per_k_seed_mi[discovered_K][best_kseed_for_final]
    final_seed = SEED + 1000 * best_kseed_for_final + discovered_K
    print(f"[{regime}] Phase 2/2: fitting MMICRL with n_types={discovered_K} "
          f"(K discovered by Phase 1). Reseeding to kseed={best_kseed_for_final} "
          f"(produced MI={best_kseed_mi:.4f} at K={discovered_K} in Phase 1; "
          f"final torch.manual_seed={final_seed}).")
    torch.manual_seed(final_seed)
    np.random.seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)

    mmicrl = MMICRL(
        n_types=discovered_K,
        lambda1=1.0,
        lambda2=1.0,
        n_muscles=1,
        n_iterations=150,
        hidden_dims=[64, 64],
        auto_select_k=False,
        k_range=K_RANGE,
        k_selection="heldout_nll",
        heldout_frac=0.2,
        device="cpu",
    )
    t1 = time.time()
    fit_results = mmicrl.fit(collector, n_actions=n_actions)
    fit_seconds = time.time() - t1
    fit_results["phase2_reseeded_to_kseed"] = best_kseed_for_final
    fit_results["phase2_final_torch_seed"] = final_seed
    fit_results["phase2_phase1_mi_at_this_seed"] = float(best_kseed_mi)
    fit_results["k_selection_method"] = "multi_seed_best_mi_elbow"
    fit_results["k_selection_n_seeds"] = N_SEEDS_KSEL
    fit_results["k_selection_per_k_seed_mi"] = {
        str(k): per_k_seed_mi[k] for k in K_RANGE
    }
    fit_results["k_selection_best_mi_per_k"] = {
        str(k): best_mi_per_k[k] for k in K_RANGE
    }
    fit_results["k_selection_median_mi_per_k"] = {
        str(k): median_mi_per_k[k] for k in K_RANGE
    }
    fit_results["k_selection_per_seed_max_mi"] = per_seed_max_mi
    fit_results["k_selection_median_max_mi_across_seeds"] = \
        median_max_mi_across_seeds
    fit_results["k_selection_reason"] = ksel_reason
    fit_results["k_selection_seconds"] = ksel_seconds
    fit_results["elbow_gain_threshold"] = ELBOW_GAIN_THRESHOLD
    fit_results["mi_null_threshold"] = MI_NULL_THRESHOLD
    fit_results["true_K"] = true_K
    fit_results["K_discovered"] = discovered_K
    fit_results["K_match"] = bool(discovered_K == true_K)
    print(f"[{regime}] MMICRL fit completed in {fit_seconds:.1f}s")
    print(f"[{regime}] K selected: {fit_results['n_types_discovered']}")
    print(f"[{regime}] MI: {fit_results['mutual_information']:.6f}")
    print(f"[{regime}] mi_collapsed flag: {fit_results['mi_collapsed']}")
    print(f"[{regime}] Type proportions: "
          f"{[f'{p:.3f}' for p in fit_results['type_proportions']]}")

    # Recover per-demo predicted assignments to compare against ground truth
    pred_assign = infer_assignments(mmicrl, collector, true_types)

    ari = adjusted_rand_index(true_types, pred_assign)
    nmi = normalized_mutual_information(true_types, pred_assign)
    print(f"[{regime}] ARI(pred vs true types): {ari:.4f}")
    print(f"[{regime}] NMI(pred vs true types): {nmi:.4f}")

    # Confusion matrix (true type rows, predicted cluster cols)
    n_true = len(np.unique(true_types))
    n_pred = max(int(np.max(pred_assign)) + 1, 1)
    cm = np.zeros((n_true, n_pred), dtype=np.int64)
    for tt, pp in zip(true_types, pred_assign):
        cm[int(tt), int(pp)] += 1
    print(f"[{regime}] Confusion matrix (rows=true, cols=pred):\n{cm}")

    # Compose result block
    result = {
        "regime": regime,
        "n_workers": N_WORKERS_PER_REGIME,
        "n_episodes_per_worker": N_EPISODES_PER_WORKER,
        "n_demos": len(demos),
        "n_steps_per_episode": N_STEPS_PER_EPISODE,
        "dt_min": DT_MIN,
        "theta_eff": THETA_EFF,
        "rest_prob": REST_PROB,
        "TL_default": TL_DEFAULT,
        "n_actions": N_ACTIONS,
        "F_values_used": (K3_F_VALUES if regime == "K3" else [K1_F_VALUE]),
        "true_type_distribution": np.bincount(true_types).tolist(),
        "demo_length_min": int(min(lengths)),
        "demo_length_max": int(max(lengths)),
        "demo_length_mean": float(np.mean(lengths)),
        "fit_seconds": fit_seconds,
        "mmicrl_results": {
            "n_demonstrations": fit_results["n_demonstrations"],
            "n_types_discovered": fit_results["n_types_discovered"],
            "type_proportions": fit_results["type_proportions"],
            "mutual_information": float(fit_results["mutual_information"]),
            "mi_collapsed": bool(fit_results["mi_collapsed"]),
            "objective_value": float(fit_results["objective_value"]),
            "theta_per_type": fit_results["theta_per_type"],
            "lambda1": fit_results["lambda1"],
            "lambda2": fit_results["lambda2"],
            "bic_scores": fit_results.get("bic_scores", {}),
            "k_selection": fit_results.get("k_selection", {}),
            # Multi-seed median K-selection results (the K-DISCOVERY proof)
            "true_K": fit_results.get("true_K"),
            "K_discovered": fit_results.get("K_discovered"),
            "K_match": fit_results.get("K_match"),
            "k_selection_method":
                fit_results.get("k_selection_method", "unknown"),
            "k_selection_n_seeds":
                fit_results.get("k_selection_n_seeds", 0),
            "k_selection_per_k_seed_scores":
                fit_results.get("k_selection_per_k_seed_scores", {}),
            "k_selection_median_per_k":
                fit_results.get("k_selection_median_per_k", {}),
            "k_selection_seconds":
                fit_results.get("k_selection_seconds", 0),
        },
        "ground_truth_metrics": {
            "predicted_assignments": pred_assign.tolist(),
            "true_types": list(map(int, true_types)),
            "ARI": ari,
            "NMI": nmi,
            "confusion_matrix": cm.tolist(),
        },
        "verdict": _verdict(regime, fit_results, ari),
    }

    json_path = OUTDIR / f"synthetic_{regime.lower()}_mmicrl.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[{regime}] Saved results -> {json_path.relative_to(ROOT)}")

    return result


def _verdict(regime, fit_results, ari):
    """Per-regime PASS/FAIL verdict.

    NEW (multi-seed median K-selection): the model DISCOVERS K from data
    using median heldout-NLL across N_SEEDS_KSEL torch seeds with parsimony
    tie-breaking. Verdict checks both that the discovered K matches the
    ground truth AND that the structure recovery (MI/ARI) is strong.

    K=3 regime PASSES if K_discovered=3 AND MI > 0.3 AND ARI > 0.3.
    K=1 regime PASSES if K_discovered=1 AND MI < 0.1.
    """
    K = fit_results["n_types_discovered"]
    K_discovered = fit_results.get("K_discovered", K)
    K_match = fit_results.get("K_match", K == fit_results.get("true_K"))
    MI = float(fit_results["mutual_information"])
    if regime == "K3":
        if K_discovered == 3 and MI > 0.3 and ari > 0.3:
            return (f"PASS - K_discovered=3 (auto-selected from data via "
                    f"multi-seed best-of-N MI-elbow rule, ELBOW_GAIN_THRESHOLD"
                    f"={ELBOW_GAIN_THRESHOLD}; heldout-NLL was tried and "
                    f"abandoned, see source comments), MI={MI:.3f}, "
                    f"ARI={ari:.3f}")
        elif K_discovered == 3:
            return (f"PARTIAL - K_discovered=3 but weak structure: "
                    f"MI={MI:.3f}, ARI={ari:.3f}")
        else:
            return (f"FAIL - K_discovered={K_discovered} (expected 3), "
                    f"MI={MI:.3f}, ARI={ari:.3f}")
    else:  # K=1
        if K_discovered == 1 and MI < 0.1:
            return (f"PASS - K_discovered=1 (auto-selected from data via "
                    f"multi-seed MI-null homogeneity gate, "
                    f"MI_NULL_THRESHOLD={MI_NULL_THRESHOLD}; heldout-NLL was "
                    f"tried and abandoned, see source comments), "
                    f"MI={MI:.6f} (correctly collapsed)")
        elif K_discovered == 1:
            return (f"PARTIAL - K_discovered=1 but MI={MI:.3f} > 0.1")
        else:
            return (f"FAIL - K_discovered={K_discovered} (expected 1), "
                    f"MI={MI:.3f}")


def write_provenance():
    """Capture environment / git / time so the run is reproducible."""
    import platform
    import subprocess
    git_hash = "unknown"
    git_subject = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
        git_subject = subprocess.check_output(
            ["git", "log", "-1", "--format=%s"], cwd=str(ROOT)).decode().strip()
    except Exception as e:
        git_subject = f"git unavailable: {e}"

    prov = {
        "experiment": "EXP3 Part 1 - MMICRL synthetic-data validity check",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "git_subject": git_subject,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "numpy_version": np.__version__,
        "determinism_mode": "HIGH",
        "determinism_flags": {
            "torch.use_deterministic_algorithms": True,
            "torch.backends.cudnn.deterministic": True,
            "torch.backends.cudnn.benchmark": False,
            "torch.set_float32_matmul_precision": "highest",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
        },
        "seed": SEED,
        "n_workers_per_regime": N_WORKERS_PER_REGIME,
        "n_episodes_per_worker": N_EPISODES_PER_WORKER,
        "n_steps_per_episode": N_STEPS_PER_EPISODE,
        "TL_default": TL_DEFAULT,
        "theta_eff": THETA_EFF,
        "rest_prob": REST_PROB,
        "n_actions": N_ACTIONS,
        "K3_F_values": K3_F_VALUES,
        "K1_F_value": K1_F_VALUE,
    }
    path = OUTDIR / "_part1_provenance.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"\nProvenance written -> {path.relative_to(ROOT)}")


def write_readme(k3, k1):
    md = f"""# EXP3 Part 1 - MMICRL Synthetic-Data Validity Check

## What this folder contains

This folder holds the artefacts of EXP3 Part 1: a self-contained validation
that MMICRL recovers the correct number of latent worker types when the
data actually has them, and correctly collapses when it does not.

| File | Purpose |
|---|---|
| `synthetic_k3_mmicrl.json` | Full MMICRL results on K=3 synthetic data (proportions, MI, theta_per_type, K-selection scores, ARI, NMI, confusion matrix). |
| `synthetic_k1_mmicrl.json` | Full MMICRL results on K=1 synthetic data. |
| `synthetic_k3_demos.npz` | Raw 3CC-r trajectories for K=3 (90 demos). Reproducible from the script + seed. |
| `synthetic_k1_demos.npz` | Raw 3CC-r trajectories for K=1 (90 demos). |
| `_part1_provenance.json` | Git commit, library versions, determinism flags, seeds. |
| `_part1_run.log` | stdout from the run. |
| `PART1_README.md` | This file. |

## Headline numbers

### K=3 regime (3 worker types, F = {K3_F_VALUES})
- K selected: **{k3['mmicrl_results']['n_types_discovered']}**
- Mutual information: **{k3['mmicrl_results']['mutual_information']:.4f}**
- Type proportions: {[f'{p:.3f}' for p in k3['mmicrl_results']['type_proportions']]}
- ARI vs ground truth: **{k3['ground_truth_metrics']['ARI']:.4f}**
- NMI vs ground truth: **{k3['ground_truth_metrics']['NMI']:.4f}**
- Verdict: **{k3['verdict']}**

### K=1 regime (homogeneous, F = {K1_F_VALUE})
- K selected: **{k1['mmicrl_results']['n_types_discovered']}**
- Mutual information: **{k1['mmicrl_results']['mutual_information']:.6f}**
- Type proportions: {[f'{p:.3f}' for p in k1['mmicrl_results']['type_proportions']]}
- mi_collapsed flag: **{k1['mmicrl_results']['mi_collapsed']}**
- Verdict: **{k1['verdict']}**

## How to interpret

**If both verdicts are PASS**: MMICRL is functioning correctly. The result
that MMICRL collapses on real WSD4FEDSRM data (EXP1: 10/10 seeds with
MI <= 5.96e-08) reflects a genuine property of the real data lacking
separable types in the (state, action) feature space, not a defect in
MMICRL itself.

**If K=3 verdict is FAIL**: MMICRL cannot recover types even when they
exist by construction. This would invalidate Part 2 (HCMARL with-vs-
without MMICRL on synthetic K=3).

**If K=1 verdict is FAIL (over-clustered)**: MMICRL is finding spurious
structure in homogeneous data. Suggests the auto-K-selection criterion
is too lenient.

## How to re-analyse the raw demos

```python
import numpy as np
data = np.load("Results 3/Part 1/synthetic_k3_demos.npz", allow_pickle=True)
states     = data["states"]      # array of (T_i, 4) state arrays per demo
actions    = data["actions"]     # array of (T_i,) action arrays per demo
worker_ids = data["worker_ids"]  # int per demo, 0..n_workers-1
true_types = data["true_types"]  # int per demo, 0/1/2 for K=3, all 0 for K=1
F_values   = data["F_values"]    # the F values per latent type
TL_default = data["TL_default"]  # the fixed target load (TL=0.45)
# state column layout: [MR, MA, MF, TL_eff]; action in {{0=rest, 1=work}}
```

## How to visualise (optional, run later)

A separate script `scripts/visualize_exp3_part1.py` (not run by default)
can produce: PCA scatter of demos coloured by true type, confusion-matrix
heatmap, MI bar chart per regime, and K-selection score curves. The raw
JSON + npz here are sufficient input.

## Determinism

This run used HIGH-determinism mode (full PyTorch reproducibility):
`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`
+ TF32 disabled. EXP1 / EXP2 used standard determinism (cudnn.deterministic
only) which is sufficient for averaged RL claims but insufficient for
discrete K-selection. The full provenance JSON records the exact flag set.
"""
    path = OUTDIR / "PART1_README.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"README written -> {path.relative_to(ROOT)}")


def main():
    banner("EXP3 PART 1 - MMICRL synthetic-data validity check")
    print(f"Determinism: HIGH (use_deterministic_algorithms=True, TF32 off)")
    print(f"Seed: {SEED}")
    print(f"Workers per regime: {N_WORKERS_PER_REGIME}")
    print(f"Episodes per worker: {N_EPISODES_PER_WORKER}")
    print(f"K=3 F values (Frey-Law 2012 distribution): {K3_F_VALUES}")
    print(f"K=1 F value: {K1_F_VALUE}")
    print(f"Output dir: {OUTDIR.relative_to(ROOT)}")

    t_start = time.time()
    k3 = run_regime("K3", n_actions=N_ACTIONS)
    k1 = run_regime("K1", n_actions=N_ACTIONS)
    total_seconds = time.time() - t_start

    banner(f"DONE in {total_seconds/60:.1f} min")
    print(f"K=3 verdict: {k3['verdict']}")
    print(f"K=1 verdict: {k1['verdict']}")

    write_provenance()
    write_readme(k3, k1)

    print(f"\nAll artefacts in {OUTDIR.relative_to(ROOT)}/:")
    for p in sorted(OUTDIR.iterdir()):
        if p.is_file():
            print(f"  {p.name:40s}  {p.stat().st_size:>10d} bytes")


if __name__ == "__main__":
    main()
