"""
HC-MARL Phase 2 (#27): W&B Integration Logger
Logs 9 metrics per step: violation_rate, cumulative_cost, safety_rate,
tasks_completed, cumulative_reward, jain_fairness, peak_fatigue,
forced_rest_rate, constraint_recovery_time.
"""
import numpy as np
from typing import Dict, List, Optional, Any
import csv
import os
from collections import defaultdict


class HCMARLLogger:
    """Unified logger supporting W&B, CSV, and console output."""

    METRIC_NAMES = [
        "violation_rate", "cumulative_cost", "safety_rate",
        "tasks_completed", "cumulative_reward", "jain_fairness",
        "peak_fatigue", "forced_rest_rate", "constraint_recovery_time",
    ]

    # S-36: Fixed CSV column set — all 9 declared metrics + known extras.
    # New columns from agent types (e.g. "lambda" from Lagrangian) are included.
    # extrasaction="ignore" drops any undeclared fields safely.
    CSV_COLUMNS = sorted([
        "episode", "global_step", "wall_time",
        "violation_rate", "cumulative_cost", "safety_rate",
        "tasks_completed", "cumulative_reward", "jain_fairness",
        "peak_fatigue", "forced_rest_rate", "constraint_recovery_time",
        "safety_autonomy_index", "ecbf_interventions",
        "lambda", "cost_ema", "actor_loss", "critic_loss", "cost_critic_loss",
        "policy_loss", "value_loss", "entropy",
        # D4: per-agent task-selection entropy + lazy-agent kill-switch flag.
        # `per_agent_entropy_min` is what the kill-switch watches — if any one
        # agent collapses to a single-task policy, the MEAN can stay healthy
        # while the MIN drops below threshold.
        "per_agent_entropy_mean", "per_agent_entropy_min", "lazy_agent_flag",
        # Appendix F continuous-mode probe diagnostics.
        # All three default to empty/0 in discrete-mode runs and are populated
        # only by the continuous-mode probe (Appendix F).
        # gaussian_log_std_mean: mean log_std of the Gaussian actor over the
        #   episode (continuous mode only; clamped to [-5.0, 2.0] in
        #   GaussianActorNetwork). Healthy exploration: roughly [-2.0, 0.0].
        # nan_inf_events: per-episode count of NaN/Inf detections in actions.
        #   Strictly 0 expected; non-zero halts the run with a traceback.
        # allocator_calls: per-episode count of NSWF allocator invocations.
        #   Deterministic from K=30 + max_steps; included to evidence the
        #   math-doc-faithful coupling fires in continuous mode (vs decoupled
        #   in discrete; see Section 4 of paper for detail).
        "gaussian_log_std_mean", "nan_inf_events", "allocator_calls",
    ])

    def __init__(self, log_dir="logs", use_wandb=False, wandb_project="hcmarl",
                 wandb_entity=None, run_name=None, config=None):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.step_count = 0
        self.episode_count = 0
        self.history = defaultdict(list)
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self._csv_columns = list(self.CSV_COLUMNS)

        # S-37: detect existing CSV and resume (append) instead of overwriting
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            # Validate existing header matches expected columns
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                try:
                    existing_header = next(reader)
                except StopIteration:
                    existing_header = []
            if existing_header == self._csv_columns:
                self._csv_initialized = True
            else:
                # Header mismatch — existing file has different schema.
                # Rename old file to avoid data loss, start fresh.
                backup = self.csv_path + ".bak"
                os.replace(self.csv_path, backup)
                self._csv_initialized = False
        else:
            self._csv_initialized = False

        # M4: keep the CSV file handle open for the lifetime of the logger.
        # Writing one episode per open/close was ~50K syscalls over 5M steps.
        # We still flush() after every row so a Colab crash never loses more
        # than the most recent episode.
        self._csv_file = None
        self._csv_writer = None

        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, entity=wandb_entity,
                           name=run_name, config=config)
                self.wandb = wandb
            except ImportError:
                print("wandb not installed, falling back to CSV-only logging")
                self.use_wandb = False

    def _ensure_csv_writer(self):
        if self._csv_writer is not None:
            return
        mode = "a" if self._csv_initialized else "w"
        self._csv_file = open(self.csv_path, mode, newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self._csv_columns,
            extrasaction="ignore", restval="",
        )
        if not self._csv_initialized:
            self._csv_writer.writeheader()
            self._csv_initialized = True

    def log_step(self, metrics: Dict[str, float]):
        self.step_count += 1
        for k, v in metrics.items():
            self.history[k].append(v)
        if self.use_wandb:
            self.wandb.log(metrics, step=self.step_count)

    def log_episode(self, metrics: Dict[str, float]):
        self.episode_count += 1
        metrics["episode"] = self.episode_count
        self._ensure_csv_writer()
        self._csv_writer.writerow({
            k: f"{v:.6f}" if isinstance(v, float) else v
            for k, v in metrics.items() if k in self._csv_columns
        })
        self._csv_file.flush()
        if self.use_wandb:
            self.wandb.log(metrics, step=self.episode_count)

    def compute_episode_metrics(self, episode_data: Dict) -> Dict[str, float]:
        """Compute all 9 HC-MARL metrics from raw episode data."""
        n_steps = episode_data.get("n_steps", 1)
        n_workers = episode_data.get("n_workers", 1)
        n_muscles = episode_data.get("n_muscles", 6)
        total_slots = n_steps * n_workers * n_muscles

        metrics = {}
        metrics["violation_rate"] = episode_data.get("total_violations", 0) / max(1, total_slots)
        metrics["cumulative_cost"] = float(episode_data.get("total_violations", 0))
        metrics["safety_rate"] = episode_data.get("safe_steps", 0) / max(1, n_steps)
        metrics["tasks_completed"] = float(episode_data.get("tasks_completed", 0))
        metrics["cumulative_reward"] = float(episode_data.get("total_reward", 0.0))

        tasks_per_worker = episode_data.get("tasks_per_worker", np.ones(n_workers))
        tpw = np.array(tasks_per_worker)
        n = len(tpw)
        metrics["jain_fairness"] = float((tpw.sum()**2) / (n * (tpw**2).sum() + 1e-8)) if tpw.sum() > 0 else 1.0
        metrics["peak_fatigue"] = float(episode_data.get("peak_fatigue", 0.0))
        metrics["forced_rest_rate"] = episode_data.get("forced_rests", 0) / max(1, n_steps * n_workers)
        recovery = episode_data.get("recovery_times", [])
        metrics["constraint_recovery_time"] = float(np.mean(recovery)) if recovery else 0.0
        return metrics

    def close(self):
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            finally:
                self._csv_file = None
                self._csv_writer = None
        if self.use_wandb:
            self.wandb.finish()
