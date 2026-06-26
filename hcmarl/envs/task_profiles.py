"""
HC-MARL Phase 2 (#20): Task Demand Profiles
T_L,g per task j per muscle group g (Eq 34).

All %MVC values sourced from published ergonomics literature:
  - Granata & Marras (1995) J Biomech 28(11):1309-17 — trunk EMG during lifting
  - de Looze et al. (2000) Ergonomics 43(3):377-90 — shoulder loads during pushing
  - Hoozemans et al. (2004) Appl Ergon 35(3):231-37 — shoulder/grip during cart pushing
  - Snook & Ciriello (1991) Ergonomics 34(9):1197-213 — Liberty Mutual MMH tables
  - Nordander et al. (2000) Int Arch Occup Environ Health 73:507-14 — light repetitive work
  - Anton et al. (2001) Appl Ergon 32(6):549-58 — overhead work shoulder loads
  - McGill et al. (2013) Clin Biomech 28(1):1-7 — trunk stabilisation demands
"""
import numpy as np
import yaml
from typing import Dict, List, Optional
from pathlib import Path


class TaskProfileManager:
    """Manages task demand profiles: T_L,g for each task-muscle pair."""

    # Default profiles: fraction of MVC demanded per muscle
    # See module docstring for citation sources per value
    DEFAULT_PROFILES = {
        "heavy_lift":    {"shoulder": 0.45, "ankle": 0.10, "knee": 0.40, "elbow": 0.30, "trunk": 0.50, "grip": 0.55},  # Granata 1995 (trunk), Hoozemans 2004 (shoulder/grip)
        "light_sort":    {"shoulder": 0.10, "ankle": 0.05, "knee": 0.05, "elbow": 0.15, "trunk": 0.10, "grip": 0.20},  # Nordander et al. 2000
        "carry":         {"shoulder": 0.25, "ankle": 0.20, "knee": 0.25, "elbow": 0.20, "trunk": 0.30, "grip": 0.45},  # Snook & Ciriello 1991
        "overhead_reach": {"shoulder": 0.55, "ankle": 0.05, "knee": 0.10, "elbow": 0.35, "trunk": 0.15, "grip": 0.30},  # Anton et al. 2001
        "push_cart":     {"shoulder": 0.20, "ankle": 0.15, "knee": 0.20, "elbow": 0.15, "trunk": 0.25, "grip": 0.40},  # Hoozemans 2004, de Looze 2000
        "rest":          {"shoulder": 0.00, "ankle": 0.00, "knee": 0.00, "elbow": 0.00, "trunk": 0.00, "grip": 0.00},
    }

    def __init__(self, profiles: Optional[Dict] = None, config_path: Optional[str] = None):
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.profiles = yaml.safe_load(f).get("task_profiles", self.DEFAULT_PROFILES)
        elif profiles:
            self.profiles = profiles
        else:
            self.profiles = self.DEFAULT_PROFILES.copy()

        self.task_names = list(self.profiles.keys())
        self.muscle_names = list(next(iter(self.profiles.values())).keys())
        self.n_tasks = len(self.task_names)
        self.n_muscles = len(self.muscle_names)

    def get_demand(self, task_name: str, muscle: str) -> float:
        """Get T_L,g for task j and muscle g."""
        return self.profiles[task_name][muscle]

    def get_demand_vector(self, task_name: str) -> np.ndarray:
        """Get [T_L,g1, ..., T_L,gK] for task j."""
        return np.array([self.profiles[task_name][m] for m in self.muscle_names], dtype=np.float32)

    def get_demand_matrix(self) -> np.ndarray:
        """Get M x G demand matrix."""
        return np.array([[self.profiles[t][m] for m in self.muscle_names] for t in self.task_names], dtype=np.float32)

    def get_productive_tasks(self) -> List[str]:
        """Return task names excluding rest."""
        return [t for t in self.task_names if t != "rest"]

    def task_intensity(self, task_name: str) -> float:
        """Total load across all muscles (proxy for task difficulty)."""
        return sum(self.profiles[task_name].values())
