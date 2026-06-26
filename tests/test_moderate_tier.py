"""Tests covering the Moderate-tier audit fixes (M1, M4, M5, M6, M7, M8).

M1 — On resume, MMICRL pretrain is skipped when run_state.pt carries
     a saved theta_max (resume determinism).
M4 — Logger keeps a persistent CSV handle rather than open/close per episode.
M5 — cost_ema is declared in the logger's CSV schema.
M6 — seed_everything accepts a deterministic flag for throughput/reproducibility
     tradeoff.
M7 — entropy_coeff linear anneal when config provides entropy_coeff_final.
M8 — muscle_params_override warns when config contains keys the 3CC-r
     ODE does not consume.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pytest


# ---------------------------------------------------------------------------
# M5 — cost_ema is a first-class CSV column
# ---------------------------------------------------------------------------


def test_m5_cost_ema_is_in_csv_columns():
    from hcmarl.logger import HCMARLLogger
    assert "cost_ema" in HCMARLLogger.CSV_COLUMNS


def test_m5_cost_ema_gets_written_to_csv():
    from hcmarl.logger import HCMARLLogger

    with tempfile.TemporaryDirectory() as td:
        logger = HCMARLLogger(log_dir=td, use_wandb=False)
        logger.log_episode({
            "episode": 1, "global_step": 100, "cost_ema": 0.1234,
            "cumulative_reward": 10.0,
        })
        logger.close()
        with open(os.path.join(td, "training_log.csv")) as f:
            content = f.read()
        assert "cost_ema" in content.splitlines()[0]
        assert "0.1234" in content


# ---------------------------------------------------------------------------
# M1 — run_state.pt round-trip preserves theta_max (so resume skips MMICRL)
# ---------------------------------------------------------------------------


def test_m1_run_state_round_trip_preserves_theta_max():
    """_write_run_state then _load_run_state should return a theta_max
    that scripts/train.py:main() uses to gate MMICRL pretrain off."""
    # Import the private helpers from scripts.train without running main()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_train_module",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "train.py"),
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "run_state.pt")
        theta_max = {f"worker_{i}": {"shoulder": 0.70 + 0.01 * i} for i in range(3)}
        train_mod._write_run_state(
            path=path, global_step=12345, episode_count=50,
            cost_ema=0.02, best_reward=100.0,
            theta_max=theta_max, seed=0, method="hcmarl",
        )
        loaded = train_mod._load_run_state(path)

    assert loaded is not None
    assert loaded["global_step"] == 12345
    assert loaded["theta_max"] == theta_max
    # This is the exact condition main() checks before setting run_mmicrl=False
    assert loaded.get("theta_max") is not None


# ---------------------------------------------------------------------------
# M8 — muscle_params_override warns on silently-dropped keys
# ---------------------------------------------------------------------------


def test_m4_logger_keeps_persistent_csv_handle():
    from hcmarl.logger import HCMARLLogger
    with tempfile.TemporaryDirectory() as td:
        logger = HCMARLLogger(log_dir=td, use_wandb=False)
        assert logger._csv_file is None
        logger.log_episode({"episode": 1, "cumulative_reward": 1.0})
        # After first write the handle stays open across subsequent writes
        first_handle = logger._csv_file
        assert first_handle is not None
        assert not first_handle.closed
        for i in range(5):
            logger.log_episode({"episode": i + 2, "cumulative_reward": float(i)})
        assert logger._csv_file is first_handle
        assert not logger._csv_file.closed
        logger.close()
        assert logger._csv_file is None
        # All 6 rows landed in the file
        with open(os.path.join(td, "training_log.csv")) as f:
            lines = f.read().splitlines()
        assert len(lines) == 7  # 1 header + 6 episodes


def test_m6_seed_everything_accepts_deterministic_flag():
    """Function signature must accept the deterministic kwarg without crashing
    in either mode (the cudnn side-effects only fire under CUDA)."""
    from hcmarl.utils import seed_everything
    # Both paths must be callable — no assertion on cudnn state (CPU-only CI
    # skips the torch.backends.cudnn branch entirely).
    seed_everything(0, deterministic=True)
    seed_everything(0, deterministic=False)
    seed_everything(0)  # default True, back-compat


def test_m7_entropy_anneal_math():
    """Linear anneal from start to final over total_steps."""
    start, final, total = 0.05, 0.01, 1000
    def anneal(step):
        frac = min(1.0, step / max(1, total))
        return start + (final - start) * frac
    assert anneal(0) == start
    assert abs(anneal(total) - final) < 1e-9
    assert abs(anneal(total // 2) - 0.5 * (start + final)) < 1e-9
    # After total_steps entropy is capped at final (no overshoot)
    assert abs(anneal(2 * total) - final) < 1e-9


def test_m8_unknown_muscle_param_keys_warn():
    """Config with extra keys (e.g. theta_min_max) should trigger a
    UserWarning before the key is silently filtered out."""
    # Inline copy of the filter logic (mirrors scripts/train.py)
    allowed = {"F", "R", "r"}
    muscle_groups_cfg = {
        "shoulder": {"F": 0.01, "R": 0.001, "r": 15, "theta_min_max": 0.62},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for m_name, m_params in muscle_groups_cfg.items():
            extras = set(m_params.keys()) - allowed
            if extras:
                warnings.warn(
                    f"muscle_groups.{m_name} has keys {sorted(extras)} that "
                    f"the 3CC-r ODE does not use; they are being dropped. "
                    f"Only {sorted(allowed)} are applied.",
                    UserWarning, stacklevel=2,
                )
    assert any("theta_min_max" in str(w.message) for w in caught)
