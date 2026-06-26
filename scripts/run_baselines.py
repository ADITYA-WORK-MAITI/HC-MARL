"""Headline grid launcher — Experiment A (HC-MARL vs. baselines).

Reads `config/experiment_matrix.yaml` as the single source of truth and
launches every (method, seed) pair under `headline:`. That file is canonical — adding a new method or seed is a YAML
edit, never a code edit here.

The default grid is:
    methods : hcmarl, mappo, ippo, mappo_lag    (4)
    seeds   : 0..9                              (10)
    total   : 40 runs

Each run is launched as a sequential subprocess invocation of
scripts/train.py. Logs land at logs/{method}/seed_{seed}/training_log.csv,
which is what scripts/aggregate_learning_curves.py reads from.

Usage:
    python scripts/run_baselines.py --device cuda
    python scripts/run_baselines.py --methods mappo ippo --seeds 0 1 2
    python scripts/run_baselines.py --dry-run
    python scripts/run_baselines.py --device cuda \\
        --drive-backup-dir /content/drive/MyDrive/hcmarl_backup

Exit codes:
    0 — every launched subprocess returned 0.
    1 — at least one subprocess returned non-zero (failures listed at end).
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import yaml

# Repo root — used to resolve logs/ and checkpoints/ under --fresh-logs.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_one(cmd: List[str], thread_cap: Optional[int] = None) -> int:
    """Worker body for ProcessPoolExecutor: run a single train.py subprocess
    and return its exit code. Must be top-level (picklable) for fork/spawn.

    If thread_cap is set, OMP/MKL/OPENBLAS env vars are passed to the child
    BEFORE torch is imported there, capping its BLAS thread pool. This is needed because torch defaults to one thread per vCPU (25 on L4),
    so 4 parallel seeds x 25 threads = 100 threads on 25 cores, which
    oversubscribes and regresses throughput by 25-30%. With thread_cap=6
    (or total_cores // max_par), 4 parallel x 6 = 24 threads - no contention.
    """
    env = os.environ.copy()
    if thread_cap and thread_cap > 0:
        s = str(thread_cap)
        env["OMP_NUM_THREADS"] = s
        env["MKL_NUM_THREADS"] = s
        env["OPENBLAS_NUM_THREADS"] = s
    return subprocess.run(cmd, env=env).returncode

# experiment_matrix.yaml lives next to this script's parent.
DEFAULT_MATRIX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "experiment_matrix.yaml",
)


def _load_matrix(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", default=DEFAULT_MATRIX,
                        help="Path to experiment_matrix.yaml (default: repo config/).")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Override method list. Default: every key under matrix['headline']['methods'].")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seed list. Default: matrix['headline']['seeds'].")
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu, cuda, or auto (default: auto).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without launching subprocesses.")
    parser.add_argument("--drive-backup-dir", type=str, default=None,
                        help="Mirror checkpoints+logs here every checkpoint interval "
                             "(use /content/drive/MyDrive/hcmarl_backup on Colab).")
    parser.add_argument("--budget-cap", type=float, default=0.0,
                        help="Hard kill-switch: halt each run when wall-clock spend reaches "
                             "this currency-units amount. Default 0 disables.")
    parser.add_argument("--cost-per-hour", type=float, default=49.0,
                        help="GPU instance hourly rate (currency-units per hour). Default 49.0.")
    parser.add_argument("--budget-margin", type=float, default=0.95,
                        help="Trip kill-switch at budget_cap * margin (default 0.95).")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Max concurrent train.py subprocesses. Default 1 (serial, "
                             "backward-compatible). Raise to 4-8 on big-VRAM GPUs where "
                             "per-run mem << total (e.g. L4 at ~300 MiB/run leaves headroom "
                             "for 6-8 concurrent seeds). Stacks multiplicatively with T1 "
                             "(vectorised envs) on per-run SPS.")
    parser.add_argument("--fresh-logs", action="store_true",
                        help="Before any subprocess is launched, rm -rf "
                             "logs/<method_key>/ and checkpoints/<method_key>/ "
                             "for every method in the grid. Guarantees a "
                             "clean-slate CSV — prevents the 'append new good "
                             "rows onto contaminated rows' pathology. "
                             "Required for any re-run of a grid whose logs or "
                             "checkpoints already exist.")
    parser.add_argument("--thread-cap", type=int, default=None,
                        help="Max BLAS/OMP threads per child process. Defaults to "
                             "total_vcpus // max_parallel (auto). Set to 0 to disable "
                             "capping. Prevents torch's default-per-process thread pool "
                             "(= vcpu count) from oversubscribing the host when running "
                             "multiple concurrent seeds.")
    args = parser.parse_args()

    matrix = _load_matrix(args.matrix)
    headline = matrix.get("headline") or {}
    methods_map: Dict[str, Dict] = headline.get("methods") or {}
    if not methods_map:
        print(f"ERROR: matrix file {args.matrix} has no headline.methods section",
              file=sys.stderr)
        return 1

    method_keys: List[str] = list(args.methods) if args.methods else list(methods_map.keys())
    seeds: List[int] = list(args.seeds) if args.seeds else list(headline.get("seeds") or [])
    if not seeds:
        print(f"ERROR: matrix file {args.matrix} has no headline.seeds list",
              file=sys.stderr)
        return 1

    # Validate methods against the matrix before we burn any GPU minutes.
    unknown = [m for m in method_keys if m not in methods_map]
    if unknown:
        print(f"ERROR: methods {unknown} not declared in matrix headline.methods "
              f"(known: {list(methods_map.keys())})", file=sys.stderr)
        return 1

    total = len(method_keys) * len(seeds)
    print(f"Headline grid: {len(method_keys)} methods x {len(seeds)} seeds = {total} runs")
    print(f"Methods: {method_keys}")
    print(f"Seeds:   {seeds}")
    if args.drive_backup_dir:
        print(f"Drive backup: {args.drive_backup_dir}")
    if args.dry_run:
        print("(dry-run mode — no subprocesses will be launched)\n")

    # --fresh-logs: wipe per-method logs/ and checkpoints/ BEFORE any
    # subprocess is launched. HCMARLLogger appends to a CSV whose header
    # matches the expected schema; re-running a seed without wiping
    # concatenates new good rows onto prior contaminated rows. Skipped
    # in --dry-run so it stays safe to preview commands.
    if args.fresh_logs and not args.dry_run:
        print("--fresh-logs: wiping prior per-method logs/ and checkpoints/")
        for mk in method_keys:
            for subdir in ("logs", "checkpoints"):
                target = os.path.join(_REPO_ROOT, subdir, mk)
                if os.path.isdir(target):
                    shutil.rmtree(target)
                    print(f"  removed {subdir}/{mk}/")
                else:
                    print(f"  (nothing to remove at {subdir}/{mk}/)")
        print("")

    # Build the full job list up-front so we can feed it to the parallel pool
    # (or iterate serially when --max-parallel=1).
    jobs: List[Tuple[int, str, int, List[str], str]] = []
    for i, method_key in enumerate(method_keys):
        spec = methods_map[method_key]
        config_path = spec["config"]
        method_arg = spec.get("method", method_key)

        for j, seed in enumerate(seeds):
            run_id = i * len(seeds) + j + 1
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", config_path,
                "--method", method_arg,
                "--seed", str(seed),
                "--device", args.device,
                # run_name = method_key keeps the headline contract:
                # logs/{method_key}/seed_{seed}/ — what aggregate_learning_curves.py
                # iterates over.
                "--run-name", method_key,
            ]
            if args.drive_backup_dir:
                cmd += ["--drive-backup-dir", args.drive_backup_dir]
            if args.budget_cap > 0:
                cmd += ["--budget-cap", str(args.budget_cap),
                        "--cost-per-hour", str(args.cost_per_hour),
                        "--budget-margin", str(args.budget_margin)]
            jobs.append((run_id, method_key, seed, cmd, config_path))

    failures: List[tuple] = []
    max_par = max(1, int(args.max_parallel))

    # T2b thread-cap: torch defaults to one BLAS thread per vCPU. With
    # max_par>1 that oversubscribes the host. Auto-cap at total_vcpus//max_par
    # unless the user passed --thread-cap explicitly (0 disables).
    if args.thread_cap is None:
        try:
            total_cores = multiprocessing.cpu_count()
        except Exception:
            total_cores = 0
        auto_cap = (total_cores // max_par) if (total_cores > 0 and max_par > 1) else 0
        thread_cap = auto_cap if auto_cap >= 1 else None
    else:
        thread_cap = args.thread_cap if args.thread_cap > 0 else None
    if thread_cap and max_par > 1:
        print(f"Thread cap per child: OMP/MKL/OPENBLAS_NUM_THREADS={thread_cap} "
              f"(max_parallel={max_par})")

    if args.dry_run or max_par == 1:
        # Serial path — unchanged behaviour for backward-compat + dry-run preview.
        # Serial runs don't need a thread cap (only one process competing).
        for run_id, method_key, seed, cmd, config_path in jobs:
            print(f"\n[{run_id}/{total}] {method_key} seed={seed}")
            print(f"  Config: {config_path}")
            print(f"  Command: {' '.join(cmd)}")
            if args.dry_run:
                continue
            rc = _run_one(cmd, thread_cap=None)
            if rc != 0:
                print(f"  FAILED (exit code {rc})")
                failures.append((method_key, seed, rc))
            else:
                print(f"  DONE")
    else:
        # Parallel path — T2. One ProcessPoolExecutor worker per concurrent run.
        # Each worker invokes subprocess.run(train.py), so CUDA contexts are
        # isolated per-subprocess (no shared-state issues). Stdout from the
        # children interleaves on the pool's console; per-run CSVs stay clean.
        print(f"\nLaunching {len(jobs)} runs with --max-parallel={max_par} "
              f"(concurrent train.py subprocesses)")
        with ProcessPoolExecutor(max_workers=max_par) as ex:
            fut2meta = {
                ex.submit(_run_one, cmd, thread_cap): (run_id, method_key, seed, config_path)
                for (run_id, method_key, seed, cmd, config_path) in jobs
            }
            for fut in as_completed(fut2meta):
                run_id, method_key, seed, config_path = fut2meta[fut]
                try:
                    rc = fut.result()
                except Exception as exc:
                    print(f"[{run_id}/{total}] {method_key} seed={seed} CRASHED: {exc}")
                    failures.append((method_key, seed, -1))
                    continue
                if rc != 0:
                    print(f"[{run_id}/{total}] {method_key} seed={seed} FAILED (exit {rc})")
                    failures.append((method_key, seed, rc))
                else:
                    print(f"[{run_id}/{total}] {method_key} seed={seed} DONE")

    print(f"\nAll {total} jobs {'would be ' if args.dry_run else ''}complete.")
    if failures:
        print(f"{len(failures)} failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
