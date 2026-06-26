"""Remove-one ablation launcher — single-component attribution (EXP2).

Reads `config/experiment_matrix.yaml` as the single source of truth and
launches every (rung, seed) pair under `ablation:`. That file is canonical — adding a rung or changing a seed list
is a YAML edit, never a code edit here.

Each rung removes EXACTLY one component from the full HC-MARL stack.
The reference (rung 0 = full HCMARL) is provided by the EXP1 headline
runs at logs/hcmarl/seed_<s>/. Current ladder:

    rung name           component removed             config
    -----------------   --------------------------   ---------------------------------
    no_ecbf             ECBF safety filter            config/ablation_no_ecbf.yaml
    no_nswf             NSWF welfare allocator        config/ablation_no_nswf.yaml
    no_divergent        divergent disagreement        config/ablation_no_divergent.yaml
    no_reperfusion      r=15/r=30 reperfusion         config/ablation_no_reperfusion.yaml
    no_mmicrl           MMICRL type inference         config/ablation_no_mmicrl.yaml

5 rungs x 10 seeds = 50 runs.

Each rung is launched with `--run-name ablation_<rung>` so the logs land
at `logs/ablation_<rung>/seed_<s>/training_log.csv`. This keeps them
isolated from the headline runs at `logs/{method}/seed_{seed}/`
(in particular, ablation rung "mappo" would otherwise overwrite the
headline mappo logs at logs/mappo/seed_0/).

Usage:
    python scripts/run_ablations.py --device cuda
    python scripts/run_ablations.py --rungs plus_ecbf full_hcmarl
    python scripts/run_ablations.py --dry-run
    python scripts/run_ablations.py --device cuda \\
        --drive-backup-dir /content/drive/MyDrive/hcmarl_backup

Exit codes:
    0 — every launched subprocess returned 0.
    1 — at least one subprocess returned non-zero.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import yaml


def _run_one(cmd: List[str], thread_cap: Optional[int] = None) -> int:
    """Worker body for ProcessPoolExecutor: run a single train.py subprocess
    and return its exit code. Must be top-level (picklable) for fork/spawn.

    thread_cap: if set, OMP/MKL/OPENBLAS env vars are passed to the child
    before torch imports there. See scripts/run_baselines.py::_run_one for
    the full rationale (T2b concurrency fix).
    """
    env = os.environ.copy()
    if thread_cap and thread_cap > 0:
        s = str(thread_cap)
        env["OMP_NUM_THREADS"] = s
        env["MKL_NUM_THREADS"] = s
        env["OPENBLAS_NUM_THREADS"] = s
    return subprocess.run(cmd, env=env).returncode

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
    parser.add_argument("--rungs", nargs="+", default=None,
                        help="Override rung-name list. Default: every rung under matrix['ablation']['rungs'].")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override seed list. Default: matrix['ablation']['seeds'].")
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu, cuda, or auto (default: auto).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without launching subprocesses.")
    parser.add_argument("--drive-backup-dir", type=str, default=None,
                        help="Mirror checkpoints+logs here every checkpoint interval.")
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
                             "per-run mem << total. Stacks multiplicatively with T1 on "
                             "per-run SPS.")
    parser.add_argument("--thread-cap", type=int, default=None,
                        help="Max BLAS/OMP threads per child. Defaults to "
                             "total_vcpus // max_parallel (auto). 0 disables. Prevents "
                             "torch's vcpu-sized thread pool from oversubscribing when "
                             "running multiple concurrent seeds.")
    parser.add_argument("--fresh-logs", action="store_true",
                        help="Before launching, rm -rf logs/ablation_<rung>/ "
                             "and checkpoints/ablation_<rung>/ for every rung "
                             "in the matrix. REQUIRED for clean reruns -- "
                             "without it, HCMARLLogger appends new rows onto "
                             "stale CSVs from prior contaminated runs.")
    args = parser.parse_args()

    matrix = _load_matrix(args.matrix)
    ablation = matrix.get("ablation") or {}
    rungs_list: List[Dict] = ablation.get("rungs") or []
    if not rungs_list:
        print(f"ERROR: matrix file {args.matrix} has no ablation.rungs section",
              file=sys.stderr)
        return 1

    rungs_by_name: Dict[str, Dict] = {r["name"]: r for r in rungs_list}
    rung_names: List[str] = list(args.rungs) if args.rungs else [r["name"] for r in rungs_list]
    seeds: List[int] = list(args.seeds) if args.seeds else list(ablation.get("seeds") or [])
    if not seeds:
        print(f"ERROR: matrix file {args.matrix} has no ablation.seeds list",
              file=sys.stderr)
        return 1

    unknown = [r for r in rung_names if r not in rungs_by_name]
    if unknown:
        print(f"ERROR: rungs {unknown} not declared in matrix ablation.rungs "
              f"(known: {list(rungs_by_name.keys())})", file=sys.stderr)
        return 1

    total = len(rung_names) * len(seeds)
    print(f"Ablation ladder: {len(rung_names)} rungs x {len(seeds)} seeds = {total} runs")
    print(f"Rungs: {rung_names}")
    print(f"Seeds: {seeds}")
    if args.drive_backup_dir:
        print(f"Drive backup: {args.drive_backup_dir}")
    if args.dry_run:
        print("(dry-run mode — no subprocesses will be launched)\n")

    # --fresh-logs: wipe per-rung log + checkpoint dirs BEFORE launching any
    # subprocess. Mirrors run_baselines.py:156-166 to prevent stale-CSV
    # contamination on rerun. Mandatory after the incident where
    # appended rows from a contaminated baseline run silently corrupted
    # downstream aggregation. Safe to no-op in dry-run.
    if args.fresh_logs and not args.dry_run:
        import shutil
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rung_name in rung_names:
            run_name = f"ablation_{rung_name}"
            for sub in ("logs", "checkpoints"):
                target = os.path.join(repo_root, sub, run_name)
                if os.path.isdir(target):
                    print(f"  --fresh-logs: removed {sub}/{run_name}/")
                    shutil.rmtree(target, ignore_errors=True)

    jobs: List[Tuple[int, str, int, List[str], str]] = []
    for i, rung_name in enumerate(rung_names):
        spec = rungs_by_name[rung_name]
        config_path = spec["config"]
        method_arg = spec["method"]
        # Prefix the per-rung log dir so ablation rung "mappo" does not
        # collide with headline method "mappo" at logs/mappo/seed_0/.
        run_name = f"ablation_{rung_name}"

        for j, seed in enumerate(seeds):
            run_id = i * len(seeds) + j + 1
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", config_path,
                "--method", method_arg,
                "--seed", str(seed),
                "--device", args.device,
                "--run-name", run_name,
            ]
            if args.drive_backup_dir:
                cmd += ["--drive-backup-dir", args.drive_backup_dir]
            if args.budget_cap > 0:
                cmd += ["--budget-cap", str(args.budget_cap),
                        "--cost-per-hour", str(args.cost_per_hour),
                        "--budget-margin", str(args.budget_margin)]
            jobs.append((run_id, rung_name, seed, cmd, config_path))

    failures: List[tuple] = []
    max_par = max(1, int(args.max_parallel))

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
        for run_id, rung_name, seed, cmd, config_path in jobs:
            print(f"\n[{run_id}/{total}] {rung_name} seed={seed}")
            print(f"  Config: {config_path}")
            print(f"  Command: {' '.join(cmd)}")
            if args.dry_run:
                continue
            rc = _run_one(cmd, thread_cap=None)
            if rc != 0:
                print(f"  FAILED (exit code {rc})")
                failures.append((rung_name, seed, rc))
            else:
                print(f"  DONE")
    else:
        print(f"\nLaunching {len(jobs)} runs with --max-parallel={max_par} "
              f"(concurrent train.py subprocesses)")
        with ProcessPoolExecutor(max_workers=max_par) as ex:
            fut2meta = {
                ex.submit(_run_one, cmd, thread_cap): (run_id, rung_name, seed, config_path)
                for (run_id, rung_name, seed, cmd, config_path) in jobs
            }
            for fut in as_completed(fut2meta):
                run_id, rung_name, seed, config_path = fut2meta[fut]
                try:
                    rc = fut.result()
                except Exception as exc:
                    print(f"[{run_id}/{total}] {rung_name} seed={seed} CRASHED: {exc}")
                    failures.append((rung_name, seed, -1))
                    continue
                if rc != 0:
                    print(f"[{run_id}/{total}] {rung_name} seed={seed} FAILED (exit {rc})")
                    failures.append((rung_name, seed, rc))
                else:
                    print(f"[{run_id}/{total}] {rung_name} seed={seed} DONE")

    print(f"\nAll {total} jobs {'would be ' if args.dry_run else ''}complete.")
    if failures:
        print(f"{len(failures)} failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
