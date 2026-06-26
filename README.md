# HC-MARL: Human-Centric Multi-Agent Reinforcement Learning

Companion code for the paper *Fatigue-Aware Cooperative Multi-Agent RL on a
Calibrated 3CC-r Warehouse Benchmark: An Empirical Audit of Four Modular
Components*, by Aditya Maiti, Amrit Pal Singh, Amar Arora, and Arshpreet Kaur.

The codebase implements a four-component framework for fatigue-aware
cooperative task allocation in a calibrated warehouse benchmark:

1. **3CC-r physiological fatigue model** — three-compartment ODE with a
   reperfusion factor (`hcmarl/three_cc_r.py`).
2. **Exponential Control Barrier Function (ECBF) safety filter** — dual-
   barrier CBF-QP solved with OSQP (`hcmarl/ecbf_filter.py`).
3. **Nash Social Welfare (NSWF) allocator** — Hungarian assignment with a
   divergent disagreement utility (`hcmarl/nswf_allocator.py`).
4. **Multi-Modal Inverse Constrained RL (MMICRL)** — CFDE normalising flows
   for per-worker safety thresholds (`hcmarl/mmicrl.py`).

## Quick start

```bash
python -m venv venv
source venv/bin/activate            # Linux/macOS
# venv\Scripts\activate              # Windows
pip install -r requirements.txt
pip install -e .
pytest tests/ -q
```

Expected: `576 passed, 2 skipped, 0 failed` on CPU (578 collected; 7 parametrized cases trimmed for the release) (one skip is the
constants-audit muscle-groups guard on the `dry_run_50k.yaml` config; the
other is a CUDA-only deterministic-log assertion that skips on CPU).

## Reproducing the paper

The configs are matrix-driven from `config/experiment_matrix.yaml`.

### EXP1 — headline (5 methods × 10 seeds, 2M steps each)

```bash
python scripts/run_baselines.py --matrix config/experiment_matrix.yaml
python scripts/aggregate_learning_curves.py
```

Expected wall-clock on a single L4-class GPU: ~28 hours total at 6-way
parallelism (~1.5 hr per seed-run). Total compute: ~165 GPU-hours.

### EXP2 — per-component remove-one ablation (5 rungs × 10 seeds)

```bash
python scripts/run_ablations.py --matrix config/experiment_matrix.yaml
```

### EXP3 Part A — synthetic K-recovery for MMICRL (CPU)

```bash
python scripts/run_exp3_part1.py
```

### Appendix F — continuous-mode probe (3 seeds × 50K steps)

```bash
python scripts/train.py --config config/exp2_continuous_probe_hcmarl.yaml
```

### Local validation (CPU, ~6 minutes)

```bash
python scripts/experiment_0_runner.py
```

Runs a 9-stage smoke test that exercises every framework component on a
small grid and verifies determinism.

## Reproducibility provenance

- Hardware: single L4-class GPU (16 GB VRAM) for EXP1/EXP2; CPU-only for
  EXP0/EXP3 Part A.
- Determinism: `torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True`,
  `matmul=highest`. Set in `scripts/train.py` at startup.
- Seeds: `[0, 1, ..., 9]` per `config/experiment_matrix.yaml`.
- Per-seed wall time, total steps, episode counts: see
  `artifacts/exp1/<method>/seed_<i>/summary.json` and
  `artifacts/exp2/<rung>/seed_<i>/summary.json`.

## Pre-computed paper artifacts

Lightweight CSVs and JSONs that back every paper table and figure are in
`paper_artifacts/`. The 80 MB `artifacts/` directory contains the full
per-seed `training_log.csv`, `summary.json`, and resolved `config.yaml`
for each of the 110 EXP1+EXP2 runs, plus the MMICRL pretrain JSON per
seed. Reviewers can regenerate every figure in the paper from these
files with `scripts/aggregate_learning_curves.py` +
`scripts/visualize_results_4.py` without retraining.

## Statistical methodology

All headline numbers report interquartile mean (IQM) with 95% stratified-
bootstrap confidence intervals (10,000 resamples), following Agarwal et
al. (2021). Pairwise probability-of-improvement (PoI) replaces single-
point superiority claims. Implementation in `hcmarl/aggregation.py`.

## Repository structure

```
hcmarl/                        # framework library (26 .py files)
  ├── three_cc_r.py            # 3CC-r ODE
  ├── ecbf_filter.py           # ECBF dual-barrier CBF-QP
  ├── nswf_allocator.py        # Nash Social Welfare allocator
  ├── mmicrl.py                # MMICRL (CFDE flows)
  ├── pipeline.py              # end-to-end orchestrator
  ├── real_data_calibration.py # Path G WSD4FEDSRM calibration
  ├── warehouse_env.py         # single-agent Gym env
  ├── envs/                    # PettingZoo wrappers, reward functions
  ├── agents/                  # MAPPO, MAPPO-Lag, HAPPO, Shielded-MAPPO, HCMARLAgent (+ IPPO kept for compatibility; not in headline matrix)
  └── baselines/               # baseline registry

tests/                         # 35 .py files (34 test_*.py + __init__.py); 578 collected
scripts/                       # 16 .py files (15 entry-point + analysis scripts + __init__.py)
config/                        # 17 YAML/JSON configs (5 methods + 5 ablations + matrix + …)
bib/experimental_section.bib   # paper bibliography for the experimental section
data/README.md                 # WSD4FEDSRM Zenodo download instructions
paper_artifacts/               # 14 CSV/JSON files backing paper tables/figures
artifacts/                     # ~80 MB per-seed training logs (regenerate figures w/o retraining)
```

## Dataset

The 1.6 GB WSD4FEDSRM dataset (CC-BY-4.0) is redistributed by the
original authors at the Zenodo DOI listed in `data/README.md`. The
calibrated profiles (`config/pathg_profiles.json`, 34 subjects) ship with
this repository, so the headline experiments reproduce without
redownloading the raw dataset.

## License

MIT (see `LICENSE`). The WSD4FEDSRM dataset has its own CC-BY-4.0
license; see `data/README.md` for attribution requirements.
