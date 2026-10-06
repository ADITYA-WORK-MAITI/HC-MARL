# Provenance of the reported runs

These files were written on the two virtual machines that ran the experiments. E-mail addresses were removed; nothing else was changed.

- `exp1_provenance.txt`, `exp2_provenance.txt`: snapshots taken after EXP1 (4 May 2026) and EXP2 (6 May 2026): git hash, Python and PyTorch versions, GPU, CPU, memory, determinism state, timing.
- `exp1_console.log`: console output of the 50 EXP1 runs.
- `exp2_ablations_console.log`: console output of the 50 EXP2 ablation runs.
- `exp2_anchor_seed_<i>.log`: console output of the ten EXP2 anchor runs.
- `probe_*.log`: console output of the continuous-mode probe (the smoke run, the three production seeds, and the failed or stuck attempts).

Every run prints `HIGH determinism: cudnn.deterministic=True, torch.use_deterministic_algorithms(True), matmul=highest, CUBLAS_WORKSPACE_CONFIG=:4096:8` when it starts.
