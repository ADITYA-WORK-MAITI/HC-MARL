# Path G calibration data — WSD4FEDSRM

The headline experiments in this codebase do **not** require the raw dataset.
A pre-computed calibration cache is shipped at `config/pathg_profiles.json`
(60 KB, 34 worker profiles). All training, evaluation, and ablation scripts
read this cache directly when raw data is absent.

This README is for reviewers who want to regenerate the cache from raw
sensor data and verify the calibration pipeline end-to-end.

## Dataset: WSD4FEDSRM

Wearable sensor data for fatigue estimation during shoulder rotation
movements. 34 healthy adults performed dynamic shoulder internal- and
external-rotation tasks at 30–40%, 40–50%, and 50–60% of MVIC until
maximal Borg RPE (=20). The dataset includes demographics, anthropometry,
MVIC force, IMU, EMG, PPG, Borg RPE, and KSS data.

Approximate size on disk: 1.6 GB.

## Download

Zenodo record: https://zenodo.org/records/8415066
DOI:           10.5281/zenodo.8415066

Extract the archive so the directory layout is:

    <repo_root>/data/wsd4fedsrm/WSD4FEDSRM/
        Borg data/borg_data.csv
        MVIC force data/MVIC_force_data.csv
        Demographic and antropometric data/demographic.csv
        ... (EMG, IMU, PPG subfolders — not consumed by calibration)

Only three CSV files (`borg_data.csv`, `MVIC_force_data.csv`,
`demographic.csv`) are read by the calibration. The IMU/EMG/PPG raw
signals are not used by the pipeline shipped here.

## License and citation

The dataset is distributed under Creative Commons Attribution 4.0
International (CC-BY 4.0). Attribution must be given to the dataset
authors as listed in the Zenodo record. This repository ships only the
derived cache `config/pathg_profiles.json`, which contains no
personally identifying information (only opaque worker indices 0–33 and
calibrated 3CC-r parameters per muscle).

Required citation:

```bibtex
@misc{wsd4fedsrm2023,
  title     = {Wearable sensor data for fatigue estimation during
               shoulder rotation movements},
  author    = {Yasar, Merve Nur and Sica, Marco and O'Flynn, Brendan and
               Tedesco, Salvatore and Menolotto, Matteo},
  year      = {2023},
  month     = oct,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.8415066},
  url       = {https://zenodo.org/records/8415066},
  note      = {CC-BY-4.0.},
}
```

## Regenerating the cache

After extracting the dataset:

    python -c "from hcmarl.real_data_calibration import run_path_g; \
               import json; \
               r = run_path_g('data/wsd4fedsrm/WSD4FEDSRM'); \
               json.dump({'profiles': r['worker_profiles']}, \
                         open('config/pathg_profiles.json', 'w'), indent=2)"

Wall-clock cost: ~30 seconds on a single CPU core. The seeded
non-shoulder sampler (`numpy.random.RandomState(42)`) makes the output
deterministic given identical raw inputs.

## Tests

Most unit tests in `tests/test_real_data_calibration.py` use synthetic
fixtures and run without the dataset. Integration tests are gated behind
`@pytest.mark.skipif` and will skip cleanly when the dataset is absent.
`pytest tests/` is green either way.
