# Prodigy-Optimization

This repository contains an optimized implementation of [Prodigy: Towards Unsupervised Anomaly Detection in Production HPC Systems](https://dl.acm.org/doi/10.1145/3581784.3607076).

**Original Repository**: [peaclab/Prodigy](https://github.com/peaclab/Prodigy)

## What is Prodigy?

Performance variations caused by anomalies in modern High Performance Computing (HPC) systems lead to decreased efficiency, impaired application performance, and increased operational costs. Prodigy is a variational autoencoder-based anomaly detection framework that outperforms state-of-the-art alternatives by achieving a 0.95 F1-score when detecting performance anomalies.

## Optimization Goals

This optimized version maintains **exact output compatibility** with the original implementation while providing:

1. **Improved Performance**: Parallel processing enabled for feature extraction (`n_jobs=-1` in TSFresh)
2. **Enhanced Reproducibility**: Proper random seed management (`RANDOM_SEED=42`) for deterministic results
3. **Better Code Quality**: Centralized configuration in `src/config.py`, bug fixes, and cleaner code structure
4. **Bug Fix**: Fixed undefined `verbose` variable in `reproducibility_plots.py`
5. **Vectorized Operations**: Replaced Python loops with NumPy vectorized operations in the VAE prediction

See [README_OPTIMIZATIONS.md](README_OPTIMIZATIONS.md) for a detailed list of all changes.

## Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8079388.svg)](https://doi.org/10.5281/zenodo.8079388)

The dataset used for experiments can be downloaded from Zenodo. It contains telemetry data from four HPC applications (LAMMPS, sw4, sw4Lite, ExaMiniMD) with and without the "memleak" anomaly.

## Computing Environment

Python 3.6.x is required. The requirements are tested with Python 3.6.8 and 3.6.9.

```bash
python3 -m venv prodigy_py36_venv
source prodigy_py36_venv/bin/activate
pip install --upgrade pip
pip install -r py36_deployment_reqs.txt
```

For reproducible results, set the following environment variable before running:

```bash
export PYTHONHASHSEED=42
```

## Quick Start

### 1. Download the dataset

```bash
zenodo_get 10.5281/zenodo.8079388
tar -xf eclipse_small_prod_dataset.tar
```

### 2. Run experiments

```bash
cd src
python reproducibility_experiments.py
```

Configure parameters inside `reproducibility_experiments.py`:
- `data_dir`: path to the downloaded dataset
- `output_dir`: path to store results
- `repeat_nums`: list of repeat indices (reduce for faster runs)
- `expConfig_nums`: list of experiment configurations `[0..5]`

### 3. Generate plots

```bash
python reproducibility_plots.py
```

Configure `results_dir` and `plot_output_dir` inside the script.

## File Structure

```
Prodigy-Optimization/
├── LICENSE
├── README.md
├── README_OPTIMIZATIONS.md       # Detailed optimization notes
├── py36_deployment_reqs.txt
└── src/
    ├── config.py                 # NEW: centralised configuration constants
    ├── constants.py              # Column name constants (unchanged)
    ├── data_pipeline.py          # Optimized: parallel TSFresh, config usage
    ├── vae.py                    # Optimized: seeds, vectorized prediction
    ├── anomaly_detector.py       # Optimized: config usage
    ├── reproducibility_experiments.py  # Optimized: seeds, config, random_state
    ├── reproducibility_plots.py  # Fixed: undefined verbose bug
    ├── utils.py                  # Unchanged
    └── ai4hpc_predict.py         # Unchanged
```

## Original Paper

Please cite [Prodigy: Towards Unsupervised Anomaly Detection in Production HPC Systems](https://dl.acm.org/doi/10.1145/3581784.3607076):

```bibtex
@inproceedings{10.1145/3581784.3607076,
author = {Aksar, Burak and Sencan, Efe and Schwaller, Benjamin and Aaziz, Omar and Leung, Vitus J. and Brandt, Jim and Kulis, Brian and Egele, Manuel and Coskun, Ayse K.},
title = {Prodigy: Towards Unsupervised Anomaly Detection in Production HPC Systems},
year = {2023},
publisher = {Association for Computing Machinery},
doi = {10.1145/3581784.3607076},
booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
series = {SC '23}
}
```

## License

BSD 3-Clause License (see LICENSE file)
