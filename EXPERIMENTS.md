# Experiment Guide

This document provides instructions for running experiments to evaluate the TP-Transformer model.

## Overview

Two main experiments are defined:

1. **Augmentation Comparison**: Compare the current Task-Parameterized (TP) augmentation method against simple random rotation.
2. **Baseline Methods Comparison**: Compare TP-Transformer against other trajectory learning methods at varying numbers of training demonstrations.

## Prerequisites and Setup

### Environment Setup

```bash
# Create conda environment
conda create -n tp-transformer python=3.9 -y
conda activate tp-transformer

# Install dependencies
pip install -r requirements.txt

# Set PYTHONPATH (required for imports)
# Windows:
set PYTHONPATH=<path-to-repo>\src
# Linux/Mac:
export PYTHONPATH=<path-to-repo>/src
```

### Project Structure

```
TP-Transformer/
├── src/tp_transformer/
│   ├── config.py          # Training configuration
│   ├── augmentation.py    # Augmentation methods
│   ├── train.py           # Training logic
│   └── ...
├── scripts/
│   ├── train.py           # TP-Transformer training CLI
│   └── prepare_splits.py  # Writes shared train/valid/test YAML (baseline + Transformer)
├── data/
│   ├── processed/          # Demo CSV/H5 preprocessing (committed)
│   ├── splits/             # Split manifests (*.yaml)
│   └── raw/               # Demo images (partial in git — see README / baselines doc)
├── baselines/             # CNEP, CNMP — see baselines/README.md
└── transformer/            # Saved models and logs
```

---

## Experiment 1: Augmentation Comparison

### Goal

Compare the current TP-based augmentation method against simple random rotation to evaluate the effectiveness of task-parameterized augmentation.

### Current Method: TP-Augmentation

Located in `src/tp_transformer/augmentation.py`

The current augmentation method uses:
- **Variance-based frame selection**: Uses pre-computed variances to determine which object frame to use for trajectory transformation
- **Task-Parameterized transforms**: Applies `Rotation` and `Translation` transforms relative to object frames
- **Gaussian filtering**: Smooths high-variance regions

### Alternative Method: Random Rotation

Replace the TP-augmentation with your random rotation implementation.

Add a config flag in `src/tp_transformer/config.py` to switch between methods:

```python
augmentation_method: str = "tp"  # or "random"
```

### Running the Comparison

1. Train with TP-Augmentation (default)
2. Train with Random Rotation
3. Compare metrics

---

## Experiment 2: Baseline Methods Comparison

### Goal

Compare TP-Transformer against other trajectory learning methods at different numbers of training demonstrations (range: **1 to 15**). For a fair comparison, **every method must share the same train / validation / test demo IDs** for each RNG seed.

### Methods to Compare

- **TP-Transformer** (this repository)
- **TP-GMM** (Task-Parameterized Gaussian Mixture Model) — *not ported in-tree yet*
- **TP-ProMP** (Task-Parameterized Probabilistic Movement Primitives) — *not ported in-tree yet*
- **CNEP** (Conditional Neural Expert Processes)
- **CNMP** (Conditional Neural Movement Primitives)

### Shared protocol (recommended): one YAML manifest + matching baseline pickle

1. **`scripts/prepare_splits.py`** writes `data/splits/<name>.yaml` with `train` / `valid` / `test` demo IDs per `(action, seed)`. It uses a **reserve-eval-first** sampler (test → valid → shuffle train prefix) so validation and test stays fixed when you sweep `--num-train`; see **`baselines/README.md`**.

2. **`baselines/prepare_baseline_dataset.py`** reads raw + processed data and that YAML, then writes **`baselines/data/baseline_dataset_<yaml_stem>.pickle`** (default output name mirrors the YAML file stem). It needs **torchvision** and the rest of **`requirements.txt`** (same env as Experiment 2 / TP-Transformer).

3. **CNEP / CNMP** read the pickle via **`--data`** (defaults assume the canonical `n15_v3t3` bundle).

4. **TP-Transformer** reads the **same YAML** via **`scripts/train.py --splits`**; **`--seed`** must match a key under each `action_*` entry in that file (canonical seeds **9871, 9872, 9873**). With **`TrainConfig.seed`** defaulted to **9871**, **`python scripts/train.py --splits …`** (no **`--seed`**) uses manifest seed **9871**.

### End-to-end workflow (canonical 15 train / 3 valid / 3 test)

From the repo root, with **`PYTHONPATH`** set to **`src`** (see Prerequisites above):

```bash
# 1) Split manifest (shared by all methods)
python scripts/prepare_splits.py \
  --num-train 15 --num-validation 3 --num-test 3 \
  --seeds 9871 9872 9873 \
  --out data/splits/n15_v3t3.yaml

# 2) Baseline pickle (same env as above — pip install -r requirements.txt)
python baselines/prepare_baseline_dataset.py --splits data/splits/n15_v3t3.yaml
# default output: baselines/data/baseline_dataset_n15_v3t3.pickle

# 3a) CNEP / CNMP (defaults point at baseline_dataset_n15_v3t3.pickle)
python baselines/cnep/train_cnep.py
python baselines/cnmp/train_cnmp.py

# 3b) TP-Transformer — one run per manifest seed
python scripts/train.py --splits data/splits/n15_v3t3.yaml --seed 9871
python scripts/train.py --splits data/splits/n15_v3t3.yaml --seed 9872
python scripts/train.py --splits data/splits/n15_v3t3.yaml --seed 9873
```

Inference baselines:

```bash
python baselines/cnep/predict_cnep.py
python baselines/cnmp/predict_cnmp.py
```

### Sweeping number of training demos (K)

Keep **`--num-validation`** and **`--num-test`** the same across all K (e.g. 3 / 3) so eval sets stay aligned for “MSE vs. K”. For each value of **`K`** (replace `10` below with `1`, …, `15` as needed — filenames use that digit):

```bash
python scripts/prepare_splits.py --num-train 10 \
  --num-validation 3 --num-test 3 --seeds 9871 9872 9873 \
  --out data/splits/n10_v3t3.yaml

python baselines/prepare_baseline_dataset.py --splits data/splits/n10_v3t3.yaml

python baselines/cnep/train_cnep.py --data baselines/data/baseline_dataset_n10_v3t3.pickle
python baselines/cnmp/train_cnmp.py --data baselines/data/baseline_dataset_n10_v3t3.pickle

python scripts/train.py --splits data/splits/n10_v3t3.yaml --seed 9871
# repeat for seeds 9872, 9873 as needed
```

When **`--splits`** is set, demo counts come **only from the YAML**; changing **`n_train_demos`** in **`TrainConfig`** does **not** override the manifest. Omit **`--splits`** only for legacy exploratory runs (then **`n_train_demos`** and **`seed`** in config / **`--seed`** CLI control stochastic splits).

### More Detail

See **[baselines/README.md](baselines/README.md)** for split semantics, pickle schema, **`--device cuda:0`**, and teammate handoff.

---

## Evaluation Protocol

### Metrics

- **Position Error**: Average Euclidean distance (meters)
- **Orientation Error**: Average angular error (radians)

---

## Results Template

### Experiment 1: Augmentation Comparison

| Augmentation Method | Position Error (m) | Orientation Error (rad) |
|---------------------|-------------------|------------------------|
| TP-Augmentation     |                   |                        |
| Random Rotation     |                   |                        |

### Experiment 2: Baseline Comparison

| Method         | Position Error (m) | Orientation Error (rad) |
|----------------|-------------------|------------------------|
| TP-Transformer |                   |                        |
| TP-GMM         |                   |                        |
| TP-ProMP       |                   |                        |
| CNEP           |                   |                        |
| CNMP           |                   |                        |

(Create separate tables for each demo count tested)

---

## Notes

- Run each experiment with multiple random seeds and report mean +/- std (canonical manifest seeds **9871 / 9872 / 9873**).
- For Experiment 2, **share one `data/splits/<name>.yaml` and matching `baseline_dataset_<stem>.pickle`** across TP-Transformer, CNEP, and CNMP — see **`baselines/README.md`**.
- Checkpoints are saved to `transformer/<model_name>/<seed>/`
- Training logs are saved to `transformer/<model_name>/<seed>/training_log.txt`

### `environment.yml` vs `requirements.txt`

This repository ships **`requirements.txt`** only (pip packages for a normal **venv** or **`pip install`** into conda). There is no `environment.yml` in this repo.

- **`requirements.txt`** — flat list for **pip**; you choose the Python interpreter and create the env yourself (`conda create` then `pip install -r …`, or `python -m venv`). Portable and minimal.

- **`environment.yml` (conda)** — describes a **conda** environment by name; can pin **Python version**, conda-forge/binary deps (CUDA drivers, MKL), *and* optionally a pip section. Good when you want one file to reproduce a lab machine conda stack.

Use either workflow; for this repo, **`pip install -r requirements.txt`** covers TP-Transformer, the baseline pickle builder (MobileNet extraction), CNEP, and CNMP in one stack.
