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
│   └── train.py           # Training entry point
├── data/                   # Training data
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

Compare TP-Transformer against other trajectory learning methods at different numbers of training demonstrations (range: **1 to 15**).

### Methods to Compare

- **TP-Transformer** (this repository)
- **TP-GMM** (Task-Parameterized Gaussian Mixture Model)
- **TP-ProMP** (Task-Parameterized Probabilistic Movement Primitives)
- **CNEP** (Conditional Neural Expert Processes)
- **CNMP** (Conditional Neural Movement Primitives)

### Running TP-Transformer

Modify `n_train_demos` in `src/tp_transformer/config.py`:

```python
n_train_demos: int = 5  # Change to desired number (1-15)
```

Run training:
```bash
python scripts/train.py
```

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

- Run each experiment with multiple random seeds and report mean +/- std
- Ensure all methods use the same train/test split for fair comparison
- Checkpoints are saved to `transformer/<model_name>/<seed>/`
- Training logs are saved to `transformer/<model_name>/<seed>/training_log.txt`
