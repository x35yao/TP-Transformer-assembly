# TP-Transformer

Task-Parameterized Transformer for trajectory learning from demonstrations.

## Setup

```bash
# Create conda environment
conda create -n tp-transformer python=3.9 -y
conda activate tp-transformer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Set PYTHONPATH
# Windows:
set PYTHONPATH=<path-to-repo>\src
# Linux/Mac:
export PYTHONPATH=<path-to-repo>/src

# Run training
python scripts/train.py
```

### Configuration

Edit `src/tp_transformer/config.py` to modify:
- `n_train_demos`: Number of training demonstrations
- `batch_size`: Training batch size
- `total_epochs`: Total training epochs
- `model_name`: Name for saving checkpoints

### Output

Checkpoints and logs are saved to `transformer/<model_name>/<seed>/`

## Project Structure

```
TP-Transformer/
├── src/tp_transformer/
│   ├── config.py          # Training configuration
│   ├── augmentation.py    # Data augmentation
│   ├── data.py            # Data loading
│   ├── dataset.py         # PyTorch dataset
│   ├── losses.py          # Loss functions
│   ├── train.py           # Training logic
│   ├── utils.py           # Utilities
│   ├── weights.py         # Trajectory weighting
│   └── transformer/       # Transformer model
├── scripts/
│   └── train.py           # Training entry point
├── data/                  # Training data (not tracked)
└── EXPERIMENTS.md         # Experiment guide
```

## Experiments

See [EXPERIMENTS.md](EXPERIMENTS.md) for experiment instructions.
