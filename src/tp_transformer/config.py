"""
Training configuration for the TP-Transformer model.

All hyperparameters, paths, and experiment settings are centralized here
as a dataclass so they can be easily modified or overridden from CLI args.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    # --- Data paths ---
    path_config_file: str = "./data/task_config.yaml"  # YAML file defining task structure
    raw_dir: str = "./data/raw"  # Directory with raw demonstration recordings
    processed_dir: str = "./data/processed"  # Directory with preprocessed demo data (shared with baselines/)
    splits_file: Optional[str] = None  # Optional path to data/splits/<name>.yaml from scripts/prepare_splits.py. When set, train/valid/test demos are read from the manifest instead of being sampled per seed; this is how TP-Transformer shares splits with the baselines for a fair comparison.
    tasks: List[str] = field(default_factory=lambda: ["action_0", "action_1", "action_2"])  # Task/action names to train on
    all_objs: List[str] = field(default_factory=lambda: ["bolt", "nut", "bin", "jig"])  # Object names in the scene

    # --- Model & experiment ---
    model_name: str = "default"  # Name used for checkpoint folder: <output_root>/<model_name>/<seed>/
    output_root: str = "./transformer"  # Root dir for checkpoints + logs. Override to /shared/$USER/... on PCS compute nodes (which can't write to /home).
    seed: int = 9871  # RNG seed / manifest lookup (matches scripts/prepare_splits.py default seeds; use with --splits)
    max_len: int = 200  # Maximum trajectory sequence length (shorter padded, longer truncated)
    n_train_demos: int = 15  # Number of demonstrations used for training per task
    model_copies: int = 1  # Number of cross-validation folds
    kth_copy: int = 0  # Which fold to use (0-indexed)

    # --- Feature flags ---
    enable_transformation: bool = True  # Whether to apply TP-augmentation during training
    use_via_point: bool = False  # Whether to condition on intermediate via-points
    test_encoder_layers: bool = False  # Flag for ablation: test with/without encoder layers

    # --- Training hyperparameters ---
    batch_size: int = 8  # Mini-batch size for training
    learning_rate: float = 1e-4  # Initial learning rate for Adam optimizer (selected via lr_sweep on TP-aug seed 9871; was 1.25e-6 with the broken per-epoch optimizer reset)
    total_epochs: int = 50000  # Total number of training epochs
    print_interval: int = 100  # Print metrics every N epochs
    save_interval: int = 100  # Save periodic model checkpoint every N epochs
    min_lr: float = 1e-9  # Stop training when the LR scheduler floors LR at or below this value (also passed to ReduceLROnPlateau as its `min_lr`)
    save_optimizer: bool = False  # If True, periodic checkpoints also include optimizer/scheduler state

    # --- Loss weights ---
    pos_weight: float = 1.0  # Weight for position loss (x, y, z)
    quat_weight: float = 2.0  # Weight for orientation loss (quaternion)
    grasp_weight: float = 1000.0  # Weight for grasp detection loss
    action_weight: float = 100.0  # Weight for action classification loss

    # --- Augmentation ---
    augmentation_method: str = "tp"  # Augmentation method: "tp" (task-parameterized) or "random" (random rotation)
    aug_date: str = "2025-02-17"  # Date folder for pre-computed augmentation data (augmentation/<date>/)
    traj_obj_ind: int = 4  # Index of the "trajectory" object in the object list (used to skip augmenting self)
