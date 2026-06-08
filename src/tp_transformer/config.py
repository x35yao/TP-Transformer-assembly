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
    min_lr: float = 1e-7  # Stop training when the LR scheduler floors LR at or below this value (also passed to ReduceLROnPlateau as its `min_lr`). 1e-7 ≈ 10 halvings below default LR=1e-4, so a fully-stopped run has had the scheduler reduce LR 10 times with no improvement (5000 stagnant epochs) before giving up.
    scheduler_patience_steps: int = 3000  # LR-scheduler patience expressed in gradient-update steps (not epochs). Auto-scaled to epochs as `max(1, scheduler_patience_steps // batches_per_epoch)`. With dataset_size = K*3 and batch=8 this gives K=1: 3000 epochs, K=5: 1500, K=10: 750, K=15: 500. Equalizes the per-K optimization budget so the "MSE vs K" curve is fair across K.
    save_optimizer: bool = False  # If True, periodic checkpoints also include optimizer/scheduler state

    # --- Loss weights ---
    pos_weight: float = 1.0  # Weight for position loss (x, y, z)
    quat_weight: float = 2.0  # Weight for orientation loss (quaternion)
    grasp_weight: float = 1000.0  # Weight for grasp detection loss
    action_weight: float = 100.0  # Weight for action classification loss

    # --- Augmentation ---
    augmentation_method: str = "tp"  # Augmentation method: "tp" (task-parameterized), "random" (random rotation), or "none" (no augmentation; train on raw K demos)
    aug_date: str = "2025-02-17"  # Date folder for pre-computed augmentation data (augmentation/<date>/)
    traj_obj_ind: int = 4  # Index of the "trajectory" object in the object list (used to skip augmenting self)

    # --- Test-time rotation averaging (TTA) ---
    tta_rotations: int = 0  # Number of evenly-spaced rotations to average at inference. 0 or 1 = off (single deterministic pass). Used by predict (and validation when wired).
    tta_axis: str = "z"  # Rotation axis for TTA; should match the training-time augment_random_rotation axis ("z").

    # --- Model selection ---
    # Validation metric that drives best-checkpoint selection. One of:
    #   "important_dist" : L2 error at high-weight (grasp/critical) timesteps only
    #   "pose_loss"      : pos_loss + ori_loss over the whole trajectory
    #   "total_loss"     : pos + ori + grasp + action over the whole trajectory
    selection_metric: str = "important_dist"
    # Validation metric that drives the LR scheduler step and the LR-floor
    # early-stop. Same options as selection_metric; decoupled so the scheduler
    # can use a different (e.g. convergence-sensitive) signal than the
    # checkpoint selector.
    scheduler_metric: str = "important_dist"
