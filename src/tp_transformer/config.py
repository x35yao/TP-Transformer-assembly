"""
Training configuration for the TP-Transformer model.

All hyperparameters, paths, and experiment settings are centralized here
as a dataclass so they can be easily modified or overridden from CLI args.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    # --- Data paths ---
    path_config_file: str = "./data/task_config.yaml"  # YAML file defining task structure
    raw_dir: str = "./data/raw"  # Directory with raw demonstration recordings
    processed_dir: str = "./data/processed_2025-02-17"  # Directory with preprocessed demo data
    tasks: List[str] = field(default_factory=lambda: ["action_0", "action_1", "action_2"])  # Task/action names to train on
    all_objs: List[str] = field(default_factory=lambda: ["bolt", "nut", "bin", "jig"])  # Object names in the scene

    # --- Model & experiment ---
    model_name: str = "default"  # Name used for checkpoint folder: transformer/<model_name>/<seed>/
    seed: int = 9870  # Random seed for reproducibility (data splits, etc.)
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
    learning_rate: float = 1.25e-6  # Initial learning rate for Adam optimizer
    total_epochs: int = 50000  # Total number of training epochs
    print_interval: int = 100  # Print metrics every N epochs
    save_interval: int = 100  # Save model checkpoint every N epochs

    # --- Loss weights ---
    pos_weight: float = 1.0  # Weight for position loss (x, y, z)
    quat_weight: float = 2.0  # Weight for orientation loss (quaternion)
    grasp_weight: float = 1000.0  # Weight for grasp detection loss
    action_weight: float = 100.0  # Weight for action classification loss

    # --- Augmentation ---
    augmentation_method: str = "tp"  # Augmentation method: "tp" (task-parameterized) or "random" (random rotation)
    aug_date: str = "2025-02-17"  # Date folder for pre-computed augmentation data (augmentation/<date>/)
    traj_obj_ind: int = 4  # Index of the "trajectory" object in the object list (used to skip augmenting self)
