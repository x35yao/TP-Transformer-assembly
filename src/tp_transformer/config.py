from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    path_config_file: str = "./data/task_config.yaml"
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed_2025-02-17"
    tasks: List[str] = field(default_factory=lambda: ["action_0", "action_1", "action_2"])
    all_objs: List[str] = field(default_factory=lambda: ["bolt", "nut", "bin", "jig"])

    model_name: str = "default"
    seed: int = 9870
    max_len: int = 200
    n_train_demos: int = 20
    model_copies: int = 1
    kth_copy: int = 0

    enable_transformation: bool = True
    use_via_point: bool = False
    test_encoder_layers: bool = False

    batch_size: int = 8
    learning_rate: float = 1.25e-6
    total_epochs: int = 50000
    print_interval: int = 100
    save_interval: int = 100

    pos_weight: float = 1.0
    quat_weight: float = 2.0
    grasp_weight: float = 1000.0
    action_weight: float = 100.0

    aug_date: str = "2025-02-17"
    traj_obj_ind: int = 4
