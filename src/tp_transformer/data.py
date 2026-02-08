"""
Data loading and preprocessing pipeline.

This module handles the full data pipeline:
1. Load task config (which tasks, which objects)
2. Load pre-computed augmentation data (variances, transforms)
3. Load raw demonstration data (trajectories, object poses, grasp states)
4. Compute per-timestep importance weights
5. Split into train/valid/test sets
6. Normalize position data (z-score on training set)
7. Build TrajectoryDataset objects for PyTorch DataLoaders
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from .augmentation import build_labels_and_covs, build_transforms
from .config import TrainConfig
from .dataset import TrajectoryDataset
from .utils import create_chunks_of_indices, create_tags, normalize_wrapper
from .weights import get_grasp_weights, get_speed_weights, get_wrist_weights


# The 7 dimensions of robot pose: position (x,y,z) + quaternion (qx,qy,qz,qw)
TASK_DIMS = ["x", "y", "z", "qx", "qy", "qz", "qw"]


def load_task_config(path_config_file: str) -> Dict:
    """Load task configuration from YAML file."""
    with open(path_config_file) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def load_augmentation(config: TrainConfig):
    """Load pre-computed augmentation data and build transform/label structures.
    
    Loads from augmentation/<aug_date>/:
    - obj_augs.pickle: Per-object flags for which augmentations to apply
    - vars.pickle: Pre-computed variances for frame selection
    
    Returns:
        (transforms_all_actions, labels_all_actions, covs_all_actions)
    """
    with open(f"./augmentation/{config.aug_date}/obj_augs.pickle", "rb") as f:
        obj_augs = pickle.load(f)
    with open(f"./augmentation/{config.aug_date}/vars.pickle", "rb") as f:
        variances = pickle.load(f)
    transforms_all_actions = build_transforms(obj_augs, config.all_objs)
    labels_all_actions, covs_all_actions = build_labels_and_covs(variances, config.tasks, config.all_objs)
    return transforms_all_actions, labels_all_actions, covs_all_actions


def find_pick_indices(lst: List[int]) -> List[int]:
    """Find indices where gripper transitions from open (0) to closed (1)."""
    return [i for i in range(1, len(lst)) if lst[i - 1] == 0 and lst[i] == 1]


def find_release_indices(lst: List[int]) -> List[int]:
    """Find indices where gripper transitions from closed (1) to open (0)."""
    return [i for i in range(1, len(lst)) if lst[i - 1] == 1 and lst[i] == 0]


def load_task_data(config: TrainConfig, obj_tags: Dict[str, torch.Tensor], task_tags: Dict[str, torch.Tensor]) -> Dict:
    """Load all demonstration data for all tasks.
    
    For each task and each demo:
    1. Load trajectory CSV (robot pose over time)
    2. Load object poses from HDF5 file (multi-camera object detections)
    3. Compute importance weights (speed + grasp + wrist camera)
    4. Pad short trajectories to median length
    
    Args:
        config: Training configuration
        obj_tags: One-hot encodings for objects
        task_tags: One-hot encodings for tasks
    
    Returns:
        Nested dict: {task: {demo: {traj_data, obj_poses, weights, ...}}}
    """
    task_data: Dict[str, Dict[str, Dict]] = {task: {} for task in config.tasks}
    for task in config.tasks:
        if task not in ["action_0", "action_1", "action_2"]:
            continue
        
        # Load action summary (contains median trajectory length and image count)
        action_summary_file = os.path.join(config.processed_dir, task, "action_summary.pickle")
        with open(action_summary_file, "rb") as f:
            action_summary = pickle.load(f)
        median_traj_len = action_summary["median_traj_len"]
        
        # Iterate over all demos for this task
        task_root, demos, _ = next(os.walk(os.path.join(config.processed_dir, task)))
        for demo in sorted(demos):
            # --- Load trajectory ---
            traj_file = os.path.join(config.processed_dir, task, demo, demo + ".csv")
            df_traj_full = pd.read_csv(traj_file, index_col=0)
            df_traj = df_traj_full[TASK_DIMS]  # Extract pose columns only
            grasp_traj = df_traj_full["grasp_detected"].to_numpy()
            pick_inds = find_pick_indices(grasp_traj)
            release_inds = find_release_indices(grasp_traj)
            
            # Append trajectory object tag to trajectory data
            traj_len = df_traj.shape[0]
            obj_buffer = obj_tags["trajectory"].repeat([traj_len, 1])
            new_traj_data = np.concatenate([df_traj, obj_buffer], axis=1)  # Shape: (T, 7 + n_obj_tags)
            
            # --- Load object poses ---
            obj_pose_all = []
            obj_file = os.path.join(config.processed_dir, task, demo, demo + "_obj_combined.h5")
            try:
                df_obj = pd.read_hdf(obj_file, index_col=0)
            except FileNotFoundError:
                continue  # Skip demos with missing object data
            
            # img_inds: timestep indices where camera images were captured
            # These define segment boundaries for the multi-step prediction
            img_inds = list(df_obj.index)
            if len(img_inds) - 1 != action_summary["median_n_images"]:
                continue  # Skip demos with unexpected number of captures
            
            # Build object pose sequence: for each object + trajectory
            for obj_ind in config.all_objs + ["trajectory"]:
                if obj_ind != "trajectory":
                    individual_ind = obj_ind + "1"
                    obj_pose = df_obj[individual_ind][TASK_DIMS].to_numpy()
                    obj_tag_repeat = np.repeat(obj_tags[obj_ind].reshape(1, -1), len(obj_pose), axis=0)
                    obj_pose = np.concatenate([obj_pose, obj_tag_repeat], axis=1)
                    obj_pose_all.append(obj_pose)
                else:
                    # "Trajectory" object: robot pose at camera capture times
                    obj_pose = df_traj[TASK_DIMS].iloc[img_inds]
                    obj_tag_repeat = np.repeat(obj_tags[obj_ind].reshape(1, -1), len(obj_pose), axis=0)
                    obj_pose = np.concatenate([obj_pose, obj_tag_repeat], axis=1)
                    obj_pose_all.append(obj_pose)
            
            # Reshape to (n_captures, n_objs, D)
            obj_pose_all = np.array(obj_pose_all)
            obj_pose_all = np.transpose(obj_pose_all, (1, 0, 2))
            
            # --- Compute importance weights ---
            # Three weight sources, take max at each timestep:
            # - Speed: slow segments get high weight (precision matters)
            # - Wrist: near wrist camera captures get high weight
            # - Grasp: near pick/release transitions get high weight
            weights_speed = get_speed_weights(df_traj_full, ub=1000, lb=10).reshape(-1, 1)
            weights_wrist = get_wrist_weights(df_traj_full, ub=1000, lb=10).reshape(-1, 1)
            weights_grasp = get_grasp_weights(df_traj_full, ub=1000, lb=10).reshape(-1, 1)
            weights = np.max(np.concatenate([weights_speed, weights_wrist, weights_grasp], axis=1), axis=1)
            # Repeat weights across all pose dimensions (+ grasp)
            weights = np.repeat(weights.reshape(-1, 1), len(TASK_DIMS) + 1, axis=1)
            
            # --- Pad short trajectories to median length ---
            if new_traj_data.shape[0] < median_traj_len:
                repeat_count = median_traj_len - new_traj_data.shape[0]
                last_entry_traj = new_traj_data[-1].reshape(1, -1)
                new_traj_data = np.vstack([new_traj_data, np.repeat(last_entry_traj, repeat_count, axis=0)])
                last_entry_weight = weights[-1].reshape(1, -1)
                weights = np.vstack([weights, np.repeat(last_entry_weight, repeat_count, axis=0)])
                last_entry_grasp = grasp_traj[-1]
                grasp_traj = np.concatenate([grasp_traj, np.repeat(last_entry_grasp, repeat_count, axis=0)])
            
            task_data[task][demo] = {
                "pick_inds": pick_inds,
                "release_inds": release_inds,
                "grasp_traj": grasp_traj,
                "obj_pose_all": obj_pose_all,
                "new_traj_data": new_traj_data,
                "weights": weights,
                "task_tag": task_tags[task],
                "img_inds": img_inds,
            }
    return task_data


def split_task_data(
    config: TrainConfig, task_data: Dict
) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List], List[int], List[int], List[int]]:
    """Split demonstrations into train, validation, and test sets.
    
    For each task:
    1. Randomly sample n_train_demos for training
    2. Split remaining demos 50/50 into validation and test
    
    Uses config.seed for reproducible splits.
    
    Returns:
        (train, valid, test, train_splits, valid_splits, test_splits)
        Splits are cumulative start indices per task (for identifying task boundaries).
    """
    random.seed(config.seed)
    train, valid, test = {}, {}, {}
    for name in ["pick_inds", "release_inds", "grasp", "objs_pose", "traj_pose", "weights", "action_tags", "img_inds", "traj_id"]:
        train[name], valid[name], test[name] = [], [], []
    train_splits, valid_splits, test_splits = [], [], []
    
    for task in config.tasks:
        # Record current sizes as split boundaries
        test_splits.append(len(test["traj_pose"]))
        train_splits.append(len(train["traj_pose"]))
        valid_splits.append(len(valid["traj_pose"]))
        
        demos = list(task_data[task].keys())
        if not demos:
            continue
        
        # Sample training demos
        train_demos_pool = random.sample(demos, min(config.n_train_demos, len(demos)))
        remaining = [demo for demo in demos if demo not in train_demos_pool]
        
        # Cross-validation support: select a subset of training demos
        selected_indices = create_chunks_of_indices(len(train_demos_pool), len(train_demos_pool), config.model_copies)[
            config.kth_copy
        ]
        train_demos = [val for i, val in enumerate(train_demos_pool) if i in selected_indices]
        
        # Split remaining into validation and test (50/50)
        split_size = int(np.ceil(len(remaining) / 2))
        valid_demos = random.sample(remaining, split_size) if remaining else []
        test_demos = [demo for demo in remaining if demo not in valid_demos]
        
        # Assign each demo to its bucket
        for demo in demos:
            if demo not in task_data[task]:
                continue
            demo_data = task_data[task][demo]
            if demo in train_demos:
                bucket = train
            elif demo in test_demos:
                bucket = test
            else:
                bucket = valid
            bucket["pick_inds"].append(demo_data["pick_inds"])
            bucket["release_inds"].append(demo_data["release_inds"])
            bucket["grasp"].append(demo_data["grasp_traj"])
            bucket["objs_pose"].append(demo_data["obj_pose_all"])
            bucket["traj_pose"].append(demo_data["new_traj_data"])
            bucket["weights"].append(demo_data["weights"])
            bucket["action_tags"].append(demo_data["task_tag"])
            bucket["img_inds"].append(demo_data["img_inds"])
            if bucket is train:
                bucket["traj_id"].append(demo)
        
        print(
            f"{task}\\n # Training Demos: {len(train_demos)}, # Test Demos: {len(test_demos)}, # Valid Demos: {len(valid_demos)}"
        )
    return train, valid, test, train_splits, valid_splits, test_splits


def build_datasets(config: TrainConfig):
    """Main entry point: load data, split, normalize, and create Dataset objects.
    
    Pipeline:
    1. Create one-hot tags for objects and tasks
    2. Load all demonstration data
    3. Split into train/valid/test
    4. Compute normalization stats from training set (mean, std of positions)
    5. Normalize all position data (x, y, z only)
    6. Load augmentation config
    7. Create TrajectoryDataset objects (training set has augmentation enabled)
    
    Returns:
        (training_data, valid_data, test_data, train_stats)
        train_stats = {"mean": ..., "std": ...} for denormalizing predictions later
    """
    obj_tags = create_tags(config.all_objs + ["trajectory"])
    task_tags = create_tags(config.tasks)
    task_data = load_task_data(config, obj_tags, task_tags)
    train, valid, test, train_splits, valid_splits, test_splits = split_task_data(config, task_data)
    
    # Compute normalization statistics from training trajectories
    contiguous_traj = np.concatenate(train["traj_pose"]) if train["traj_pose"] else np.zeros((1, 3))
    train_mean = np.mean(contiguous_traj[:, :3], axis=0)  # Mean position
    train_std = np.std(contiguous_traj[:, :3]) / 3  # Scaled std for position
    norm_func = normalize_wrapper(train_mean, train_std)
    
    # Normalize positions in all splits (only x, y, z -- orientations untouched)
    train["objs_pose"] = list(map(norm_func, train["objs_pose"]))
    train["traj_pose"] = list(map(norm_func, train["traj_pose"]))
    valid["objs_pose"] = list(map(norm_func, valid["objs_pose"]))
    valid["traj_pose"] = list(map(norm_func, valid["traj_pose"]))
    test["objs_pose"] = list(map(norm_func, test["objs_pose"]))
    test["traj_pose"] = list(map(norm_func, test["traj_pose"]))
    
    # Load augmentation transforms and frame selection labels
    transforms_all_actions, labels_all_actions, covs_all_actions = load_augmentation(config)
    
    # Training dataset: augmentation enabled
    training_data = TrajectoryDataset(
        train["objs_pose"],
        train["traj_pose"],
        train["grasp"],
        train["action_tags"],
        len(TASK_DIMS),
        train["weights"],
        train["img_inds"],
        train["pick_inds"],
        train["release_inds"],
        obj_tags,
        max_traj_seq_len=config.max_len,
        splits=train_splits,
        train_traj_id=train["traj_id"],
        transforms_all_actions=transforms_all_actions,
        labels_all_actions=labels_all_actions,
        covs_all_actions=covs_all_actions,
        augment_data=True,
        traj_obj_ind=config.traj_obj_ind,
    )
    
    # Validation dataset: no augmentation
    valid_data = TrajectoryDataset(
        valid["objs_pose"],
        valid["traj_pose"],
        valid["grasp"],
        valid["action_tags"],
        len(TASK_DIMS),
        valid["weights"],
        valid["img_inds"],
        valid["pick_inds"],
        valid["release_inds"],
        obj_tags,
        max_traj_seq_len=config.max_len,
        splits=valid_splits,
        transforms_all_actions=transforms_all_actions,
        labels_all_actions=labels_all_actions,
        covs_all_actions=covs_all_actions,
        augment_data=False,
        traj_obj_ind=config.traj_obj_ind,
    )
    
    # Test dataset: no augmentation, no augmentation config needed
    test_data = TrajectoryDataset(
        test["objs_pose"],
        test["traj_pose"],
        test["grasp"],
        test["action_tags"],
        len(TASK_DIMS),
        test["weights"],
        test["img_inds"],
        test["pick_inds"],
        test["release_inds"],
        obj_tags,
        max_traj_seq_len=config.max_len,
        return_index=False,
    )
    
    train_stats = {"mean": train_mean, "std": train_std}
    return training_data, valid_data, test_data, train_stats
