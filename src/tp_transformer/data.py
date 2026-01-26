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


TASK_DIMS = ["x", "y", "z", "qx", "qy", "qz", "qw"]


def load_task_config(path_config_file: str) -> Dict:
    with open(path_config_file) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def load_augmentation(config: TrainConfig):
    with open(f"./augmentation/{config.aug_date}/obj_augs.pickle", "rb") as f:
        obj_augs = pickle.load(f)
    with open(f"./augmentation/{config.aug_date}/vars.pickle", "rb") as f:
        variances = pickle.load(f)
    transforms_all_actions = build_transforms(obj_augs, config.all_objs)
    labels_all_actions, covs_all_actions = build_labels_and_covs(variances, config.tasks, config.all_objs)
    return transforms_all_actions, labels_all_actions, covs_all_actions


def find_pick_indices(lst: List[int]) -> List[int]:
    return [i for i in range(1, len(lst)) if lst[i - 1] == 0 and lst[i] == 1]


def find_release_indices(lst: List[int]) -> List[int]:
    return [i for i in range(1, len(lst)) if lst[i - 1] == 1 and lst[i] == 0]


def load_task_data(config: TrainConfig, obj_tags: Dict[str, torch.Tensor], task_tags: Dict[str, torch.Tensor]) -> Dict:
    task_data: Dict[str, Dict[str, Dict]] = {task: {} for task in config.tasks}
    for task in config.tasks:
        if task not in ["action_0", "action_1", "action_2"]:
            continue
        action_summary_file = os.path.join(config.processed_dir, task, "action_summary.pickle")
        with open(action_summary_file, "rb") as f:
            action_summary = pickle.load(f)
        median_traj_len = action_summary["median_traj_len"]
        task_root, demos, _ = next(os.walk(os.path.join(config.processed_dir, task)))
        for demo in sorted(demos):
            traj_file = os.path.join(config.processed_dir, task, demo, demo + ".csv")
            df_traj_full = pd.read_csv(traj_file, index_col=0)
            df_traj = df_traj_full[TASK_DIMS]
            grasp_traj = df_traj_full["grasp_detected"].to_numpy()
            pick_inds = find_pick_indices(grasp_traj)
            release_inds = find_release_indices(grasp_traj)
            traj_len = df_traj.shape[0]
            obj_buffer = obj_tags["trajectory"].repeat([traj_len, 1])
            new_traj_data = np.concatenate([df_traj, obj_buffer], axis=1)
            obj_pose_all = []
            obj_file = os.path.join(config.processed_dir, task, demo, demo + "_obj_combined.h5")
            try:
                df_obj = pd.read_hdf(obj_file, index_col=0)
            except FileNotFoundError:
                continue
            img_inds = list(df_obj.index)
            if len(img_inds) - 1 != action_summary["median_n_images"]:
                continue
            for obj_ind in config.all_objs + ["trajectory"]:
                if obj_ind != "trajectory":
                    individual_ind = obj_ind + "1"
                    obj_pose = df_obj[individual_ind][TASK_DIMS].to_numpy()
                    obj_tag_repeat = np.repeat(obj_tags[obj_ind].reshape(1, -1), len(obj_pose), axis=0)
                    obj_pose = np.concatenate([obj_pose, obj_tag_repeat], axis=1)
                    obj_pose_all.append(obj_pose)
                else:
                    obj_pose = df_traj[TASK_DIMS].iloc[img_inds]
                    obj_tag_repeat = np.repeat(obj_tags[obj_ind].reshape(1, -1), len(obj_pose), axis=0)
                    obj_pose = np.concatenate([obj_pose, obj_tag_repeat], axis=1)
                    obj_pose_all.append(obj_pose)
            obj_pose_all = np.array(obj_pose_all)
            obj_pose_all = np.transpose(obj_pose_all, (1, 0, 2))
            weights_speed = get_speed_weights(df_traj_full, ub=1000, lb=10).reshape(-1, 1)
            weights_wrist = get_wrist_weights(df_traj_full, ub=1000, lb=10).reshape(-1, 1)
            weights_grasp = get_grasp_weights(df_traj_full, ub=1000, lb=10).reshape(-1, 1)
            weights = np.max(np.concatenate([weights_speed, weights_wrist, weights_grasp], axis=1), axis=1)
            weights = np.repeat(weights.reshape(-1, 1), len(TASK_DIMS) + 1, axis=1)
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
    random.seed(config.seed)
    train, valid, test = {}, {}, {}
    for name in ["pick_inds", "release_inds", "grasp", "objs_pose", "traj_pose", "weights", "action_tags", "img_inds", "traj_id"]:
        train[name], valid[name], test[name] = [], [], []
    train_splits, valid_splits, test_splits = [], [], []
    for task in config.tasks:
        test_splits.append(len(test["traj_pose"]))
        train_splits.append(len(train["traj_pose"]))
        valid_splits.append(len(valid["traj_pose"]))
        demos = list(task_data[task].keys())
        if not demos:
            continue
        train_demos_pool = random.sample(demos, min(config.n_train_demos, len(demos)))
        remaining = [demo for demo in demos if demo not in train_demos_pool]
        selected_indices = create_chunks_of_indices(len(train_demos_pool), len(train_demos_pool), config.model_copies)[
            config.kth_copy
        ]
        train_demos = [val for i, val in enumerate(train_demos_pool) if i in selected_indices]
        split_size = int(np.ceil(len(remaining) / 2))
        valid_demos = random.sample(remaining, split_size) if remaining else []
        test_demos = [demo for demo in remaining if demo not in valid_demos]
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
    obj_tags = create_tags(config.all_objs + ["trajectory"])
    task_tags = create_tags(config.tasks)
    task_data = load_task_data(config, obj_tags, task_tags)
    train, valid, test, train_splits, valid_splits, test_splits = split_task_data(config, task_data)
    contiguous_traj = np.concatenate(train["traj_pose"]) if train["traj_pose"] else np.zeros((1, 3))
    train_mean = np.mean(contiguous_traj[:, :3], axis=0)
    train_std = np.std(contiguous_traj[:, :3]) / 3
    norm_func = normalize_wrapper(train_mean, train_std)
    train["objs_pose"] = list(map(norm_func, train["objs_pose"]))
    train["traj_pose"] = list(map(norm_func, train["traj_pose"]))
    valid["objs_pose"] = list(map(norm_func, valid["objs_pose"]))
    valid["traj_pose"] = list(map(norm_func, valid["traj_pose"]))
    test["objs_pose"] = list(map(norm_func, test["objs_pose"]))
    test["traj_pose"] = list(map(norm_func, test["traj_pose"]))
    transforms_all_actions, labels_all_actions, covs_all_actions = load_augmentation(config)
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
