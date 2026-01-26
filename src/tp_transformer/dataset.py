from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentation import augment


def fill_nans(obj_data: np.ndarray) -> np.ndarray:
    tmp = pd.DataFrame(obj_data.reshape(obj_data.shape[0], -1))
    tmp = tmp.ffill(axis=0)
    return tmp.to_numpy().reshape(obj_data.shape[0], obj_data.shape[1], obj_data.shape[2])


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        obj_data: List[np.ndarray],
        traj_data: List[np.ndarray],
        grasp_data: List[np.ndarray],
        action_tags: List[torch.Tensor],
        transform_dims: int,
        weights: List[np.ndarray],
        img_inds: List[List[int]],
        pick_inds: List[List[int]],
        release_inds: List[List[int]],
        obj_tags: Dict[str, torch.Tensor],
        max_traj_seq_len: int = 200,
        return_index: bool = False,
        train_traj_id: List[str] | None = None,
        splits: List[int] | None = None,
        transforms_all_actions: Dict | None = None,
        labels_all_actions: Dict | None = None,
        covs_all_actions: Dict | None = None,
        augment_data: bool = False,
        traj_obj_ind: int | None = None,
    ) -> None:
        self.traj_data = traj_data
        self.obj_data = obj_data
        self.grasp_data = grasp_data
        self.dims = transform_dims
        self.max_traj_seq_len = max_traj_seq_len
        self.return_index = return_index
        self.train_traj_id = train_traj_id
        self.weights = weights
        self.splits = splits
        self.transforms_all_actions = transforms_all_actions
        self.labels_all_actions = labels_all_actions
        self.covs_all_actions = covs_all_actions
        self.action_tags = action_tags
        self.augment = augment_data
        self.img_inds = img_inds
        self.pick_inds = pick_inds
        self.release_inds = release_inds
        self.traj_obj_ind = traj_obj_ind
        self.obj_tags = obj_tags
        self.obj_moving = {
            0: np.array(
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, True, True],
                    [True, True, True, True],
                ]
            ).T,
            1: np.array(
                [
                    [False, False, False, False],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, True, True],
                ]
            ).T,
            2: np.array(
                [
                    [False, False, False, False],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, True],
                    [True, True, True, True],
                ]
            ).T,
        }

    def __len__(self) -> int:
        return len(self.obj_data)

    def __getitem__(self, idx: int):
        traj_data = self.traj_data[idx].copy()
        obj_data = self.obj_data[idx].copy()
        obj_data_filled = obj_data.copy()
        traj_data_new = traj_data.copy()
        weights = self.weights[idx].copy()
        img_inds = self.img_inds[idx].copy()
        pick_inds = self.pick_inds[idx].copy()
        release_inds = self.release_inds[idx].copy()
        grasp_data = self.grasp_data[idx].copy()

        if self.augment:
            task_ind = torch.argmax(self.action_tags[idx]).item()
            transforms = self.transforms_all_actions[f"action_{task_ind}"]
            labels = self.labels_all_actions[f"action_{task_ind}"]
            covs = self.covs_all_actions[f"action_{task_ind}"]
            init_obj_pose_all = []
            for i in range(obj_data.shape[1]):
                init_obj_pose_all.append(get_init_obj_pose(obj_data[:, i, :], cam="both"))
            traj_data_new, init_obj_pose_all_transformed = augment(
                traj_data,
                init_obj_pose_all,
                transforms,
                labels,
                covs,
                self.obj_tags,
                traj_obj_ind=self.traj_obj_ind,
            )
            if self.traj_obj_ind is not None:
                for i in range(init_obj_pose_all_transformed[self.traj_obj_ind].shape[0]):
                    ind = img_inds[i]
                    init_obj_pose_all_transformed[self.traj_obj_ind][i, :7] = traj_data[ind][:7]
            for i, init_obj_pose_transformed in enumerate(init_obj_pose_all_transformed):
                obj_data_filled[0, i] = init_obj_pose_transformed[0]
                if len(init_obj_pose_transformed) == 2:
                    first_detection_ind = np.where(~np.isnan(obj_data[1:, i, :]).any(axis=1))[0][0] + 1
                    obj_data_filled[first_detection_ind, i] = init_obj_pose_transformed[1]
            obj_moving_action = self.obj_moving[task_ind]
            for i, j in zip(*np.where(obj_moving_action)):
                displacement = traj_data_new[img_inds[i], :3] - traj_data[img_inds[i], :3]
                obj_data_filled[i, j, :3] += displacement

        obj_data_filled = fill_nans(obj_data_filled)
        obj_data_filled = np.nan_to_num(obj_data_filled, nan=0.0)
        traj_data_new = np.insert(traj_data_new, 7, grasp_data, axis=1)

        traj_data = torch.tensor(traj_data_new)
        obj_data = torch.tensor(obj_data_filled)
        weights = torch.tensor(weights)
        padding_mask = torch.zeros(traj_data.shape[0])
        if traj_data.shape[0] < self.max_traj_seq_len:
            diff = self.max_traj_seq_len - traj_data.shape[0]
            pad_traj = torch.zeros([diff, traj_data.shape[1]])
            pad_traj[:, 6] = 1
            traj_data = torch.cat([traj_data, pad_traj])
            pad_weights = torch.zeros((diff, weights.shape[1]))
            weights = torch.cat([weights, pad_weights])
            padding_mask = torch.cat([padding_mask, torch.ones(pad_traj.shape[0])])
            padding_mask = padding_mask > 0
        traj_hidden = traj_data.clone()
        traj_hidden[:, : self.dims] = 0
        traj_hidden = torch.cat((traj_hidden[:, :7], traj_hidden[:, 8:]), dim=1)
        img_inds = torch.tensor(img_inds)
        pick_inds = torch.tensor(pick_inds)
        release_inds = torch.tensor(release_inds)
        if self.return_index:
            return obj_data, traj_data, traj_hidden, weights, idx, self.action_tags[idx], padding_mask, img_inds, pick_inds, release_inds
        return obj_data, traj_data, traj_hidden, weights, self.action_tags[idx], padding_mask, img_inds, pick_inds, release_inds


def get_init_obj_pose(obj_pose: np.ndarray, cam: str = "zed") -> np.ndarray | None:
    if cam == "zed":
        return obj_pose[0, :]
    if cam == "wrist":
        tmp = obj_pose[1:, :][~np.isnan(obj_pose[1:, :]).any(axis=1)]
        if len(tmp) == 0:
            return None
        return tmp[0]
    if cam == "combined":
        tmp = obj_pose[1:][~np.isnan(obj_pose[1:, :]).any(axis=1)]
        if len(tmp) == 0:
            init_obj_pose = obj_pose[0, :]
        else:
            init_obj_pose = tmp[0]
        return np.array(init_obj_pose)
    if cam == "both":
        tmp = obj_pose[1:][~np.isnan(obj_pose[1:, :]).any(axis=1)]
        if len(tmp) == 0:
            init_obj_pose = obj_pose[0, :].reshape(1, -1)
        else:
            init_obj_pose = np.concatenate([obj_pose[0, :].reshape(1, -1), tmp[0].reshape(1, -1)], axis=0)
        return np.array(init_obj_pose)
    raise ValueError("Camera type wrong. Choose from zed, wrist, combined, or both")
