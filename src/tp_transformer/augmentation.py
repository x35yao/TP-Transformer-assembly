from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R

from .utils import (
    get_dist_traj_to_obj,
    get_label,
    get_mean_cov_hats,
    get_obj,
    homogeneous_transform,
    lintrans,
    lintrans_cov,
)


class Rotation:
    def __init__(self, degrees: Tuple[int, int] = (-30, 30), axis: str = "z") -> None:
        self.degrees = degrees
        self.axis = axis

    def transform(self, obj_pose: np.ndarray, mus_obj: np.ndarray, sigmas_obj: List[np.ndarray] | None = None):
        t = 1000 * time.time()
        random.seed(int(t) % 2**32)
        degree = random.randrange(self.degrees[0], self.degrees[1])
        rotmatrix = R.from_euler(self.axis, degree, degrees=True).as_matrix()
        H = homogeneous_transform(rotmatrix, np.zeros(3))
        for i in range(len(obj_pose)):
            pt_obj = obj_pose[i, :3].copy()
            obj_pose[i, :3] = obj_pose[i, :3] - pt_obj
            obj_pose[i, :7] = lintrans(obj_pose[i, :7].reshape(1, -1), H)
            obj_pose[i, :3] = obj_pose[i, :3] + pt_obj
        pt_traj_ind = np.argmin(get_dist_traj_to_obj(mus_obj[:, :3], obj_pose[-1, :3]))
        pt_traj = mus_obj[pt_traj_ind, :3].copy()
        mus_obj_new = mus_obj.copy()
        mus_obj_new[:, :3] = mus_obj_new[:, :3] - pt_traj
        mus_obj_new[:, :7] = lintrans(mus_obj_new[:, :7], H)
        mus_obj_new[:, :3] = mus_obj_new[:, :3] + pt_traj
        mus_obj[:, 3:] = mus_obj_new[:, 3:]
        if sigmas_obj is None:
            return obj_pose, mus_obj
        sigmas_obj = lintrans_cov(np.array(sigmas_obj), H)
        return obj_pose, mus_obj, sigmas_obj


class Translation:
    def __init__(self, xrange: Tuple[int, int] = (-3, 3), yrange: Tuple[int, int] = (-3, 3), zrange: Tuple[float, float] = (-0.06, 0.06)) -> None:
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange

    def transform(self, obj_pose: np.ndarray, mus_obj: np.ndarray, sigmas_obj: List[np.ndarray] | None = None):
        t = 1000 * time.time()
        random.seed(int(t) % 2**32)
        x_trans = random.uniform(self.xrange[0], self.xrange[1])
        y_trans = random.uniform(self.yrange[0], self.yrange[1])
        z_trans = random.uniform(self.zrange[0], self.zrange[1])
        translation = np.array([x_trans, y_trans, 0])
        H = homogeneous_transform(np.eye(3), translation)
        for i in range(len(obj_pose)):
            pt_obj = obj_pose[i, :3].copy()
            obj_pose[i, :3] = obj_pose[i, :3] - pt_obj
            obj_pose[i, :7] = lintrans(obj_pose[i, :7].reshape(1, -1), H)
            obj_pose[i, :3] = obj_pose[i, :3] + pt_obj
        pt_traj_ind = np.argmin(get_dist_traj_to_obj(mus_obj[:, :3], obj_pose[-1, :3]))
        pt_traj = mus_obj[pt_traj_ind, :3].copy()
        mus_obj_new = mus_obj.copy()
        mus_obj_new[:, :3] = mus_obj_new[:, :3] - pt_traj
        mus_obj_new[:, :7] = lintrans(mus_obj_new[:, :7], H)
        mus_obj_new[:, :3] = mus_obj_new[:, :3] + pt_traj
        mus_obj[:, :3] = mus_obj_new[:, :3]
        if sigmas_obj is None:
            return obj_pose, mus_obj
        sigmas_obj = lintrans_cov(np.array(sigmas_obj), H)
        return obj_pose, mus_obj, sigmas_obj


def build_transforms(obj_augs: Dict, all_objs: List[str]) -> Dict[str, List[List[object]]]:
    transforms_all_actions = {}
    for action in obj_augs.keys():
        objs_aug = []
        for obj in all_objs:
            obj_aug = []
            if obj_augs[action][obj]["pos"]:
                obj_aug.append(Translation())
            if obj_augs[action][obj]["ori"]:
                obj_aug.append(Rotation())
            objs_aug.append(obj_aug)
        transforms_all_actions[action] = objs_aug
    return transforms_all_actions


def build_labels_and_covs(variances: Dict, tasks: List[str], all_objs: List[str]) -> Tuple[Dict, Dict]:
    covs_all_actions = {}
    labels_all_actions = {}
    for task in tasks:
        var_pos, var_ori = [], []
        for obj in all_objs:
            var_pos.append(variances[task][obj]["pos"])
            var_ori.append(variances[task][obj]["ori"])
        var_pos.append(variances[task]["global"]["pos"])
        var_ori.append(variances[task]["global"]["ori"])
        label_pos = get_label(var_pos, "wta")
        label_ori = get_label(var_ori, "wta")
        labels = np.concatenate([label_pos, label_ori], axis=1)
        var = np.concatenate([var_pos, var_ori], axis=2)
        covs = {}
        for i in range(len(var_pos)):
            cov_obj = []
            for j in range(len(var[i])):
                cov_obj.append(np.diag(var[i][j]))
            tmp = all_objs + ["trajectory"]
            obj = tmp[i]
            covs[obj] = np.array(cov_obj)
        covs_all_actions[task] = covs
        labels_all_actions[task] = labels
    return labels_all_actions, covs_all_actions


def augment(
    traj_data: np.ndarray,
    obj_pose_data: List[np.ndarray],
    transforms: List[List[object]],
    labels: np.ndarray,
    covs: Dict,
    obj_tags: Dict[str, torch.Tensor],
    traj_obj_ind: int | None = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    traj = traj_data.copy()
    obj_poses = obj_pose_data.copy()
    mus_all, sigmas_all = [], []
    n_objs = len(obj_poses)
    for i in range(n_objs):
        obj_name = get_obj(obj_tags, obj_poses[i][0, 7:12])
        mus_obj = np.array(traj[:, :7])
        sigmas_obj = covs[obj_name]
        if i != traj_obj_ind:
            obj_pose = obj_poses[i]
            transforms_obj = transforms[i]
            for trans in transforms_obj:
                obj_pose, mus_obj, sigmas_obj = trans.transform(obj_pose, mus_obj, sigmas_obj)
            mus_all.append(mus_obj)
            sigmas_all.append(sigmas_obj)
        else:
            mus_all.append(traj[:, :7])
            sigmas_all.append(sigmas_obj)
    mus_all = np.array(mus_all)
    sigmas_all = np.array(sigmas_all)
    tmp = labels.choose(mus_all)
    traj[:, :7] = tmp[:, :7]
    mus_mean, sigmas_mean = get_mean_cov_hats(mus_all, sigmas_all)
    sigmas_sum = np.sum(sigmas_mean[:, :3, :3].reshape(sigmas_mean.shape[0], -1), axis=1)
    high_cov_mask = sigmas_sum > 0.5 * np.max(sigmas_sum)
    mus_mean_filtered = gaussian_filter(mus_mean[:, :3], sigma=2, mode="nearest", axes=0)
    traj[high_cov_mask, :3] = mus_mean_filtered[high_cov_mask]
    return traj, obj_poses
