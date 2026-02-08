"""
Task-Parameterized (TP) data augmentation.

The core idea: instead of augmenting data randomly, we augment trajectories
by transforming them relative to each object's reference frame, then select
the best frame at each timestep based on pre-computed variance.

Pipeline:
1. For each object, apply Rotation/Translation transforms to the trajectory
   expressed in that object's frame (produces one "candidate" trajectory per frame)
2. Use variance-based labels to select the best frame at each timestep
   (separately for position and orientation)
3. Smooth high-uncertainty regions with Gaussian filtering

This produces augmented demonstrations that respect the task structure.
"""

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
    """Apply a random rotation around an axis to the trajectory relative to an object frame.
    
    The rotation is applied by:
    1. Centering the object pose at origin, rotating, then restoring position
    2. Finding the closest trajectory point to the object
    3. Centering trajectory at that point, rotating orientation only, then restoring
    
    Note: Only ORIENTATION of the trajectory is updated (line 45: mus_obj[:, 3:]),
    not position. Position is handled by the Translation class.
    """
    
    def __init__(self, degrees: Tuple[int, int] = (-30, 30), axis: str = "z") -> None:
        self.degrees = degrees
        self.axis = axis

    def transform(self, obj_pose: np.ndarray, mus_obj: np.ndarray, sigmas_obj: List[np.ndarray] | None = None):
        """Apply random rotation augmentation.
        
        Args:
            obj_pose: Object poses at each camera capture (n_captures, D)
            mus_obj: Trajectory mean expressed in this object's frame (T, 7)
            sigmas_obj: Optional covariance matrices (T, D, D)
        
        Returns:
            Transformed (obj_pose, mus_obj) or (obj_pose, mus_obj, sigmas_obj)
        """
        # Seed with current time for randomness
        t = 1000 * time.time()
        random.seed(int(t) % 2**32)
        degree = random.randrange(self.degrees[0], self.degrees[1])
        rotmatrix = R.from_euler(self.axis, degree, degrees=True).as_matrix()
        H = homogeneous_transform(rotmatrix, np.zeros(3))
        
        # Rotate each object pose around its own position (in-place rotation)
        for i in range(len(obj_pose)):
            pt_obj = obj_pose[i, :3].copy()
            obj_pose[i, :3] = obj_pose[i, :3] - pt_obj  # Center at origin
            obj_pose[i, :7] = lintrans(obj_pose[i, :7].reshape(1, -1), H)  # Apply rotation
            obj_pose[i, :3] = obj_pose[i, :3] + pt_obj  # Restore position
        
        # Find trajectory point closest to last object detection
        pt_traj_ind = np.argmin(get_dist_traj_to_obj(mus_obj[:, :3], obj_pose[-1, :3]))
        pt_traj = mus_obj[pt_traj_ind, :3].copy()
        
        # Rotate trajectory orientation around the closest point
        mus_obj_new = mus_obj.copy()
        mus_obj_new[:, :3] = mus_obj_new[:, :3] - pt_traj  # Center at closest point
        mus_obj_new[:, :7] = lintrans(mus_obj_new[:, :7], H)  # Apply rotation
        mus_obj_new[:, :3] = mus_obj_new[:, :3] + pt_traj  # Restore position
        
        # KEY: Only update orientation (cols 3:), not position
        mus_obj[:, 3:] = mus_obj_new[:, 3:]
        
        if sigmas_obj is None:
            return obj_pose, mus_obj
        sigmas_obj = lintrans_cov(np.array(sigmas_obj), H)
        return obj_pose, mus_obj, sigmas_obj


class Translation:
    """Apply a random translation to the trajectory relative to an object frame.
    
    Similar structure to Rotation but:
    - Only POSITION of the trajectory is updated (line 77: mus_obj[:, :3])
    - Orientation is left unchanged
    """
    
    def __init__(self, xrange: Tuple[int, int] = (-3, 3), yrange: Tuple[int, int] = (-3, 3), zrange: Tuple[float, float] = (-0.06, 0.06)) -> None:
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange

    def transform(self, obj_pose: np.ndarray, mus_obj: np.ndarray, sigmas_obj: List[np.ndarray] | None = None):
        """Apply random translation augmentation.
        
        Args:
            obj_pose: Object poses at each camera capture (n_captures, D)
            mus_obj: Trajectory mean expressed in this object's frame (T, 7)
            sigmas_obj: Optional covariance matrices (T, D, D)
        
        Returns:
            Transformed (obj_pose, mus_obj) or (obj_pose, mus_obj, sigmas_obj)
        """
        t = 1000 * time.time()
        random.seed(int(t) % 2**32)
        x_trans = random.uniform(self.xrange[0], self.xrange[1])
        y_trans = random.uniform(self.yrange[0], self.yrange[1])
        z_trans = random.uniform(self.zrange[0], self.zrange[1])
        translation = np.array([x_trans, y_trans, 0])
        H = homogeneous_transform(np.eye(3), translation)
        
        # Translate each object pose
        for i in range(len(obj_pose)):
            pt_obj = obj_pose[i, :3].copy()
            obj_pose[i, :3] = obj_pose[i, :3] - pt_obj
            obj_pose[i, :7] = lintrans(obj_pose[i, :7].reshape(1, -1), H)
            obj_pose[i, :3] = obj_pose[i, :3] + pt_obj
        
        # Translate trajectory position around closest point to object
        pt_traj_ind = np.argmin(get_dist_traj_to_obj(mus_obj[:, :3], obj_pose[-1, :3]))
        pt_traj = mus_obj[pt_traj_ind, :3].copy()
        mus_obj_new = mus_obj.copy()
        mus_obj_new[:, :3] = mus_obj_new[:, :3] - pt_traj
        mus_obj_new[:, :7] = lintrans(mus_obj_new[:, :7], H)
        mus_obj_new[:, :3] = mus_obj_new[:, :3] + pt_traj
        
        # KEY: Only update position (cols :3), not orientation
        mus_obj[:, :3] = mus_obj_new[:, :3]
        
        if sigmas_obj is None:
            return obj_pose, mus_obj
        sigmas_obj = lintrans_cov(np.array(sigmas_obj), H)
        return obj_pose, mus_obj, sigmas_obj


def build_transforms(obj_augs: Dict, all_objs: List[str]) -> Dict[str, List[List[object]]]:
    """Build augmentation transform lists for each action based on obj_augs config.
    
    obj_augs specifies per-object, per-action whether position and/or orientation
    augmentation should be applied.
    
    Args:
        obj_augs: Dict {action_name: {obj_name: {"pos": bool, "ori": bool}}}
        all_objs: List of object names
    
    Returns:
        Dict {action_name: [[transforms for obj_0], [transforms for obj_1], ...]}
    """
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
    """Build frame selection labels and covariance matrices from pre-computed variances.
    
    For each task, computes:
    - labels: Which frame to use at each timestep (separately for position and orientation)
    - covs: Per-object covariance matrices (diagonal, from variances)
    
    Frame selection uses "winner-takes-all" (WTA): at each timestep, pick the
    frame with lowest maximum variance. Position and orientation can select
    DIFFERENT frames, since they're computed independently.
    
    Args:
        variances: Dict {task: {obj: {"pos": var_array, "ori": var_array}}}
        tasks: List of task names
        all_objs: List of object names
    
    Returns:
        (labels_all_actions, covs_all_actions) dicts keyed by task name
    """
    covs_all_actions = {}
    labels_all_actions = {}
    for task in tasks:
        # Collect position and orientation variances for each frame
        var_pos, var_ori = [], []
        for obj in all_objs:
            var_pos.append(variances[task][obj]["pos"])
            var_ori.append(variances[task][obj]["ori"])
        # Also include "global" (trajectory-centric) frame
        var_pos.append(variances[task]["global"]["pos"])
        var_ori.append(variances[task]["global"]["ori"])
        
        # Compute frame selection labels independently for position and orientation
        label_pos = get_label(var_pos, "wta")  # Shape: (T, D_pos), values are frame indices
        label_ori = get_label(var_ori, "wta")  # Shape: (T, D_ori)
        labels = np.concatenate([label_pos, label_ori], axis=1)  # Combined: (T, D_pos + D_ori)
        
        # Build diagonal covariance matrices from variances
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
    """Apply TP-augmentation to a single trajectory.
    
    Steps:
    1. For each object frame (except the trajectory itself at traj_obj_ind):
       - Apply that object's transforms (Rotation/Translation) to the trajectory
       - This produces a "candidate" augmented trajectory per frame
    2. Use labels.choose(mus_all) to select the best frame at each timestep
       - Position columns use position labels, orientation columns use orientation labels
       - This means position and orientation can come from DIFFERENT frames
    3. Compute fused mean/covariance using Gaussian product
    4. Smooth position in high-uncertainty regions with Gaussian filtering
    
    Args:
        traj_data: Original trajectory (T, D) with cols [x,y,z,qx,qy,qz,qw,...]
        obj_pose_data: List of object pose arrays, one per object
        transforms: Per-object list of transforms to apply
        labels: Frame selection labels from build_labels_and_covs (T, D)
        covs: Per-object covariance matrices
        obj_tags: One-hot tags for object identification
        traj_obj_ind: Index of the "trajectory" object (skipped for augmentation)
    
    Returns:
        (augmented_trajectory, transformed_object_poses)
    """
    traj = traj_data.copy()
    obj_poses = obj_pose_data.copy()
    mus_all, sigmas_all = [], []
    n_objs = len(obj_poses)
    
    for i in range(n_objs):
        # Identify which object this is from its one-hot tag
        obj_name = get_obj(obj_tags, obj_poses[i][0, 7:12])
        mus_obj = np.array(traj[:, :7])  # Start with original trajectory
        sigmas_obj = covs[obj_name]  # This object's covariances
        
        if i != traj_obj_ind:
            # Apply transforms (Rotation/Translation) for non-trajectory objects
            obj_pose = obj_poses[i]
            transforms_obj = transforms[i]
            for trans in transforms_obj:
                obj_pose, mus_obj, sigmas_obj = trans.transform(obj_pose, mus_obj, sigmas_obj)
            mus_all.append(mus_obj)
            sigmas_all.append(sigmas_obj)
        else:
            # Trajectory object: no augmentation, keep original
            mus_all.append(traj[:, :7])
            sigmas_all.append(sigmas_obj)
    
    mus_all = np.array(mus_all)  # Shape: (n_frames, T, 7)
    sigmas_all = np.array(sigmas_all)  # Shape: (n_frames, T, D, D)
    
    # Select best frame at each timestep using pre-computed labels
    # labels.choose(mus_all) picks mus_all[labels[t,d], t, d] for each (t, d)
    # This allows position and orientation to come from different frames
    tmp = labels.choose(mus_all)
    traj[:, :7] = tmp[:, :7]
    
    # Compute fused mean and covariance (Gaussian product across frames)
    mus_mean, sigmas_mean = get_mean_cov_hats(mus_all, sigmas_all)
    
    # Smooth position in high-uncertainty regions
    # Sum the position block of the covariance to get scalar uncertainty
    sigmas_sum = np.sum(sigmas_mean[:, :3, :3].reshape(sigmas_mean.shape[0], -1), axis=1)
    high_cov_mask = sigmas_sum > 0.5 * np.max(sigmas_sum)
    # Apply Gaussian smoothing to position only
    mus_mean_filtered = gaussian_filter(mus_mean[:, :3], sigma=2, mode="nearest", axes=0)
    # Replace high-uncertainty positions with smoothed version
    traj[high_cov_mask, :3] = mus_mean_filtered[high_cov_mask]
    
    return traj, obj_poses


# ---------------------------------------------------------------------------
# Random Rotation Augmentation (alternative to TP-augmentation)
# ---------------------------------------------------------------------------

def random_rotation(x: np.ndarray, axis: str = "x") -> np.ndarray:
    """Apply a random rotation to pose data around the given axis.
    
    Picks a random rotation angle (0-360 degrees) and a random point from the
    data as the center of rotation. All positions and orientations are rotated
    together around this center point.
    
    Unlike TP-augmentation, this does NOT use variance-based frame selection --
    it applies the SAME rotation to ALL points uniformly.
    
    Args:
        x: Pose data of shape (N, D) where D >= 7 [pos(3), quat(4), ...]
        axis: Rotation axis ('x', 'y', or 'z')
    
    Returns:
        Rotated pose data of same shape
    """
    new_x = x.copy()
    degree = random.randrange(0, 360)
    idx = random.randrange(0, x.shape[0])
    rot = R.from_euler(axis, degree, degrees=True)
    H = np.zeros([4, 4])
    H[:3, :3] = rot.as_matrix()
    # Center at a random point, rotate, then restore
    rand_pt = x[idx, :3].copy()
    new_x[:, :3] = new_x[:, :3] - rand_pt
    new_x[:, :7] = lintrans(new_x[:, :7], H)
    new_x[:, :3] = new_x[:, :3] + rand_pt
    return new_x


def augment_random_rotation(
    traj_data: np.ndarray,
    obj_pose_data: List[np.ndarray],
    axis: str = "z",
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Apply random rotation augmentation to trajectory and object poses.
    
    Concatenates object poses and trajectory into a single array, applies
    a uniform random rotation to all of them together (maintaining spatial
    relationships), then splits them back.
    
    This is a simpler alternative to TP-augmentation that doesn't use
    variance-based frame selection.
    
    Args:
        traj_data: Trajectory data (T, D) with [pos(3), quat(4), ...]
        obj_pose_data: List of object pose arrays, one per object
        axis: Rotation axis ('x', 'y', or 'z')
    
    Returns:
        (augmented_trajectory, augmented_object_poses)
    """
    traj = traj_data.copy()
    obj_poses = [p.copy() for p in obj_pose_data]
    
    # Build a single array: stack all object initial poses + trajectory
    # We only rotate the first detection of each object (used as encoder input)
    obj_first_poses = np.array([op[0, :7] for op in obj_poses])  # (n_objs, 7)
    
    # Concatenate object poses and trajectory for joint rotation
    combined = np.concatenate([obj_first_poses, traj[:, :7]], axis=0)
    rotated = random_rotation(combined, axis=axis)
    
    # Split back
    n_objs = len(obj_poses)
    rotated_obj_poses = rotated[:n_objs]
    rotated_traj = rotated[n_objs:]
    
    # Update trajectory pose columns
    traj[:, :7] = rotated_traj
    
    # Update object poses (first detection only)
    for i in range(n_objs):
        obj_poses[i][0, :7] = rotated_obj_poses[i]
    
    return traj, obj_poses
