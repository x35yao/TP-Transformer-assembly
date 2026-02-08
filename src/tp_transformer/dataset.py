"""
PyTorch Dataset for trajectory data.

TrajectoryDataset handles:
- Loading and padding trajectories to a fixed max length
- Applying TP-augmentation on-the-fly during training
- Constructing the encoder input (object sequence) and decoder input (hidden trajectory)
- Tracking which objects are moving during each action for displacement correction
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentation import augment, augment_random_rotation


def fill_nans(obj_data: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values in object pose data along the time axis.
    
    Objects may not be detected at every timestep (especially wrist camera detections).
    This fills gaps by propagating the last known pose forward.
    
    Args:
        obj_data: Object data of shape (T, n_objs, D) possibly containing NaNs
    
    Returns:
        Same shape array with NaNs forward-filled
    """
    tmp = pd.DataFrame(obj_data.reshape(obj_data.shape[0], -1))
    tmp = tmp.ffill(axis=0)
    return tmp.to_numpy().reshape(obj_data.shape[0], obj_data.shape[1], obj_data.shape[2])


class TrajectoryDataset(Dataset):
    """PyTorch Dataset that yields (obj_seq, traj_seq, traj_hidden, weights, ...) tuples.
    
    Each sample represents one demonstration and contains:
    - obj_seq: Object poses at camera capture times (n_captures, n_objs, D)
        -> Fed to the Transformer ENCODER
    - traj_seq: Full ground-truth trajectory (T, D) with grasp appended
        -> Used as the target for loss computation
    - traj_hidden: Trajectory with position/orientation zeroed out (T, D-1)
        -> Fed to the Transformer DECODER as "hidden" input (only sees non-pose features)
    - weights: Per-timestep importance weights (T, D)
    - action_tag: One-hot action/task label
    - padding_mask: Boolean mask for padded timesteps
    - img_inds: Camera capture indices (segment boundaries)
    - pick_inds: Timestep indices where gripper closes (pick events)
    - release_inds: Timestep indices where gripper opens (release events)
    """
    
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
        augmentation_method: str = "tp",
    ) -> None:
        self.traj_data = traj_data
        self.obj_data = obj_data
        self.grasp_data = grasp_data
        self.dims = transform_dims  # Number of pose dimensions to zero out (7 for [x,y,z,qx,qy,qz,qw])
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
        self.augmentation_method = augmentation_method  # "tp" or "random"
        
        # obj_moving: per-action mask of which objects move between camera captures.
        # Shape: (n_objs, n_captures) -- True means the object moves during that segment.
        # Used to propagate trajectory displacement to moving objects after augmentation.
        # Rows = objects [bolt, nut, bin, jig], Cols = camera capture segments
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

        # --- Apply augmentation if enabled (training only) ---
        if self.augment:
            if self.augmentation_method == "random":
                # --- Random rotation augmentation ---
                # Collect initial object poses from both cameras
                init_obj_pose_all = []
                for i in range(obj_data.shape[1]):
                    init_obj_pose_all.append(get_init_obj_pose(obj_data[:, i, :], cam="both"))
                
                # Apply random rotation to trajectory + object poses jointly
                traj_data_new, init_obj_pose_all_transformed = augment_random_rotation(
                    traj_data,
                    init_obj_pose_all,
                )
                
                # Update object data with augmented initial poses
                for i, init_obj_pose_transformed in enumerate(init_obj_pose_all_transformed):
                    obj_data_filled[0, i] = init_obj_pose_transformed[0]
                    if len(init_obj_pose_transformed) == 2:
                        first_detection_ind = np.where(~np.isnan(obj_data[1:, i, :]).any(axis=1))[0][0] + 1
                        obj_data_filled[first_detection_ind, i] = init_obj_pose_transformed[1]
            else:
                # --- TP-augmentation (default) ---
                task_ind = torch.argmax(self.action_tags[idx]).item()
                transforms = self.transforms_all_actions[f"action_{task_ind}"]
                labels = self.labels_all_actions[f"action_{task_ind}"]
                covs = self.covs_all_actions[f"action_{task_ind}"]
                
                # Get initial object poses from both cameras (zed + wrist)
                init_obj_pose_all = []
                for i in range(obj_data.shape[1]):
                    init_obj_pose_all.append(get_init_obj_pose(obj_data[:, i, :], cam="both"))
                
                # Apply augmentation to trajectory and object poses
                traj_data_new, init_obj_pose_all_transformed = augment(
                    traj_data,
                    init_obj_pose_all,
                    transforms,
                    labels,
                    covs,
                    self.obj_tags,
                    traj_obj_ind=self.traj_obj_ind,
                )
                
                # For the trajectory object, override with actual trajectory poses at capture times
                if self.traj_obj_ind is not None:
                    for i in range(init_obj_pose_all_transformed[self.traj_obj_ind].shape[0]):
                        ind = img_inds[i]
                        init_obj_pose_all_transformed[self.traj_obj_ind][i, :7] = traj_data[ind][:7]
                
                # Update object data with augmented initial poses
                for i, init_obj_pose_transformed in enumerate(init_obj_pose_all_transformed):
                    obj_data_filled[0, i] = init_obj_pose_transformed[0]  # Zed camera detection
                    if len(init_obj_pose_transformed) == 2:
                        # Wrist camera detection: find first non-NaN and update
                        first_detection_ind = np.where(~np.isnan(obj_data[1:, i, :]).any(axis=1))[0][0] + 1
                        obj_data_filled[first_detection_ind, i] = init_obj_pose_transformed[1]
                
                # Propagate trajectory displacement to moving objects
                obj_moving_action = self.obj_moving[task_ind]
                for i, j in zip(*np.where(obj_moving_action)):
                    displacement = traj_data_new[img_inds[i], :3] - traj_data[img_inds[i], :3]
                    obj_data_filled[i, j, :3] += displacement

        # --- Post-processing ---
        # Fill any remaining NaNs in object data (forward-fill + zero-fill)
        obj_data_filled = fill_nans(obj_data_filled)
        obj_data_filled = np.nan_to_num(obj_data_filled, nan=0.0)
        
        # Insert grasp state as column 7 (between orientation and object tag)
        traj_data_new = np.insert(traj_data_new, 7, grasp_data, axis=1)

        # Convert to tensors
        traj_data = torch.tensor(traj_data_new)
        obj_data = torch.tensor(obj_data_filled)
        weights = torch.tensor(weights)
        
        # --- Pad to max_traj_seq_len ---
        padding_mask = torch.zeros(traj_data.shape[0])
        if traj_data.shape[0] < self.max_traj_seq_len:
            diff = self.max_traj_seq_len - traj_data.shape[0]
            pad_traj = torch.zeros([diff, traj_data.shape[1]])
            pad_traj[:, 6] = 1  # Pad quaternion w-component to 1 (identity rotation)
            traj_data = torch.cat([traj_data, pad_traj])
            pad_weights = torch.zeros((diff, weights.shape[1]))
            weights = torch.cat([weights, pad_weights])
            padding_mask = torch.cat([padding_mask, torch.ones(pad_traj.shape[0])])
            padding_mask = padding_mask > 0
        
        # --- Build decoder input (traj_hidden) ---
        # Zero out the pose dimensions (first self.dims columns) so the decoder
        # must predict them from the encoder output, not just copy the input.
        # Also remove column 7 (grasp) to get a different dimensionality for decoder input.
        traj_hidden = traj_data.clone()
        traj_hidden[:, : self.dims] = 0  # Zero out position + orientation
        traj_hidden = torch.cat((traj_hidden[:, :7], traj_hidden[:, 8:]), dim=1)  # Remove grasp column
        
        img_inds = torch.tensor(img_inds)
        pick_inds = torch.tensor(pick_inds)
        release_inds = torch.tensor(release_inds)
        
        if self.return_index:
            return obj_data, traj_data, traj_hidden, weights, idx, self.action_tags[idx], padding_mask, img_inds, pick_inds, release_inds
        return obj_data, traj_data, traj_hidden, weights, self.action_tags[idx], padding_mask, img_inds, pick_inds, release_inds


def get_init_obj_pose(obj_pose: np.ndarray, cam: str = "zed") -> np.ndarray | None:
    """Extract initial object pose from one or both cameras.
    
    Objects are detected by cameras at different times:
    - Row 0: Zed (overhead) camera detection (always available)
    - Rows 1+: Wrist camera detections (may contain NaNs if not detected)
    
    Args:
        obj_pose: Object pose time series (T, D), row 0 = zed, rows 1+ = wrist
        cam: Which camera to use:
            "zed": Return only zed camera pose (row 0)
            "wrist": Return first non-NaN wrist camera pose
            "combined": Return wrist if available, else zed (single pose)
            "both": Return both zed and first wrist detection (1-2 poses)
    
    Returns:
        Object pose(s) as array. Shape depends on cam mode.
    """
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
            init_obj_pose = obj_pose[0, :].reshape(1, -1)  # Only zed
        else:
            init_obj_pose = np.concatenate([obj_pose[0, :].reshape(1, -1), tmp[0].reshape(1, -1)], axis=0)  # Zed + wrist
        return np.array(init_obj_pose)
    raise ValueError("Camera type wrong. Choose from zed, wrist, combined, or both")
