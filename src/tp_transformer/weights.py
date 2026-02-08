"""
Trajectory weighting functions.

These functions assign per-timestep importance weights to trajectory points.
Higher weights mean the model is penalized more for errors at those timesteps.
The final weight per timestep is the max across all three weighting schemes
(speed, grasp, wrist camera).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def find_change_points(arr: np.ndarray) -> list[int]:
    """Find indices where the value changes from one timestep to the next.
    
    Used to detect gripper open/close transitions and wrist camera capture events.
    """
    change_points = []
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            change_points.append(i)
    return change_points


def get_speed_weights(
    df_traj: pd.DataFrame, lb: float = 1, ub: float = 2, sigma: float = 5, p: float = 0.25, fill_value: float = 0.1
) -> np.ndarray:
    """Assign high weights to low-speed (near-stationary) trajectory segments.
    
    Low-speed regions typically correspond to critical manipulation phases
    (approaching, grasping, placing) where precision matters most.
    
    Args:
        df_traj: DataFrame with columns ["x", "y", "z", "time"]
        lb: Lower bound weight for fast-moving segments
        ub: Upper bound weight for slow segments
        sigma: Gaussian smoothing sigma for speed profile
        p: Threshold fraction of max speed -- below this is considered "slow"
        fill_value: Replacement for zero time differences to avoid division by zero
    
    Returns:
        Weight array of shape (T,) with values lb or ub per timestep
    """
    robot_pos = df_traj[["x", "y", "z"]].to_numpy()
    robot_pos_diff = np.diff(robot_pos, axis=0)
    time_diff = np.diff(df_traj["time"].to_numpy())
    time_diff = np.where(time_diff == 0, fill_value, time_diff)
    speed = np.linalg.norm(robot_pos_diff, axis=1) / time_diff
    speed = np.concatenate([[0.1], speed])  # Pad first timestep
    speed = gaussian_filter(speed, sigma=sigma)
    weights = speed < p * np.max(speed)  # True where speed is below threshold
    weights = weights.astype(float)
    weights[weights > 0] = ub  # Slow segments get high weight
    weights[weights == 0] = lb  # Fast segments get low weight
    return weights


def get_grasp_weights(
    df_traj: pd.DataFrame, lb: float = 1, ub: float = 2, length_ahead: int = 5, length_after: int = 1
) -> np.ndarray:
    """Assign high weights around gripper open/close transitions.
    
    The timesteps immediately before and after a grasp state change are critical
    for learning precise pick and place behaviors.
    
    Args:
        df_traj: DataFrame with column "grasp_detected" (0 or 1)
        lb: Lower bound weight for non-grasp regions
        ub: Upper bound weight for grasp transition regions
        length_ahead: Number of timesteps BEFORE the transition to upweight
        length_after: Number of timesteps AFTER the transition to upweight
    
    Returns:
        Weight array of shape (T,) with values lb or ub per timestep
    """
    weights = np.ones(len(df_traj)) * lb
    gripper_state = df_traj["grasp_detected"].to_numpy()
    gripper_state_change_inds = find_change_points(gripper_state)
    for ind in gripper_state_change_inds:
        weights[ind - length_ahead : ind] = ub
        weights[ind : ind + length_after] = ub
    return weights


def get_wrist_weights(
    df_traj: pd.DataFrame, lb: float = 1, ub: float = 2, length: int = 5
) -> np.ndarray:
    """Assign high weights around wrist camera capture events.
    
    When the wrist camera captures an image, the robot is typically close to
    an object of interest -- these are important waypoints to learn accurately.
    
    Args:
        df_traj: DataFrame with column "wrist_camara_capture" (0 or 1)
        lb: Lower bound weight for normal regions
        ub: Upper bound weight for wrist capture regions
        length: Number of timesteps before and after the capture to upweight
    
    Returns:
        Weight array of shape (T,) with values lb or ub per timestep
    """
    weights = np.ones(len(df_traj)) * lb
    wrist_state = df_traj["wrist_camara_capture"].to_numpy()
    wrist_state_change_inds = find_change_points(wrist_state)
    for ind in wrist_state_change_inds:
        weights[ind - length : ind] = ub
        weights[ind : ind + length] = ub
    return weights
