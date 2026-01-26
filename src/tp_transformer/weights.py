from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def find_change_points(arr: np.ndarray) -> list[int]:
    change_points = []
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            change_points.append(i)
    return change_points


def get_speed_weights(df_traj: pd.DataFrame, lb: float = 1, ub: float = 2, sigma: float = 5, p: float = 0.25, fill_value: float = 0.1) -> np.ndarray:
    robot_pos = df_traj[["x", "y", "z"]].to_numpy()
    robot_pos_diff = np.diff(robot_pos, axis=0)
    time_diff = np.diff(df_traj["time"].to_numpy())
    time_diff = np.where(time_diff == 0, fill_value, time_diff)
    speed = np.linalg.norm(robot_pos_diff, axis=1) / time_diff
    speed = np.concatenate([[0.1], speed])
    speed = gaussian_filter(speed, sigma=sigma)
    weights = speed < p * np.max(speed)
    weights = weights.astype(float)
    weights[weights > 0] = ub
    weights[weights == 0] = lb
    return weights


def get_grasp_weights(df_traj: pd.DataFrame, lb: float = 1, ub: float = 2, length_ahead: int = 5, length_after: int = 1) -> np.ndarray:
    weights = np.ones(len(df_traj)) * lb
    gripper_state = df_traj["grasp_detected"].to_numpy()
    gripper_state_change_inds = find_change_points(gripper_state)
    for ind in gripper_state_change_inds:
        weights[ind - length_ahead : ind] = ub
        weights[ind : ind + length_after] = ub
    return weights


def get_wrist_weights(df_traj: pd.DataFrame, lb: float = 1, ub: float = 2, length: int = 5) -> np.ndarray:
    weights = np.ones(len(df_traj)) * lb
    wrist_state = df_traj["wrist_camara_capture"].to_numpy()
    wrist_state_change_inds = find_change_points(wrist_state)
    for ind in wrist_state_change_inds:
        weights[ind - length : ind] = ub
        weights[ind : ind + length] = ub
    return weights
