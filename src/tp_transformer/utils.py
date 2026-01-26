from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from numpy.linalg import norm as l2norm
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal as mvn


def normalize_quats(quats: np.ndarray) -> np.ndarray:
    norm = l2norm(quats, axis=1)
    return quats / norm[:, None]


def force_rotation_axis_direction(quat: np.ndarray, axis: Iterable[float] = (1, 0, 0)) -> np.ndarray:
    vect = quat[:3]
    if vect @ np.array(axis) > 0:
        return quat
    return -quat


def force_smooth_quats(quats: np.ndarray) -> np.ndarray:
    for i, quat_current in enumerate(quats):
        if i == 0:
            continue
        quat_previous = quats[i - 1]
        if quat_current[:3] @ quat_previous[:3] > 0:
            quats[i] = quat_current
        else:
            quats[i] = -quat_current
    return quats


def process_quaternions(quats: np.ndarray, sigma: float | None = None, normalize: bool = True) -> np.ndarray:
    quats[0] = force_rotation_axis_direction(quats[0])
    quats_new = force_smooth_quats(quats)
    if sigma is not None:
        from scipy import ndimage

        quats_new = ndimage.gaussian_filter(quats_new, sigma=sigma)
    if normalize:
        return normalize_quats(quats_new)
    return quats_new


def get_quat_matrix(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat
    result = np.zeros((4, 4))
    result[0, :] = [qw, -qz, qy, qx]
    result[1, :] = [qz, qw, -qx, qy]
    result[2, :] = [-qy, qx, qw, qz]
    result[3, :] = [-qx, -qy, -qz, qw]
    return result


def get_rot_quat_matrix(r: R) -> np.ndarray:
    rotmatrix = r.as_matrix()
    quat = r.as_quat()
    result = np.zeros((7, 7))
    result[:3, :3] = rotmatrix
    result[3:, 3:] = get_quat_matrix(quat)
    return result


def homogeneous_position(vect: np.ndarray) -> np.ndarray:
    temp = np.array(vect)
    if temp.ndim == 1:
        ones = np.ones(1)
        return np.r_[temp, ones].reshape(-1, 1)
    if temp.ndim == 2:
        num_rows, num_cols = temp.shape
        if num_cols != 3:
            raise ValueError(f"vect is not N by 3, it is {num_rows}x{num_cols}")
        ones = np.ones(num_rows).reshape(-1, 1)
        return np.c_[temp, ones].T
    raise ValueError("vect must be 1D or 2D")


def lintrans(a: np.ndarray, H: np.ndarray) -> np.ndarray:
    D = a.shape[1]
    if D == 3:
        a_homo = homogeneous_position(a)
        return (H @ a_homo).T[:, :3]
    if D == 4:
        r_quat = R.from_matrix(H[:3, :3]).as_quat()
        quat_matrix = get_quat_matrix(r_quat)
        ori = (quat_matrix @ a.T).T
        return R.from_matrix(R.from_quat(ori).as_matrix()).as_quat()
    if D == 7:
        a_homo = homogeneous_position(a[:, :3])
        pos = (H @ a_homo).T[:, :3]
        r = R.from_matrix(H[:3, :3])
        quat_matrix = get_quat_matrix(r.as_quat())
        ori = (quat_matrix @ a[:, 3:].T).T
        try:
            ori_new = R.from_matrix(R.from_quat(ori).as_matrix()).as_quat()
        except ValueError:
            ori_new = ori
        return np.concatenate((pos, ori_new), axis=1)
    raise ValueError(f"Unsupported dimension: {D}")


def lintrans_cov(sigmas: np.ndarray, H: np.ndarray) -> List[np.ndarray]:
    rotmatrix = H[:3, :3]
    D = sigmas.shape[1]
    if D == 3:
        return [rotmatrix @ cov @ rotmatrix.T for cov in sigmas]
    if D == 4:
        r_quat = R.from_matrix(rotmatrix).as_quat()
        r_quat_inv = quat_conjugate(r_quat)
        quat_matrix = get_quat_matrix(r_quat)
        quat_matrix_inv = get_quat_matrix(r_quat_inv)
        return [quat_matrix @ cov @ quat_matrix_inv for cov in sigmas]
    if D == 7:
        r = R.from_matrix(rotmatrix)
        r_inv = r.inv()
        rot_quat_matrix = get_rot_quat_matrix(r)
        rot_quat_matrix_inv = get_rot_quat_matrix(r_inv)
        return [rot_quat_matrix @ cov @ rot_quat_matrix_inv for cov in sigmas]
    raise ValueError(f"Unsupported dimension: {D}")


def homogeneous_transform(rotmat: np.ndarray, vect: Iterable[float]) -> np.ndarray:
    if not isinstance(vect, list):
        vect = list(vect)
    H = np.zeros((4, 4))
    H[0:3, 0:3] = rotmat
    frame_displacement = vect + [1]
    D = np.array(frame_displacement).reshape(1, 4)
    H[:, 3] = D
    return H


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]])


def get_quaternion_difference_per_step(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return np.array([norm_diff_quat(q1[i], q2[i]) for i in range(q1.shape[0])])


def get_position_difference_per_step(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(d1 - d2, axis=1)


def swap_dict_level(data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    temp: Dict[str, Dict[str, np.ndarray]] = {}
    for obj, value in data.items():
        for demo, traj in value.items():
            temp.setdefault(demo, {})[obj] = traj
    return temp


def create_tags(objs: List[str], dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    one_hots = torch.eye(len(objs), dtype=dtype)
    return {obj: one_hots[i] for i, obj in enumerate(objs)}


def to_obj_index(onehot: torch.Tensor) -> int:
    for i in range(onehot.shape[0]):
        if onehot[i] == 1:
            return i
    return 0


def find_task_by_index(demo_index: int, breaks: List[int]) -> int:
    task_index = -1
    for br in breaks:
        if br > demo_index:
            break
        task_index += 1
    return task_index


def get_n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def create_chunks_of_indices(size: int, total: int, splits: int) -> List[List[int]]:
    offset = int(total / splits)
    all_indices = list(range(total))
    output = []
    for i in range(splits):
        holder = all_indices[offset * i : offset * i + size] + all_indices[: max(offset * i + size - total, 0)]
        output.append(sorted(holder))
    return output


def rescale_array(values: np.ndarray, m: float, n1: float, n2: float) -> np.ndarray:
    if n1 == m:
        raise ValueError("Upper bound of the original range (n1) cannot be equal to the lower bound (m).")
    if n2 == m:
        raise ValueError("Upper bound of the target range (n2) cannot be equal to the lower bound (m).")
    return (((values - m) / (n1 - m)) * (n2 - m)) + m


def get_dist_traj_to_obj(traj_pos: np.ndarray, obj_pos: np.ndarray) -> np.ndarray:
    return np.linalg.norm(traj_pos - obj_pos, axis=1)


def get_gaussians_per_step(traj_pose: np.ndarray, covs: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    mus, sigmas = [], []
    D = traj_pose.shape[1]
    for i, p in enumerate(traj_pose):
        mus.append(p)
        sigmas.append(np.eye(D) * covs[i])
    return mus, sigmas


def get_gaussians_per_obj(obj_pos: np.ndarray, sigma: float = 0.8) -> List[mvn]:
    return [mvn(obj_pos[i], np.eye(obj_pos.shape[1]) * sigma) for i in range(obj_pos.shape[0])]


def get_gaussians_per_obj_new(obj_pos: np.ndarray, sigmas: np.ndarray) -> List[mvn]:
    return [mvn(obj_pos[i], np.eye(obj_pos.shape[1]) * sigmas[i]) for i in range(obj_pos.shape[0])]


def get_likelihood_per_step(traj: np.ndarray, distributions: List[mvn]) -> np.ndarray:
    lls = np.zeros((len(distributions), traj.shape[0]))
    for i, dist in enumerate(distributions):
        lls[i] = dist.pdf(traj)
    return lls / np.sum(lls, axis=0)


def get_mean_cov_hats(ref_means: np.ndarray, ref_covs: np.ndarray, min_len: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    ref_pts = len(ref_means)
    if min_len is None:
        min_len = min(len(r) for r in ref_means)
    inv_covs = np.linalg.inv(np.array(ref_covs))
    mean_hats = np.empty((min_len, *ref_means[0][0].shape))
    sigma_hats = np.empty((min_len, *ref_covs[0][0].shape))
    for p in range(min_len):
        inv_sum = np.sum(inv_covs[:, p, :, :], axis=0)
        sigma_hat = np.linalg.inv(inv_sum)
        sigma_hats[p] = sigma_hat
        mean_w_sum = np.zeros(ref_means[0][0].shape)
        for ref in range(ref_pts):
            mean_w_sum += np.dot(inv_covs[ref][p], ref_means[ref][p])
        mean_hats[p] = np.dot(sigma_hat, mean_w_sum)
    return mean_hats, sigma_hats


def winner_takes_all(mus_all: np.ndarray, var: np.ndarray) -> np.ndarray:
    winner = np.argmin(np.max(var, axis=2), axis=0)
    winner_stack = np.repeat(winner.reshape(-1, 1), repeats=mus_all.shape[-1], axis=1)
    mus_all = np.array(mus_all)
    return winner_stack.choose(mus_all)


def get_label(var: np.ndarray, method: str = "wta") -> np.ndarray:
    var = np.array(var)
    if method == "wta":
        tmp = np.argmin(np.max(var, axis=2), axis=0)
        return np.repeat(tmp.reshape(-1, 1), repeats=var.shape[-1], axis=1)
    if method == "independent":
        return np.argmin(var, axis=0)
    raise ValueError(f"Unknown method: {method}")


def get_obj(obj_tags: Dict[str, torch.Tensor], tag: np.ndarray) -> str:
    values = list(obj_tags.values())
    ind = [i for i in range(len(values)) if (values[i].numpy() == tag).all()][0]
    return list(obj_tags.keys())[ind]


def map_range(array: Iterable[float], a1: float, a2: float, b1: float, b2: float) -> List[float]:
    if a1 == a2:
        raise ValueError("a1 and a2 must be different values.")
    if b1 == b2:
        raise ValueError("b1 and b2 must be different values.")
    return [b1 + (b2 - b1) * (x - a1) / (a2 - a1) for x in array]


def normalize_3d(entry: np.ndarray, average: np.ndarray, std: np.ndarray) -> np.ndarray:
    if entry.ndim == 3:
        entry[:, :, :3] = (entry[:, :, :3] - average) / std
    elif entry.ndim == 2:
        entry[:, :3] = (entry[:, :3] - average) / std
    return entry


def normalize_wrapper(average: np.ndarray, std: np.ndarray):
    return lambda x: normalize_3d(x, average, std)


def norm_diff_quat(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return min(np.linalg.norm(q1 + q2), np.linalg.norm(q1 - q2))


def quat_to_rotvect(traj: np.ndarray) -> np.ndarray:
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    traj_new = np.zeros((len(traj), 6))
    traj_new[:, :3] = traj[:, :3]
    traj_new[:, 3:] = R.from_quat(traj[:, 3:]).as_rotvec()
    return traj_new


def rotvect_to_quat(traj: np.ndarray) -> np.ndarray:
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    traj_new = np.zeros((len(traj), 7))
    traj_new[:, :3] = traj[:, :3]
    traj_new[:, 3:] = R.from_rotvec(traj[:, 3:]).as_quat()
    return traj_new
