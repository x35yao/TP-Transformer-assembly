"""Build `baseline_dataset.pickle` consumed by the CNEP / CNMP baselines.

Port of `D:\\project\\assembly\\split_data_assembly.ipynb` (cell 12 -- the only
cell that actually writes the baseline dataset). The notebook is ~12 cells of
mixed exploratory code; this script keeps only the *builder* path and drops
all dead code (transformer-side TrajectoryDataset, augmentation class
instantiation, plotting, hyperparameter search, etc. were either re-computed
in cell 12 but never used or only relevant to other experiments).

Pipeline
--------
For each (action, seed) combination:
    1. Walk `<processed-dir>/<action>/<demo>/` and load per-demo
       (T, 7) trajectories from `<demo>.csv` and (n_img, 4_obj, 7)
       initial object poses from `<demo>_obj_combined.h5`.
       Demos in the bad-demo list are dropped; demos with the wrong
       number of detected images (per `action_summary.pickle`) are also
       dropped.
    2. Pad each demo to `median_traj_len` by repeating the last pose
       (matches notebook behaviour).
    3. Random split with the given seed: `--num-train` train demos
       (default 15), `--num-validation` validation demos (default 4),
       1 test demo (notebook hardcodes test set to `test_demos[:1]`).
    4. Centre positions on `train_mean / train_std` (over xyz only;
       std is shrunk by 3, matching the notebook).
    5. Min-max normalise the concatenated train+valid+test pose
       trajectories per dim to [-1, 1] -> populates `*_traj_global_cnep`
       and the `minmax7` denormaliser.
    6. Crop the per-demo `0_left.png` to (left=400, upper=100,
       right=1900, lower=1200), ImageNet-normalise, and extract
       1280-d MobileNetV2 features -> populates `*_feats`.
    7. Re-express each train trajectory in 5 reference frames (4 real
       objects + global) using the homogeneous transforms derived from
       the demo's initial object poses -> populates `train_traj_tp_gmm`
       (list of 5 (15*T, 8) arrays) and `train_traj_tp_pmp`
       ((15, T, 35) array, 5 frames * 7 dims concatenated). The same
       transforms are recorded for test (`HTs_test`, list of 5 (4, 4))
       and validation (`HTs_validation`, (4, 5, 4, 4)).

Inputs (defaults assume the canonical layout at the repo root `data/`):
    --raw-dir         data/raw
                          cnep_action_0/<demo>/0_left.png
                          cnep_action_1/<demo>/0_left.png
                          cnep_action_2/<demo>/0_left.png
    --processed-dir   data/processed
                          action_0/action_summary.pickle
                          action_0/<demo>/<demo>.csv
                          action_0/<demo>/<demo>_obj_combined.h5
                          (and same for action_1, action_2)
    --task-config     data/task_config.yaml
                          (optional; falls back to a hardcoded object
                          list of [bin, bolt, jig, nut] -- matches the
                          notebook's `sorted(['bolt','nut','bin','jig'])`)

The `data/` directory is shared with the TP-Transformer pipeline
(`src/tp_transformer/data.py`), so both methods read the same source
trajectories and object poses.

Output:
    --out             baselines/data/baseline_dataset.pickle
        Dict shaped:
            {
                'action_0': [split_seed_0, split_seed_1, split_seed_2],
                'action_1': [...],
                'action_2': [...],
            }
        Each split is the 17-key dict that CNEP/CNMP consume; see
        `baselines/cnep/train_cnep.py` for the field-by-field contract.

The script is deterministic given the seeds and the inputs; the only
non-deterministic factor is the MobileNetV2 forward pass, which can drift
by ~1e-6 between CPU and CUDA runs.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_TASK_CONFIG = REPO_ROOT / "data" / "task_config.yaml"
DEFAULT_OUT = HERE / "data" / "baseline_dataset.pickle"

# Notebook-pinned constants. These come straight from
# `split_data_assembly.ipynb` cell 12 and are *not* CLI-exposed because
# changing them invalidates the existing checkpoints under
# `baselines/cnep/outputs/` and `baselines/cnmp/outputs/`.
DEFAULT_SEEDS: Tuple[int, ...] = (9871, 9872, 9873)
DEFAULT_ACTIONS: Tuple[str, ...] = ("action_0", "action_1", "action_2")
DEFAULT_NUM_TRAIN = 15
DEFAULT_NUM_VALIDATION = 4
TASK_DIMS: Tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw")
IMG_CROP = (400, 100, 1900, 1200)  # (left, upper, right, lower) -> 1500 x 1100
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Per-action demo blacklist. Reasons retained from the notebook for traceability.
BAD_DEMOS: Dict[str, List[str]] = {
    # nut detection wrong: 1724352860
    # wrong image numbers: 1724183761, 1724191561
    "action_0": ["1724352860", "1724183761", "1724191561"],
    # moves horizontally before grasping: 1724359405, 1724278229
    # wrong end:                          1724192868, 1724191561
    # missing image:                      1724280967
    # bad alignment:                      1724195677, 1724278798, 1724280457, 1724359951
    "action_1": [
        "1724359405", "1724278229", "1724280967", "1724195677",
        "1724278798", "1724280457", "1724359951", "1724192868",
        "1724191561",
    ],
    # did not go back to default:         1724267351, 1724359951
    # moves horizontally before grasping: 1724353947
    # wrong image numbers:                1724360506, 1724182947, 1724192244,
    #                                     1724195677, 1724354521
    # wrong image timing:                 1724183761, 1724194979, 1724191561
    "action_2": [
        "1724267351", "1724359951", "1724353947", "1724360506",
        "1724182947", "1724192244", "1724195677", "1724354521",
        "1724183761", "1724194979", "1724191561",
    ],
}


# ---------- pose / frame math (extracted from notebook cells 5, 6) ----------

def homogeneous_transform(rotmat: np.ndarray, vect) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and a 3-vec."""
    if not isinstance(vect, list):
        vect = list(vect)
    H = np.zeros((4, 4))
    H[:3, :3] = rotmat
    H[:, 3] = np.array(vect + [1]).reshape(4)
    return H


def inverse_homogeneous_transform(H: np.ndarray) -> np.ndarray:
    R_ = H[:3, :3]
    t = H[:3, 3].reshape(3, 1)
    Rt = R_.T
    return homogeneous_transform(Rt, list((-Rt @ t).flatten()))


def _homogeneous_position(vect: np.ndarray) -> np.ndarray:
    """Reshape (N, 3) positions to (4, N) homogeneous columns."""
    a = np.asarray(vect)
    if a.ndim == 1:
        return np.r_[a, 1.0].reshape(-1, 1)
    n = a.shape[0]
    return np.c_[a, np.ones(n).reshape(-1, 1)].T


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    """Left-quaternion-multiplication matrix for the [qx, qy, qz, qw] convention."""
    qx, qy, qz, qw = quat
    return np.array([
        [qw, -qz,  qy, qx],
        [qz,  qw, -qx, qy],
        [-qy, qx,  qw, qz],
        [-qx, -qy, -qz, qw],
    ])


def lintrans(a: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to (N, D) trajectories. D in {3, 4, 7}."""
    D = a.shape[1]
    if D == 3:
        return (H @ _homogeneous_position(a)).T[:, :3]
    if D == 4:
        r_quat = R.from_matrix(H[:3, :3]).as_quat()
        ori = (_quat_matrix(r_quat) @ a.T).T
        return R.from_matrix(R.from_quat(ori).as_matrix()).as_quat()
    if D == 7:
        pos = (H @ _homogeneous_position(a[:, :3])).T[:, :3]
        r_quat = R.from_matrix(H[:3, :3]).as_quat()
        ori = (_quat_matrix(r_quat) @ a[:, 3:].T).T
        try:
            ori = R.from_matrix(R.from_quat(ori).as_matrix()).as_quat()
        except ValueError:
            pass
        return np.concatenate([pos, ori], axis=1)
    raise ValueError(f"Unsupported D={D}")


def get_init_obj_pose(obj_pose: np.ndarray, cam: str = "zed"):
    """Pick the initial object pose for a demo. The notebook builder only ever
    uses `cam='zed'` (the first detection from the ZED), so we keep just that.
    """
    if cam != "zed":
        raise NotImplementedError("builder only uses zed-camera detections")
    return obj_pose[0, :]


# ---------- IO helpers ----------

def _load_task_config(path: Path) -> List[str]:
    """Read the object list from task_config.yaml; fall back to the notebook default."""
    if not path.exists():
        return sorted(["bolt", "nut", "bin", "jig"])
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    objs = cfg.get("objects", ["bolt", "nut", "bin", "jig"])
    return sorted(objs)


def _load_action_summary(processed_dir: Path, action: str) -> dict:
    path = processed_dir / action / "action_summary.pickle"
    with open(path, "rb") as f:
        return pickle.load(f)


def _enumerate_demos(processed_dir: Path, action: str) -> List[str]:
    action_dir = processed_dir / action
    bad = set(BAD_DEMOS.get(action, []))
    return sorted(
        d.name for d in action_dir.iterdir()
        if d.is_dir() and d.name not in bad
    )


def _load_demo(
    processed_dir: Path,
    action: str,
    demo: str,
    all_objs: List[str],
    median_traj_len: int,
    median_n_images: int,
) -> dict | None:
    """Load and pad one demo. Returns None if its image-count header doesn't
    match the action's median (matches notebook drop-condition)."""
    demo_dir = processed_dir / action / demo
    df_full = pd.read_csv(demo_dir / f"{demo}.csv", index_col=0)
    traj = df_full[list(TASK_DIMS)].to_numpy(dtype=np.float64)

    h5_path = demo_dir / f"{demo}_obj_combined.h5"
    if not h5_path.exists():
        return None
    df_obj = pd.read_hdf(h5_path)
    img_inds = list(df_obj.index)
    if len(img_inds) - 1 != median_n_images:
        return None

    # Stack object poses: (n_imgs, n_objs, 7) where the final object slot is
    # the trajectory at the image indices (i.e. the "global" frame).
    obj_pose_all = []
    for obj in all_objs:
        individual = obj + "1"
        obj_pose = df_obj[individual][list(TASK_DIMS)].to_numpy(dtype=np.float64)
        obj_pose_all.append(obj_pose)
    obj_pose_all.append(traj[img_inds][:, :7])  # 'trajectory' object
    obj_pose_all = np.transpose(np.array(obj_pose_all), (1, 0, 2))

    # Pad trajectory to median length by repeating the final pose.
    if traj.shape[0] < median_traj_len:
        pad = median_traj_len - traj.shape[0]
        traj = np.vstack([traj, np.repeat(traj[-1:], pad, axis=0)])

    return {
        "traj_pose": traj,           # (median_traj_len, 7)
        "obj_pose_all": obj_pose_all,  # (n_imgs, n_objs+1, 7)
    }


def _load_demo_dataset(
    processed_dir: Path,
    action: str,
    all_objs: List[str],
) -> Tuple[Dict[str, dict], int]:
    summary = _load_action_summary(processed_dir, action)
    median_traj_len = int(summary["median_traj_len"])
    median_n_images = int(summary["median_n_images"])
    out = {}
    for demo in _enumerate_demos(processed_dir, action):
        loaded = _load_demo(
            processed_dir, action, demo, all_objs,
            median_traj_len, median_n_images,
        )
        if loaded is not None:
            out[demo] = loaded
    return out, median_traj_len


# ---------- MobileNetV2 image features ----------

def _build_mobilenet(device: str) -> torch.nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    net = models.mobilenet_v2(weights=weights).to(device)
    net.classifier = torch.nn.Identity()
    net.eval()
    return net


def _build_img_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _extract_features(
    net: torch.nn.Module,
    img_transform: transforms.Compose,
    raw_dir: Path,
    action: str,
    demos: List[str],
    device: str,
) -> torch.Tensor:
    feats = []
    with torch.no_grad():
        for demo in demos:
            img_path = raw_dir / f"cnep_{action}" / demo / "0_left.png"
            img = Image.open(img_path).convert("RGB")
            img = img.crop(IMG_CROP)
            x = img_transform(img).unsqueeze(0).to(device)
            feats.append(net(x).flatten())
    return torch.stack(feats, dim=0)


# ---------- split + pack ----------

def _split_demos(
    demos: List[str],
    n_train: int,
    n_validation: int,
    rng: "random.Random",
) -> Tuple[List[str], List[str], List[str]]:
    """Replicate the notebook's split exactly:

        train_demos_pool = random.sample(demos, n_train)         # cell 12 line 310
        test_valid_pool  = [d for d in demos if d not in pool]
        valid_demos      = random.sample(test_valid_pool, n_val) # line 317
        test_demos       = [test_demos[0]]                       # line 320

    Important: the caller seeds the RNG ONCE per seed and reuses the same RNG
    across all 3 actions (matching cell 12 line 283 `random.seed(seed)` placed
    OUTSIDE the inner `for task in tasks:` loop). Calling `random.seed` per
    action would advance the state independently and produce different splits
    for action_1/action_2 than the canonical pickle.

    Then we SORT train/valid alphabetically before returning. The notebook
    discards `random.sample`'s order by iterating over the sorted demo pool
    and appending only demos in each split (cell 12 line 321 `for demo in demos`),
    which sorts the output as a side effect. Without this sort, our trajectories
    end up in a different per-demo order than the canonical pickle.
    """
    train = rng.sample(demos, n_train)
    test_valid = [d for d in demos if d not in train]
    valid = rng.sample(test_valid, n_validation)
    test = [d for d in test_valid if d not in valid][:1]
    return sorted(train), sorted(valid), test


def _per_frame_local_traj(
    train_trajs_global: np.ndarray,
    train_obj_poses: np.ndarray,
    obj_idx: int,
    is_traj_frame: bool,
    times_col: np.ndarray,
) -> np.ndarray:
    """For one reference frame, project each training trajectory into it and
    prepend the time column (matches notebook's TP-GMM 8-dim layout)."""
    out = []
    for k in range(train_trajs_global.shape[0]):
        if is_traj_frame:
            local = train_trajs_global[k].copy()
        else:
            obj_pose = get_init_obj_pose(train_obj_poses[k, :, obj_idx, :7]).copy()
            rotmat = R.from_quat(obj_pose[3:]).as_matrix()
            H = homogeneous_transform(rotmat, obj_pose[:3])
            H_inv = inverse_homogeneous_transform(H)
            local = lintrans(train_trajs_global[k][:, :7].copy(), H_inv)
        out.append(np.concatenate([times_col, local], axis=1))
    return np.array(out)  # (n_train, T, 8)


def _frame_HT_for_test(test_obj_poses: np.ndarray, obj_idx: int, is_traj_frame: bool) -> np.ndarray:
    if is_traj_frame:
        return homogeneous_transform(np.eye(3), np.zeros(3))
    obj_pose = get_init_obj_pose(test_obj_poses[:, obj_idx, :7]).copy()
    rotmat = R.from_quat(obj_pose[3:]).as_matrix()
    return homogeneous_transform(rotmat, obj_pose[:3])


def _frame_HTs_for_valid(valid_obj_poses: np.ndarray, obj_idx: int, is_traj_frame: bool) -> List[np.ndarray]:
    HTs = []
    for mm in range(valid_obj_poses.shape[0]):
        if is_traj_frame:
            HTs.append(homogeneous_transform(np.eye(3), np.zeros(3)))
        else:
            obj_pose = get_init_obj_pose(valid_obj_poses[mm, :, obj_idx, :7]).copy()
            rotmat = R.from_quat(obj_pose[3:]).as_matrix()
            HTs.append(homogeneous_transform(rotmat, obj_pose[:3]))
    return HTs


def _normalise_train_stats(train_traj_pose_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Notebook formula: mean over xyz of all stacked train trajectories,
    std as a *single scalar* over the same elements divided by 3."""
    contiguous = np.concatenate(train_traj_pose_list)
    train_mean = np.mean(contiguous[:, :3], axis=0)
    train_std = float(np.std(contiguous[:, :3]) / 3.0)
    return train_mean, train_std


def _apply_xyz_normalisation(trajs: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
    """Centre xyz only; orientation columns are left untouched."""
    out = trajs.copy()
    out[..., :3] = (out[..., :3] - mean) / std
    return out


def _pack_one_split(
    action: str,
    seed: int,
    train_demos: List[str],
    valid_demos: List[str],
    test_demos: List[str],
    demo_dataset: Dict[str, dict],
    all_objs: List[str],
    feats_train: torch.Tensor,
    feats_valid: torch.Tensor,
    feats_test: torch.Tensor,
    train_mean: np.ndarray,
    train_std: float,
) -> dict:
    """Compute the 17-key per-split dict for one (action, seed) cell.

    `train_mean` / `train_std` are SHARED across all actions for the same seed
    (see notebook cell 12 lines 361-363: the stats are computed once after the
    inner per-task loop has accumulated train_traj_pose for all 3 actions).
    """
    n_train = len(train_demos)
    n_valid = len(valid_demos)
    n_objs_with_traj = len(all_objs) + 1  # +1 for the 'trajectory' frame
    traj_obj_ind = n_objs_with_traj - 1

    raw_train_trajs = [demo_dataset[d]["traj_pose"] for d in train_demos]
    raw_valid_trajs = [demo_dataset[d]["traj_pose"] for d in valid_demos]
    raw_test_trajs = [demo_dataset[d]["traj_pose"] for d in test_demos]

    raw_train_objs = np.array([demo_dataset[d]["obj_pose_all"] for d in train_demos])
    raw_valid_objs = np.array([demo_dataset[d]["obj_pose_all"] for d in valid_demos])
    raw_test_objs = np.array(demo_dataset[test_demos[0]]["obj_pose_all"])

    norm_train_trajs = np.array([
        _apply_xyz_normalisation(t, train_mean, train_std) for t in raw_train_trajs
    ])
    norm_valid_trajs = np.array([
        _apply_xyz_normalisation(t, train_mean, train_std) for t in raw_valid_trajs
    ])
    norm_test_traj = _apply_xyz_normalisation(raw_test_trajs[0], train_mean, train_std)

    norm_train_objs = _apply_xyz_normalisation(raw_train_objs, train_mean, train_std)
    norm_valid_objs = _apply_xyz_normalisation(raw_valid_objs, train_mean, train_std)
    norm_test_objs = _apply_xyz_normalisation(raw_test_objs, train_mean, train_std)

    # ---- CNEP-side: stack and min-max normalise to [-1, 1] ----
    traj_len = norm_train_trajs.shape[1]
    train_global = norm_train_trajs[:, :, :7]
    valid_global = norm_valid_trajs[:, :, :7]
    test_global = norm_test_traj[:, :7]
    traj_all = torch.from_numpy(np.concatenate([
        train_global,
        valid_global,
        test_global[None],
    ]))
    minmax7 = torch.zeros(7, 2)
    for k in range(7):
        lo = float(traj_all[:, :, k].min())
        hi = float(traj_all[:, :, k].max())
        traj_all[:, :, k] = 2 * (traj_all[:, :, k] - lo) / (hi - lo) - 1
        minmax7[k] = torch.tensor([lo, hi], dtype=torch.float32)
    train_traj_global_cnep = traj_all[:n_train]
    valid_traj_global_cnep = traj_all[n_train: n_train + n_valid]
    test_traj_global_cnep = traj_all[-1]

    # ---- TP-GMM / TP-PMP: re-express train trajs in 5 reference frames ----
    times_col = (np.arange(traj_len) / traj_len).reshape(-1, 1)
    train_data_in_all_rfs_tp_gmm = []
    train_data_in_all_rfs_tp_pmp = []
    test_HTs = []
    validation_HTs = []
    for jj in range(n_objs_with_traj):
        is_traj_frame = (jj == traj_obj_ind)
        per_frame_train = _per_frame_local_traj(
            train_global, norm_train_objs, jj, is_traj_frame, times_col,
        )  # (n_train, T, 8)
        train_data_in_all_rfs_tp_gmm.append(per_frame_train.reshape(-1, 1 + 7))  # (n_train*T, 8)
        train_data_in_all_rfs_tp_pmp.append(per_frame_train[:, :, 1:])           # (n_train, T, 7)

        test_HTs.append(_frame_HT_for_test(norm_test_objs, jj, is_traj_frame))
        validation_HTs.append(_frame_HTs_for_valid(norm_valid_objs, jj, is_traj_frame))

    # validation_HTs is list[5] of list[n_valid] -> stack to (n_valid, 5, 4, 4)
    validation_HTs_arr = np.array(validation_HTs).swapaxes(0, 1)
    # tp_pmp concat across frame axis -> (n_train, T, 5*7=35)
    train_data_in_all_rfs_tp_pmp = np.concatenate(train_data_in_all_rfs_tp_pmp, axis=2)

    train_times_tp_pmp = [times_col.flatten() for _ in range(n_train)]

    return {
        "train_traj_tp_gmm": train_data_in_all_rfs_tp_gmm,
        "train_traj_tp_pmp": train_data_in_all_rfs_tp_pmp,
        "test_t": times_col.flatten(),
        "HTs_test": test_HTs,
        "HTs_validation": validation_HTs_arr,
        "test_traj_global": norm_test_traj[:, :7],
        "validation_traj_global": valid_global,
        "seed": int(seed),
        "train_stat": {"mean": train_mean, "std": np.float64(train_std)},
        "train_times_tp_pmp": train_times_tp_pmp,
        "train_feats": feats_train,
        "valid_feats": feats_valid,
        "test_feats": feats_test,
        "minmax7": minmax7,
        "train_traj_global_cnep": train_traj_global_cnep,
        "valid_traj_global_cnep": valid_traj_global_cnep,
        "test_traj_global_cnep": test_traj_global_cnep,
    }


# ---------- main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help="Root holding cnep_action_{0,1,2}/<demo>/0_left.png "
                        "(default: data/raw).")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
                   help="Root holding processed action_{0,1,2}/... CSVs/H5s "
                        "(default: data/processed).")
    p.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG,
                   help="task_config.yaml with the `objects:` list "
                        "(default: data/task_config.yaml; if missing, "
                        "falls back to ['bin','bolt','jig','nut']).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output pickle path (default: baselines/data/baseline_dataset.pickle).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                   help="Per-split RNG seeds. Default reproduces the canonical pickle.")
    p.add_argument("--actions", type=str, nargs="+", default=list(DEFAULT_ACTIONS))
    p.add_argument("--num-train", type=int, default=DEFAULT_NUM_TRAIN)
    p.add_argument("--num-validation", type=int, default=DEFAULT_NUM_VALIDATION)
    p.add_argument("--device", type=str, default=None,
                   help="torch device for MobileNetV2 (default: cpu - matches the "
                        "notebook; switch to 'cuda:0' for ~10x speedup with "
                        "negligible numerical drift).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or "cpu"
    print(f"raw-dir       : {args.raw_dir}")
    print(f"processed-dir : {args.processed_dir}")
    print(f"task-config   : {args.task_config}")
    print(f"out           : {args.out}")
    print(f"seeds         : {args.seeds}")
    print(f"actions       : {args.actions}")
    print(f"num-train     : {args.num_train}")
    print(f"num-validation: {args.num_validation}")
    print(f"device        : {device}")

    all_objs = _load_task_config(args.task_config)
    print(f"object list   : {all_objs}")

    net = _build_mobilenet(device)
    img_transform = _build_img_transform()

    Data_all_task: Dict[str, list] = {a: [] for a in args.actions}

    # Cache per-action demo dataset + features (independent of seed).
    per_action_demo_data: Dict[str, dict] = {}
    per_action_feats_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    for action in args.actions:
        print(f"\n=== loading {action} ===")
        demos, median_traj_len = _load_demo_dataset(args.processed_dir, action, all_objs)
        print(f"  {len(demos)} valid demos, median_traj_len={median_traj_len}")
        per_action_demo_data[action] = demos

    for seed in args.seeds:
        # The notebook seeds Python's `random` ONCE per seed (cell 12 line 283,
        # placed OUTSIDE the per-task loop). All 3 actions share that RNG state,
        # so action_1's split depends on action_0 having consumed 2 sample calls
        # first. We replicate that here with an explicit per-seed RNG instance.
        rng = random.Random(seed)

        # Pass 1: do all splits, share the RNG across actions.
        per_action_split: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
        for action in args.actions:
            demos = list(per_action_demo_data[action].keys())
            per_action_split[action] = _split_demos(
                demos, args.num_train, args.num_validation, rng,
            )

        # Pass 2: compute the seed-global train_mean / train_std over the
        # train trajectories from ALL actions concatenated (notebook cell 12
        # lines 361-363). All per-action `train_stat`s share this value.
        all_train_trajs: List[np.ndarray] = []
        for action in args.actions:
            train, _, _ = per_action_split[action]
            demo_dataset = per_action_demo_data[action]
            for d in train:
                all_train_trajs.append(demo_dataset[d]["traj_pose"])
        train_mean, train_std = _normalise_train_stats(all_train_trajs)

        # Pass 3: pack each per-action split with the global stats and features.
        for action in args.actions:
            train, valid, test = per_action_split[action]
            demo_dataset = per_action_demo_data[action]

            def _feats(role_demos: List[str]) -> torch.Tensor:
                # cache key keyed by demo set so we don't re-run MobileNet
                # for the same image across seeds
                key = (action, "|".join(role_demos))
                if key not in per_action_feats_cache:
                    per_action_feats_cache[key] = _extract_features(
                        net, img_transform, args.raw_dir, action, role_demos, device,
                    )
                return per_action_feats_cache[key]

            split = _pack_one_split(
                action, seed, train, valid, test, demo_dataset, all_objs,
                feats_train=_feats(train),
                feats_valid=_feats(valid),
                feats_test=_feats(test),
                train_mean=train_mean,
                train_std=train_std,
            )
            Data_all_task[action].append(split)
            print(f"  seed={seed} {action}: "
                  f"train={len(train)} valid={len(valid)} test={len(test)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(Data_all_task, f)
    print(f"\nwrote {args.out} "
          f"({os.path.getsize(args.out) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
