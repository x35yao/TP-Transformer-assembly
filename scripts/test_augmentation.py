"""
Test script to visualize augmentation methods (TP and random rotation).

Loads one demo, applies both augmentations, and plots 3D trajectory + object
positions before and after augmentation side by side.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Add project root to path so we can import tp_transformer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tp_transformer.augmentation import (
    augment,
    augment_random_rotation,
    build_labels_and_covs,
    build_transforms,
)
from src.tp_transformer.config import TrainConfig
from src.tp_transformer.utils import create_tags

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
config = TrainConfig()
TASK_DIMS = ["x", "y", "z", "qx", "qy", "qz", "qw"]
TASK = "action_0"
OBJ_COLORS = {"bolt": "red", "nut": "blue", "bin": "green", "jig": "orange", "trajectory": "purple"}

# ---------------------------------------------------------------------------
# Load one demo
# ---------------------------------------------------------------------------
task_root = os.path.join(config.processed_dir, TASK)
demos = sorted([d for d in os.listdir(task_root) if os.path.isdir(os.path.join(task_root, d))])
demo = demos[0]
print(f"Using demo: {demo}")

# Trajectory
traj_file = os.path.join(task_root, demo, f"{demo}.csv")
df_traj = pd.read_csv(traj_file, index_col=0)
traj_pose = df_traj[TASK_DIMS].to_numpy()

# Object tags
obj_tags = create_tags(config.all_objs + ["trajectory"])

# Append trajectory object tag
traj_len = traj_pose.shape[0]
obj_buffer = np.tile(obj_tags["trajectory"].numpy(), (traj_len, 1))
traj_data = np.concatenate([traj_pose, obj_buffer], axis=1)

# Object poses
obj_file = os.path.join(task_root, demo, f"{demo}_obj_combined.h5")
df_obj = pd.read_hdf(obj_file, index_col=0)
img_inds = list(df_obj.index)

obj_pose_all = []
for obj_ind in config.all_objs + ["trajectory"]:
    if obj_ind != "trajectory":
        individual_ind = obj_ind + "1"
        obj_pose = df_obj[individual_ind][TASK_DIMS].to_numpy()
        obj_tag_repeat = np.tile(obj_tags[obj_ind].numpy().reshape(1, -1), (len(obj_pose), 1))
        obj_pose = np.concatenate([obj_pose, obj_tag_repeat], axis=1)
        obj_pose_all.append(obj_pose)
    else:
        obj_pose = df_traj[TASK_DIMS].iloc[img_inds].to_numpy()
        obj_tag_repeat = np.tile(obj_tags[obj_ind].numpy().reshape(1, -1), (len(obj_pose), 1))
        obj_pose = np.concatenate([obj_pose, obj_tag_repeat], axis=1)
        obj_pose_all.append(obj_pose)

obj_pose_all = np.array(obj_pose_all)
obj_pose_all = np.transpose(obj_pose_all, (1, 0, 2))  # (n_captures, n_objs, D)

# Extract initial object poses (first non-NaN for each object)
def get_init_obj_pose_both(obj_pose):
    tmp = obj_pose[1:][~np.isnan(obj_pose[1:, :]).any(axis=1)]
    if len(tmp) == 0:
        return obj_pose[0, :].reshape(1, -1)
    return np.concatenate([obj_pose[0, :].reshape(1, -1), tmp[0].reshape(1, -1)], axis=0)

init_obj_poses = []
for i in range(obj_pose_all.shape[1]):
    init_obj_poses.append(get_init_obj_pose_both(obj_pose_all[:, i, :]))

# ---------------------------------------------------------------------------
# Apply random rotation augmentation
# ---------------------------------------------------------------------------
traj_rand, obj_rand = augment_random_rotation(traj_data.copy(), init_obj_poses)
print("Random rotation augmentation applied successfully!")

# ---------------------------------------------------------------------------
# Apply TP augmentation
# ---------------------------------------------------------------------------
try:
    with open(f"./augmentation/{config.aug_date}/obj_augs.pickle", "rb") as f:
        obj_augs = pickle.load(f)
    with open(f"./augmentation/{config.aug_date}/vars.pickle", "rb") as f:
        variances = pickle.load(f)
    transforms_all_actions = build_transforms(obj_augs, config.all_objs)
    labels_all_actions, covs_all_actions = build_labels_and_covs(variances, config.tasks, config.all_objs)

    transforms = transforms_all_actions[TASK]
    labels = labels_all_actions[TASK]
    covs = covs_all_actions[TASK]

    traj_tp, obj_tp = augment(
        traj_data.copy(),
        [p.copy() for p in init_obj_poses],
        transforms,
        labels,
        covs,
        obj_tags,
        traj_obj_ind=config.traj_obj_ind,
    )
    tp_available = True
    print("TP augmentation applied successfully!")
except Exception as e:
    tp_available = False
    print(f"TP augmentation not available: {e}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
n_cols = 3 if tp_available else 2
fig = plt.figure(figsize=(7 * n_cols, 6))

def plot_3d(ax, traj, obj_poses, title):
    """Plot trajectory and object positions in 3D."""
    # Trajectory
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "k-", alpha=0.6, linewidth=1, label="trajectory")
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c="lime", s=60, marker="^", zorder=5, label="start")
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c="black", s=60, marker="v", zorder=5, label="end")

    # Objects (first detection of each)
    obj_names = config.all_objs + ["trajectory"]
    for i, obj_name in enumerate(obj_names):
        if i < len(obj_poses):
            pos = obj_poses[i][0, :3]
            color = OBJ_COLORS.get(obj_name, "gray")
            ax.scatter(*pos, c=color, s=100, marker="o", edgecolors="black", label=obj_name)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")

# 1) Original
ax1 = fig.add_subplot(1, n_cols, 1, projection="3d")
plot_3d(ax1, traj_data, init_obj_poses, "Original")

# 2) Random Rotation
ax2 = fig.add_subplot(1, n_cols, 2, projection="3d")
plot_3d(ax2, traj_rand, obj_rand, "Random Rotation")

# 3) TP Augmentation (if available)
if tp_available:
    ax3 = fig.add_subplot(1, n_cols, 3, projection="3d")
    plot_3d(ax3, traj_tp, obj_tp, "TP Augmentation")

plt.tight_layout()

# Save figure
os.makedirs("figures", exist_ok=True)
save_path = os.path.join("figures", "augmentation_comparison.png")
plt.savefig(save_path, dpi=150)
print(f"Figure saved to {save_path}")
plt.show()
