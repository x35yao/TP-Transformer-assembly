"""Per-(subtask, row, arm) bare 3D panels for the augmentation experiment.

Emits one bare PNG/EPS per panel so the 2x3-per-subtask grids can be arranged
freely in LaTeX (column headers = arms, row labels = train/pred, added in
LaTeX), plus a single shared legend image.

Each panel shows a gray dashed reference (original demo for the training row,
test GT for the prediction row) plus the arm line, coloured by type:
  training row -> augmented demo in red
  prediction row -> prediction in blue
Objects are squares (bolt green, nut yellow, bin black, jig purple);
start = circle, end = cross.

Uses the important_dist (reported) models; the (seed, demo) cell per subtask
is the one where TP-aug is most representative (lowest ADE).

Outputs (results/figures/):
  aug3d_<action>_<row>_<arm>.{png,eps}   18 bare panels (3 actions x 2 rows x 3 arms)
  aug3d_legend.{png,eps}                 one shared legend strip
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from tp_transformer.config import TrainConfig
from tp_transformer.data import build_datasets
from evaluate_predictions import _metrics_for_trajectory, _to_3d

USER = os.environ["USER"]
EVAL = Path(f"/shared/{USER}/RingAIAutoAnnotation/eval")
IMP = EVAL / "exp2" / "archive_important_dist"
DATA = "baselines/data/baseline_dataset_n15_v3t3.pickle"
OUT = EVAL / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [9871, 9872, 9873, 9874, 9875]
# Object -> colour (bolt green, nut yellow, bin/box black, jig purple).
OBJ_COLOR = {"bolt": "#2ca02c", "nut": "#e8d100", "bin": "#000000", "jig": "#9467bd"}
# Object axis ordering differs by source:
#   - live dataset (row 1, obj_data): config.all_objs order [bolt, nut, bin, jig]
#   - baseline pickle (row 2, HTs_test): alphabetical (prepare_baseline_dataset
#     sorts the object list), i.e. [bin, bolt, jig, nut].
DATASET_OBJ_ORDER = ["bolt", "nut", "bin", "jig"]
HTS_OBJ_ORDER = ["bin", "bolt", "jig", "nut"]
COLS = ["No-aug", "Random rotation", "TP-aug"]
# Trajectory colours encode the *type* of line (not the column):
#   reference  = gray dashed (original demo in row 1 / test GT in row 2)
#   row 1 arm  = red   (the augmented training trajectory)
#   row 2 arm  = blue  (the model prediction)
REF_COLOR = "#808080"     # gray, dashed
AUG_COLOR = "#d62728"     # red  -- augmented training trajectory (row 1)
PRED_COLOR = "#1f77b4"    # blue -- prediction (row 2)
VIEW = dict(elev=22, azim=-60)

dset = pickle.load(open(DATA, "rb"))


def best_cell(action, action_idx):
    """(seed_idx, seed, demo) minimising TP-aug ADE for this action."""
    best = None
    for si, seed in enumerate(SEEDS):
        e = dset[action][si]
        gt = _to_3d(e["test_traj_global"]); m = np.asarray(e["train_stat"]["mean"]).reshape(3); s = float(e["train_stat"]["std"])
        p = IMP / "tp" / str(seed) / "predictions.pickle"
        if not p.exists():
            continue
        arr = _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))
        for d in range(min(arr.shape[0], gt.shape[0])):
            ade, _ = _metrics_for_trajectory(arr[d], gt[d], m, s)
            if best is None or ade < best[0]:
                best = (ade, si, seed, d)
    return best[1], best[2], best[3]


def train_traj_objs(method, seed, action_idx):
    cfg = TrainConfig()
    cfg.splits_file = "data/splits/n15_v3t3.yaml"; cfg.seed = seed
    cfg.augmentation_method = method
    train_ds, _, _, stats = build_datasets(cfg)
    mean = np.asarray(stats["mean"]).reshape(3); std = float(stats["std"])
    for i in range(len(train_ds)):
        obj_data, traj_data, _, _, atag, pad, _, _, _ = train_ds[i]
        if int(torch.argmax(atag).item()) == action_idx:
            mask = ~pad.numpy().astype(bool)
            return (traj_data.numpy()[mask][:, :3] * std + mean,
                    obj_data.numpy()[0, :4, :3] * std + mean)
    raise RuntimeError("no sample")


# Per-action rotation angle (deg, about z) used for the illustrative
# "Random rotation" training column, chosen so the start/end displacement vs the
# original is visually obvious. action_1's default already shows it clearly; we
# fix all three for reproducibility.
RANDOM_ROT_DEG = {0: 150.0, 1: 130.0, 2: 150.0}


def rotate_for_display(traj_mm, objs_mm, deg):
    """Rotate a (T,3) trajectory and (n,3) object positions jointly about their
    shared centroid on the z-axis by `deg` degrees (same geometry as the
    training-time random rotation, but with a fixed, clearly-visible angle)."""
    from scipy.spatial.transform import Rotation as R
    pts = np.vstack([traj_mm, objs_mm])
    center = pts.mean(axis=0)
    Rz = R.from_euler("z", deg, degrees=True).as_matrix()
    def rot(a):
        return (a - center) @ Rz.T + center
    return rot(traj_mm), rot(objs_mm)


ARM_FNAME = {"No-aug": "none", "Random rotation": "random", "TP-aug": "tp"}


def _draw_panel(ax, ref, traj, objs, obj_order, arm_color, lo, hi):
    """Draw one bare 3D panel: gray dashed reference + arm line + objects."""
    ax.plot(*ref.T, color=REF_COLOR, lw=2.0, ls="--", alpha=0.9, zorder=2)
    if traj is not None:
        ax.plot(*traj.T, color=arm_color, lw=2.5, zorder=3)
        ax.scatter(*traj[0], color=arm_color, s=70, marker="o", edgecolor="k", zorder=5)
        ax.scatter(*traj[-1], color=arm_color, s=90, marker="x", linewidths=2.5, zorder=5)
    for i, nm in enumerate(obj_order):
        ax.scatter(*objs[i], color=OBJ_COLOR[nm], s=70, marker="s", edgecolor="k", zorder=7)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(**VIEW)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.set_xlabel("x (mm)", fontsize=15, labelpad=-8)
    ax.set_ylabel("y (mm)", fontsize=15, labelpad=-8)
    ax.set_zlabel("z (mm)", fontsize=15, labelpad=-8)


def make_fig(action, action_idx):
    si, seed, demo = best_cell(action, action_idx)
    e = dset[action][si]
    mean = np.asarray(e["train_stat"]["mean"]).reshape(3); std = float(e["train_stat"]["std"])
    gt = np.asarray(e["test_traj_global"])[demo][:, :3] * std + mean
    gt_objs = np.asarray(e["HTs_test"])[demo, :4, :3, 3] * std + mean

    # Training row: reference = raw (unaugmented) demo; arm = the trajectory the
    # model trains on (red).
    raw = train_traj_objs("none", seed, action_idx)
    r1_ref = raw[0]
    r1 = {"No-aug": raw, "TP-aug": train_traj_objs("tp", seed, action_idx)}
    rt, ro = rotate_for_display(raw[0], raw[1], RANDOM_ROT_DEG[action_idx])
    r1["Random rotation"] = (rt, ro)

    # Prediction row: reference = test GT; arm = prediction (blue).
    r2_ref = gt
    r2 = {}
    for col, arm in [("No-aug", "none"), ("Random rotation", "random"), ("TP-aug", "tp")]:
        p = IMP / arm / str(seed) / "predictions.pickle"
        arr = _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))
        r2[col] = (arr[demo][:, :3] * std + mean, gt_objs)

    def limits(rowdata, ref):
        allpts = ([ref] + [t for (t, _) in rowdata.values() if t is not None]
                  + [o for (_, o) in rowdata.values()])
        allpts = np.vstack(allpts)
        lo, hi = allpts.min(0), allpts.max(0)
        pad = (hi - lo) * 0.02 + 1
        return lo - pad, hi + pad

    rows = [("train", r1, r1_ref, DATASET_OBJ_ORDER, AUG_COLOR, limits(r1, r1_ref)),
            ("pred", r2, r2_ref, HTS_OBJ_ORDER, PRED_COLOR, limits(r2, r2_ref))]
    for row_name, rowdata, ref, obj_order, arm_color, (lo, hi) in rows:
        for col in COLS:
            traj, objs = rowdata[col]
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            _draw_panel(ax, ref, traj, objs, obj_order, arm_color, lo, hi)
            fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
            out = OUT / f"aug3d_{action}_{row_name}_{ARM_FNAME[col]}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}  (seed {seed}, demo {demo})")


def make_legend():
    """One shared horizontal legend strip for the 18-panel augmentation grid."""
    handles = [
        Line2D([], [], color=REF_COLOR, lw=2.0, ls="--", label="Reference (GT)"),
        Line2D([], [], color=AUG_COLOR, lw=2.5, label="Augmented demo"),
        Line2D([], [], color=PRED_COLOR, lw=2.5, label="Prediction"),
        Line2D([], [], color="none", marker="o", markerfacecolor="0.7",
               markeredgecolor="k", markersize=9, label="start"),
        Line2D([], [], color="none", marker="x", markeredgecolor="0.3",
               markeredgewidth=2.0, markersize=9, label="end"),
        Line2D([], [], color="none", marker="s", markerfacecolor=OBJ_COLOR["bolt"],
               markeredgecolor="k", markersize=11, label="bolt"),
        Line2D([], [], color="none", marker="s", markerfacecolor=OBJ_COLOR["nut"],
               markeredgecolor="k", markersize=11, label="nut"),
        Line2D([], [], color="none", marker="s", markerfacecolor=OBJ_COLOR["bin"],
               markeredgecolor="k", markersize=11, label="bin"),
        Line2D([], [], color="none", marker="s", markerfacecolor=OBJ_COLOR["jig"],
               markeredgecolor="k", markersize=11, label="jig"),
    ]
    fig = plt.figure(figsize=(15, 0.6))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               fontsize=15, handletextpad=0.4, columnspacing=1.3)
    out = OUT / "aug3d_legend.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


for ai, action in enumerate(["action_0", "action_1", "action_2"]):
    make_fig(action, ai)
make_legend()
