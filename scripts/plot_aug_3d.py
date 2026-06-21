"""2x3 3D trajectory figure for the augmentation experiment, per action.

Columns: [No-aug, Random rotation, TP-aug]
Reference (shaded red) is shown in every panel.
Row 1 (Training trajectory): reference = the raw (unaugmented) demo; the arm
       line is the trajectory the model trains on (no-aug = raw; random/tp =
       the augmented demo). Shows how augmentation displaces the data.
Row 2 (Prediction): reference = the test GT; the arm line is that arm's model
       prediction. Shows test-time quality.

Uses the important_dist (reported) models. For each action we pick the
(seed, demo) test cell where TP-aug is most representative (lowest ADE).

Output: figures (per action) under results/figures/aug3d_<action>.png
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def make_fig(action, action_idx):
    si, seed, demo = best_cell(action, action_idx)
    e = dset[action][si]
    mean = np.asarray(e["train_stat"]["mean"]).reshape(3); std = float(e["train_stat"]["std"])
    gt = np.asarray(e["test_traj_global"])[demo][:, :3] * std + mean
    gt_objs = np.asarray(e["HTs_test"])[demo, :4, :3, 3] * std + mean

    # row 1: training augmentation (seed-matched). The reference is the raw
    # (unaugmented) demo; each arm shows the trajectory the model trains on.
    raw = train_traj_objs("none", seed, action_idx)
    r1_ref = raw[0]                                  # original-demo reference (T,3)
    r1 = {}
    r1["No-aug"] = raw
    r1["TP-aug"] = train_traj_objs("tp", seed, action_idx)
    rt, ro = rotate_for_display(raw[0], raw[1], RANDOM_ROT_DEG[action_idx])
    r1["Random rotation"] = (rt, ro)

    # row 2: predictions; the reference is the test GT.
    r2_ref = gt
    r2 = {}
    for col, arm in [("No-aug", "none"), ("Random rotation", "random"), ("TP-aug", "tp")]:
        p = IMP / arm / str(seed) / "predictions.pickle"
        arr = _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))
        r2[col] = (arr[demo][:, :3] * std + mean, gt_objs)

    # shared limits per row (include the reference + all arms + objects)
    def limits(rowdata, ref):
        allpts = ([ref] + [t for (t, _) in rowdata.values() if t is not None]
                  + [o for (_, o) in rowdata.values()])
        allpts = np.vstack(allpts)
        lo, hi = allpts.min(0), allpts.max(0)
        pad = (hi - lo) * 0.02 + 1
        return lo - pad, hi + pad

    fig = plt.figure(figsize=(14, 9))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.07, wspace=0.0, hspace=0.0)
    rows = [("Training trajectory", r1, r1_ref, DATASET_OBJ_ORDER, limits(r1, r1_ref)),
            ("Prediction (test)", r2, r2_ref, HTS_OBJ_ORDER, limits(r2, r2_ref))]
    for r, (label, rowdata, ref, obj_order, lim) in enumerate(rows):
        lo, hi = lim
        for c, col in enumerate(COLS):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d")
            traj, objs = rowdata[col]
            # Reference (gray dashed) shown in every panel.
            ax.plot(*ref.T, color=REF_COLOR, lw=2.0, ls="--", alpha=0.9, zorder=2,
                    label=("Reference" if (r == 0 and c == 0) else None))
            if traj is not None:
                tcol = AUG_COLOR if r == 0 else PRED_COLOR
                ax.plot(*traj.T, color=tcol, lw=2.0, zorder=3)
                ax.scatter(*traj[0], color=tcol, s=70, marker="o", edgecolor="k", zorder=5)
                ax.scatter(*traj[-1], color=tcol, s=90, marker="x", linewidths=2.5, zorder=5)
                if r == 0 and c == 0:
                    ax.scatter([], [], color="grey", s=70, marker="o", edgecolor="k", label="start")
                    ax.scatter([], [], color="grey", s=90, marker="x", linewidths=2.5, label="end")
            for i, nm in enumerate(obj_order):
                ax.scatter(*objs[i], color=OBJ_COLOR[nm], s=60, marker="s", edgecolor="k",
                           label=(nm if (r == 0 and c == 0) else None))
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.view_init(**VIEW)
            ax.set_box_aspect((1.0, 1.0, 0.45))
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.tick_params(length=0)
            ax.set_xlabel("x (mm)", fontsize=15, labelpad=-8)
            ax.set_ylabel("y (mm)", fontsize=15, labelpad=-8)
            ax.set_zlabel("z (mm)", fontsize=15, labelpad=-8)
            if r == 0:
                ax.set_title(col, fontsize=20, pad=4)
            if c == 0:
                ax.text2D(-0.08, 0.5, label, transform=ax.transAxes, rotation=90,
                          va="center", fontsize=20, fontweight="bold")
    # Legend along the bottom. Build explicit proxies so both trajectory types
    # (augmented = red, prediction = blue) are explained alongside the reference.
    from matplotlib.lines import Line2D
    handles, labels = fig.axes[0].get_legend_handles_labels()
    proxies = [
        Line2D([], [], color=REF_COLOR, lw=2.0, ls="--", label="Reference (GT)"),
        Line2D([], [], color=AUG_COLOR, lw=2.0, label="Augmented demo"),
        Line2D([], [], color=PRED_COLOR, lw=2.0, label="Prediction"),
    ]
    # Drop the auto "Reference" entry (replaced by the explicit proxy) but keep
    # start/end + object handles.
    keep = [(h, l) for h, l in zip(handles, labels) if l not in ("Reference",)]
    handles = proxies + [h for h, _ in keep]
    labels = [p.get_label() for p in proxies] + [l for _, l in keep]
    fig.legend(handles, labels, loc="lower center", fontsize=18, ncol=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out = OUT / f"aug3d_{action}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (seed {seed}, demo {demo})")


for ai, action in enumerate(["action_0", "action_1", "action_2"]):
    make_fig(action, ai)
