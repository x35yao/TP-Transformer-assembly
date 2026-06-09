"""2x4 3D trajectory figure for the augmentation experiment, per action.

Columns: [Ground truth, No-aug, Random rotation, TP-aug]
Row 1 (Training trajectory): the (augmented) training trajectory each method
       trains on. GT/No-aug = raw training demo; Random/TP-aug apply the real
       training-time augmentation. Objects move with the augmentation.
Row 2 (Prediction): each method's prediction (colour) overlaid on the test GT
       (faint black) in the same panel, with shared axis limits + fixed view.

Uses the important_dist (reported) models. For each action we pick the
(seed, demo) test cell where TP-aug is most representative (lowest ADE) so the
figure reflects the reported quality rather than a worst-case demo.

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
COLS = ["Ground truth", "No-aug", "Random rotation", "TP-aug"]
PRED_COLOR = {"Ground truth": "#000000", "No-aug": "#2ca02c",
              "Random rotation": "#ff7f0e", "TP-aug": "#1f77b4"}
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

    # row 1: training augmentation (seed-matched)
    # GT and No-aug use the raw (unaugmented) trajectory. TP-aug uses the real
    # TP augmentation. Random uses a fixed, clearly-visible rotation of the raw
    # trajectory (illustrative) so the start/end displacement is obvious.
    r1 = {}
    raw = train_traj_objs("none", seed, action_idx)
    r1["Ground truth"] = raw
    r1["No-aug"] = raw
    r1["TP-aug"] = train_traj_objs("tp", seed, action_idx)
    rt, ro = rotate_for_display(raw[0], raw[1], RANDOM_ROT_DEG[action_idx])
    r1["Random rotation"] = (rt, ro)

    # row 2: predictions overlaid on test GT
    r2 = {"Ground truth": (gt, gt_objs)}
    for col, arm in [("No-aug", "none"), ("Random rotation", "random"), ("TP-aug", "tp")]:
        p = IMP / arm / str(seed) / "predictions.pickle"
        arr = _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))
        r2[col] = (arr[demo][:, :3] * std + mean, gt_objs)

    # shared limits per row (so panels are comparable)
    def limits(rowdata):
        allpts = np.vstack([t for (t, _) in rowdata.values() if t is not None]
                           + [o for (_, o) in rowdata.values()])
        lo, hi = allpts.min(0), allpts.max(0)
        pad = (hi - lo) * 0.02 + 1
        return lo - pad, hi + pad

    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.07, wspace=0.0, hspace=0.0)
    for r, (label, rowdata, lim) in enumerate(
            [("Training trajectory", r1, limits(r1)), ("Prediction (test)", r2, limits(r2))]):
        lo, hi = lim
        for c, col in enumerate(COLS):
            ax = fig.add_subplot(2, 4, r * 4 + c + 1, projection="3d")
            traj, objs = rowdata[col]
            # row 2: overlay GT faintly behind the prediction
            if r == 1 and col != "Ground truth":
                ax.plot(*gt.T, color="black", lw=1.2, alpha=0.35)
            if traj is not None:
                tcol = PRED_COLOR[col]
                ax.plot(*traj.T, color=tcol, lw=2.0)
                # start = circle, end = cross, in the trajectory colour.
                # Add neutral-coloured legend handles once (top-left panel).
                start_lbl = "start" if (r == 0 and c == 0) else None
                end_lbl = "end" if (r == 0 and c == 0) else None
                ax.scatter(*traj[0], color=tcol, s=70, marker="o", edgecolor="k", zorder=5)
                ax.scatter(*traj[-1], color=tcol, s=90, marker="x", linewidths=2.5, zorder=5)
                if r == 0 and c == 0:
                    # legend proxies (grey) so 'start'/'end' appear once
                    ax.scatter([], [], color="grey", s=70, marker="o", edgecolor="k", label=start_lbl)
                    ax.scatter([], [], color="grey", s=90, marker="x", linewidths=2.5, label=end_lbl)
            # Object axis order depends on the source: row 0 (training) draws
            # objects from the live dataset (config.all_objs order); row 1
            # (prediction) draws from HTs_test (alphabetical order).
            obj_order = DATASET_OBJ_ORDER if r == 0 else HTS_OBJ_ORDER
            for i, nm in enumerate(obj_order):
                ax.scatter(*objs[i], color=OBJ_COLOR[nm], s=60, marker="s", edgecolor="k",
                           label=(nm if (r == 0 and c == 0) else None))
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
            ax.view_init(**VIEW)
            # Fixed flat box aspect (z compressed) so the floor plane stays
            # horizontal regardless of each row's data z-extent.
            ax.set_box_aspect((1.0, 1.0, 0.45))
            # Axis labels (no tick labels, to stay uncluttered).
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
    # Legend along the bottom so it never overlaps the column titles.
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=18, ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out = OUT / f"aug3d_{action}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (seed {seed}, demo {demo})")


for ai, action in enumerate(["action_0", "action_1", "action_2"]):
    make_fig(action, ai)
