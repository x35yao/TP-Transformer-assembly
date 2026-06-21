"""Per-subtask 3D figure comparing TP-Transformer at K=1, K=5, K=15.

One PNG per subtask (action): a single 3D axis showing
  - ground truth                red    dashed (reference)
  - TP-Transformer K=1          light blue solid
  - TP-Transformer K=5          medium blue solid
  - TP-Transformer K=15         dark blue solid
Objects are squares (bolt green, nut yellow, bin black, jig purple);
trajectory start = circle, end = cross. Same fixed flat view and the same
(seed, demo) cell per subtask as plot_sample_traj_3d (so the scene matches).
A separate shared legend strip is emitted as k1k5k15_legend.

Output: results/figures/k1vk15_<action>.{png,eps}  (+ k1vk15_legend.{png,eps})
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from evaluate_predictions import _to_3d

USER = os.environ["USER"]
EVAL = Path(f"/shared/{USER}/RingAIAutoAnnotation/eval")
DATA = "baselines/data/baseline_dataset_n15_v3t3.pickle"
OUT = EVAL / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [9871, 9872, 9873, 9874, 9875]

OBJ_COLOR = {"bolt": "#2ca02c", "nut": "#e8d100", "bin": "#000000", "jig": "#9467bd"}
# HTs_test object axis is alphabetical (prepare_baseline_dataset sorts the list).
HTS_OBJ_ORDER = ["bin", "bolt", "jig", "nut"]
GT_COLOR = "#d62728"   # red, dashed reference
# Blue ramp: more demonstrations -> darker blue.
K_VALUES = [1, 5, 15]
K_COLOR = {1: "#9ecae1", 5: "#4292c6", 15: "#08519c"}
VIEW = dict(elev=22, azim=-60)

# Same cell per subtask as plot_sample_traj_3d.
CELL_OVERRIDE = {"action_0": (4, 0), "action_1": (3, 0), "action_2": (2, 0)}

dset = pickle.load(open(DATA, "rb"))


def k_pred(action, seed, K):
    """TP-Transformer prediction at a given K for this action/seed."""
    if K == 15:
        p = EVAL / "exp2" / "archive_important_dist" / "tp" / str(seed) / "predictions.pickle"
    else:
        p = EVAL / "exp1" / "archive_important_dist" / "tp_transformer" / str(K) / str(seed) / "predictions.pickle"
    if not p.exists():
        return None
    return _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))


def make_fig(action, action_idx):
    si, demo = CELL_OVERRIDE[action]
    seed = SEEDS[si]
    e = dset[action][si]
    mean = np.asarray(e["train_stat"]["mean"]).reshape(3); std = float(e["train_stat"]["std"])
    gt = np.asarray(e["test_traj_global"])[demo][:, :3] * std + mean
    objs = np.asarray(e["HTs_test"])[demo, :4, :3, 3] * std + mean

    preds = {}
    for K in K_VALUES:
        arr = k_pred(action, seed, K)
        if arr is not None:
            preds[K] = arr[demo][:, :3] * std + mean

    allpts = [gt, objs] + list(preds.values())
    allpts = np.vstack(allpts)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = (hi - lo) * 0.02 + 1
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Ground truth: red dashed reference.
    ax.plot(*gt.T, color=GT_COLOR, lw=2.0, ls="--", zorder=3)
    ax.scatter(*gt[0], color=GT_COLOR, s=70, marker="o", edgecolor="k", zorder=5)
    ax.scatter(*gt[-1], color=GT_COLOR, s=90, marker="x", linewidths=2.5, zorder=5)
    # K variants: light -> dark blue.
    for j, K in enumerate(K_VALUES):
        if K not in preds:
            continue
        pr = preds[K]
        col = K_COLOR[K]
        ax.plot(*pr.T, color=col, lw=2.5, zorder=4 + j)
        ax.scatter(*pr[0], color=col, s=70, marker="o", edgecolor="k", zorder=6 + j)
        ax.scatter(*pr[-1], color=col, s=90, marker="x", linewidths=2.5, zorder=6 + j)
    for i, nm in enumerate(HTS_OBJ_ORDER):
        ax.scatter(*objs[i], color=OBJ_COLOR[nm], s=80, marker="s", edgecolor="k", zorder=9)

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(**VIEW)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.set_xlabel("x (mm)", fontsize=15, labelpad=-8)
    ax.set_ylabel("y (mm)", fontsize=15, labelpad=-8)
    ax.set_zlabel("z (mm)", fontsize=15, labelpad=-8)
    ax.set_title(f"Subtask {action_idx + 1}", fontsize=22, pad=2)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.02)
    out = OUT / f"k1vk15_{action}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (seed {seed}, demo {demo})")


def make_legend():
    """One shared horizontal legend strip for the K=1/5/15 comparison."""
    handles = [
        Line2D([], [], color=GT_COLOR, lw=2.0, ls="--", label="Ground truth"),
        Line2D([], [], color=K_COLOR[1], lw=2.5, label="TP-Transformer (K=1)"),
        Line2D([], [], color=K_COLOR[5], lw=2.5, label="TP-Transformer (K=5)"),
        Line2D([], [], color=K_COLOR[15], lw=2.5, label="TP-Transformer (K=15)"),
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
    fig = plt.figure(figsize=(16, 0.6))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               fontsize=14, handletextpad=0.4, columnspacing=1.2)
    out = OUT / "k1vk15_legend.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


for ai, action in enumerate(["action_0", "action_1", "action_2"]):
    make_fig(action, ai)
make_legend()
