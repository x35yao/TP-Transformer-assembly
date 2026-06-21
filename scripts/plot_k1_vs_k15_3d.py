"""Per-subtask 3D figure comparing TP-Transformer at K=1 vs K=15.

One PNG per subtask (action): a single 3D axis showing
  - ground truth                gray   dashed
  - TP-Transformer K=1          light blue solid
  - TP-Transformer K=15         blue   solid
Objects are squares (bolt green, nut yellow, bin black, jig purple);
trajectory start = circle, end = cross. Same fixed flat view and the same
(seed, demo) cell per subtask as plot_sample_traj_3d (so the scene matches).

Output: results/figures/k1vk15_<action>.{png,eps}
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
K15_COLOR = "#1f77b4"   # blue
K1_COLOR = "#9ecae1"    # light blue
VIEW = dict(elev=22, azim=-60)

# Same cell per subtask as plot_sample_traj_3d.
CELL_OVERRIDE = {"action_0": (4, 0), "action_1": (3, 0), "action_2": (2, 0)}

dset = pickle.load(open(DATA, "rb"))


def k15_pred(action, seed):
    p = EVAL / "exp2" / "archive_important_dist" / "tp" / str(seed) / "predictions.pickle"
    return _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))


def k1_pred(action, seed):
    p = EVAL / "exp1" / "archive_important_dist" / "tp_transformer" / "1" / str(seed) / "predictions.pickle"
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
    p15 = k15_pred(action, seed)[demo][:, :3] * std + mean
    a1 = k1_pred(action, seed)
    p1 = a1[demo][:, :3] * std + mean if a1 is not None else None

    allpts = [gt, objs, p15] + ([p1] if p1 is not None else [])
    allpts = np.vstack(allpts)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = (hi - lo) * 0.02 + 1
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(*gt.T, color="0.5", lw=2.0, ls="--", label="Ground truth")
    ax.scatter(*gt[0], color="0.5", s=70, marker="o", edgecolor="k", zorder=5)
    ax.scatter(*gt[-1], color="0.5", s=90, marker="x", linewidths=2.5, zorder=5)
    if p1 is not None:
        ax.plot(*p1.T, color=K1_COLOR, lw=2.5, label="TP-Transformer (K=1)")
        ax.scatter(*p1[0], color=K1_COLOR, s=70, marker="o", edgecolor="k", zorder=6)
        ax.scatter(*p1[-1], color=K1_COLOR, s=90, marker="x", linewidths=2.5, zorder=6)
    ax.plot(*p15.T, color=K15_COLOR, lw=2.5, label="TP-Transformer (K=15)")
    ax.scatter(*p15[0], color=K15_COLOR, s=70, marker="o", edgecolor="k", zorder=7)
    ax.scatter(*p15[-1], color=K15_COLOR, s=90, marker="x", linewidths=2.5, zorder=7)
    for i, nm in enumerate(HTS_OBJ_ORDER):
        ax.scatter(*objs[i], color=OBJ_COLOR[nm], s=80, marker="s", edgecolor="k",
                   zorder=8, label=nm)

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(**VIEW)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.set_xlabel("x (mm)", fontsize=15, labelpad=-8)
    ax.set_ylabel("y (mm)", fontsize=15, labelpad=-8)
    ax.set_zlabel("z (mm)", fontsize=15, labelpad=-8)
    ax.set_title(f"Subtask {action_idx + 1}", fontsize=22, pad=2)

    # Two-row legend: trajectory lines on top, object squares below.
    line_handles = [
        Line2D([], [], color="0.5", lw=2.0, ls="--", label="Ground truth"),
        Line2D([], [], color=K1_COLOR, lw=2.5, label="TP-Transformer (K=1)"),
        Line2D([], [], color=K15_COLOR, lw=2.5, label="TP-Transformer (K=15)"),
    ]
    obj_handles = [
        Line2D([], [], color="none", marker="s", markerfacecolor=OBJ_COLOR[nm],
               markeredgecolor="k", markersize=11, label=nm)
        for nm in ["bolt", "nut", "bin", "jig"]
    ]
    leg1 = fig.legend(handles=line_handles, loc="lower center", fontsize=13,
                      ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.07))
    fig.add_artist(leg1)
    fig.legend(handles=obj_handles, loc="lower center", fontsize=13,
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.14)
    out = OUT / f"k1vk15_{action}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (seed {seed}, demo {demo})")


for ai, action in enumerate(["action_0", "action_1", "action_2"]):
    make_fig(action, ai)
