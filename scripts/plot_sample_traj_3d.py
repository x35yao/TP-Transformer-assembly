"""Single-method 3D sample-trajectory panels (K=15), for a 5x3 LaTeX grid.

One bare PNG/EPS per (subtask, method): a single 3D axis showing the
ground-truth trajectory (red dashed -- a visualization reference)
and that method's prediction (solid, method colour), with objects as squares
(bolt green, nut yellow, bin black, jig purple) and start=circle / end=cross
markers. The panels carry NO title and NO legend (those are added once in
LaTeX: column headers = subtasks, row labels = methods, plus a single shared
legend image) but DO keep the x/y/z (mm) axis labels.

We pick the (seed, demo) cell per subtask (see CELL_OVERRIDE) and use the same
cell for every method so the panels are comparable.

Outputs (results/figures/):
  sample3d_<action>_<method>.{png,eps}   bare panels (15)
  sample3d_legend.{png,eps}              one shared legend strip
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
from evaluate_predictions import _metrics_for_trajectory, _to_3d

USER = os.environ["USER"]
EVAL = Path(f"/shared/{USER}/RingAIAutoAnnotation/eval")
DATA = "baselines/data/baseline_dataset_n15_v3t3.pickle"
OUT = EVAL / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [9871, 9872, 9873, 9874, 9875]
K = 15

# Object square colours (squares, so they don't conflict with the dashed
# trajectory lines): bolt green, nut yellow, bin black, jig purple.
OBJ_COLOR = {"bolt": "#2ca02c", "nut": "#e8d100", "bin": "#000000", "jig": "#9467bd"}
# HTs_test object axis is alphabetical (prepare_baseline_dataset sorts the list).
HTS_OBJ_ORDER = ["bin", "bolt", "jig", "nut"]
# Global method colour convention (shared across the paper figures).
GT_COLOR = "#d62728"  # red, dashed -- a visualization reference line
METHODS = ["TP-Transformer", "TP-GMM", "TP-ProMP", "CNEP", "CNMP"]
METHOD_COLOR = {"TP-Transformer": "#1f77b4", "TP-GMM": "#2ca02c", "TP-ProMP": "#9467bd",
                "CNEP": "#17becf", "CNMP": "#ff7f0e"}
VIEW = dict(elev=22, azim=-60)

dset = pickle.load(open(DATA, "rb"))


def tpt_pred(action, seed):
    p = EVAL / "exp2" / "archive_important_dist" / "tp" / str(seed) / "predictions.pickle"
    return _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))


def classical_pred(method, action, seed):
    p = EVAL / "exp1" / method / str(K) / str(seed) / "predictions.pickle"
    if not p.exists():
        return None
    return _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))


def deep_pred(method, action, seed_idx):
    p = EVAL / "exp1" / "predictions" / method / "random" / str(K) / f"{method}_predictions.pickle"
    if not p.exists():
        return None
    arrs = pickle.load(open(p, "rb"))[action]
    return _to_3d(np.asarray(arrs[seed_idx]))


def best_cell(action):
    best = None
    for si, seed in enumerate(SEEDS):
        e = dset[action][si]
        gt = _to_3d(e["test_traj_global"]); m = np.asarray(e["train_stat"]["mean"]).reshape(3); s = float(e["train_stat"]["std"])
        arr = tpt_pred(action, seed)
        for d in range(min(arr.shape[0], gt.shape[0])):
            ade, _ = _metrics_for_trajectory(arr[d], gt[d], m, s)
            if best is None or ade < best[0]:
                best = (ade, si, seed, d)
    return best[1], best[2], best[3]


# Manual (seed_idx, demo) override per action so the chosen test example shows
# the trajectory/objects clearly in the fixed view. None -> auto best_cell.
# best seeds: action_0=9875(idx4), action_1=9874(idx3), action_2=9873(idx2).
# demo 0 reads cleanest in the fixed view (objects in front, traj visible).
CELL_OVERRIDE = {
    "action_0": (4, 0),
    "action_1": (3, 0),
    "action_2": (2, 0),
}


def pick_cell(action):
    ov = CELL_OVERRIDE.get(action)
    if ov is not None:
        si, demo = ov
        return si, SEEDS[si], demo
    return best_cell(action)


def get_pred(method, action, si, seed, demo, mean, std):
    if method == "TP-Transformer":
        arr = tpt_pred(action, seed)
    elif method == "TP-ProMP":
        arr = classical_pred("tp_promp", action, seed)
    elif method == "TP-GMM":
        arr = classical_pred("tp_gmm", action, seed)
    elif method == "CNEP":
        arr = deep_pred("cnep", action, si)
    elif method == "CNMP":
        arr = deep_pred("cnmp", action, si)
    if arr is None:
        return None
    return arr[demo][:, :3] * std + mean


def make_figs(action, action_idx):
    si, seed, demo = pick_cell(action)
    e = dset[action][si]
    mean = np.asarray(e["train_stat"]["mean"]).reshape(3); std = float(e["train_stat"]["std"])
    gt = np.asarray(e["test_traj_global"])[demo][:, :3] * std + mean
    objs = np.asarray(e["HTs_test"])[demo, :4, :3, 3] * std + mean

    preds = {m: get_pred(m, action, si, seed, demo, mean, std) for m in METHODS}

    # shared limits (GT + all preds + objects) so every method panel for this
    # subtask uses identical axes.
    allpts = [gt, objs] + [p for p in preds.values() if p is not None]
    allpts = np.vstack(allpts)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = (hi - lo) * 0.02 + 1
    lo, hi = lo - pad, hi + pad

    fname_method = {"TP-Transformer": "tp_transformer", "TP-GMM": "tp_gmm",
                    "TP-ProMP": "tp_promp", "CNEP": "cnep", "CNMP": "cnmp"}
    for m in METHODS:
        pr = preds[m]
        col = METHOD_COLOR[m]
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")

        # Ground truth: red dashed -- a visualization reference.
        ax.plot(*gt.T, color=GT_COLOR, lw=2.0, ls="--", alpha=0.9, zorder=2)
        ax.scatter(*gt[0], color=GT_COLOR, s=70, marker="o", edgecolor="k", zorder=5)
        ax.scatter(*gt[-1], color=GT_COLOR, s=90, marker="x", linewidths=2.5, zorder=5)
        # Prediction: solid, method colour.
        if pr is not None:
            ax.plot(*pr.T, color=col, lw=2.5, ls="-")
            ax.scatter(*pr[0], color=col, s=70, marker="o", edgecolor="k", zorder=6)
            ax.scatter(*pr[-1], color=col, s=90, marker="x", linewidths=2.5, zorder=6)
        # Objects: squares.
        for i, nm in enumerate(HTS_OBJ_ORDER):
            ax.scatter(*objs[i], color=OBJ_COLOR[nm], s=80, marker="s", edgecolor="k",
                       zorder=7)

        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.view_init(**VIEW)
        ax.set_box_aspect((1.0, 1.0, 0.45))
        # No titles/legends (added in LaTeX), but keep axis labels.
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.tick_params(length=0)
        ax.set_xlabel("x (mm)", fontsize=15, labelpad=-8)
        ax.set_ylabel("y (mm)", fontsize=15, labelpad=-8)
        ax.set_zlabel("z (mm)", fontsize=15, labelpad=-8)
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        out = OUT / f"sample3d_{action}_{fname_method[m]}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}  (seed {seed}, demo {demo})")


def make_legend():
    """One shared horizontal legend strip for the whole 5x3 grid."""
    handles = [
        Line2D([], [], color=GT_COLOR, lw=2.0, ls="--", label="Ground truth"),
        Line2D([], [], color="black", lw=2.5, ls="-", label="Prediction"),
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
    fig = plt.figure(figsize=(13, 0.6))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               fontsize=15, handletextpad=0.4, columnspacing=1.4)
    out = OUT / "sample3d_legend.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


for ai, action in enumerate(["action_0", "action_1", "action_2"]):
    make_figs(action, ai)
make_legend()
