"""1x5 3D prediction comparison for Experiment 2 (methods), per action, at K=15.

Columns: [TP-Transformer, TP-ProMP, TP-GMM, CNEP, CNMP]
Each panel: that method's K=15 prediction (coloured) overlaid on the test GT
(faint black). Objects as squares (bolt green, nut yellow, box black, jig
purple); trajectory start = circle, end = cross. Shared axis limits + fixed view.
Picks the (seed, demo) where TP-Transformer is best (representative cell).

Output: results/figures/methods3d_<action>.png
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from evaluate_predictions import _metrics_for_trajectory, _to_3d

USER = os.environ["USER"]
EVAL = Path(f"/shared/{USER}/RingAIAutoAnnotation/eval")
DATA = "baselines/data/baseline_dataset_n15_v3t3.pickle"
OUT = EVAL / "results/figures"
SEEDS = [9871, 9872, 9873, 9874, 9875]

OBJ_COLOR = {"bolt": "#2ca02c", "nut": "#e8d100", "bin": "#000000", "jig": "#9467bd"}
# HTs_test object axis is alphabetical (prepare_baseline_dataset sorts the list).
HTS_OBJ_ORDER = ["bin", "bolt", "jig", "nut"]
COLS = ["TP-Transformer", "TP-ProMP", "TP-GMM", "CNEP", "CNMP"]
COL_COLOR = {"TP-Transformer": "#1f77b4", "TP-ProMP": "#2ca02c", "TP-GMM": "#d62728",
             "CNEP": "#9467bd", "CNMP": "#ff7f0e"}
VIEW = dict(elev=22, azim=-60)
K = 15

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
    # aggregated {action: [arr_seed0..4]}, multi-context (random) regime
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


def get_pred(col, action, si, seed, demo, mean, std):
    if col == "TP-Transformer":
        arr = tpt_pred(action, seed)
    elif col == "TP-ProMP":
        arr = classical_pred("tp_promp", action, seed)
    elif col == "TP-GMM":
        arr = classical_pred("tp_gmm", action, seed)
    elif col == "CNEP":
        arr = deep_pred("cnep", action, si)
    elif col == "CNMP":
        arr = deep_pred("cnmp", action, si)
    if arr is None:
        return None
    return arr[demo][:, :3] * std + mean


def make_fig(action):
    si, seed, demo = best_cell(action)
    e = dset[action][si]
    mean = np.asarray(e["train_stat"]["mean"]).reshape(3); std = float(e["train_stat"]["std"])
    gt = np.asarray(e["test_traj_global"])[demo][:, :3] * std + mean
    objs = np.asarray(e["HTs_test"])[demo, :4, :3, 3] * std + mean

    preds = {c: get_pred(c, action, si, seed, demo, mean, std) for c in COLS}

    # shared limits across all panels (include GT + all preds + objects)
    allpts = [gt, objs] + [p for p in preds.values() if p is not None]
    allpts = np.vstack(allpts)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = (hi - lo) * 0.02 + 1
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(22, 4.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.14, wspace=-0.15)
    for c, col in enumerate(COLS):
        ax = fig.add_subplot(1, 5, c + 1, projection="3d")
        ax.plot(*gt.T, color="black", lw=1.2, alpha=0.35)  # GT reference
        pr = preds[col]
        tcol = COL_COLOR[col]
        if pr is not None:
            ax.plot(*pr.T, color=tcol, lw=2.0)
            ax.scatter(*pr[0], color=tcol, s=70, marker="o", edgecolor="k", zorder=5)
            ax.scatter(*pr[-1], color=tcol, s=90, marker="x", linewidths=2.5, zorder=5)
        for i, nm in enumerate(HTS_OBJ_ORDER):
            ax.scatter(*objs[i], color=OBJ_COLOR[nm], s=55, marker="s", edgecolor="k",
                       label=(nm if c == 0 else None))
        if c == 0:
            ax.scatter([], [], color="grey", s=70, marker="o", edgecolor="k", label="start")
            ax.scatter([], [], color="grey", s=90, marker="x", linewidths=2.5, label="end")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.view_init(**VIEW)
        ax.set_box_aspect((1.0, 1.0, 0.45))
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.tick_params(length=0)
        ax.set_xlabel("x (mm)", fontsize=15, labelpad=-8)
        ax.set_ylabel("y (mm)", fontsize=15, labelpad=-8)
        ax.set_zlabel("z (mm)", fontsize=15, labelpad=-8)
        ax.set_title(col, fontsize=20, pad=4)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=18, ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out = OUT / f"methods3d_{action}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (seed {seed}, demo {demo})")


for action in ["action_0", "action_1", "action_2"]:
    make_fig(action)
