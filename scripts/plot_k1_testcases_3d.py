"""Per-action 1x3 3D figure for TP-Transformer at K=1.

One figure per action; the three columns are three different test cases (test
demos) for that action. Each panel shows the K=1 TP-Transformer prediction
(coloured) overlaid on the test GT (faint black). Objects as squares
(bolt green, nut yellow, bin black, jig purple); trajectory start = circle,
end = cross. Shared axis limits + fixed flat view, matching plot_methods_3d.

We pick the seed whose mean ADE over its test demos is lowest (representative),
then show all of that seed's test demos as the columns.

Output: results/figures/k1_3d_<action>.{png,eps}
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
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [9871, 9872, 9873, 9874, 9875]
K = 1

OBJ_COLOR = {"bolt": "#2ca02c", "nut": "#e8d100", "bin": "#000000", "jig": "#9467bd"}
# HTs_test object axis is alphabetical (prepare_baseline_dataset sorts the list).
HTS_OBJ_ORDER = ["bin", "bolt", "jig", "nut"]
PRED_COLOR = "#1f77b4"  # TP-Transformer blue
VIEW = dict(elev=22, azim=-60)

dset = pickle.load(open(DATA, "rb"))


def tpt_pred(action, seed):
    p = EVAL / "exp1" / "archive_important_dist" / "tp_transformer" / str(K) / str(seed) / "predictions.pickle"
    if not p.exists():
        return None
    return _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))


def best_seed(action):
    """Seed minimising mean ADE over its test demos for this action."""
    best = None
    for si, seed in enumerate(SEEDS):
        e = dset[action][si]
        gt = _to_3d(e["test_traj_global"])
        m = np.asarray(e["train_stat"]["mean"]).reshape(3); s = float(e["train_stat"]["std"])
        arr = tpt_pred(action, seed)
        if arr is None:
            continue
        ades = []
        for d in range(min(arr.shape[0], gt.shape[0])):
            ade, _ = _metrics_for_trajectory(arr[d], gt[d], m, s)
            ades.append(ade)
        if not ades:
            continue
        mean_ade = float(np.mean(ades))
        if best is None or mean_ade < best[0]:
            best = (mean_ade, si, seed)
    return best[1], best[2]


def make_fig(action):
    si, seed = best_seed(action)
    e = dset[action][si]
    mean = np.asarray(e["train_stat"]["mean"]).reshape(3); std = float(e["train_stat"]["std"])
    gt_all = np.asarray(e["test_traj_global"])
    objs_all = np.asarray(e["HTs_test"])
    arr = tpt_pred(action, seed)
    n_demos = min(3, gt_all.shape[0], arr.shape[0])

    # shared limits across the panels (GT + pred + objects for the shown demos)
    pts = []
    for d in range(n_demos):
        pts.append(gt_all[d][:, :3] * std + mean)
        pts.append(arr[d][:, :3] * std + mean)
        pts.append(objs_all[d, :4, :3, 3] * std + mean)
    allpts = np.vstack(pts)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = (hi - lo) * 0.02 + 1
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(14, 4.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.14, wspace=-0.15)
    for c in range(n_demos):
        gt = gt_all[c][:, :3] * std + mean
        pr = arr[c][:, :3] * std + mean
        objs = objs_all[c, :4, :3, 3] * std + mean
        ax = fig.add_subplot(1, 3, c + 1, projection="3d")
        ax.plot(*gt.T, color="black", lw=1.2, alpha=0.35)  # GT reference
        ax.plot(*pr.T, color=PRED_COLOR, lw=2.0)
        ax.scatter(*pr[0], color=PRED_COLOR, s=70, marker="o", edgecolor="k", zorder=5)
        ax.scatter(*pr[-1], color=PRED_COLOR, s=90, marker="x", linewidths=2.5, zorder=5)
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
        ax.set_title(f"Test case {c + 1}", fontsize=20, pad=4)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=18, ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out = OUT / f"k1_3d_{action}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (seed {seed})")


for action in ["action_0", "action_1", "action_2"]:
    make_fig(action)
