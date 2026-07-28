"""3D sample-trajectory panel for Diffusion Policy (per-subtask), single cell.

Mirrors scripts/plot_sample_traj_3d.py exactly (same objects/colours/view/cell)
but plots the DP prediction so we can eyeball how DP tracks the trajectory
relative to the TP-Transformer panel. Default: action_0, cell = TP-TF's
CELL_OVERRIDE (seed_idx 4 = 9875, demo 0).

Outputs (results/figures/):
  sample3d_<action>_dp_persub.{png,eps}
"""
import os, sys, pickle, argparse
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
K = 15

OBJ_COLOR = {"bolt": "#2ca02c", "nut": "#e8d100", "bin": "#000000", "jig": "#9467bd"}
HTS_OBJ_ORDER = ["bin", "bolt", "jig", "nut"]
GT_COLOR = "#d62728"
DP_COLOR = "#8c564b"   # brown -- new method colour, distinct from existing ones
VIEW = dict(elev=22, azim=-60)

# same cells the TP-Transformer panels use, so the DP panel is comparable.
CELL_OVERRIDE = {"action_0": (4, 0), "action_1": (3, 0), "action_2": (2, 0)}

dset = pickle.load(open(DATA, "rb"))


def dp_pred(model_dir, action, seed):
    p = EVAL / "exp1" / "predictions" / model_dir / "tp" / str(K) / str(seed) / "predictions.pickle"
    if not p.exists():
        return None
    return _to_3d(np.asarray(pickle.load(open(p, "rb"))[action]))


def make_fig(action, model_dir, tag):
    si, demo = CELL_OVERRIDE[action]
    seed = SEEDS[si]
    e = dset[action][si]
    mean = np.asarray(e["train_stat"]["mean"]).reshape(3); std = float(e["train_stat"]["std"])
    gt = np.asarray(e["test_traj_global"])[demo][:, :3] * std + mean
    objs = np.asarray(e["HTs_test"])[demo, :4, :3, 3] * std + mean

    arr = dp_pred(model_dir, action, seed)
    if arr is None:
        raise SystemExit(f"no DP predictions for {model_dir} {action} seed {seed}")
    pr = arr[demo][:, :3] * std + mean

    ade, _ = _metrics_for_trajectory(arr[demo], np.asarray(e["test_traj_global"])[demo], mean, std)

    allpts = np.vstack([gt, objs, pr])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = (hi - lo) * 0.02 + 1
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*gt.T, color=GT_COLOR, lw=2.0, ls="--", alpha=0.9, zorder=2)
    ax.scatter(*gt[0], color=GT_COLOR, s=70, marker="o", edgecolor="k", zorder=5)
    ax.scatter(*gt[-1], color=GT_COLOR, s=90, marker="x", linewidths=2.5, zorder=5)
    ax.plot(*pr.T, color=DP_COLOR, lw=2.5, ls="-")
    ax.scatter(*pr[0], color=DP_COLOR, s=70, marker="o", edgecolor="k", zorder=6)
    ax.scatter(*pr[-1], color=DP_COLOR, s=90, marker="x", linewidths=2.5, zorder=6)
    for i, nm in enumerate(HTS_OBJ_ORDER):
        ax.scatter(*objs[i], color=OBJ_COLOR[nm], s=80, marker="s", edgecolor="k", zorder=7)

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.view_init(**VIEW)
    ax.set_box_aspect((1.0, 1.0, 0.45))
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.set_xlabel("x (mm)", fontsize=15, labelpad=-8)
    ax.set_ylabel("y (mm)", fontsize=15, labelpad=-8)
    ax.set_zlabel("z (mm)", fontsize=15, labelpad=-8)
    ax.set_title(f"DP ({tag})  {action}  ADE={ade:.1f} mm", fontsize=12)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.0)
    out = OUT / f"sample3d_{action}_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (seed {seed}, demo {demo}, ADE={ade:.1f} mm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="action_0")
    ap.add_argument("--model-dir", default="dpB_persub", help="predictions/<model-dir>/tp/<K>/...")
    ap.add_argument("--tag", default="dp_persub", help="output filename suffix")
    args = ap.parse_args()
    make_fig(args.action, args.model_dir, args.tag)
