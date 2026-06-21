"""Quaternion-vs-time overlay for Subtask 3 (action_2) at K=15.

Four stacked rows (qx, qy, qz, qw). Ground truth solid black; each method
overlaid in its colour. Uses the same (seed, demo) cell as the sample-traj
figures so the example matches. Quaternion sign is aligned to GT per demo to
avoid spurious flips from the q/-q double cover.

Output: results/figures/ori_sample_traj_action_2.{png,eps}
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
# Reuse the SAME quaternion processing as the evaluator that produced the
# reported NDQ numbers, so the figure is consistent with them.
from evaluate_predictions import process_quaternions

USER = os.environ["USER"]
EVAL = Path(f"/shared/{USER}/RingAIAutoAnnotation/eval")
DATA = "baselines/data/baseline_dataset_n15_v3t3.pickle"
OUT = EVAL / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [9871, 9872, 9873, 9874, 9875]
K = 15
ACTION = "action_2"
CELL = (2, 0)  # (seed_idx, demo) -- same as sample-traj Subtask 3

QUAT_COLS = [3, 4, 5, 6]
# Quaternions are stored scalar-LAST [qx,qy,qz,qw]. Verified physically: during
# the screwing phase (t~100-150 in Subtask 3) the relative rotation under this
# ordering is purely about +z, matching the screw motion.
QUAT_NAMES = ["$q_x$", "$q_y$", "$q_z$", "$q_w$"]
# Global method colour convention (matches the paper's other figures).
GT_COLOR = "#d62728"   # red, solid
# Method -> (display, colour); all methods drawn dashed.
METHODS = [
    ("TP-TF",    "#1f77b4"),   # blue
    ("TP-GMM",   "#2ca02c"),   # green
    ("TP-ProMP", "#9467bd"),   # purple
    ("CNEP",     "#17becf"),   # cyan
    ("CNMP",     "#ff7f0e"),   # orange
]

dset = pickle.load(open(DATA, "rb"))


def raw_pred(path):
    return np.asarray(pickle.load(open(path, "rb"))[ACTION])


def pred_for(method, si, seed):
    if method == "TP-TF":
        return raw_pred(EVAL / "exp2" / "archive_important_dist" / "tp" / str(seed) / "predictions.pickle")
    if method == "TP-GMM":
        return raw_pred(EVAL / "exp1" / "tp_gmm" / str(K) / str(seed) / "predictions.pickle")
    if method == "TP-ProMP":
        return raw_pred(EVAL / "exp1" / "tp_promp" / str(K) / str(seed) / "predictions.pickle")
    if method in ("CNEP", "CNMP"):
        name = method.lower()
        p = EVAL / "exp1" / "predictions" / name / "random" / str(K) / f"{name}_predictions.pickle"
        arrs = pickle.load(open(p, "rb"))[ACTION]
        return np.asarray(arrs[si])
    return None


def main():
    si, demo = CELL
    seed = SEEDS[si]
    e = dset[ACTION][si]
    gt = np.asarray(e["test_traj_global"])[demo]          # (T, 7)
    # Same processing as the evaluator: force-smooth (no consecutive sign
    # flips) + normalise, applied to columns 3:7.
    gt_q = process_quaternions(gt[:, QUAT_COLS].astype(np.float64))

    preds = {}
    for name, _ in METHODS:
        arr = pred_for(name, si, seed)
        if arr is None:
            continue
        q = process_quaternions(arr[demo][:, QUAT_COLS].astype(np.float64))
        # Globally align each (already-smoothed) series to GT so overlapping
        # curves share the same sign branch.
        if np.sum(q * gt_q) < 0:
            q = -q
        preds[name] = q

    T = gt_q.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for r, (ax, col_name) in enumerate(zip(axes, QUAT_NAMES)):
        ax.plot(t, gt_q[:, r], color=GT_COLOR, lw=2.0, ls="-", label="Ground truth",
                zorder=3)
        for name, color in METHODS:
            if name not in preds:
                continue
            zo = 5 if name == "TP-TF" else 4
            lw = 2.0 if name == "TP-TF" else 1.6
            ax.plot(t, preds[name][:, r], color=color, lw=lw, ls="--", label=name,
                    alpha=0.9, zorder=zo)
        ax.set_ylabel(col_name, fontsize=16, rotation=0, labelpad=16, va="center")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
    axes[-1].set_xlabel("Timestep", fontsize=15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT / "ori_sample_traj_action_2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (seed {seed}, demo {demo})")


if __name__ == "__main__":
    main()
