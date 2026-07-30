"""Generate paper-ready plots for exp1 K-sweep + exp2 augmentation comparison.

Outputs to /shared/$USER/RingAIAutoAnnotation/eval/results/figures/.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

USER = os.environ["USER"]
RESULTS = Path(f"/shared/{USER}/RingAIAutoAnnotation/eval/results")
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# TP-Transformer + classical-baseline summaries use the important_dist
# (waypoint-error) selection -- the numbers reported in the paper. Those CSVs
# were archived to results/archive_important_dist/ when the pose_loss/total_loss
# re-runs overwrote the live results/exp1 and results/exp2. The deep-baseline
# CNEP/CNMP summaries (exp1_deep) were not re-run and stay in the live location.
IMPDIST = RESULTS / "archive_important_dist"

ACTIONS = ["action_0", "action_1", "action_2"]
KS = [1, 2, 5, 10, 15]
EXP1_METHODS_DISPLAY = {
    "tp_gmm": "TP-GMM",
    "tp_promp": "TP-ProMP",
    # The TP-Transformer rows in summary.csv use the K value as model name.
    "tp_transformer": "TP-Transformer",
}
# Deep baselines live in a separate results tree, one dir per sampling regime.
# We show multi-context (the paper-faithful regime) as the primary curve and
# single-context as a lighter dashed line of the same colour.
DEEP_REGIMES = ["multi", "single"]   # subdir names: results/exp1_deep/<regime>/k<K>/
DEEP_REGIME_DIR = {"multi": "random", "single": "fixed"}  # on-disk dir names
COLORS = {
    "TP-Transformer": "#1f77b4",
    "TP-GMM":         "#2ca02c",
    "TP-ProMP":       "#9467bd",
    # CNEP / CNMP: shared hue (magenta/pink), two shades. Kept clear of the
    # blue TP-Transformer, purple TP-ProMP and green TP-GMM.
    "CNEP":           "#c2185b",   # dark magenta
    "CNMP":           "#f06fa8",   # light pink
    # Diffusion Policy: shared hue (brown/orange family), four shades. No-aug
    # arms are also dashed, so shade + style together separate all four.
    "DP (CNN)":                   "#8c3b12",   # dark brown  (CNN, aug)
    "DP (Transformer)":           "#d1622b",   # mid orange-brown (TF, aug)
    "DP (CNN, no-aug)":           "#b5651d",   # medium brown (CNN, no-aug)
    "DP (Transformer, no-aug)":   "#e6a06b",   # light tan   (TF, no-aug)
    "TP-aug":         "#1f77b4",
    "No-aug":         "#2ca02c",
    "Random rotation": "#d62728",
}

# Per-method plot style for the K-sweep curves: (linestyle, linewidth, zorder).
# No-aug arms dashed; TP-Transformer drawn last (high zorder) with a wider line.
def _curve_style(method: str):
    if method == "TP-Transformer":
        return ("-", 3.6, 10)          # wider + on top
    if "no-aug" in method:
        return ("--", 2.0, 3)          # dashed for non-augmented DP arms
    return ("-", 2.0, 3)


def load_summary(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["ade_mean_mm"] = float(r["ade_mean_mm"])
            r["ade_std_mm"] = float(r["ade_std_mm"])
            r["ndq_mean"] = float(r["ndq_mean"])
            r["ndq_std"] = float(r["ndq_std"])
            r["n_seeds"] = int(r["n_seeds"])
            rows.append(r)
    return rows


def collect_exp1():
    """Return {(method_display, regime): {action: {K: (ade_m, ade_s, ndq_m, ndq_s)}}}.

    regime is None for the classical/transformer methods (which have a single
    result), and 'multi'/'single' for the deep CNEP/CNMP baselines.
    """
    out: dict = defaultdict(lambda: defaultdict(dict))
    # --- TP-Transformer + classical baselines (important_dist results) ---
    for K in KS:
        rows = load_summary(IMPDIST / "exp1" / f"k{K}" / "summary.csv")
        for r in rows:
            model = r["model"]
            if model in ("tp_gmm", "tp_promp"):
                disp = EXP1_METHODS_DISPLAY[model]
            else:
                disp = "TP-Transformer"
            out[(disp, None)][r["action"]][K] = (
                r["ade_mean_mm"], r["ade_std_mm"], r["ndq_mean"], r["ndq_std"]
            )
    # --- deep baselines: one dir per regime, model column is 'cnep'/'cnmp' ---
    deep_disp = {"cnep": "CNEP", "cnmp": "CNMP"}
    for regime in DEEP_REGIMES:
        sub = DEEP_REGIME_DIR[regime]
        for K in KS:
            rows = load_summary(RESULTS / "exp1_deep" / sub / f"k{K}" / "summary.csv")
            for r in rows:
                disp = deep_disp.get(r["model"])
                if disp is None:
                    continue
                out[(disp, regime)][r["action"]][K] = (
                    r["ade_mean_mm"], r["ade_std_mm"], r["ndq_mean"], r["ndq_std"]
                )
    # --- Diffusion Policy: 4 arms (CNN/Transformer x tp/no-aug), per-subtask,
    # faithful 7-D config. All share one colour; no-aug arms are dashed.
    dp_arms = [
        ("DP (CNN)",              "exp1_dp_cnn",      "dp_cnn"),
        ("DP (Transformer)",      "exp1_dp_tf",       "dp_tf"),
        ("DP (CNN, no-aug)",      "exp1_dp_cnn_none", "dp_cnn"),
        ("DP (Transformer, no-aug)", "exp1_dp_tf_none", "dp_tf"),
    ]
    for disp, subdir, model in dp_arms:
        for K in KS:
            rows = load_summary(RESULTS / subdir / f"k{K}" / "summary.csv")
            for r in rows:
                if r["model"] != model:
                    continue
                out[(disp, None)][r["action"]][K] = (
                    r["ade_mean_mm"], r["ade_std_mm"], r["ndq_mean"], r["ndq_std"]
                )
    return out


def collect_exp2():
    """Return list of (method_display, action -> (ade_mean, ade_std, ndq_mean, ndq_std))."""
    rows = load_summary(IMPDIST / "exp2" / "summary.csv")
    out: dict = defaultdict(dict)
    label = {"tp": "TP-aug", "none": "No-aug", "random": "Random rotation"}
    for r in rows:
        disp = label.get(r["model"], r["model"])
        out[disp][r["action"]] = (r["ade_mean_mm"], r["ade_std_mm"], r["ndq_mean"], r["ndq_std"])
    return out


# ---------------- Exp1 K-curve plots ----------------

def plot_exp1_curve(metric_idx: int, ylabel: str, fname: str, log_scale: bool = False, ymax: float | None = None):
    data = collect_exp1()
    # Larger fonts + shorter panels: the figure occupies less vertical space
    # while the text stays legible at reduced size in the paper.
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2), sharey=True)
    for i, action in enumerate(ACTIONS):
        ax = axes[i]
        ax.set_title(f"Subtask {i + 1}")
        # Single shared x-label under the middle panel (avoid repeating it 3x).
        if i == 1:
            ax.set_xlabel("K (training demos per action)")
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.set_xticks(KS)
        # Draw TP-Transformer last so its wider line sits on top of the others.
        items = sorted(data.items(), key=lambda kv: kv[0][0] == "TP-Transformer")
        for (method, regime), by_action in items:
            if action not in by_action:
                continue
            # Plots show only the paper-faithful multi-context regime for the
            # deep baselines (the table reports both sc and mc).
            if regime == "single":
                continue
            by_K = by_action[action]
            xs = sorted(by_K.keys())
            means = [by_K[k][metric_idx * 2] for k in xs]
            stds = [by_K[k][metric_idx * 2 + 1] for k in xs]
            ls, lw, zo = _curve_style(method)
            ax.errorbar(
                xs, means, yerr=stds,
                marker="o", capsize=4, lw=lw, ls=ls, zorder=zo,
                label=method, color=COLORS.get(method, None),
            )
        if log_scale:
            ax.set_yscale("log")
            # Raise the lower limit to just under 10^1 so the curves spread out
            # (ADE panel); the smallest TP-Transformer values sit ~10-20 mm.
            ax.set_ylim(bottom=8)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        # ADE/NDQ are non-negative; keep the linear-scale floor at 0 so wide
        # error bars (e.g. TP-GMM at low K) don't pull the axis negative.
        if not log_scale:
            ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_exp1_legend(fname: str = "exp1_methods_legend.png"):
    """Standalone horizontal legend strip for the K-sweep method curves."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=COLORS["TP-Transformer"], lw=3.6, marker="o",
               markersize=9, label="TP-Transformer"),
        Line2D([], [], color=COLORS["TP-GMM"], lw=2.0, marker="o",
               markersize=9, label="TP-GMM"),
        Line2D([], [], color=COLORS["TP-ProMP"], lw=2.0, marker="o",
               markersize=9, label="TP-ProMP"),
        Line2D([], [], color=COLORS["CNEP"], lw=2.0, marker="o",
               markersize=9, label="CNEP"),
        Line2D([], [], color=COLORS["CNMP"], lw=2.0, marker="o",
               markersize=9, label="CNMP"),
        Line2D([], [], color=COLORS["DP (CNN)"], lw=2.0, ls="-", marker="o",
               markersize=9, label="DP (CNN)"),
        Line2D([], [], color=COLORS["DP (Transformer)"], lw=2.0, ls="-", marker="o",
               markersize=9, label="DP (Transformer)"),
        Line2D([], [], color=COLORS["DP (CNN, no-aug)"], lw=2.0, ls="--", marker="o",
               markersize=9, label="DP (CNN, no-aug)"),
        Line2D([], [], color=COLORS["DP (Transformer, no-aug)"], lw=2.0, ls="--", marker="o",
               markersize=9, label="DP (Transformer, no-aug)"),
    ]
    fig = plt.figure(figsize=(20, 0.7))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               fontsize=17, handletextpad=0.4, columnspacing=1.1)
    out = FIG_DIR / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".eps"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------- Exp2 bar chart ----------------

def plot_exp2_bar():
    data = collect_exp2()
    methods = [m for m in ["TP-aug", "No-aug", "Random rotation"] if m in data]
    n_m = len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ADE panel
    ax = axes[0]
    width = 0.8 / n_m
    x = np.arange(len(ACTIONS))
    for j, m in enumerate(methods):
        means = [data[m][a][0] for a in ACTIONS]
        stds = [data[m][a][1] for a in ACTIONS]
        offset = (j - (n_m - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=m, color=COLORS.get(m, None))
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIONS)
    ax.set_ylabel("ADE (mm)")
    ax.set_title("Position error (ADE)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # NDQ panel
    ax = axes[1]
    for j, m in enumerate(methods):
        means = [data[m][a][2] for a in ACTIONS]
        stds = [data[m][a][3] for a in ACTIONS]
        offset = (j - (n_m - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=m, color=COLORS.get(m, None))
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIONS)
    ax.set_ylabel("NDQ")
    ax.set_title("Orientation error (NDQ)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("TP-augmentation vs. no-augmentation vs. random rotation (K=15, 5 seeds)")
    fig.tight_layout()
    out = FIG_DIR / "exp2_augmentation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    print("=== K-sweep curves ===")
    # ADE: log-scale only (the linear-capped variant is dropped).
    plot_exp1_curve(0, "ADE (mm)", "exp1_ade_log.png", log_scale=True)
    # NDQ
    plot_exp1_curve(1, "NDQ", "exp1_ndq.png", log_scale=False)
    # Shared legend (curves no longer carry an in-plot legend).
    plot_exp1_legend()

    print("\n=== Augmentation bar chart ===")
    plot_exp2_bar()
    print(f"\nAll figures under {FIG_DIR}")
