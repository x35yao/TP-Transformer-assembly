"""Compare a fresh TP-GMM prediction pickle against the legacy run.

Usage:
    python scripts/compare_tp_gmm_to_legacy.py \\
        --new baselines/tp_gmm/predictions/tp_gmm_predictions_legacy_n15_v4t1.pickle \\
        --old "D:/project/trajectory_modeling/tests/assembly/tp_gmm_predictions_new.pickle"

Old pickle stores ``{action: [arr_seed_0, arr_seed_1, arr_seed_2]}`` where
each ``arr`` is ``(T, 7)`` (single test demo). New pickle has the same outer
shape but each ``arr`` is ``(N_test, T, 7)``. When ``N_test == 1`` we squeeze
to ``(T, 7)`` for a direct elementwise comparison.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _np(x):
    return np.asarray(x)


def _squeeze_to_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    return a


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--old", type=Path, default=Path(r"D:\project\trajectory_modeling\tests\assembly\tp_gmm_predictions_new.pickle"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.old, "rb") as f:
        old = pickle.load(f)
    with open(args.new, "rb") as f:
        new = pickle.load(f)

    print(f"old: {args.old}")
    print(f"new: {args.new}")

    overall_pos = []
    overall_quat = []
    for action in ("action_0", "action_1", "action_2"):
        if action not in old or action not in new:
            print(f"\n=== {action}: SKIP (missing) ===")
            continue
        print(f"\n=== {action} ===")
        oa, na = old[action], new[action]
        n_entries = min(len(oa), len(na))
        for i in range(n_entries):
            o = _squeeze_to_2d(_np(oa[i]))
            n = _squeeze_to_2d(_np(na[i]))
            if o.shape != n.shape:
                print(f"  entry[{i}] SHAPE MISMATCH old={o.shape} new={n.shape}")
                continue
            pos_diff = np.abs(o[:, :3] - n[:, :3]).mean()
            pos_max = np.abs(o[:, :3] - n[:, :3]).max()
            quat_diff = np.abs(o[:, 3:] - n[:, 3:]).mean()
            quat_max = np.abs(o[:, 3:] - n[:, 3:]).max()
            overall_pos.append(pos_diff)
            overall_quat.append(quat_diff)
            print(f"  entry[{i}] shape={o.shape}  "
                  f"pos mean={pos_diff:.4e} max={pos_max:.4e}  "
                  f"quat mean={quat_diff:.4e} max={quat_max:.4e}")

    if overall_pos:
        print(f"\n=== overall mean across entries ===")
        print(f"  position mean-abs-diff:    {np.mean(overall_pos):.4e}")
        print(f"  quaternion mean-abs-diff:  {np.mean(overall_quat):.4e}")
        print(f"\n(interpretation: 0 == bit-identical TP-GMM EM convergence;"
              f"\n small nonzero (~1e-3 or below in normalised xyz / radians)"
              f"\n means the model converged to essentially the same solution;"
              f"\n large values (>~1e-1) mean different EM optima.)")


if __name__ == "__main__":
    main()
