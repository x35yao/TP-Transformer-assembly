"""Fit TP-ProMP (with validation sigma selection) and generate test predictions.

Reads `baselines/data/baseline_dataset_*.pickle` and writes predictions as:
`baselines/tp_promp/predictions/tp_pmp_predictions*.pickle`

Output schema:
    {action: [arr_seed_0, arr_seed_1, ...]}
Each `arr_seed_i` has shape `(N_test, T, 7)`.
"""

from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from TP_PMP import PMP
from quaternion_metric import norm_diff_quat
from utils import get_position_difference_per_step


CANONICAL_DATASET_NAME = "baseline_dataset_n15_v3t3.pickle"
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data" / CANONICAL_DATASET_NAME
DEFAULT_OUT_DIR = HERE / "predictions"
DEFAULT_ACTIONS = ["action_0", "action_1", "action_2"]
OBJS = ["bin", "bolt", "jig", "nut", "global"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TP-ProMP on assembly baseline dataset.")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                   help=f"Packaged baseline pickle (default: baselines/data/{CANONICAL_DATASET_NAME}).")
    p.add_argument("--actions", nargs="+", default=DEFAULT_ACTIONS)
    p.add_argument("--seed-indices", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--sigma-max", type=float, default=0.20)
    p.add_argument("--sigma-count", type=int, default=5)
    p.add_argument("--n-basis", type=int, default=21)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--out-suffix", type=str, default="")
    return p.parse_args()


def _to_3d(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 2:
        return arr[None, ...]
    return arr


def _build_basis(sigma: float, n_basis: int) -> dict:
    basis_center = np.arange(n_basis) / (n_basis - 1)
    return {
        "conf": [
            {"type": "sqexp", "nparams": n_basis + 1, "conf": {"dim": n_basis}},
            {"type": "poly", "nparams": 0, "conf": {"order": 3}},
        ],
        "params": np.concatenate([[np.log(sigma)], basis_center]),
    }


def _validation_score(model: PMP, entry: dict) -> float:
    t = np.asarray(entry["test_t"])
    val_traj = _to_3d(entry["validation_traj_global"])
    HTs_val = _to_3d(entry["HTs_validation"])

    pos_errs: List[float] = []
    ori_errs: List[float] = []
    for v_idx in range(val_traj.shape[0]):
        mu_v, _ = model.predict(t, HTs_val[v_idx], OBJS)
        gt = val_traj[v_idx]
        pos_errs.append(float(np.mean(get_position_difference_per_step(mu_v[:, :3], gt[:, :3]))))
        ori_errs.append(float(np.mean(norm_diff_quat(gt[:, 3:7], mu_v[:, 3:7]))))
    return float(np.mean(pos_errs) + np.mean(ori_errs))


def _predict_tests(model: PMP, entry: dict) -> np.ndarray:
    t = np.asarray(entry["test_t"])
    test_traj = _to_3d(entry["test_traj_global"])
    HTs_test = _to_3d(entry["HTs_test"])
    if HTs_test.shape[0] != test_traj.shape[0]:
        raise ValueError(f"HTs_test and test_traj_global mismatch: {HTs_test.shape[0]} vs {test_traj.shape[0]}")

    out = []
    for k in range(test_traj.shape[0]):
        mu_k, _ = model.predict(t, HTs_test[k], OBJS)
        out.append(np.asarray(mu_k))
    return np.stack(out, axis=0)


def run_one_entry(action: str, entry: dict, args: argparse.Namespace) -> np.ndarray:
    train_traj = np.asarray(entry["train_traj_tp_pmp"])
    train_times = entry["train_times_tp_pmp"]
    if train_traj.shape[-1] != len(OBJS) * 7:
        raise ValueError(
            f"train_traj_tp_pmp last dim is {train_traj.shape[-1]}; "
            f"expected {len(OBJS) * 7} = n_rfs({len(OBJS)}) * dof(7)."
        )
    dof = 7
    sigma_grid = np.linspace(args.sigma_min, args.sigma_max, args.sigma_count)

    best_model = None
    best_score = np.inf
    best_sigma = None
    for sigma in tqdm(sigma_grid, desc=f"{action} seed={entry['seed']}", leave=False):
        basis = _build_basis(float(sigma), args.n_basis)
        model = PMP(
            train_traj,
            train_times,
            dof,
            OBJS,
            sigma=float(sigma),
            n_components=1,
            full_basis=basis,
            covariance_type="diag",
            max_iter=args.max_iter,
            gmm=False,
        )
        model.train(print_lowerbound=False)
        score = _validation_score(model, entry)
        if score < best_score:
            best_score = score
            best_sigma = float(sigma)
            if best_model is not None:
                del best_model
            best_model = model
        else:
            del model
        gc.collect()

    if best_model is None:
        raise RuntimeError(f"No TP-ProMP model selected for {action} seed={entry['seed']}.")

    preds = _predict_tests(best_model, entry)
    print(f"  [{action}/seed={entry['seed']}] best_sigma={best_sigma:.4f} score={best_score:.6f} pred_shape={preds.shape}")
    del best_model
    gc.collect()
    return preds


def main() -> None:
    args = parse_args()
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    predictions: Dict[str, List[np.ndarray]] = {}
    for action in args.actions:
        if action not in data:
            print(f"WARNING: action '{action}' missing in dataset; skipping.")
            continue
        predictions[action] = []
        entries = data[action]
        for idx in args.seed_indices:
            if idx < 0 or idx >= len(entries):
                print(f"WARNING: seed index {idx} out of range for '{action}' (len={len(entries)}).")
                continue
            predictions[action].append(run_one_entry(action, entries[idx], args))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tp_pmp_predictions{args.out_suffix}.pickle"
    with open(out_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
