"""Fit TP-GMM (with validation model selection) and generate test predictions.

Reads `baselines/data/baseline_dataset_*.pickle` and writes predictions as:
`baselines/tp_gmm/predictions/tp_gmm_predictions*.pickle`

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

from TP_GMM import TP_GMM
from quaternion_metric import norm_diff_quat
from utils import get_position_difference_per_step


CANONICAL_DATASET_NAME = "baseline_dataset_n15_v3t3.pickle"
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data" / CANONICAL_DATASET_NAME
DEFAULT_OUT_DIR = HERE / "predictions"
DEFAULT_ACTIONS = ["action_0", "action_1", "action_2"]
OBJS = ["bin", "bolt", "jig", "nut", "global"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TP-GMM on assembly baseline dataset.")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                   help=f"Packaged baseline pickle (default: baselines/data/{CANONICAL_DATASET_NAME}).")
    p.add_argument("--actions", nargs="+", default=DEFAULT_ACTIONS)
    p.add_argument("--seed-indices", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--min-states", type=int, default=2)
    p.add_argument("--max-states", type=int, default=50)
    p.add_argument("--reg", type=float, default=1e-6)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--out-suffix", type=str, default="")
    return p.parse_args()


def _to_3d(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 2:
        return arr[None, ...]
    return arr


def _validation_score(model: TP_GMM, entry: dict) -> float:
    t = np.asarray(entry["test_t"])
    val_traj = _to_3d(entry["validation_traj_global"])
    HTs_val = _to_3d(entry["HTs_validation"])

    pos_errs: List[float] = []
    ori_errs: List[float] = []
    for v_idx in range(val_traj.shape[0]):
        mu_v, _ = model.predict(t, HTs_val[v_idx], OBJS)
        gt = val_traj[v_idx]
        pos_errs.append(float(np.mean(get_position_difference_per_step(gt[:, :3], mu_v[:, :3]))))
        ori_errs.append(float(np.mean(norm_diff_quat(gt[:, 3:7], mu_v[:, 3:7]))))
    return float(np.mean(pos_errs) + np.mean(ori_errs))


def _predict_tests(model: TP_GMM, entry: dict) -> np.ndarray:
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
    train_traj = np.asarray(entry["train_traj_tp_gmm"])[:, :, :8]
    n_dims = train_traj.shape[-1] - 1

    best_model = None
    best_score = np.inf
    best_states = None
    for nb_states in tqdm(range(args.min_states, args.max_states + 1), desc=f"{action} seed={entry['seed']}", leave=False):
        model = TP_GMM(train_traj, OBJS, nb_states, n_dims + 1)
        model.train(reg=args.reg, maxiter=args.maxiter, verbose=False)
        score = _validation_score(model, entry)
        if score < best_score:
            best_score = score
            best_states = nb_states
            if best_model is not None:
                del best_model
            best_model = model
        else:
            del model
        gc.collect()

    if best_model is None:
        raise RuntimeError(f"No TP-GMM model selected for {action} seed={entry['seed']}.")

    preds = _predict_tests(best_model, entry)
    print(f"  [{action}/seed={entry['seed']}] best_states={best_states} score={best_score:.6f} pred_shape={preds.shape}")
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
    out_path = out_dir / f"tp_gmm_predictions{args.out_suffix}.pickle"
    with open(out_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
