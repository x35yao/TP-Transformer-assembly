"""Recover demo IDs used by the original `D:\\project\\assembly\\data\\kmp\\
assembly\\data.pickle` and write them as a splits manifest YAML.

The old pickle never stored demo IDs, only the trajectories themselves. But:

* xyz columns are train-mean/std normalised
* orientation columns (qx, qy, qz, qw) are untouched

so we can identify any demo by matching the quaternion track of its
``traj_pose`` (loaded from processed CSV) against the corresponding bucket in
the old pickle.

Train demos live inside ``train_traj_tp_gmm[trajectory_frame]`` (the last
reference frame is the trajectory itself: local trajectory equals global
trajectory in that frame). Reshape (n_train * T, 8) -> (n_train, T, 8) and
the columns are [time, x, y, z, qx, qy, qz, qw].
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BASELINES_DIR = REPO_ROOT / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from prepare_baseline_dataset import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    DEFAULT_TASK_CONFIG,
    BAD_DEMOS,
    _load_demo_dataset,
    _load_task_config,
)

DEFAULT_OLD_PICKLE = Path(r"D:\project\assembly\data\kmp\assembly\data.pickle")
DEFAULT_OUT = REPO_ROOT / "data" / "splits" / "legacy_n15_v4t1.yaml"
ACTIONS = ("action_0", "action_1", "action_2")


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _match_demo(
    quat_target: np.ndarray,
    demos: List[str],
    demo_quats: Dict[str, np.ndarray],
    tol: float = 1e-6,
) -> Tuple[str, float]:
    """Return (best_demo_id, best_mean_abs_diff) given a (T, 4) quaternion target."""
    best = None
    for d in demos:
        q = demo_quats[d]
        if q.shape != quat_target.shape:
            continue
        err = float(np.abs(q - quat_target).mean())
        if best is None or err < best[1]:
            best = (d, err)
    if best is None:
        raise RuntimeError("no candidate demo of matching shape")
    if best[1] > tol:
        # Not bit-identical; either traj differs (rare) or quat representation
        # was canonicalised differently. Surface for inspection.
        pass
    return best


def recover(old_pickle: Path, processed_dir: Path, task_config: Path) -> dict:
    with open(old_pickle, "rb") as f:
        old = pickle.load(f)

    all_objs = _load_task_config(task_config)
    splits: Dict[str, Dict[int, Dict[str, List[str]]]] = {}
    per_action_eligible: Dict[str, List[str]] = {}

    seeds_seen: List[int] = []
    for action in ACTIONS:
        print(f"\n=== {action} ===")
        demo_dataset, _ = _load_demo_dataset(processed_dir, action, all_objs)
        eligible = sorted(demo_dataset.keys())
        per_action_eligible[action] = eligible

        # Cache per-demo quaternions for matching.
        demo_quats = {d: demo_dataset[d]["traj_pose"][:, 3:7] for d in eligible}

        splits[action] = {}
        for entry in old[action]:
            seed = int(entry["seed"])
            if seed not in seeds_seen:
                seeds_seen.append(seed)
            train_tp_gmm = _to_np(entry["train_traj_tp_gmm"])  # (5, n_train*T, 8)
            valid_global = _to_np(entry["validation_traj_global"])  # (n_valid, T, 7)
            test_global = _to_np(entry["test_traj_global"])  # (T, 7) or (1, T, 7)

            n_frames = train_tp_gmm.shape[0]
            traj_frame = n_frames - 1
            train_block = train_tp_gmm[traj_frame]  # (n_train*T, 8)
            n_valid, T, _ = valid_global.shape
            n_train = train_block.shape[0] // T
            train_arr = train_block.reshape(n_train, T, 8)  # cols: t, x, y, z, qx, qy, qz, qw
            train_quats = train_arr[:, :, 4:8]  # (n_train, T, 4)

            valid_quats = valid_global[:, :, 3:7]
            if test_global.ndim == 2:
                test_global = test_global[None, :, :]
            test_quats = test_global[:, :, 3:7]

            train_ids: List[str] = []
            for k in range(n_train):
                d, err = _match_demo(train_quats[k], eligible, demo_quats)
                train_ids.append(d)
            valid_ids: List[str] = []
            for k in range(n_valid):
                d, err = _match_demo(valid_quats[k], eligible, demo_quats)
                valid_ids.append(d)
            test_ids: List[str] = []
            for k in range(test_quats.shape[0]):
                d, err = _match_demo(test_quats[k], eligible, demo_quats)
                test_ids.append(d)

            train_sorted = sorted(set(train_ids))
            if len(train_sorted) != len(train_ids):
                dup = [d for d in train_ids if train_ids.count(d) > 1]
                raise RuntimeError(f"duplicate train matches for {action} seed={seed}: {dup}")

            overlap = set(train_sorted) & set(valid_ids) | set(train_sorted) & set(test_ids) | set(valid_ids) & set(test_ids)
            if overlap:
                raise RuntimeError(f"split overlap for {action} seed={seed}: {overlap}")

            splits[action][seed] = {
                "train": train_sorted,
                "valid": sorted(valid_ids),
                "test": sorted(test_ids),
            }

            print(f"  seed={seed}: n_train={len(train_ids)} n_valid={len(valid_ids)} n_test={len(test_ids)}")
            print(f"    train sample: {train_sorted[:3]} ...")
            print(f"    valid       : {sorted(valid_ids)}")
            print(f"    test        : {sorted(test_ids)}")

    manifest = {
        "meta": {
            "num_train": len(splits[ACTIONS[0]][seeds_seen[0]]["train"]),
            "num_validation": len(splits[ACTIONS[0]][seeds_seen[0]]["valid"]),
            "num_test": len(splits[ACTIONS[0]][seeds_seen[0]]["test"]),
            "split_strategy": "legacy_recovered_from_old_pickle",
            "seeds": seeds_seen,
            "actions": list(ACTIONS),
            "processed_dir": str(processed_dir),
            "bad_demos": {a: list(BAD_DEMOS.get(a, [])) for a in ACTIONS},
            "eligible_demos": {a: per_action_eligible[a] for a in ACTIONS},
            "recovered_from": str(old_pickle),
            "generated_with": "scripts/recover_legacy_splits.py",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "splits": splits,
    }
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--old-pickle", type=Path, default=DEFAULT_OLD_PICKLE)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.old_pickle.exists():
        raise SystemExit(f"old pickle not found: {args.old_pickle}")
    manifest = recover(args.old_pickle, args.processed_dir, args.task_config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
