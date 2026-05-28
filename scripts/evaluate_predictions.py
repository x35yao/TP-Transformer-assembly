"""Evaluate test-set predictions for the baselines + TP-Transformer.

Reads `test_traj_global` (ground truth) and per-(action, seed) `train_stat`
from the baseline dataset pickle, loads each model's saved predictions, and
reports:

  * ADE (mm)  — Average Displacement Error: per-step Euclidean distance in
                position, averaged over timesteps, in physical millimetres
                (de-normalised using the seed's `train_stat['mean'/'std']`).
  * NDQ       — Norm of Difference Quaternions metric, per-step, averaged
                over timesteps. Values live in [0, sqrt(2)]; lower is better.

Aggregation is two-level so that the reported std reflects seed-to-seed
variability (the quantity you usually want for a model-comparison plot)
rather than pooling per-demo variability into the same number:
    1. Per (model, action, seed): mean ADE / NDQ across the N_test demos.
    2. Per (model, action): `mean ± std` (ddof=1) across seeds.
For the legacy split (1 test demo per seed) this collapses back to the
original `D:\\project\\trajectory_modeling\\assembly_comparison.py` numbers;
for n15_v3t3 (3 test demos per seed) it does the right thing.

Unit conventions (verified against the predict scripts):
    Every prediction file in the repo stores xyz in the SAME centred + scaled
    space as `test_traj_global` ((raw - train_mean) / train_std). Quaternions
    are stored raw (not necessarily unit-norm). We de-normalise xyz back to mm
    using the seed's `train_stat`, and pipe the predicted quaternion through
    the same `process_quaternions` (force smooth + normalise) as the legacy
    evaluator before computing NDQ.

Default file layout (override with --baselines / --transformer-* flags):
    baselines/<name>/predictions/<name>_predictions_<stem>.pickle
        (tp_promp is special: tp_pmp_predictions_<stem>.pickle)
    <transformer-output-root>/<model_name>/<seed>/predictions.pickle

`<stem>` is the part of the baseline-data filename after `baseline_dataset_`,
e.g. `baselines/data/baseline_dataset_n15_v3t3.pickle` -> stem `n15_v3t3`.

Example:
    python scripts/evaluate_predictions.py \\
        --baseline-data baselines/data/baseline_dataset_n15_v3t3.pickle \\
        --baselines cnep cnmp tp_gmm tp_promp \\
        --transformer-models tp_aug random_aug \\
        --transformer-output-root transformer \\
        --out-dir eval_results/n15_v3t3
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm as l2norm


# ---------- inline copy of D:\\project\\trajectory_modeling helpers ----------
# Inlined (not imported) so the script works without any extra sys.path
# rigging and stays robust if we later trim `baselines/tp_gmm/quaternion_metric.py`.

def _force_rotation_axis_direction(quat: np.ndarray, axis=(1.0, 0.0, 0.0)) -> np.ndarray:
    return quat if (quat[:3] @ np.asarray(axis)) > 0 else -quat


def _force_smooth_quats(quats: np.ndarray) -> np.ndarray:
    out = quats.copy()
    for i in range(1, len(out)):
        if out[i, :3] @ out[i - 1, :3] < 0:
            out[i] = -out[i]
    return out


def process_quaternions(quats: np.ndarray) -> np.ndarray:
    """Match legacy `process_quaternions(quats, sigma=None, normalize=True)`."""
    quats = quats.copy()
    quats[0] = _force_rotation_axis_direction(quats[0])
    quats = _force_smooth_quats(quats)
    norms = l2norm(quats, axis=1)
    return quats / norms[:, None]


def norm_diff_quat(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Per-step Norm-of-Difference-Quaternions distance, q1/q2 shape (T, 4)."""
    q1 = q1 / l2norm(q1, axis=1)[:, None]
    q2 = q2 / l2norm(q2, axis=1)[:, None]
    return np.minimum(l2norm(q1 + q2, axis=1), l2norm(q1 - q2, axis=1))


def position_distance_per_step(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Per-step Euclidean distance, p1/p2 shape (T, 3)."""
    return l2norm(p1 - p2, axis=1)


# ---------- file loaders ----------

BASELINE_PRED_PREFIXES = {
    "cnep": "cnep_predictions",
    "cnmp": "cnmp_predictions",
    "tp_gmm": "tp_gmm_predictions",
    "tp_promp": "tp_pmp_predictions",  # legacy filename keeps `pmp` short form
}


def _dataset_stem(baseline_data_path: Path) -> str:
    """`baseline_dataset_n15_v3t3.pickle` -> `n15_v3t3`."""
    stem = baseline_data_path.stem
    prefix = "baseline_dataset_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def _default_baseline_pred_path(repo_root: Path, name: str, stem: str) -> Path:
    if name not in BASELINE_PRED_PREFIXES:
        raise KeyError(
            f"Unknown baseline '{name}'. Known: {sorted(BASELINE_PRED_PREFIXES)}. "
            f"Pass an explicit path with --baseline-pred {name}=<path>."
        )
    prefix = BASELINE_PRED_PREFIXES[name]
    return repo_root / "baselines" / name / "predictions" / f"{prefix}_{stem}.pickle"


def _parse_kv_overrides(items: Optional[List[str]]) -> Dict[str, str]:
    """Parse `name=path` overrides from --baseline-pred / --transformer-pred."""
    out: Dict[str, str] = {}
    if not items:
        return out
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Expected 'name=path', got {raw!r}.")
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_3d(arr) -> np.ndarray:
    """Normalise (T, 7) -> (1, T, 7); leave (N, T, 7) unchanged."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a[None, ...]
    return a


# ---------- metric core ----------

def _metrics_for_trajectory(
    pred: np.ndarray,            # (T, 7), normalised xyz + raw quat
    gt: np.ndarray,              # (T, 7), normalised xyz + raw quat
    train_mean: np.ndarray,      # (3,)
    train_std: float,
) -> Tuple[float, float]:
    """Return (ADE in mm, NDQ) reduced over T."""
    T_pred = pred.shape[0]
    T_gt = gt.shape[0]
    T = min(T_pred, T_gt)
    pred = pred[:T]
    gt = gt[:T]

    pred_pos_mm = pred[:, :3] * train_std + train_mean
    gt_pos_mm = gt[:, :3] * train_std + train_mean
    ade = float(np.mean(position_distance_per_step(pred_pos_mm, gt_pos_mm)))

    pred_quat = process_quaternions(pred[:, 3:7].astype(np.float64))
    gt_quat = gt[:, 3:7].astype(np.float64)
    ndq = float(np.mean(norm_diff_quat(pred_quat, gt_quat)))
    return ade, ndq


# ---------- per-(model, action, seed) evaluation ----------

def _eval_one_cell(
    model_name: str,
    action: str,
    seed: int,
    preds_for_cell: np.ndarray,  # (N_test, T, 7)
    entry: dict,                  # baseline_data[action][seed_idx]
) -> List[dict]:
    """Return a per-test-demo list of metric rows."""
    gt = _to_3d(entry["test_traj_global"])          # (N_test, T, 7)
    train_mean = np.asarray(entry["train_stat"]["mean"]).reshape(3)
    train_std = float(entry["train_stat"]["std"])
    preds_for_cell = _to_3d(preds_for_cell)         # (N_test_pred, T_pred, 7)

    if preds_for_cell.shape[0] != gt.shape[0]:
        print(
            f"  WARNING [{model_name}/{action}/seed={seed}]: pred N_test="
            f"{preds_for_cell.shape[0]} vs gt N_test={gt.shape[0]}; "
            f"using min."
        )
    n = min(preds_for_cell.shape[0], gt.shape[0])
    rows: List[dict] = []
    for i in range(n):
        ade, ndq = _metrics_for_trajectory(
            preds_for_cell[i], gt[i], train_mean, train_std,
        )
        rows.append({
            "model": model_name,
            "action": action,
            "seed": int(seed),
            "demo_idx": i,
            "ade_mm": ade,
            "ndq": ndq,
        })
    return rows


def evaluate(
    baseline_data: dict,
    actions: List[str],
    seeds_in_order: List[int],
    baseline_preds: Dict[str, dict],
    transformer_preds: Dict[str, Dict[int, dict]],
) -> List[dict]:
    """Run metric computation across all configured models.

    Args:
        baseline_data: Loaded `baseline_dataset_<stem>.pickle`.
        actions: Actions to score (must be in baseline_data).
        seeds_in_order: Canonical seed order from the baseline data (one per
            seed_idx position).
        baseline_preds: name -> loaded prediction dict
            ({action: [arr_seed_0, arr_seed_1, ...]}).
        transformer_preds: name -> {seed -> loaded prediction dict
            ({action: arr(N_test, T, 7)})}.

    Returns:
        Flat list of per-trajectory rows (one per
        (model, action, seed, demo_idx) cell).
    """
    rows: List[dict] = []

    # --- per-seed sanity: each baseline_data entry's `seed` matches the
    # corresponding index in seeds_in_order; otherwise the positional
    # alignment we use below for baseline preds is meaningless.
    for action in actions:
        entries = baseline_data[action]
        actual = [int(e["seed"]) for e in entries]
        if actual != seeds_in_order:
            raise ValueError(
                f"Seed order mismatch for action='{action}'. "
                f"Expected {seeds_in_order} (from first action), "
                f"got {actual}. Cannot align baseline preds positionally."
            )

    # --- baselines (positional alignment per action) ---
    for name, preds in baseline_preds.items():
        for action in actions:
            if action not in preds:
                print(f"  WARNING [{name}]: action '{action}' missing in predictions; skipping.")
                continue
            per_seed_arrays = preds[action]
            if not isinstance(per_seed_arrays, (list, tuple)):
                # Some pickles may have a single (N_test, T, 7) when only one
                # seed was predicted. Wrap in a list.
                per_seed_arrays = [per_seed_arrays]
            for seed_idx, arr in enumerate(per_seed_arrays):
                if seed_idx >= len(seeds_in_order):
                    print(
                        f"  WARNING [{name}/{action}]: pred has more seeds "
                        f"({len(per_seed_arrays)}) than baseline data "
                        f"({len(seeds_in_order)}); skipping seed_idx={seed_idx}."
                    )
                    continue
                seed = seeds_in_order[seed_idx]
                entry = baseline_data[action][seed_idx]
                rows.extend(_eval_one_cell(name, action, seed, arr, entry))

    # --- TP-Transformer (one file per seed) ---
    for name, per_seed in transformer_preds.items():
        for action in actions:
            for seed_idx, seed in enumerate(seeds_in_order):
                if seed not in per_seed:
                    continue
                preds_dict = per_seed[seed]
                if action not in preds_dict:
                    print(f"  WARNING [{name}/seed={seed}]: action '{action}' missing in predictions; skipping.")
                    continue
                arr = preds_dict[action]
                entry = baseline_data[action][seed_idx]
                rows.extend(_eval_one_cell(name, action, seed, arr, entry))

    return rows


# ---------- aggregation + output ----------

def aggregate_per_seed(rows: List[dict]) -> List[dict]:
    """Step 1 reduction: per (model, action, seed) mean across N_test demos.

    Returns one row per (model, action, seed) cell. This is the "right"
    per-seed score; downstream `aggregate_across_seeds` averages these.
    """
    buckets: Dict[Tuple[str, str, int], List[Tuple[float, float]]] = {}
    for r in rows:
        key = (r["model"], r["action"], int(r["seed"]))
        buckets.setdefault(key, []).append((r["ade_mm"], r["ndq"]))
    out: List[dict] = []
    for (model, action, seed), vals in sorted(buckets.items()):
        ades = np.array([v[0] for v in vals], dtype=np.float64)
        ndqs = np.array([v[1] for v in vals], dtype=np.float64)
        out.append({
            "model": model,
            "action": action,
            "seed": seed,
            "n_demos": len(vals),
            "ade_mean_mm": float(np.mean(ades)),
            "ndq_mean": float(np.mean(ndqs)),
        })
    return out


def aggregate_across_seeds(per_seed_rows: List[dict]) -> List[dict]:
    """Step 2 reduction: (model, action) -> `mean ± std` across seeds.

    Std uses ddof=1 to match the legacy `assembly_comparison.py` chart.
    """
    buckets: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for r in per_seed_rows:
        buckets.setdefault((r["model"], r["action"]), []).append(
            (r["ade_mean_mm"], r["ndq_mean"])
        )
    out: List[dict] = []
    for (model, action), vals in sorted(buckets.items()):
        ades = np.array([v[0] for v in vals], dtype=np.float64)
        ndqs = np.array([v[1] for v in vals], dtype=np.float64)
        ade_std = float(np.std(ades, ddof=1)) if len(ades) > 1 else 0.0
        ndq_std = float(np.std(ndqs, ddof=1)) if len(ndqs) > 1 else 0.0
        out.append({
            "model": model,
            "action": action,
            "n_seeds": len(vals),
            "ade_mean_mm": float(np.mean(ades)),
            "ade_std_mm": ade_std,
            "ndq_mean": float(np.mean(ndqs)),
            "ndq_std": ndq_std,
        })
    return out


def _write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        print(f"  (nothing to write to {path}; empty)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}  ({len(rows)} rows)")


def _print_summary_table(summary: List[dict]) -> None:
    if not summary:
        print("  (no summary rows)")
        return
    print()
    print(f"{'model':<14} {'action':<10} {'n_seeds':>7}  {'ADE (mm)':>17}  {'NDQ':>17}")
    print("-" * 74)
    for r in summary:
        ade_str = f"{r['ade_mean_mm']:8.2f} ± {r['ade_std_mm']:6.2f}"
        ndq_str = f"{r['ndq_mean']:8.4f} ± {r['ndq_std']:6.4f}"
        print(
            f"{r['model']:<14} {r['action']:<10} {r['n_seeds']:>7}  "
            f"{ade_str:>17}  {ndq_str:>17}"
        )
    print("  (std is across seeds, ddof=1; per-seed score is mean across "
          "N_test demos.)")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--baseline-data", type=Path, required=True,
                   help="Packaged baseline pickle (provides test_traj_global + train_stat).")
    p.add_argument("--actions", nargs="+",
                   default=["action_0", "action_1", "action_2"])
    p.add_argument("--baselines", nargs="*", default=[],
                   choices=sorted(BASELINE_PRED_PREFIXES.keys()),
                   help="Baseline names to evaluate. Predictions are read from "
                        "baselines/<name>/predictions/<prefix>_<stem>.pickle by default.")
    p.add_argument("--baseline-pred", nargs="*", default=[],
                   help="Override default baseline prediction path with "
                        "name=path entries (e.g. cnep=runs/cnep.pkl).")
    p.add_argument("--transformer-models", nargs="*", default=[],
                   help="TP-Transformer model_name(s) to evaluate. Predictions "
                        "are read from <transformer-output-root>/<model>/<seed>/predictions.pickle.")
    p.add_argument("--transformer-output-root", type=Path, default=Path("transformer"),
                   help="Root for TP-Transformer per-seed checkpoint folders "
                        "(default: ./transformer).")
    p.add_argument("--transformer-pred", nargs="*", default=[],
                   help="Override default TP-Transformer prediction path with "
                        "name=path entries; <path> is the model root directory "
                        "containing <seed>/predictions.pickle (e.g. "
                        "tp_aug=/abs/path/to/tp_aug).")
    p.add_argument("--repo-root", type=Path, default=Path("."),
                   help="Repo root (used to resolve default baseline pred paths). "
                        "Default: cwd.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write per_trajectory.csv + summary.csv. "
                        "Default: alongside --baseline-data as eval_results/<stem>/.")
    return p.parse_args()


def _resolve_baseline_pred_paths(
    args: argparse.Namespace, stem: str,
) -> Dict[str, Path]:
    overrides = _parse_kv_overrides(args.baseline_pred)
    paths: Dict[str, Path] = {}
    for name in args.baselines:
        if name in overrides:
            paths[name] = Path(overrides[name])
        else:
            paths[name] = _default_baseline_pred_path(args.repo_root, name, stem)
    # Allow override-only baselines that weren't listed in --baselines.
    for name, raw in overrides.items():
        if name not in paths:
            paths[name] = Path(raw)
    return paths


def _resolve_transformer_roots(args: argparse.Namespace) -> Dict[str, Path]:
    overrides = _parse_kv_overrides(args.transformer_pred)
    roots: Dict[str, Path] = {}
    for name in args.transformer_models:
        roots[name] = Path(overrides[name]) if name in overrides else (
            args.transformer_output_root / name
        )
    for name, raw in overrides.items():
        if name not in roots:
            roots[name] = Path(raw)
    return roots


def main() -> int:
    args = parse_args()

    if not args.baseline_data.exists():
        print(f"ERROR: baseline data not found: {args.baseline_data}", file=sys.stderr)
        return 1

    baseline_data = _load_pickle(args.baseline_data)
    stem = _dataset_stem(args.baseline_data)
    print(f"Baseline data: {args.baseline_data}  (stem='{stem}')")

    actions = [a for a in args.actions if a in baseline_data]
    if not actions:
        print(f"ERROR: no requested actions {args.actions} are in baseline data.", file=sys.stderr)
        return 1

    # Establish seed order from the first action. We check the other actions
    # match this order inside `evaluate(...)`.
    first = baseline_data[actions[0]]
    seeds_in_order = [int(e["seed"]) for e in first]
    print(f"Actions: {actions}")
    print(f"Seeds (in pickle order): {seeds_in_order}")

    # --- load baseline predictions ---
    baseline_pred_paths = _resolve_baseline_pred_paths(args, stem)
    baseline_preds: Dict[str, dict] = {}
    for name, path in baseline_pred_paths.items():
        if not path.exists():
            print(f"  WARNING [{name}]: predictions not found at {path}; skipping.")
            continue
        baseline_preds[name] = _load_pickle(path)
        print(f"  loaded baseline '{name}' from {path}")

    # --- load TP-Transformer predictions ---
    transformer_roots = _resolve_transformer_roots(args)
    transformer_preds: Dict[str, Dict[int, dict]] = {}
    for name, root in transformer_roots.items():
        per_seed: Dict[int, dict] = {}
        for seed in seeds_in_order:
            cand = root / str(seed) / "predictions.pickle"
            if not cand.exists():
                print(f"  WARNING [{name}/seed={seed}]: predictions not found at {cand}; skipping.")
                continue
            per_seed[seed] = _load_pickle(cand)
            print(f"  loaded transformer '{name}/seed={seed}' from {cand}")
        if per_seed:
            transformer_preds[name] = per_seed

    if not baseline_preds and not transformer_preds:
        print("ERROR: no predictions loaded; nothing to evaluate.", file=sys.stderr)
        return 1

    # --- compute metrics ---
    rows = evaluate(baseline_data, actions, seeds_in_order, baseline_preds, transformer_preds)
    per_seed = aggregate_per_seed(rows)
    summary = aggregate_across_seeds(per_seed)

    out_dir = args.out_dir or Path("eval_results") / stem
    _write_csv(rows, out_dir / "per_trajectory.csv")
    _write_csv(per_seed, out_dir / "per_seed.csv")
    _write_csv(summary, out_dir / "summary.csv")
    _print_summary_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
