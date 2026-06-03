"""Test-set prediction for the TP-Transformer model.

Mirrors `train.py::train_model` for the model build + dataset construction,
but loads a trained checkpoint and runs inference on the test set instead of
training. Output schema matches the baselines convention
(`baselines/{cnep,cnmp,tp_gmm,tp_promp}/predictions/<...>_predictions.pickle`)
so the same evaluator can consume both.

Inference pipeline:
    1. Build datasets exactly as in training (same `splits_file` + `seed` ->
       same test demos, same per-seed train_mean/train_std).
    2. Build the model with `build_model` and load the checkpoint into it.
       Supports both new (state_dict-only) and legacy (full module pickle)
       checkpoint formats.
    3. Iterate over the test set with batch_size=1. For each sample, replay
       the segment-by-segment forward pass from `valid_traj_epoch`:
         result = traj_hidden + zero column
         for each camera segment i:
             output_seq, _ = model(obj_seq[:, i], traj_hidden, ...)
             result[:, start:end, :8] = output_seq[:, start:end]
       Slice `result[..., :7]` by the padding mask to get the actual
       (T_action, 7) pose trajectory.
    4. Group samples by ground-truth action tag (one-hot in the sample), then
       stack each group into ``(N_test_for_action, T_action, 7)``.

Output:
    <output_root>/<model_name>/<seed>/predictions.pickle
        dict {action_name: np.ndarray of shape (N_test, T, 7)}
    Same xyz normalisation as the baselines (centred by train_mean,
    scaled by train_std). The evaluator denormalises both sides using
    `train_stat`.

The function deliberately re-uses `build_datasets` (rather than poking at
`test_traj_pose` directly) so the same RNG / split / normalisation path that
trained the model also produces the test samples. That guarantees the model
sees exactly the same inputs at predict time as at train-validation time.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import TASK_DIMS, build_datasets
from .inference import tta_assemble_result
from .train import build_model
from .utils import get_n_params


def _load_checkpoint_into(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> dict:
    """Load `ckpt_path` into `model` in-place, tolerating legacy schemas.

    Returns the loaded ckpt dict (for logging best_epoch / v_important_dist
    if present).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt
    if isinstance(ckpt, dict) and "model" in ckpt:
        # Legacy: train.py used to save the live nn.Module under "model".
        legacy_model = ckpt["model"]
        if hasattr(legacy_model, "state_dict"):
            model.load_state_dict(legacy_model.state_dict())
        else:
            # Or maybe a state_dict directly under "model".
            model.load_state_dict(legacy_model)
        return ckpt
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # Bare state_dict-shaped dict (no wrapper keys).
        try:
            model.load_state_dict(ckpt)
            return {"loaded_from": "bare_state_dict"}
        except Exception:
            pass
    raise ValueError(
        f"Unrecognised checkpoint schema at {ckpt_path}: top-level keys = "
        f"{sorted(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__}"
    )


def _slice_by_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Slice (T_max, D) by 1-D bool mask of length T_max -> (T_valid, D)."""
    return arr[mask]


def predict_test_set(
    config: TrainConfig,
    checkpoint: Optional[str] = None,
    out_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Run the TP-Transformer on the test set defined by `config` and save predictions.

    Args:
        config: Same TrainConfig used for training (splits_file + seed must match).
        checkpoint: Path to model_best.pth (or any state_dict pickle). Default:
            <output_root>/<model_name>/<seed>/model_best.pth.
        out_path: Where to save the predictions pickle. Default:
            <output_root>/<model_name>/<seed>/predictions.pickle.

    Returns:
        Dict {action_name: np.ndarray(N_test, T, 7)} -- also written to `out_path`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder = os.path.join(config.output_root, config.model_name, str(config.seed))
    if checkpoint is None:
        checkpoint = os.path.join(folder, "model_best.pth")
    if out_path is None:
        out_path = os.path.join(folder, "predictions.pickle")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    # 1. Build datasets (this also writes train_stat.pickle as a side effect of
    # build_datasets calling load_augmentation; safe to call here without
    # touching the model). The seed + splits must match what was trained on.
    _train_data, _valid_data, test_data, train_stats = build_datasets(config)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 2. Build model + load checkpoint.
    model = build_model(config, device)
    ckpt_meta = _load_checkpoint_into(model, checkpoint, device)
    model.eval()

    n_dims = len(TASK_DIMS)  # 7

    print(f"Device: {device}")
    print(f"Model parameters: {get_n_params(model)}")
    print(f"Checkpoint: {checkpoint}")
    if isinstance(ckpt_meta, dict):
        for k in ("epoch", "v_important_dist", "loaded_from"):
            if k in ckpt_meta:
                print(f"  ckpt.{k}: {ckpt_meta[k]}")
    print(f"Test samples: {len(test_data)}")

    # 3. Run inference, grouped by ground-truth action tag.
    # `config.tasks` is the canonical order; the one-hot action_tag in each
    # sample indexes into it.
    per_action_preds: Dict[str, List[np.ndarray]] = {t: [] for t in config.tasks}

    with torch.no_grad():
        for sample_idx, sample_batched in enumerate(test_loader):
            obj_seq, traj_seq, traj_hidden, weights, action_tag_gt, padding_mask, img_inds, _pick_inds, _release_inds = sample_batched
            obj_seq = obj_seq.to(device)
            traj_hidden = traj_hidden.to(device)
            traj_seq = traj_seq.to(device)
            padding_mask = padding_mask.to(device)

            # Segment-by-segment decode, optionally with test-time rotation
            # averaging. num_rotations <= 1 is a single deterministic pass
            # identical to the previous inlined loop.
            result = tta_assemble_result(
                model,
                obj_seq,
                traj_hidden,
                padding_mask,
                img_inds,
                n_dims,
                device,
                num_rotations=config.tta_rotations,
                axis=config.tta_axis,
            )

            # Slice by padding mask to get the actual (T_valid, 7) prediction.
            pred = result[0, :, :n_dims].cpu().numpy()
            mask = (~padding_mask[0].cpu().numpy().astype(bool))
            pred_valid = _slice_by_mask(pred, mask)  # (T_valid, 7)

            # Group by ground-truth action.
            action_idx = int(torch.argmax(action_tag_gt[0]).item())
            action_name = config.tasks[action_idx]
            per_action_preds[action_name].append(pred_valid)
            print(
                f"  [{sample_idx + 1}/{len(test_data)}] action={action_name} "
                f"T_valid={pred_valid.shape[0]}"
            )

    # 4. Stack each action's predictions to (N_test_for_action, T, 7).
    predictions: Dict[str, np.ndarray] = {}
    for action_name, lst in per_action_preds.items():
        if not lst:
            print(f"  WARNING: no test samples for action='{action_name}'")
            continue
        T_lengths = {arr.shape[0] for arr in lst}
        if len(T_lengths) != 1:
            raise ValueError(
                f"action='{action_name}': test demos have inconsistent T "
                f"after padding-mask slicing: {sorted(T_lengths)}. The test "
                f"trajectories for the same action are expected to be padded "
                f"to the action's median_traj_len before being added to the "
                f"dataset, so post-slice lengths must all match."
            )
        predictions[action_name] = np.stack(lst, axis=0)
        print(
            f"  action={action_name}: stacked shape={predictions[action_name].shape}  "
            f"(N_test={len(lst)}, T={predictions[action_name].shape[1]}, 7)"
        )

    # Persist train_stats alongside predictions for convenience (so the
    # evaluator can de-normalise without separately loading the baseline
    # pickle). Already saved by train.py; we re-write here so this script
    # also works on a checkpoint folder that wasn't produced by `train.py`.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    train_stat_path = os.path.join(os.path.dirname(out_path) or ".", "train_stat.pickle")
    if not os.path.exists(train_stat_path):
        with open(train_stat_path, "wb") as f:
            pickle.dump(train_stats, f)
        print(f"Wrote {train_stat_path}")

    with open(out_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Wrote {out_path}")
    return predictions
