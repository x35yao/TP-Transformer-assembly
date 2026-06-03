"""Shared inference helpers for TP-Transformer (used by train.py + predict.py).

Two entry points:

- `assemble_result`: run the segment-by-segment forward pass for ONE sample
  (batch size 1) and return the assembled trajectory buffer of shape
  (1, T, n_dims+1) = (1, T, 8) with columns [x,y,z, qx,qy,qz,qw, grasp].
  This is the exact loop that train.py's valid loop and the old predict.py
  inlined.

- `tta_assemble_result`: test-time rotation averaging. For `num_rotations`
  samples, rotate the ENCODER object poses, run `assemble_result`, rotate the
  predicted trajectory BACK by the same angle, and average the back-transformed
  trajectories. By default (`sampling="match_train"`) the angle and pivot are
  drawn the same way as the training-time `augment_random_rotation` (random
  angle in [0, 360), a random existing object point as pivot, axis 'z'), so TTA
  averages over the same transformation distribution the model was trained on.
  An `"even"` mode (evenly-spaced angles about the object centroid) is also
  available for a deterministic variant.

  Ported (and corrected) from the multi-task notebook's TTA loop. The notebook
  rotated `obj_seq_input` IN PLACE and re-read it each iteration, so rotations
  compounded while only the latest was inverted (only the first sample landed
  back in the canonical frame). Here every iteration rotates a fresh copy of
  the ORIGINAL `obj_seq`, so all samples are correctly back-transformed before
  averaging.

All rotations happen in the model's normalized pose space (xyz centred by
train_mean and scaled by the scalar train_std). Because train_std is a single
isotropic scale, rotation commutes with the normalization, so rotating in
normalized space is equivalent to rotating in physical space.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .augmentation import rotate_pose_about


def assemble_result(
    model: torch.nn.Module,
    obj_seq: torch.Tensor,
    traj_hidden: torch.Tensor,
    padding_mask: torch.Tensor,
    img_inds: torch.Tensor,
    n_dims: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the segment-by-segment decode and return the (B, T, n_dims+1) buffer.

    Mirrors the inner loop of `valid_traj_epoch`: for each camera-capture
    segment, the encoder sees that segment's object poses, the decoder fills in
    the trajectory, and the segment's slice is written into a running result
    buffer initialised from `traj_hidden` (with a zero grasp column prepended).
    """
    result = traj_hidden.clone()
    zero_col = torch.zeros((result.shape[0], result.shape[1], 1), device=device)
    result = torch.cat((zero_col, result), dim=2)

    n_segments = obj_seq.shape[1]
    for i in range(n_segments):
        loss_start_inds = img_inds[:, i]
        if i == n_segments - 1:
            loss_end_inds = torch.ones(obj_seq.shape[0], dtype=torch.int) * (-1)
        else:
            loss_end_inds = img_inds[:, i + 1]

        output_seq, _action_logits = model(
            obj_seq[:, i, :, :],
            traj_hidden,
            tgt_padding_mask=padding_mask,
            predict_action=True,
        )
        for (start, end) in zip(loss_start_inds, loss_end_inds):
            result[:, start:end, : n_dims + 1] = output_seq[:, start:end]
    return result


def _average_quaternions(quats: np.ndarray) -> np.ndarray:
    """Sign-aligned mean of (N, T, 4) quaternions -> (T, 4), renormalised.

    Quaternions q and -q encode the same rotation, so before averaging we flip
    each sample's quaternion (per timestep) to the hemisphere of the first
    sample. Then mean + renormalise. (Downstream NDQ also applies
    process_quaternions, but aligning here keeps the *position-frame* average
    meaningful and avoids cancellation.)
    """
    ref = quats[0]  # (T, 4)
    dots = np.sum(quats * ref[None], axis=2)  # (N, T)
    flips = np.where(dots < 0.0, -1.0, 1.0)[..., None]  # (N, T, 1)
    aligned = quats * flips
    mean_q = aligned.mean(axis=0)  # (T, 4)
    norms = np.linalg.norm(mean_q, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mean_q / norms


def tta_assemble_result(
    model: torch.nn.Module,
    obj_seq: torch.Tensor,
    traj_hidden: torch.Tensor,
    padding_mask: torch.Tensor,
    img_inds: torch.Tensor,
    n_dims: int,
    device: torch.device,
    num_rotations: int = 0,
    axis: str = "z",
    center: Optional[np.ndarray] = None,
    sampling: str = "match_train",
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Test-time rotation averaging; returns the same (B, T, n_dims+1) buffer.

    `num_rotations <= 1` falls back to a single deterministic pass (identical to
    `assemble_result`). Assumes batch size 1 (validation + predict both use
    batch_size=1).

    For each of `num_rotations` samples:
      1. rotate every object pose (cols 0:7) in every segment by +deg about a
         pivot on `axis`,
      2. run `assemble_result` on the rotated scene,
      3. rotate the predicted trajectory (cols 0:7 = pos+quat) back by -deg
         about the SAME pivot,
    then average positions (and grasp) and sign-aligned-average quaternions.

    `sampling` controls the angle + pivot distribution:
      - "match_train" (default): mirror the training-time `random_rotation`
        augmentation -- a random angle in [0, 360) and a random *existing*
        object point as the pivot, drawn fresh per sample. This averages over
        the same transformation distribution the model was trained on.
      - "even": evenly-spaced angles (0, 360/N, ...) about the fixed object
        centroid (deterministic; angle 0 is included).
    `center`, if given, overrides the pivot for ALL samples (mainly for tests).
    """
    if num_rotations is None or num_rotations <= 1:
        return assemble_result(model, obj_seq, traj_hidden, padding_mask, img_inds, n_dims, device)

    if obj_seq.shape[0] != 1:
        raise ValueError(
            f"tta_assemble_result expects batch size 1, got {obj_seq.shape[0]}."
        )

    obj_np = obj_seq.detach().cpu().numpy()  # (1, S, O, D)
    obj_pts = obj_np[0, :, :, :3].reshape(-1, 3)  # all object xyz across segments
    centroid = obj_pts.mean(axis=0)
    if rng is None:
        rng = np.random.default_rng()

    if sampling == "even":
        angles = (np.arange(num_rotations) * (360.0 / num_rotations)).tolist()
        centers = [centroid if center is None else np.asarray(center).reshape(3)] * num_rotations
    elif sampling == "match_train":
        # Random angle in [0, 360) and a random existing object point as pivot,
        # matching the training-time random_rotation augmentation.
        angles = [float(rng.integers(0, 360)) for _ in range(num_rotations)]
        if center is not None:
            fixed = np.asarray(center, dtype=np.float64).reshape(3)
            centers = [fixed] * num_rotations
        else:
            idxs = rng.integers(0, obj_pts.shape[0], size=num_rotations)
            centers = [obj_pts[i].astype(np.float64) for i in idxs]
    else:
        raise ValueError(f"unknown sampling={sampling!r}; expected 'match_train' or 'even'")

    n_seg, n_obj = obj_np.shape[1], obj_np.shape[2]

    samples = []  # each (T, n_dims+1)
    for deg, ctr in zip(angles, centers):
        deg = float(deg)
        ctr = np.asarray(ctr, dtype=np.float64).reshape(3)
        # --- rotate the ORIGINAL object scene (fresh copy: no accumulation) ---
        obj_rot = obj_np.copy()
        flat = obj_rot[0, :, :, :7].reshape(-1, 7)
        flat = rotate_pose_about(flat, deg, ctr, axis=axis)
        obj_rot[0, :, :, :7] = flat.reshape(n_seg, n_obj, 7)
        obj_rot_t = torch.as_tensor(obj_rot, dtype=obj_seq.dtype, device=device)

        # --- forward pass on the rotated scene ---
        res = assemble_result(model, obj_rot_t, traj_hidden, padding_mask, img_inds, n_dims, device)
        res_np = res.detach().cpu().numpy()[0]  # (T, n_dims+1)

        # --- rotate the predicted trajectory back by -deg about the same pivot ---
        res_np[:, :7] = rotate_pose_about(res_np[:, :7], -deg, ctr, axis=axis)
        samples.append(res_np)

    stacked = np.stack(samples, axis=0)  # (N, T, n_dims+1)
    avg = stacked.mean(axis=0)  # position + grasp (+ raw quat, overwritten below)
    avg[:, 3:7] = _average_quaternions(stacked[:, :, 3:7])

    out = torch.as_tensor(avg[None, ...], dtype=traj_hidden.dtype, device=device)
    return out
