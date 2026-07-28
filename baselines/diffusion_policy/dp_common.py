"""Shared helpers for the Diffusion Policy baseline (Approach B).

We reuse the *official* upstream implementation for everything scientifically
relevant -- the policy's training loss and receding-horizon inference
(`DiffusionUnetLowdimPolicy.compute_loss` / `.predict_action`), the denoising
network (`ConditionalUnet1D`), the DDPM scheduler, the `LinearNormalizer`
(mode='limits' -> [-1,1], required by clip_sample=True), and the `EMAModel`.
Only the data adapter (segment -> fixed-horizon windows) and the outer
train/predict driver are ours, so results plug into the TP-Transformer
evaluator unchanged.

Conditioning (our adaptation of DP's low-dim `obs`): a per-window observation
vector = the object poses observed at the current camera capture (all n_objs
frames, pose part) concatenated with the gripper pose at the window start
(proprioception / anchor). n_obs_steps=1 (the obs is a single static vector).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
UPSTREAM = REPO_ROOT / "baselines" / "diffusion_policy_upstream"

# --- dimensions (mirror tp_transformer.train.build_model) ---
N_DIMS = 7          # pose: x,y,z,qx,qy,qz,qw
N_OBJS = 5          # bolt, nut, bin, jig, trajectory
ACTION_DIM = 7      # pose only (x,y,z,qx,qy,qz,qw) -- matches the classical
                    # baselines (TP-GMM/TP-ProMP), which model the 7-D pose and
                    # no grasp. (TP-Transformer predicts 8-D pose+grasp but is
                    # likewise scored on the 7-D pose; DP is scored on 7-D too.)

# Whether the observation includes the gripper anchor (proprioception). When
# False, obs = object poses only (no gripper pose) -- the "anchor-free" ablation
# that forces the policy to condition on object poses like TP-Transformer
# (which never sees the gripper pose as an input). Toggle via set_use_anchor().
USE_ANCHOR = True

def set_use_anchor(flag: bool):
    """Set global anchor mode. Call before build_policy / windows_from_sample."""
    global USE_ANCHOR
    USE_ANCHOR = bool(flag)

def obs_dim() -> int:
    """Current observation dim: object poses (35) [+ gripper anchor (7)]."""
    return N_OBJS * N_DIMS + (N_DIMS if USE_ANCHOR else 0)

# Backwards-compatible module constant (anchor on). Prefer obs_dim() at runtime.
OBS_DIM = N_OBJS * N_DIMS + N_DIMS   # object poses (35) + gripper anchor (7) = 42

# DP low-dim CNN default horizon params (paper Tab. 7 / lowdim config).
HORIZON = 16
N_OBS_STEPS = 2      # official CNN lowdim default (transformer uses 3); set via set_n_obs_steps()
N_ACTION_STEPS = 8

def set_n_obs_steps(n: int):
    """Set number of observation steps (2 for CNN, 3 for TF per official yamls)."""
    global N_OBS_STEPS
    N_OBS_STEPS = int(n)


def import_upstream():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    return ConditionalUnet1D, DiffusionUnetLowdimPolicy, EMAModel, LinearNormalizer, DDPMScheduler


def import_transformer():
    """Upstream Transformer backbone + its lowdim policy wrapper."""
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
    from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
    return TransformerForDiffusion, DiffusionTransformerLowdimPolicy


def _make_scheduler(DDPMScheduler):
    return DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )


def _patch_mask_generator_device(policy, device):
    # ModuleAttrMixin.device reads next(self.parameters()).device; the
    # LowdimMaskGenerator's only param is an empty nn.Parameter() whose device
    # doesn't reliably follow .to() on GPU, leaving mask generation on CPU and
    # breaking the boolean index-assign in compute_loss. Wrap its forward to
    # move the generated mask onto the target device (content is a no-op mask in
    # our obs_as_cond setup; only its device matters).
    _dev = torch.device(device)
    mg = policy.mask_generator
    _orig_forward = mg.forward
    def _forward_on_device(shape, seed=None, _f=_orig_forward, _d=_dev):
        return _f(shape, seed=seed).to(_d)
    mg.forward = _forward_on_device


def build_policy(device: str = "cuda", num_inference_steps: int = 100,
                 backbone: str = "cnn"):
    """Construct the official DP lowdim policy with the paper's hyperparameters,
    adapted to our obs/action dims. backbone='cnn' -> DiffusionUnetLowdimPolicy
    (ConditionalUnet1D); backbone='transformer' -> DiffusionTransformerLowdimPolicy
    (TransformerForDiffusion)."""
    if backbone == "transformer":
        return _build_transformer_policy(device, num_inference_steps)
    ConditionalUnet1D, DiffusionUnetLowdimPolicy, _EMA, _LN, DDPMScheduler = import_upstream()
    _obs_dim = obs_dim()
    net = ConditionalUnet1D(
        input_dim=ACTION_DIM,
        local_cond_dim=None,
        global_cond_dim=_obs_dim * N_OBS_STEPS,   # obs_as_global_cond
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )
    scheduler = _make_scheduler(DDPMScheduler)
    policy = DiffusionUnetLowdimPolicy(
        model=net,
        noise_scheduler=scheduler,
        horizon=HORIZON,
        obs_dim=_obs_dim,
        action_dim=ACTION_DIM,
        n_action_steps=N_ACTION_STEPS,
        n_obs_steps=N_OBS_STEPS,
        num_inference_steps=num_inference_steps,
        obs_as_global_cond=True,
        oa_step_convention=True,
    ).to(device)
    _patch_mask_generator_device(policy, device)
    return policy


def _build_transformer_policy(device: str, num_inference_steps: int):
    """Official DiffusionTransformerLowdimPolicy with the paper's lowdim
    transformer hyperparameters (train_diffusion_transformer_lowdim: n_layer=8,
    n_head=4, n_emb=256, p_drop_attn=0.3, causal_attn=True, obs_as_cond=True)."""
    _CU, _Pol, _EMA, _LN, DDPMScheduler = import_upstream()
    TransformerForDiffusion, DiffusionTransformerLowdimPolicy = import_transformer()
    _obs_dim = obs_dim()
    net = TransformerForDiffusion(
        input_dim=ACTION_DIM,
        output_dim=ACTION_DIM,
        horizon=HORIZON,
        n_obs_steps=N_OBS_STEPS,
        cond_dim=_obs_dim,          # obs_as_cond -> cond_dim>0
        n_layer=8,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=0,
    )
    scheduler = _make_scheduler(DDPMScheduler)
    policy = DiffusionTransformerLowdimPolicy(
        model=net,
        noise_scheduler=scheduler,
        horizon=HORIZON,
        obs_dim=_obs_dim,
        action_dim=ACTION_DIM,
        n_action_steps=N_ACTION_STEPS,
        n_obs_steps=N_OBS_STEPS,
        num_inference_steps=num_inference_steps,
        obs_as_cond=True,
    ).to(device)
    _patch_mask_generator_device(policy, device)
    return policy


def _obs_vector(obj_data_capture: np.ndarray, anchor_pose: np.ndarray) -> np.ndarray:
    """(obs_dim,) obs = object poses at this capture [+ gripper anchor pose].

    When USE_ANCHOR is False the gripper anchor is omitted so the policy must
    condition on object poses alone (TP-Transformer-style, no proprioception)."""
    objs = obj_data_capture[:, :N_DIMS].reshape(-1)
    if not USE_ANCHOR:
        return objs.astype(np.float32)
    anchor = np.asarray(anchor_pose[:N_DIMS]).reshape(-1)
    return np.concatenate([objs, anchor]).astype(np.float32)


def segment_bounds(obj_data, img_inds, real_len):
    """List of (capture_idx, start, end) for the non-empty segments."""
    n_captures = obj_data.shape[0]
    out = []
    for i in range(n_captures):
        s = int(img_inds[i]); e = int(img_inds[i + 1]) if i < n_captures - 1 else int(real_len)
        if e > s:
            out.append((i, s, e))
    return out


def _obs_stack(obj_data_capture: np.ndarray, traj_data: np.ndarray,
               obs_step_inds: List[int]) -> np.ndarray:
    """(N_OBS_STEPS, obs_dim) obs = for each of the To observation timesteps,
    the capture's (static) object poses [+ gripper pose at that timestep].

    obs_step_inds are the To buffer timestep indices (already clamped to the
    segment range for front/back padding, mirroring DP's SequenceSampler which
    repeats the edge frame when a window overruns the episode)."""
    rows = []
    for t in obs_step_inds:
        anchor = traj_data[t, :N_DIMS]
        rows.append(_obs_vector(obj_data_capture, anchor))
    return np.stack(rows, 0).astype(np.float32)         # (To, obs_dim)


def windows_from_sample(obj_data: np.ndarray, traj_data: np.ndarray,
                        img_inds: np.ndarray, padding_mask: np.ndarray
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Slice a demonstration into DP-style (obs, action) training windows,
    following the official SequenceSampler convention (per capture-segment).

    For each segment [s,e) we mirror DP's create_indices/sample_sequence with
    pad_before = N_OBS_STEPS-1 and pad_after = N_ACTION_STEPS-1, and the
    oa_step_convention (obs occupies the first To steps; the executed action
    chunk begins at step To-1). Concretely, for each window start index `idx`
    running from -(To-1) to (seg_len - HORIZON + Ta-1):
      - obs    = (To, obs_dim): object poses (static in-segment) + gripper pose
                 at buffer steps [idx, idx+To), edge-repeated when out of range.
      - action = (HORIZON, ACTION_DIM): the trajectory chunk [idx, idx+HORIZON),
                 edge-repeated (front/back) when out of range -- DP pads with
                 edge values, not zeros.
    Stride 1 (every start index), matching DP's SequenceSampler.
    """
    real_len = int((~padding_mask.astype(bool)).sum())
    To = N_OBS_STEPS
    Ta = N_ACTION_STEPS
    pad_before = To - 1
    pad_after = Ta - 1
    pairs = []
    for ci, s, e in segment_bounds(obj_data, img_inds, real_len):
        seg_len = e - s
        # DP: idx in [-pad_before, seg_len - horizon + pad_after]
        min_start = -pad_before
        max_start = seg_len - HORIZON + pad_after
        for idx in range(min_start, max_start + 1):
            # buffer timesteps for the HORIZON action chunk, clamped into the
            # segment and edge-repeated outside it (matches sample_sequence).
            act = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
            for j in range(HORIZON):
                t = min(max(idx + j, 0), seg_len - 1) + s
                act[j] = traj_data[t, :ACTION_DIM]
            # obs timesteps: first To steps of the window, same clamping.
            obs_inds = [min(max(idx + j, 0), seg_len - 1) + s for j in range(To)]
            obs = _obs_stack(obj_data[ci], traj_data, obs_inds)     # (To, obs_dim)
            pairs.append((obs, act))
    return pairs
