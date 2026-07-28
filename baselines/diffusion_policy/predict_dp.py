"""Predict test-set trajectories with the trained Diffusion Policy (Approach B),
using the official receding-horizon `policy.predict_action`, and write
predictions in the TP-Transformer schema
(`<root>/dp/<seed>/predictions.pickle` = {action: (N_test, T, 7)}).

For each test demo we roll out closed-loop over the trajectory: at each control
step the observation is the object poses of the *current* camera-capture segment
plus the gripper anchor (the last executed pose); the policy predicts a HORIZON
window and we execute N_ACTION_STEPS of it, then advance. The capture (object
poses) switches when we cross the next img_ind, exactly mirroring the
TP-Transformer's segment structure. This matches DP's receding-horizon
inference while consuming the same per-capture object poses as TP-Transformer.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from tp_transformer.config import TrainConfig
from tp_transformer.data import build_datasets, TASK_DIMS
import dp_common as C


def capture_for_step(bounds, step):
    """Return the capture index whose segment [s,e) contains `step`
    (clamp to the last capture beyond the final boundary)."""
    for ci, s, e in bounds:
        if s <= step < e:
            return ci
    return bounds[-1][0]


@torch.no_grad()
def rollout_demo(policy, obj_np, traj_np, img, real_len, n_dims, device):
    """Receding-horizon rollout for one demo -> (real_len, ACTION_DIM).

    Standard behavior-cloning setup: the policy is TRAINED conditioning on the
    ground-truth gripper history (windows_from_sample), and at INFERENCE it
    conditions on its own previously-executed poses (offline analog of Diffusion
    Policy's env.step state feedback -- we have no simulator). Object poses come
    from the current camera-capture segment. With N_OBS_STEPS>1 the obs is a
    stack of the last To gripper poses (edge-repeated at the very start, like
    DP's pad_before) + the (static per-segment) object poses.
    """
    bounds = C.segment_bounds(obj_np, img, real_len)
    result = np.zeros((real_len, C.ACTION_DIM), dtype=np.float32)
    To = C.N_OBS_STEPS
    # gripper-pose history buffer, seeded with the true initial pose repeated
    # (mirrors training front-padding pad_before = To-1).
    hist = [traj_np[0, :n_dims].copy() for _ in range(To)]
    step = 0
    while step < real_len:
        ci = capture_for_step(bounds, step)
        obs = C._obs_stack(obj_np[ci], np.stack(hist, 0),
                           list(range(To)))                         # (To, obs_dim)
        obs_t = torch.from_numpy(obs[None, ...]).float().to(device)  # (1, To, Do)
        out = policy.predict_action({"obs": obs_t})
        act = out["action"][0].cpu().numpy()                         # (N_ACTION_STEPS, ACTION_DIM)
        take = min(C.N_ACTION_STEPS, real_len - step)
        result[step:step + take] = act[:take]
        # advance the gripper history with the executed poses (own predictions)
        for k in range(take):
            hist.append(result[step + k, :n_dims].copy())
        hist = hist[-To:]
        step += take
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--augmentation", choices=["tp", "none"], default="none")
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--num-inference-steps", type=int, default=100)
    ap.add_argument("--per-subtask", action="store_true",
                    help="Load one model per action from "
                         "<model-root>/<seed>/action_<idx>/dp_model.pt and route "
                         "each demo to its action's model (Chi et al. one-policy-"
                         "per-task convention).")
    ap.add_argument("--no-anchor", action="store_true",
                    help="Force anchor-free obs (object poses only). Normally "
                         "auto-detected from the checkpoint's 'use_anchor' flag.")
    ap.add_argument("--backbone", choices=["cnn", "transformer"], default=None,
                    help="Force backbone. Normally auto-detected from the "
                         "checkpoint's 'backbone' flag (default cnn if absent).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    _CU, _Pol, _EMA, LinearNormalizer, _Sched = C.import_upstream()

    def _apply_anchor_mode(meta):
        """Set the global anchor mode + obs-history from checkpoint meta."""
        if args.no_anchor:
            C.set_use_anchor(False)
        else:
            C.set_use_anchor(bool(meta.get("use_anchor", True)))
        C.set_n_obs_steps(int(meta.get("n_obs_steps", 1)))

    def _backbone_for(meta):
        return args.backbone or meta.get("backbone", "cnn")

    def _load_policy(ckpt_path: Path):
        meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _apply_anchor_mode(meta)
        pol = C.build_policy(device=device, num_inference_steps=args.num_inference_steps,
                             backbone=_backbone_for(meta))
        # official EMA-weight evaluation
        pol.load_state_dict(meta["policy_ema"] if "policy_ema" in meta else meta["policy"])
        pol.to(device)
        pol.eval()
        return pol

    cfg = TrainConfig()
    cfg.splits_file = args.splits; cfg.seed = args.seed
    cfg.augmentation_method = args.augmentation
    _tr, _va, test_ds, stats = build_datasets(cfg)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    n_dims = len(TASK_DIMS)

    seed_dir = Path(args.model_root) / str(args.seed)
    policies: Dict[int, object] = {}
    if args.per_subtask:
        # one model per action; routed by the demo's action tag
        for aidx in range(len(cfg.tasks)):
            policies[aidx] = _load_policy(seed_dir / f"action_{aidx}" / "dp_model.pt")
        print(f"Loaded {len(policies)} per-subtask models from {seed_dir}")
    else:
        shared = _load_policy(seed_dir / "dp_model.pt")
        policies = None  # signal: use shared for all

    per_action: Dict[str, List[np.ndarray]] = {t: [] for t in cfg.tasks}
    for sidx, batch in enumerate(test_loader):
        obj_seq, traj_seq, _th, _w, atag, pad, img_inds, _pk, _rl = batch
        obj_np = obj_seq[0].numpy(); traj_np = traj_seq[0].numpy()
        img = img_inds[0].numpy(); real_len = int((~pad[0].numpy().astype(bool)).sum())
        aidx = int(torch.argmax(atag[0]).item())
        policy = policies[aidx] if args.per_subtask else shared
        res = rollout_demo(policy, obj_np, traj_np, img, real_len, n_dims, device)
        pred_valid = res[:, :n_dims]
        per_action[cfg.tasks[aidx]].append(pred_valid)
        print(f"  [{sidx+1}/{len(test_ds)}] action={cfg.tasks[aidx]} T_valid={pred_valid.shape[0]}")

    predictions: Dict[str, np.ndarray] = {}
    for a, lst in per_action.items():
        if lst:
            predictions[a] = np.stack(lst, 0)
            print(f"  action={a}: {predictions[a].shape}")

    out_dir = Path(args.out_root) / str(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_stat.pickle", "wb") as f:
        pickle.dump(stats, f)
    with open(out_dir / "predictions.pickle", "wb") as f:
        pickle.dump(predictions, f)
    print(f"Wrote {out_dir/'predictions.pickle'}")


if __name__ == "__main__":
    main()
