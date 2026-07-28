"""Train the Diffusion Policy (CNN, low-dim) baseline -- Approach B.

Reuses the official DiffusionUnetLowdimPolicy.compute_loss for the diffusion
training objective (masking, conditioning, add_noise -> predict-epsilon -> MSE
are all upstream code), the official LinearNormalizer (mode='limits' -> [-1,1]),
and the official EMAModel. We only provide the sliding-window dataset adapter
(fixed HORIZON windows over the TP-Transformer data pipeline, same splits and
same on-the-fly TP-augmentation) and the outer epoch loop.

Usage:
    python baselines/diffusion_policy/train_dp.py \
        --splits data/splits/n15_v3t3.yaml --seed 9871 --augmentation none \
        --output-root /shared/$USER/.../eval/exp1/dp/none/15
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from tp_transformer.config import TrainConfig
from tp_transformer.data import build_datasets
import dp_common as C


class WindowDataset(Dataset):
    """Fixed-HORIZON (obs, action) windows over the TP-Transformer dataset.

    Augmentation is expensive (TP transforms in TrajectoryDataset.__getitem__),
    so we augment each demonstration **once per epoch** and cache all of its
    windows. `resample()` (called at each epoch start) does a single
    augmentation pass over every demo and flattens the windows; __getitem__ is
    then a cheap lookup. This resamples the augmentation every epoch (matching
    TP-Transformer) without re-augmenting per window.

    If `action_idx` is given, only demos of that subtask are included (used for
    per-subtask DP models, matching Chi et al.'s one-policy-per-task convention).
    """

    def __init__(self, base_ds, action_idx=None):
        self.base = base_ds
        self.action_idx = action_idx
        self._windows = []   # flat list of (obs, action) for the current epoch
        self.resample()

    def _keep(self, si):
        if self.action_idx is None:
            return True
        _o, _t, _th, _w, atag, _p, _i, _pk, _rl = self.base[si]
        return int(atag.argmax().item()) == self.action_idx

    def resample(self):
        wins = []
        for si in range(len(self.base)):
            obj, traj, _th, _w, atag, pad, img, _pk, _rl = self.base[si]
            if self.action_idx is not None and int(atag.argmax().item()) != self.action_idx:
                continue
            wins.extend(C.windows_from_sample(
                obj.numpy(), traj.numpy(), img.numpy(), pad.numpy()))
        self._windows = wins

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        obs, act = self._windows[idx]
        return {"obs": torch.from_numpy(obs), "action": torch.from_numpy(act)}


def fit_normalizer(base_ds, LinearNormalizer, action_idx=None):
    """Fit the official LinearNormalizer on obs+action over a scan of windows."""
    obs_all, act_all = [], []
    for si in range(len(base_ds)):
        obj, traj, _th, _w, atag, pad, img, _pk, _rl = base_ds[si]
        if action_idx is not None and int(atag.argmax().item()) != action_idx:
            continue
        for obs, act in C.windows_from_sample(obj.numpy(), traj.numpy(), img.numpy(), pad.numpy()):
            obs_all.append(obs); act_all.append(act)
    data = {
        "obs": torch.from_numpy(np.concatenate(obs_all, 0)),      # (Nw, OBS_DIM)
        "action": torch.from_numpy(np.concatenate(act_all, 0)),   # (Nw, HORIZON, ACTION_DIM)
    }
    norm = LinearNormalizer()
    norm.fit(data, last_n_dims=1, mode="limits")
    return norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--augmentation", choices=["tp", "none"], default="none")
    ap.add_argument("--action-idx", type=int, default=None,
                    help="If set, train a per-subtask model on this action only "
                         "(0/1/2). Default None = joint model over all subtasks.")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--backbone", choices=["cnn", "transformer"], default="cnn",
                    help="DP denoising backbone: 'cnn' (ConditionalUnet1D, "
                         "default) or 'transformer' (TransformerForDiffusion).")
    ap.add_argument("--n-obs-steps", type=int, default=None,
                    help="Observation history length. Default: 2 for cnn, 3 for "
                         "transformer (official lowdim configs).")
    ap.add_argument("--no-anchor", action="store_true",
                    help="Drop the gripper anchor from obs -> condition on "
                         "object poses only (TP-Transformer-style, no "
                         "proprioception). Tests whether DP uses object poses.")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    _CU, _Pol, EMAModel, LinearNormalizer, _Sched = C.import_upstream()
    C.set_use_anchor(not args.no_anchor)
    # official lowdim obs-history: CNN uses n_obs_steps=2, transformer uses 3.
    C.set_n_obs_steps(args.n_obs_steps if args.n_obs_steps is not None
                      else (3 if args.backbone == "transformer" else 2))

    cfg = TrainConfig()
    cfg.splits_file = args.splits; cfg.seed = args.seed
    cfg.augmentation_method = args.augmentation
    train_ds, _v, _t, stats = build_datasets(cfg)

    device = args.device
    policy = C.build_policy(device=device, backbone=args.backbone)
    normalizer = fit_normalizer(train_ds, LinearNormalizer, action_idx=args.action_idx)
    policy.set_normalizer(normalizer)
    # set_normalizer loads CPU-fit buffers into the policy; move everything
    # (incl. normalizer scale/offset + mask generator) to the device again so
    # compute_loss runs entirely on GPU.
    policy.to(device)

    win_ds = WindowDataset(train_ds, action_idx=args.action_idx)
    # num_workers=0: __getitem__ is now a trivial cached lookup (augmentation is
    # done once per epoch via win_ds.resample() in the main process), so workers
    # add no benefit and would only risk serving a stale window cache.
    loader = DataLoader(win_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Both official lowdim configs use a cosine LR schedule with linear warmup
    # (CNN warmup 500 / transformer 1000). The transformer additionally uses the
    # paper's param-group weight-decay split (get_optimizer); the CNN uses AdamW.
    if args.backbone == "transformer":
        opt = policy.get_optimizer(weight_decay=1e-3, learning_rate=args.lr,
                                   betas=(0.9, 0.95))
        warmup = 1000
    else:
        opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999),
                                weight_decay=1e-6)
        warmup = 500
    try:
        from diffusers.optimization import get_scheduler
        total_steps = args.epochs * max(len(loader), 1)
        lr_sched = get_scheduler("cosine", optimizer=opt,
                                 num_warmup_steps=warmup,
                                 num_training_steps=total_steps)
    except Exception:
        lr_sched = None
    ema = EMAModel(model=copy.deepcopy(policy), update_after_step=0, inv_gamma=1.0,
                   power=0.75, min_value=0.0, max_value=0.9999)

    out_dir = Path(args.output_root) / str(args.seed)
    if args.action_idx is not None:
        out_dir = out_dir / f"action_{args.action_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "training_log.txt", "w")

    def logline(s):
        print(s); log.write(s + "\n"); log.flush()

    logline(f"DP-{args.backbone.upper()}(B) | seed={args.seed} aug={args.augmentation} "
            f"anchor={not args.no_anchor} obs_dim={C.obs_dim()} n_obs_steps={C.N_OBS_STEPS} "
            f"windows={len(win_ds)} epochs={args.epochs} horizon={C.HORIZON} "
            f"n_act={C.N_ACTION_STEPS} device={device}")

    policy.train()
    for epoch in range(args.epochs):
        win_ds.resample()   # one augmentation pass over all demos this epoch
        tot = 0.0; nb = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = policy.compute_loss(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            if lr_sched is not None:
                lr_sched.step()
            ema.step(policy)
            tot += float(loss.item()); nb += 1
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            logline(f"epoch {epoch:5d}  loss {tot / max(nb,1):.5f}")

    torch.save({
        # Only the EMA weights are used for evaluation (predict_dp loads
        # 'policy_ema'); the raw 'policy' state is redundant and ~doubles the
        # checkpoint size, so we omit it to conserve disk quota.
        "policy_ema": ema.averaged_model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "train_mean": np.asarray(stats["mean"]).reshape(3),
        "train_std": float(stats["std"]),
        "augmentation": args.augmentation, "seed": args.seed,
        "use_anchor": not args.no_anchor,
        "backbone": args.backbone,
        "n_obs_steps": C.N_OBS_STEPS,
    }, out_dir / "dp_model.pt")
    logline(f"saved {out_dir/'dp_model.pt'}")
    log.close()


if __name__ == "__main__":
    main()
