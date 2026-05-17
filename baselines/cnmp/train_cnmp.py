"""Train the CNMP baseline on the assembly tasks.

Adapted from upstream `cnep-master/baxter/train_cnep_with_mobilenet_v2_assembly.py`.
Key changes vs upstream:
- All paths resolved relative to this file (no implicit CWD dependence).
- CLI flags for action, seeds, epoch budget, number of demos, output root.
- Removed dead MobileNetV2 model loading (image features come pre-extracted
  from `../data/baseline_dataset.pickle`; the upstream script loaded the CNN but never used it).
- Trains CNMP only (CNEP has its own script under `baselines/cnep/`).

Data dependency:
    ../data/baseline_dataset.pickle  (shared with the CNEP baseline)
        - dict keyed by 'action_0' / 'action_1' / 'action_2'
        - each value: list of 3 pre-baked splits (one per seed)
        - each entry has keys (T = 150/186/197 for action_0/1/2):
            train_traj_global_cnep : (15, T, 7)    torch.float64  min-max normalised to [-1, 1]
            valid_traj_global_cnep : (4,  T, 7)    torch.float64  min-max normalised to [-1, 1]
            test_traj_global_cnep  : (T, 7)        torch.float64
            train_feats            : (15, 1280)    torch.float32  MobileNetV2 feats
            valid_feats            : (4,  1280)
            test_feats             : (1,  1280)
            minmax7                : (7, 2)        per-dim min/max for de-normalisation
            seed                   : int           (9871/9872/9873 in the current pickle)

Per-step input layout fed to CNMP (image-conditioned `_g` variant):
    obs      (B, n_max=1, dx+dy)   with obs[:, 0] = (t=0, pose@t=0)
    tar_x    (B, m_max=T-1, dx)    with tar_x[:, m] = (m+1)/T
    img_in   (B, 1280)             = train_feats[traj_id]   (paired with obs traj)
    targets  (B, m_max, dy)        = pose at steps 1 .. T-1
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from cnmp_g import CNMP


DEFAULT_DATA = HERE.parent / "data" / "baseline_dataset.pickle"
DEFAULT_OUTPUT_ROOT = HERE / "outputs"


def log_training_stats(log_message: str, log_file: Path) -> None:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNMP baseline on assembly data.")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                   help="Path to baseline_dataset.pickle (default: baselines/data/baseline_dataset.pickle).")
    p.add_argument("--actions", nargs="+", default=["action_0", "action_1", "action_2"])
    p.add_argument("--seed-indices", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num-demos", type=int, default=15,
                   help="Number of training demos to use (max 15; for the 1-15 sweep in Experiment 2).")
    p.add_argument("--num-valid", type=int, default=4,
                   help="Number of validation demos to use (max 4 due to valid_feats size).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-per-epoch", type=int, default=1000)
    p.add_argument("--snapshot-per-epoch", type=int, default=100_000)
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                   help="Root for run output directories: <root>/<action>/<seed>/cnmp.pt")
    p.add_argument("--device", type=str, default=None,
                   help="torch device, e.g. 'cuda:0' or 'cpu'. Defaults to cuda if available.")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def train_one_split(
    action: str,
    data: dict,
    args: argparse.Namespace,
    device: str,
) -> None:
    seed = int(data["seed"])
    num_demos = min(args.num_demos, data["train_traj_global_cnep"].shape[0])
    v_num_demos = min(args.num_valid, data["valid_feats"].shape[0])
    batch_size = args.batch_size

    train_trajs = data["train_traj_global_cnep"][:num_demos]
    val_trajs = data["valid_traj_global_cnep"][:v_num_demos]
    train_feats = data["train_feats"][:num_demos]
    val_feats = data["valid_feats"][:v_num_demos]

    t_steps = train_trajs.shape[1]
    dy = train_trajs.shape[-1]
    dx, dg, dims = 1, 256, train_feats.shape[-1]
    n_max = 1
    m_max = t_steps - 1

    obs = torch.zeros((batch_size, n_max, dx + dy), device=device)
    tar_x = torch.zeros((batch_size, m_max, dx), device=device)
    tar_y = torch.zeros((batch_size, m_max, dy), device=device)
    obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
    tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)
    img_in = torch.zeros((batch_size, dims), device=device)

    val_obs = torch.zeros((batch_size, n_max, dx + dy), device=device)
    val_tar_x = torch.zeros((batch_size, m_max, dx), device=device)
    val_tar_y = torch.zeros((batch_size, m_max, dy), device=device)
    val_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
    val_img_in = torch.zeros((batch_size, dims), device=device)

    m_ids_full = torch.arange(1, t_steps)

    def prepare_masked_batch(traj_ids):
        obs.zero_(); tar_x.zero_(); tar_y.zero_()
        obs_mask.zero_(); tar_mask.zero_(); img_in.zero_()
        for i, traj_id in enumerate(traj_ids):
            traj = train_trajs[traj_id]
            obs[i, 0, :dx] = 0.0
            obs[i, 0, dx:] = traj[0]
            obs_mask[i, 0] = True
            img_in[i] = train_feats[traj_id]
            tar_x[i, :, :dx] = (m_ids_full.float() / t_steps).unsqueeze(1)
            tar_y[i] = traj[m_ids_full]
            tar_mask[i] = True

    def prepare_masked_val_batch(traj_ids):
        val_obs.zero_(); val_tar_x.zero_(); val_tar_y.zero_()
        val_obs_mask.zero_(); val_img_in.zero_()
        for i, traj_id in enumerate(traj_ids):
            traj = val_trajs[traj_id]
            val_obs[i, 0, :dx] = 0.0
            val_obs[i, 0, dx:] = traj[0]
            val_obs_mask[i, 0] = True
            val_img_in[i] = val_feats[traj_id]
            val_tar_x[i, :, :dx] = (m_ids_full.float() / t_steps).unsqueeze(1)
            val_tar_y[i] = traj[m_ids_full]

    cnmp_ = CNMP(
        dx + dg, dy, n_max, m_max, [512, 512],
        decoder_hidden_dims=[512, 512], batch_size=batch_size, device=device,
    )
    opt = torch.optim.Adam(lr=args.lr, params=cnmp_.parameters())

    n_params = sum(p.numel() for p in cnmp_.parameters())
    print(f"[{action}/{seed}] cnmp params: {n_params:,}")

    if args.compile and torch.__version__ >= "2.0":
        cnmp = torch.compile(cnmp_)
    else:
        cnmp = cnmp_

    root = Path(args.output_root) / action / str(seed)
    root.mkdir(parents=True, exist_ok=True)
    log_file = root / "training_log.txt"
    tl_path = root / "cnmp_training_loss.pt"
    ve_path = root / "cnmp_validation_error.pt"
    best_path = root / "cnmp.pt"
    snapshot_path = root / "last_cnmp.pt"

    epoch_iter = max(1, num_demos // batch_size)
    min_vl = np.inf
    avg_loss = 0.0
    tl, ve = [], []

    # Deterministic full-coverage validation: every eval pass iterates over ALL
    # v_num_demos validation demos in fixed order, padding the final partial batch
    # by repeating its last id (those padded slots are excluded from the metric
    # via n_real). This avoids the upstream bug where each eval drew a random
    # subset of v_num_demos and picked "best" from a noisy single snapshot.
    val_batches = []
    for start in range(0, v_num_demos, batch_size):
        ids = list(range(start, min(start + batch_size, v_num_demos)))
        n_real = len(ids)
        while len(ids) < batch_size:
            ids.append(ids[-1])
        val_batches.append((ids, n_real))

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        traj_ids = torch.randperm(num_demos)[: batch_size * epoch_iter].chunk(epoch_iter)

        for i in range(epoch_iter):
            prepare_masked_batch(traj_ids[i])
            opt.zero_grad()
            pred = cnmp(obs, tar_x, obs_mask, img_in)
            loss = cnmp.loss(pred, tar_y, tar_mask)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        epoch_loss /= epoch_iter
        tl.append(epoch_loss)
        avg_loss += epoch_loss

        if epoch % args.val_per_epoch == 0:
            with torch.no_grad():
                sse = 0.0
                n_elem = 0
                for ids, n_real in val_batches:
                    prepare_masked_val_batch(ids)
                    p = cnmp.val(val_obs, val_tar_x, val_obs_mask, val_img_in)
                    vp_means = torch.nan_to_num(p[:n_real, :, :dy])
                    err = vp_means - val_tar_y[:n_real]
                    sse += err.pow(2).sum().item()
                    n_elem += err.numel()
                ve_avg = sse / n_elem if n_elem else float("inf")
                if ve_avg < min_vl:
                    min_vl = ve_avg
                    print(f"  [{action}/{seed}] CNMP new best: {min_vl:.6f}")
                    torch.save(cnmp_.state_dict(), best_path)
                ve.append(ve_avg)

            message = (
                f"Epoch: {epoch}, Loss: {avg_loss/args.val_per_epoch:.4f}, "
                f"Val MSE: {ve_avg:.6f}, Min Err: {min_vl:.6f}\n"
            )
            print(message, end="")
            log_training_stats(message, log_file)
            avg_loss = 0.0

        if epoch % args.snapshot_per_epoch == 0 and epoch > 1:
            torch.save(cnmp_.state_dict(), snapshot_path)

    torch.save(torch.Tensor(tl), tl_path)
    torch.save(torch.Tensor(ve), ve_path)


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading data from: {args.data}")
    with open(args.data, "rb") as f:
        all_data = pickle.load(f)

    for action in args.actions:
        if action not in all_data:
            print(f"WARNING: action '{action}' not in data; skipping")
            continue
        entries = all_data[action]
        for ii in args.seed_indices:
            if ii < 0 or ii >= len(entries):
                print(f"WARNING: seed index {ii} out of range for '{action}' (len={len(entries)})")
                continue
            t0 = time.time()
            train_one_split(action, entries[ii], args, device)
            print(f"[{action}/seed_idx={ii}] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
