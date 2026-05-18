"""Generate CNEP test-set predictions using a trained checkpoint per (action, seed).

Adapted from upstream `cnep-master/baxter/cnep_assembly_predictions.py`.
Reads checkpoints from <output-root>/<action>/<seed>/cnep.pt, loads the test
trajectories + features from the packaged baseline pickle (canonical:
``baseline_dataset_n15_v3t3.pickle``), runs inference per test demo,
and de-normalises each predicted trajectory using `minmax7`.

Output:
    <out>/cnep_predictions<suffix>.pickle : dict {action: [arr_seed_0, arr_seed_1, ...]}
Each `arr_seed_i` is a numpy array of shape (N_test, T, 7) in the original
pose units (x, y, z, qx, qy, qz, qw), comparable to ``test_traj_global``.
``N_test`` follows the splits manifest used to build the pickle.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from cnep_g import CNEP


CANONICAL_DATASET_NAME = "baseline_dataset_n15_v3t3.pickle"
DEFAULT_DATA = HERE.parent / "data" / CANONICAL_DATASET_NAME
DEFAULT_OUTPUT_ROOT = HERE / "outputs"
DEFAULT_OUT_DIR = HERE / "predictions"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CNEP test predictions.")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                   help=f"Packaged baseline data "
                        f"(default: baselines/data/{CANONICAL_DATASET_NAME}).")
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                   help="Where to find <action>/<seed>/cnep.pt")
    p.add_argument("--actions", nargs="+", default=["action_0", "action_1", "action_2"])
    p.add_argument("--seed-indices", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num-decoders", type=int, default=2)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--out-suffix", type=str, default="",
                   help="Optional suffix for prediction filename, e.g. '_resaved'.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for inference (cpu is fine here).")
    return p.parse_args()


def _as_3d_test_traj(test_traj: torch.Tensor) -> torch.Tensor:
    """Normalise legacy (T, 7) cells to the new (N_test, T, 7) layout."""
    if test_traj.dim() == 2:
        return test_traj.unsqueeze(0)
    return test_traj


def _as_2d_test_feats(test_feats: torch.Tensor) -> torch.Tensor:
    """Normalise (1280,) or (1, 1280) feats to (N_test, 1280)."""
    if test_feats.dim() == 1:
        return test_feats.unsqueeze(0)
    return test_feats


def predict_one_split(
    action: str,
    data: dict,
    args: argparse.Namespace,
    device: str,
) -> np.ndarray:
    seed = int(data["seed"])
    test_trajs = _as_3d_test_traj(data["test_traj_global_cnep"])   # (N, T, 7)
    test_feats = _as_2d_test_feats(data["test_feats"])             # (N, 1280)
    n_test, t_steps, dy = test_trajs.shape
    if test_feats.shape[0] != n_test:
        raise ValueError(
            f"{action} seed={seed}: test_feats has N={test_feats.shape[0]} but "
            f"test_traj_global_cnep has N={n_test}. Rebuild the pickle with "
            f"the latest prepare_baseline_dataset.py."
        )

    dx, dg, dims = 1, 256, test_feats.shape[-1]
    n_max = 1
    m_max = t_steps - 1
    batch_size = 1

    cnep = CNEP(
        dx + dg, dy, n_max, m_max, [512, 512],
        num_decoders=args.num_decoders, decoder_hidden_dims=[512, 512],
        batch_size=batch_size, scale_coefs=True, device=device,
    )
    ckpt = Path(args.output_root) / action / str(seed) / "cnep.pt"
    cnep.load_state_dict(torch.load(ckpt, map_location=device))

    mm_raw = data["minmax7"]
    minmax = mm_raw.cpu().numpy() if isinstance(mm_raw, torch.Tensor) else np.asarray(mm_raw)

    out_trajs: List[np.ndarray] = []
    for kk in range(n_test):
        test_traj = test_trajs[kk]      # (T, 7)
        test_feat = test_feats[kk]      # (1280,)

        obs = torch.zeros((batch_size, n_max, dx + dy), device=device)
        tar_x = torch.zeros((batch_size, m_max, dx), device=device)
        obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
        img_in = torch.zeros((batch_size, dims), device=device)

        obs[0, 0, :dx] = 0.0
        obs[0, 0, dx:] = test_traj[0]
        obs_mask[0, 0] = True
        img_in[0] = test_feat

        m_ids = torch.arange(1, t_steps)
        tar_x[0, :, :dx] = (m_ids.float() / t_steps).unsqueeze(1)

        with torch.no_grad():
            pred, gate = cnep.val(obs, tar_x, obs_mask, img_in)
            dec_id = torch.argmax(gate.squeeze(1), dim=-1)
            traj = pred[dec_id, 0, :, :dy].squeeze(0)

        traj = np.concatenate([test_traj[0].reshape(1, -1).cpu().numpy(), traj.cpu().numpy()])
        for i in range(dy):
            lo, hi = float(minmax[i][0]), float(minmax[i][1])
            traj[:, i] = 0.5 * (traj[:, i] + 1) * (hi - lo) + lo
        out_trajs.append(traj)

    return np.stack(out_trajs, axis=0)   # (N_test, T, 7)


def main() -> None:
    args = parse_args()
    device = args.device
    print(f"Device: {device}")
    print(f"Loading data from: {args.data}")
    with open(args.data, "rb") as f:
        all_data = pickle.load(f)

    predictions: dict = {}
    for action in args.actions:
        if action not in all_data:
            print(f"WARNING: action '{action}' not in data; skipping")
            continue
        predictions[action] = []
        entries = all_data[action]
        for ii in args.seed_indices:
            if ii < 0 or ii >= len(entries):
                print(f"WARNING: seed index {ii} out of range for '{action}' (len={len(entries)})")
                continue
            t = predict_one_split(action, entries[ii], args, device)
            predictions[action].append(t)
            print(f"  [{action}/seed_idx={ii}] CNEP shape={t.shape}  (N_test, T, 7)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cnep_predictions{args.out_suffix}.pickle"
    with open(out_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
