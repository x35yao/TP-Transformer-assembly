"""CLI wrapper for `tp_transformer.predict.predict_test_set`.

Mirrors `scripts/train.py`'s flag set so you can re-use the exact same
arguments you trained with -- in particular `--splits`, `--seed`,
`--model-name`, and `--output-root` -- and the predict script will pick up
the matching checkpoint at
`<output_root>/<model_name>/<seed>/model_best.pth`.

Example (Experiment 2, TP-aug, seed 9871):

    python scripts/predict.py \
        --splits data/splits/n15_v3t3.yaml \
        --seed 9871 \
        --model-name tp_aug \
        --output-root ./transformer \
        --augmentation-method tp \
        --aug-date 2025-02-17

Output:
    ./transformer/tp_aug/9871/predictions.pickle
        dict {action_name: np.ndarray(N_test, T, 7)}
    ./transformer/tp_aug/9871/train_stat.pickle  (if missing; written by training already)

Override the checkpoint or output path explicitly with --checkpoint / --out
if you want to predict from a non-default location (e.g. model_last.pth,
or a per-epoch checkpoint).
"""

from __future__ import annotations

import argparse

from tp_transformer.config import TrainConfig
from tp_transformer.predict import predict_test_set


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    if args.model_name:
        cfg.model_name = args.model_name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.processed_dir:
        cfg.processed_dir = args.processed_dir
    if args.raw_dir:
        cfg.raw_dir = args.raw_dir
    if args.config_path:
        cfg.path_config_file = args.config_path
    if args.splits:
        cfg.splits_file = args.splits
    if args.augmentation_method:
        cfg.augmentation_method = args.augmentation_method
    if args.aug_date:
        cfg.aug_date = args.aug_date
    if args.output_root:
        cfg.output_root = args.output_root
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TP-Transformer on the test set and save predictions."
    )
    parser.add_argument("--model-name", type=str, default=None,
                        help="Checkpoint folder name "
                             "(<output_root>/<model_name>/<seed>/).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed used at training time -- selects the "
                             "matching (action, seed) cell in the manifest "
                             "and the matching checkpoint subfolder.")
    parser.add_argument("--splits", type=str, default=None,
                        help="Splits manifest from scripts/prepare_splits.py "
                             "(must be the same one used at training).")
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--augmentation-method", type=str, default=None,
                        choices=["tp", "random", "none"],
                        help="Must match the augmentation method used at "
                             "training time so the model's expected input "
                             "shape matches.")
    parser.add_argument("--aug-date", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root dir for checkpoints + logs "
                             "(<output_root>/<model_name>/<seed>/). Default "
                             "matches scripts/train.py.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override the default "
                             "<output_root>/<model_name>/<seed>/model_best.pth.")
    parser.add_argument("--out", type=str, default=None,
                        help="Override the default "
                             "<output_root>/<model_name>/<seed>/predictions.pickle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    predict_test_set(cfg, checkpoint=args.checkpoint, out_path=args.out)


if __name__ == "__main__":
    main()
