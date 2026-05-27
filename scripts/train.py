import argparse

from tp_transformer.config import TrainConfig
from tp_transformer.train import train_model


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    if args.model_name:
        cfg.model_name = args.model_name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.total_epochs is not None:
        cfg.total_epochs = args.total_epochs
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
    if args.save_interval is not None:
        cfg.save_interval = args.save_interval
    if args.print_interval is not None:
        cfg.print_interval = args.print_interval
    if args.save_optimizer:
        cfg.save_optimizer = True
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.min_lr is not None:
        cfg.min_lr = args.min_lr
    if args.output_root:
        cfg.output_root = args.output_root
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TP-Transformer model.")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="When --splits is set, this seed selects which "
                             "(action, seed) cell in the manifest to use (so "
                             "pass a seed that exists in the manifest).")
    parser.add_argument("--splits", type=str, default=None,
                        help="Optional YAML manifest from scripts/prepare_splits.py. "
                             "When set, train/valid/test demos come from the "
                             "manifest instead of random sampling -- this is "
                             "how TP-Transformer stays aligned with the "
                             "baselines for a fair head-to-head comparison.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--total-epochs", type=int, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--augmentation-method", type=str, default=None,
                        choices=["tp", "random"],
                        help="Augmentation method: 'tp' (task-parameterized) or "
                             "'random' (random rotation). Used for Experiment 1.")
    parser.add_argument("--aug-date", type=str, default=None,
                        help="Date folder under augmentation/<date>/ "
                             "(only relevant when --augmentation-method=tp).")
    parser.add_argument("--save-interval", type=int, default=None,
                        help="Save periodic + last checkpoint every N epochs.")
    parser.add_argument("--print-interval", type=int, default=None,
                        help="Log metrics every N epochs.")
    parser.add_argument("--save-optimizer", action="store_true",
                        help="If set, periodic checkpoints also include "
                             "optimizer/scheduler state (larger files).")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None,
                        help="LR scheduler floor; training stops once LR "
                             "reaches this value (default 1e-9).")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root dir for checkpoints + logs "
                             "(<output_root>/<model_name>/<seed>/). "
                             "On PCS compute nodes set this to "
                             "/shared/$USER/RingAIAutoAnnotation/eval "
                             "since /home is read-only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    train_model(cfg)


if __name__ == "__main__":
    main()
