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
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TP-Transformer model.")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--total-epochs", type=int, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    train_model(cfg)


if __name__ == "__main__":
    main()
