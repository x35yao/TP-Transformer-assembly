"""Generate a shared train/valid/test split manifest for both pipelines.

Both `baselines/prepare_baseline_dataset.py` (CNEP / CNMP / TP-GMM / TP-ProMP)
and `src/tp_transformer/data.py` (TP-Transformer) can consume the YAML this
script writes via their `--splits` flag. Using the same manifest in both
pipelines guarantees they evaluate on the same train / valid / test demos,
which is the only way a head-to-head MSE comparison is meaningful.

Design: "reserve eval first"
----------------------------
For each (action, seed) we partition the eligible demos in this order:
  1. Reserve `--num-test` demos for the test set.
  2. Reserve `--num-validation` demos for validation (model selection).
  3. Shuffle the remaining demos deterministically into the train pool.
  4. Take the first `--num-train` demos from that shuffled pool.

Two properties this guarantees:
  * Test / valid sets are FIXED across n_train sweeps. If you regenerate
    the manifest with the same seeds and the same num_validation/num_test
    but a different num_train, the test and valid lists are bit-identical
    -- only the train list changes. This is what makes "MSE vs n_train"
    curves meaningful.
  * The train list for n_train = K1 is a PREFIX of the train list for
    n_train = K2 when K1 < K2, so smaller-train experiments use a strict
    subset of the larger-train experiments' data.

Capacity check: each action must have at least
`num_train + num_validation + num_test` eligible demos; action_2 is the
bottleneck (21 demos), so 15 + 3 + 3 = 21 is the tightest feasible config
at num_train = 15.

Output schema (YAML):

    meta:
      num_train: 15
      num_validation: 3
      num_test: 3
      split_strategy: reserve_eval
      seeds: [9871, 9872, 9873]
      actions: [action_0, action_1, action_2]
      processed_dir: data/processed
      generated_with: scripts/prepare_splits.py
    splits:
      action_0:
        9871:
          train: [<demo_id>, ...]   # length = num_train
          valid: [<demo_id>, ...]   # length = num_validation
          test:  [<demo_id>, ...]   # length = num_test
        9872: {...}
        9873: {...}
      action_1: {...}
      action_2: {...}

CLI:
    python scripts\\prepare_splits.py                               # 15 / 3 / 3
    python scripts\\prepare_splits.py --num-train 10 \\
        --out data\\splits\\n10_v3t3.yaml                           # sweep point

TODO (follow-up commit): the shared helpers below currently live in
`baselines/prepare_baseline_dataset.py` and are imported via a `sys.path`
tweak. Factor them (and `BAD_DEMOS`) into a small `src/dataset_common.py` so
both pipelines + this script import from one canonical location, and lift
the bad-demo blacklist into `data/task_config.yaml` so the filter becomes
data, not code.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BASELINES_DIR = REPO_ROOT / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from prepare_baseline_dataset import (  # noqa: E402
    DEFAULT_PROCESSED_DIR,
    DEFAULT_TASK_CONFIG,
    DEFAULT_SEEDS,
    DEFAULT_ACTIONS,
    DEFAULT_NUM_TRAIN,
    BAD_DEMOS,
    _load_task_config,
    _load_demo_dataset,
)


DEFAULT_OUT = REPO_ROOT / "data" / "splits" / "n15_v3t3.yaml"
DEFAULT_NUM_VALIDATION = 3
DEFAULT_NUM_TEST = 3


def _split_demos_reserve_eval(
    demos: List[str],
    num_train: int,
    num_validation: int,
    num_test: int,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    """Partition `demos` into (train, valid, test) using reserve-eval-first.

    Order of RNG consumption (do not reorder; it determines reproducibility):
      1. `rng.sample(pool, num_test)`               -> reserve test
      2. `rng.sample(remaining, num_validation)`    -> reserve valid
      3. `rng.shuffle(train_pool)`                  -> deterministic order
      4. `train_pool[:num_train]`                   -> take prefix

    Each bucket is alphabetically sorted on the way out so downstream
    consumers see a stable, diff-friendly order regardless of sample order.
    """
    n_required = num_train + num_validation + num_test
    if n_required > len(demos):
        raise ValueError(
            f"Need {num_train}+{num_validation}+{num_test}={n_required} demos "
            f"but only {len(demos)} eligible. Lower one of the counts or pick "
            f"a different action."
        )

    pool = list(demos)
    test = rng.sample(pool, num_test)
    after_test = [d for d in pool if d not in test]
    valid = rng.sample(after_test, num_validation)
    train_pool = [d for d in after_test if d not in valid]
    rng.shuffle(train_pool)
    train = train_pool[:num_train]

    return sorted(train), sorted(valid), sorted(test)


def build_splits(
    processed_dir: Path,
    task_config: Path,
    actions: List[str],
    seeds: List[int],
    num_train: int,
    num_validation: int,
    num_test: int,
) -> Tuple[Dict, Dict[str, List[str]]]:
    """Return (splits_dict, per_action_eligible).

    `splits_dict` is shaped:
        {action: {seed: {'train': [...], 'valid': [...], 'test': [...]}}}
    """
    all_objs = _load_task_config(task_config)

    per_action_demos: Dict[str, List[str]] = {}
    for action in actions:
        demo_dataset, _ = _load_demo_dataset(processed_dir, action, all_objs)
        per_action_demos[action] = list(demo_dataset.keys())
        n_required = num_train + num_validation + num_test
        if n_required > len(per_action_demos[action]):
            raise ValueError(
                f"action={action}: {len(per_action_demos[action])} eligible "
                f"demos but {n_required} requested "
                f"(train={num_train}, valid={num_validation}, test={num_test})."
            )
        print(f"  {action}: {len(per_action_demos[action])} eligible -> "
              f"train={num_train}, valid={num_validation}, test={num_test}")

    splits: Dict[str, Dict[int, Dict[str, List[str]]]] = {a: {} for a in actions}
    for seed in seeds:
        # CRITICAL: one Random per seed, shared across all actions. Re-seeding
        # per action would make action_1's split independent of action_0's
        # RNG consumption, but here we deliberately let actions share state
        # so the manifest is a single deterministic function of (seed,
        # num_train, num_validation, num_test, actions order).
        rng = random.Random(seed)
        for action in actions:
            demos = list(per_action_demos[action])
            train, valid, test = _split_demos_reserve_eval(
                demos, num_train, num_validation, num_test, rng,
            )
            splits[action][seed] = {
                "train": list(train),
                "valid": list(valid),
                "test": list(test),
            }

    return splits, per_action_demos


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
                   help="Root holding action_{0,1,2}/... (default: data/processed).")
    p.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG,
                   help="task_config.yaml (default: data/task_config.yaml).")
    p.add_argument("--actions", nargs="+", default=list(DEFAULT_ACTIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                   help="One split per seed; canonical pickle uses 9871 9872 9873.")
    p.add_argument("--num-train", type=int, default=DEFAULT_NUM_TRAIN,
                   help="Train demos per (action, seed). Default 15.")
    p.add_argument("--num-validation", type=int, default=DEFAULT_NUM_VALIDATION,
                   help="Valid demos per (action, seed). Default 3 (uniform).")
    p.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST,
                   help="Test demos per (action, seed). Default 3 (uniform). "
                        "Bumping this beyond ~3 leaves no headroom for "
                        "num_train=15 on action_2 (only 21 eligible demos).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output YAML path (default: data/splits/n15_v3t3.yaml).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"processed-dir  : {args.processed_dir}")
    print(f"task-config    : {args.task_config}")
    print(f"actions        : {args.actions}")
    print(f"seeds          : {args.seeds}")
    print(f"num-train      : {args.num_train}")
    print(f"num-validation : {args.num_validation}")
    print(f"num-test       : {args.num_test}")
    print(f"out            : {args.out}")

    splits, per_action_demos = build_splits(
        processed_dir=args.processed_dir,
        task_config=args.task_config,
        actions=args.actions,
        seeds=args.seeds,
        num_train=args.num_train,
        num_validation=args.num_validation,
        num_test=args.num_test,
    )

    manifest = {
        "meta": {
            "num_train": args.num_train,
            "num_validation": args.num_validation,
            "num_test": args.num_test,
            "split_strategy": "reserve_eval",
            "seeds": args.seeds,
            "actions": args.actions,
            "processed_dir": str(args.processed_dir),
            "bad_demos": {a: list(BAD_DEMOS.get(a, [])) for a in args.actions},
            "eligible_demos": {a: per_action_demos[a] for a in args.actions},
            "generated_with": "scripts/prepare_splits.py",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "splits": splits,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    print(f"\nwrote {args.out}")

    first = args.actions[0]
    first_seed = args.seeds[0]
    cell = splits[first][first_seed]
    print(f"\nsample -- {first} / seed={first_seed}:")
    print(f"  train ({len(cell['train'])}): {cell['train']}")
    print(f"  valid ({len(cell['valid'])}): {cell['valid']}")
    print(f"  test  ({len(cell['test'])}): {cell['test']}")


if __name__ == "__main__":
    main()
