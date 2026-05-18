# Baselines

Reference implementations and shared data for the methods we compare against
TP-Transformer on the assembly task:

| Folder | Method | Status |
|---|---|---|
| `cnep/` | Conditional Neural Expert Processes | ported, trains end-to-end |
| `cnmp/` | Conditional Neural Movement Primitives | ported, trains end-to-end |
| `tp_gmm/` | Task-Parameterised GMM | not yet added |
| `tp_promp/` | Task-Parameterised ProMP | not yet added |
| `eval/` | Cross-method metric aggregation | not yet added |

All baselines consume pickles produced by
[`prepare_baseline_dataset.py`](./prepare_baseline_dataset.py), written under
`baselines/data/`. **The default output name mirrors the split manifest**
(`data/splits/n15_v3t3.yaml` → `baselines/data/baseline_dataset_n15_v3t3.pickle`);
use a different YAML or pass `--out` explicitly for other experiments.

The script reads source trajectories, object poses, and demo images from
the repo-root `data/` directory, which is **shared with the TP-Transformer
pipeline** (`src/tp_transformer/data.py`). Both methods therefore start from
the same demos; the baseline builder just packages them into the format
CNEP / CNMP expect.

### Sharing the same split with TP-Transformer

For an apples-to-apples comparison, both pipelines must agree on which demos
land in train / valid / test for a given seed. The canonical workflow:

```powershell
# 1. Generate ONE shared splits manifest (consumed by both pipelines).
python scripts\prepare_splits.py `
    --num-train 15 --num-validation 3 --num-test 3 `
    --seeds 9871 9872 9873 `
    --out data\splits\n15_v3t3.yaml

# 2. Build the baseline dataset (default --out: baseline_dataset_n15_v3t3.pickle).
python baselines\prepare_baseline_dataset.py `
    --splits data\splits\n15_v3t3.yaml

# 3. Train TP-Transformer against the SAME manifest.
python scripts\train.py --splits data\splits\n15_v3t3.yaml --seed 9871
python scripts\train.py --splits data\splits\n15_v3t3.yaml --seed 9872
python scripts\train.py --splits data\splits\n15_v3t3.yaml --seed 9873

# 4. Train baselines (defaults use baselines/data/baseline_dataset_n15_v3t3.pickle).
python baselines\cnep\train_cnep.py
python baselines\cnmp\train_cnmp.py

# Non-canonical pickles: pass `--data baselines/data/baseline_dataset_<stem>.pickle`.
```

### Split-sampling design ("reserve eval first")

`scripts/prepare_splits.py` partitions each `(action, seed)`'s eligible
demos in this order:

  1. Sample `--num-test` demos → **test set**
  2. Sample `--num-validation` demos from the rest → **validation set**
  3. Shuffle the remaining demos → **train pool**
  4. Take the first `--num-train` from that pool

Two properties this guarantees, both important for the comparison study:

  * **Test / valid sets are fixed across `n_train` sweeps.** Regenerate the
    manifest with the same seeds and `(num_validation, num_test)` but a
    different `num_train` and the test/valid lists are bit-identical. Only
    then is "MSE vs. n_train" a meaningful curve.
  * **Train sets are nested across sweeps.** The `n_train = K1` train list
    is a strict subset of the `n_train = K2` list when `K1 < K2`, so smaller
    runs see a subset of the data that larger runs see.

### Sweeping `n_train`

```powershell
# Same test/valid demos as n15_v3t3, just a smaller train pool prefix.
python scripts\prepare_splits.py --num-train 10 `
    --out data\splits\n10_v3t3.yaml
python baselines\prepare_baseline_dataset.py `
    --splits data\splits\n10_v3t3.yaml
# default --out → baselines\data\baseline_dataset_n10_v3t3.pickle
python baselines\cnep\train_cnep.py `
    --data baselines\data\baseline_dataset_n10_v3t3.pickle
python baselines\cnmp\train_cnmp.py `
    --data baselines\data\baseline_dataset_n10_v3t3.pickle
```

Capacity check: `num_train + num_validation + num_test ≤ 21` (action_2's
eligible pool is the bottleneck), so `15 + 3 + 3 = 21` is the tightest
config at `num_train = 15`. Keep `(num_validation, num_test) = (3, 3)`
across all sweep points so the eval sets stay frozen.

---

## Directory layout

```text
data/                                <- SHARED source data (both pipelines read here)
├── task_config.yaml                 <- object / camera / action manifest
├── raw/
│   ├── cnep_action_{0,1,2}/<demo>/0_left.png  (one image per demo, in git)
│   ├── cnep_action_{0,1,2}/<demo>/{0_right.png, *_left.png, ...}  (gitignored)
│   └── README.md
└── processed/                       <- preprocessed CSVs / H5s / action_summary.pickle

baselines/
├── README.md                        <- you are here
├── prepare_baseline_dataset.py      <- writes baselines/data/baseline_dataset_<stem>.pickle (this README)
│
├── data/
│   └── baseline_dataset_<stem>.pickle <- one file per splits manifest (~20 MB, gitignored)
│
├── cnep/                            <- CNEP training + inference
│   ├── train_cnep.py
│   ├── predict_cnep.py
│   └── models/
└── cnmp/                            <- CNMP training + inference
    ├── train_cnmp.py
    ├── predict_cnmp.py
    └── models/
```

---

## `prepare_baseline_dataset.py` — building packaged baseline pickles (`baseline_dataset_<stem>.pickle`)

This is a port of `D:\project\assembly\split_data_assembly.ipynb` (cell 12).
It walks the raw + processed demo data, builds train / valid / test splits per
seed, extracts MobileNetV2 image features, expresses trajectories in 5
reference frames (4 objects + global), and writes one pickle that **every
baseline reads**.

### Environment

Use the **same** Python environment as TP-Transformer. From the repo root:

```powershell
pip install -r requirements.txt
```

(`requirements.txt` already includes PyTorch, **torchvision**, pandas, scipy, PyYAML, Pillow, and **`tables`** — needed by `pandas.read_hdf` for the `_obj_combined.h5` files.)

### Inputs (where the data has to live)

| Argument | Default | What it must contain |
|---|---|---|
| `--raw-dir` | `data/raw` (repo root) | `cnep_action_{0,1,2}/<demo>/0_left.png` for every demo used in any split |
| `--processed-dir` | `data/processed` (repo root) | `action_{0,1,2}/action_summary.pickle` plus per-demo `<demo>.csv` and `<demo>_obj_combined.h5` |
| `--task-config` | `data/task_config.yaml` (repo root) | YAML with an `objects:` list. If missing, the builder falls back to `sorted(['bolt','nut','bin','jig'])` — the same default the notebook used |
| `--out` | auto from splits | Omit to use **`baselines/data/baseline_dataset_<yaml_stem>.pickle`** (e.g. `n15_v3t3.yaml` → `baseline_dataset_n15_v3t3.pickle`). Inline fallback: `baseline_dataset_legacy_inline.pickle` |

All paths are resolved relative to wherever you launch Python from; the
defaults above are anchored to this folder via `Path(__file__).resolve()`,
so running the script from any cwd uses the same defaults.

### Run

Build the canonical pickle from the shared manifest (recommended):

```powershell
cd C:\Users\xyao0\Desktop\TP-Transformer
python baselines\prepare_baseline_dataset.py --splits data\splits\n15_v3t3.yaml
```

Quick smoke test against the manifest (one action, one seed):

```powershell
python scripts\prepare_splits.py `
    --actions action_0 --seeds 9871 `
    --out data\splits\smoke.yaml
python baselines\prepare_baseline_dataset.py `
    --splits data\splits\smoke.yaml
# default --out → baselines\data\baseline_dataset_smoke.pickle
```

GPU MobileNetV2 forward (~10× faster, numerically within ~1e-6 of CPU):

```powershell
python baselines\prepare_baseline_dataset.py `
    --splits data\splits\n15_v3t3.yaml --device cuda:0
```

Inline mode (no manifest) is supported only as a legacy fallback that
reproduces the original notebook's 15/4/1 split — every new experiment
should go through `scripts/prepare_splits.py`.

### CLI reference

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--splits PATH` | path | `None` | YAML manifest from `scripts/prepare_splits.py`. **Strongly recommended**: makes the build deterministic and aligns the splits with TP-Transformer. When set, the flags below (`--seeds / --actions / --num-train / --num-validation`) are read from the manifest, not the CLI. |
| `--raw-dir PATH` | path | `data/raw` | root of `cnep_action_{0,1,2}/` |
| `--processed-dir PATH` | path | `data/processed` | root of `action_{0,1,2}/` |
| `--task-config PATH` | path | `data/task_config.yaml` | optional |
| `--out PATH` | path | auto-named from `--splits` stem (see Inputs) | Pass explicitly only to override |
| `--seeds N [N ...]` | ints | `9871 9872 9873` | one split per seed; ignored under `--splits` |
| `--actions A [A ...]` | strs | `action_0 action_1 action_2` | ignored under `--splits` |
| `--num-train N` | int | `15` | ignored under `--splits` |
| `--num-validation N` | int | `4` | ignored under `--splits` (inline-only; legacy default) |
| `--device DEV` | str | `cpu` | passed to MobileNetV2 only; trajectories are pure numpy |

### Output schema

```python
import pickle
data = pickle.load(open('baselines/data/baseline_dataset_n15_v3t3.pickle', 'rb'))
# data is a dict:
#   data['action_0']  -> list of len(seeds) per-split dicts
#   data['action_1']  -> same
#   data['action_2']  -> same

split = data['action_0'][0]   # action_0, seeds[0]
# 17-key dict consumed by CNEP/CNMP and (later) TP-GMM/TP-ProMP:
```

Shapes use `N_train / N_valid / N_test` from the manifest. The canonical
pickle (15/3/3) has `N_train = 15`, `N_valid = 3`, `N_test = 3` uniformly
across the three actions.

| Key | Type / shape | What it is |
|---|---|---|
| `train_traj_global_cnep` | `Tensor (N_train, T, 7)` | train trajectories, min-max normalised to `[-1, 1]` |
| `valid_traj_global_cnep` | `Tensor (N_valid, T, 7)` | validation trajectories, same normalisation |
| `test_traj_global_cnep`  | `Tensor (N_test,  T, 7)` | held-out test trajectories, same normalisation |
| `train_feats`            | `Tensor (N_train, 1280)` | MobileNetV2 features for each demo's `0_left.png` |
| `valid_feats`            | `Tensor (N_valid, 1280)` | same for validation |
| `test_feats`             | `Tensor (N_test,  1280)` | same for the test demos |
| `train_traj_tp_gmm`      | `list[5] of (N_train*T, 8)` | train trajs in 5 reference frames, time-augmented (col 0 = `t/T`) |
| `train_traj_tp_pmp`      | `(N_train, T, 35)` | per-frame trajs concatenated across the 5 frames (5 × 7 dims) |
| `train_times_tp_pmp`     | `list[N_train] of (T,)` | per-train-demo normalised time vector |
| `test_t`                 | `(T,)` | normalised time vector (same for all demos in this action) |
| `HTs_test`               | `np.ndarray (N_test,  5, 4, 4)` | per-test-demo, per-frame homogeneous transforms |
| `HTs_validation`         | `np.ndarray (N_valid, 5, 4, 4)` | per-valid-demo, per-frame homogeneous transforms |
| `test_traj_global`       | `(N_test,  T, 7)` | test trajs before min-max (still xyz-centred) |
| `validation_traj_global` | `(N_valid, T, 7)` | validation trajs before min-max |
| `train_stat`             | `{'mean': (3,), 'std': float}` | xyz centring stats |
| `minmax7`                | `Tensor (7, 2)` | per-dim `[lo, hi]` used for min-max → invertible denormalisation |
| `seed`                   | `int` | the seed for this split |

The `*_cnep` tensors are what `cnep/train_cnep.py` and `cnmp/train_cnmp.py`
consume directly (training only touches `train_*` and `valid_*`). The
`predict_*.py` scripts iterate over `N_test` test demos and emit
predictions of shape `(N_test, T, 7)` per `(action, seed)`. The `tp_gmm` /
`tp_pmp` / `HTs_*` fields are pre-computed for the classical baselines
that will live under `tp_gmm/` and `tp_promp/`.

### Reproducibility notes

- **Same `(splits.yaml, raw + processed data)` ⇒ same pickle.** All four
  baselines (CNEP, CNMP, TP-GMM, TP-ProMP) share splits via the manifest +
  pickle, and TP-Transformer reads the same `--splits` manifest. Don't
  regenerate splits casually — checkpoints trained against the old splits
  will silently mismatch.
- **Defaults reproduce the canonical 15/3/3 split.** `seeds=(9871, 9872,
  9873)`, `num_train=15`, `num_validation=3`, `num_test=3`, and the
  per-action bad-demo blacklist (constants at the top of
  `prepare_baseline_dataset.py`) determine the canonical
  `data/splits/n15_v3t3.yaml`.
- **Seed/RNG ordering matters.** Python's `random.Random(seed)` is
  instantiated *once per seed* and reused across all 3 actions, so
  `action_1`'s split depends on `action_0` having consumed its share of
  random calls first. Don't reseed per action — you'll get a different
  manifest.
- **`train_mean` / `train_std` are seed-global, not per-action.** They are
  computed once over the train trajectories of all 3 actions combined and
  then applied to every per-action split. The original notebook does the
  same; we preserve it so the normalisation matches across pipelines.
- **MobileNetV2 features may drift across OSes.** Cosine similarity to the
  reference pickle stays ≥ 0.995, but L2 distance can be ~1.5–2.2 — this is
  PNG decoding differences in `libpng` / Pillow across Linux vs Windows, not
  a bug in the script.

### Inspecting the result

```powershell
python -c "import pickle, sys; d = pickle.load(open(r'baselines\data\baseline_dataset_n15_v3t3.pickle','rb')); print({k: len(v) for k, v in d.items()}); s = d['action_0'][0]; print({k: (tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in s.items()})"
```
