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

All baselines consume the same dataset, produced by
[`prepare_baseline_dataset.py`](./prepare_baseline_dataset.py) and written to
`baselines/data/baseline_dataset.pickle`.

The script reads source trajectories, object poses, and demo images from
the repo-root `data/` directory, which is **shared with the TP-Transformer
pipeline** (`src/tp_transformer/data.py`). Both methods therefore start from
the same demos; the baseline builder just packages them into the format
CNEP / CNMP expect.

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
├── prepare_baseline_dataset.py      <- baseline_dataset.pickle builder (this README)
│
├── data/
│   └── baseline_dataset.pickle      <- canonical baseline dataset, ~20 MB (gitignored)
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

## `prepare_baseline_dataset.py` — building `baseline_dataset.pickle`

This is a port of `D:\project\assembly\split_data_assembly.ipynb` (cell 12).
It walks the raw + processed demo data, builds train / valid / test splits per
seed, extracts MobileNetV2 image features, expresses trajectories in 5
reference frames (4 objects + global), and writes one pickle that **every
baseline reads**.

### Environment

The builder needs PyTorch + torchvision + pandas + scipy + PyYAML + Pillow.
The repo's `tp-transformer` env does **not** have `torchvision`; use the
`cnep` conda env (already set up on this machine):

```powershell
conda activate cnep
```

If you're setting this up fresh:

```powershell
conda create -n cnep python=3.10
conda activate cnep
pip install torch torchvision pandas numpy scipy pyyaml pillow tables
```

(`tables` is needed by `pandas.read_hdf` for the `_obj_combined.h5` files.)

### Inputs (where the data has to live)

| Argument | Default | What it must contain |
|---|---|---|
| `--raw-dir` | `data/raw` (repo root) | `cnep_action_{0,1,2}/<demo>/0_left.png` for every demo used in any split |
| `--processed-dir` | `data/processed` (repo root) | `action_{0,1,2}/action_summary.pickle` plus per-demo `<demo>.csv` and `<demo>_obj_combined.h5` |
| `--task-config` | `data/task_config.yaml` (repo root) | YAML with an `objects:` list. If missing, the builder falls back to `sorted(['bolt','nut','bin','jig'])` — the same default the notebook used |
| `--out` | `baselines/data/baseline_dataset.pickle` | Output pickle path |

All paths are resolved relative to wherever you launch Python from; the
defaults above are anchored to this folder via `Path(__file__).resolve()`,
so running the script from any cwd uses the same defaults.

### Run

Reproduce the canonical pickle (3 actions × 3 seeds × 15/4/1 split):

```powershell
cd C:\Users\xyao0\Desktop\TP-Transformer
python baselines\prepare_baseline_dataset.py
```

Quick smoke test (one action, one seed, ~15 s on CPU):

```powershell
python baselines\prepare_baseline_dataset.py `
    --actions action_0 `
    --seeds 9871 `
    --out baselines\data\baseline_dataset.smoke.pickle
```

GPU MobileNetV2 forward (~10× faster, numerically within ~1e-6 of CPU):

```powershell
python baselines\prepare_baseline_dataset.py --device cuda:0
```

Custom split sizes (will produce a pickle the existing CNEP/CNMP checkpoints
cannot consume — only use for new experiments):

```powershell
python baselines\prepare_baseline_dataset.py --num-train 10 --num-validation 5 `
    --out baselines\data\baseline_dataset_10x5.pickle
```

### CLI reference

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--raw-dir PATH` | path | `data/raw` | root of `cnep_action_{0,1,2}/` |
| `--processed-dir PATH` | path | `data/processed` | root of `action_{0,1,2}/` |
| `--task-config PATH` | path | `data/task_config.yaml` | optional |
| `--out PATH` | path | `baselines/data/baseline_dataset.pickle` | parents created on demand |
| `--seeds N [N ...]` | ints | `9871 9872 9873` | one split per seed; canonical pickle uses these three |
| `--actions A [A ...]` | strs | `action_0 action_1 action_2` | restrict the build |
| `--num-train N` | int | `15` | train demos per (action, seed) |
| `--num-validation N` | int | `4` | validation demos per (action, seed); 1 test demo is hardcoded |
| `--device DEV` | str | `cpu` | passed to MobileNetV2 only; trajectories are pure numpy |

### Output schema

```python
import pickle
data = pickle.load(open('baselines/data/baseline_dataset.pickle', 'rb'))
# data is a dict:
#   data['action_0']  -> list of len(seeds) per-split dicts
#   data['action_1']  -> same
#   data['action_2']  -> same

split = data['action_0'][0]   # action_0, seeds[0]
# 17-key dict consumed by CNEP/CNMP and (later) TP-GMM/TP-ProMP:
```

| Key | Type / shape | What it is |
|---|---|---|
| `train_traj_global_cnep` | `Tensor (15, T, 7)` | train trajectories, min-max normalised to `[-1, 1]` |
| `valid_traj_global_cnep` | `Tensor (4,  T, 7)` | validation trajectories, same normalisation |
| `test_traj_global_cnep`  | `Tensor (T, 7)`     | held-out test trajectory, same normalisation |
| `train_feats`            | `Tensor (15, 1280)` | MobileNetV2 features for the train demos' `0_left.png` |
| `valid_feats`            | `Tensor (4,  1280)` | same for validation |
| `test_feats`             | `Tensor (1,  1280)` | same for the single test demo |
| `train_traj_tp_gmm`      | `list[5] of (15*T, 8)` | train trajs in 5 reference frames, time-augmented (col 0 = `t/T`) |
| `train_traj_tp_pmp`      | `(15, T, 35)` | per-frame trajs concatenated across the 5 frames (5 × 7 dims) |
| `train_times_tp_pmp`     | `list[15] of (T,)` | per-train-demo normalised time vector |
| `test_t`                 | `(T,)` | normalised time vector |
| `HTs_test`               | `list[5] of (4, 4)` | object→world homogeneous transforms for the test demo |
| `HTs_validation`         | `(4, 5, 4, 4)` | per-valid-demo, per-frame homogeneous transforms |
| `test_traj_global`       | `(T, 7)` | test traj before min-max (still xyz-centred) |
| `validation_traj_global` | `(4, T, 7)` | validation traj before min-max |
| `train_stat`             | `{'mean': (3,), 'std': float}` | xyz centring stats |
| `minmax7`                | `Tensor (7, 2)` | per-dim `[lo, hi]` used for min-max → invertible denormalisation |
| `seed`                   | `int` | the seed for this split |

The `*_cnep` tensors are what `cnep/train_cnep.py` and `cnmp/train_cnmp.py`
consume directly. The `tp_gmm` / `tp_pmp` / `HTs_*` fields are pre-computed
for the classical baselines that will live under `tp_gmm/` and `tp_promp/`.

### Reproducibility notes

- **Same `baseline_dataset.pickle` ⇒ same numbers.** All four baselines share
  splits via this single pickle. Don't regenerate it casually — the existing
  CNEP/CNMP checkpoints under `cnep/models/` and `cnmp/models/` will silently
  mismatch if the splits change.
- **Defaults reproduce the canonical pickle.** `seeds=(9871, 9872, 9873)`,
  `num_train=15`, `num_validation=4`, and the per-action bad-demo blacklist
  (constants at the top of `prepare_baseline_dataset.py`) are pinned to match
  the notebook output.
- **Seed/RNG ordering matters.** Python's `random` is seeded *once per seed*
  and reused across all 3 actions, so `action_1`'s split depends on
  `action_0` having consumed two `random.sample` calls first. Don't reseed
  per action — you'll get a different pickle.
- **`train_mean` / `train_std` are seed-global, not per-action.** They are
  computed once over the train trajectories of all 3 actions combined and
  then applied to every per-action split. The notebook does the same.
- **MobileNetV2 features may drift across OSes.** Cosine similarity to the
  reference pickle stays ≥ 0.995, but L2 distance can be ~1.5–2.2 — this is
  PNG decoding differences in `libpng` / Pillow across Linux vs Windows, not
  a bug in the script.

### Inspecting the result

```powershell
python -c "import pickle, sys; d = pickle.load(open(r'baselines\data\baseline_dataset.pickle','rb')); print({k: len(v) for k, v in d.items()}); s = d['action_0'][0]; print({k: (tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k,v in s.items()})"
```
