# Raw CNEP demo data

Source: `F:\backup\assembly\data\cnep\` — the recordings used to build
`baselines/data/baseline_dataset.pickle` for the CNEP / CNMP baselines.
Also consumed (indirectly, via `data/processed/`) by the TP-Transformer
pipeline in `src/tp_transformer/data.py`.

## What's here

32 demo folders, each named by a Unix timestamp (the start time of the
recording). For each demo we store the **minimum** needed to re-extract
the MobileNetV2 features the baselines consume:

```
<timestamp>/
├── 0_left.png        first-frame stereo image (ZED camera, left lens)
├── 0_right.png       first-frame stereo image (ZED camera, right lens)
└── <timestamp>.csv   full trajectory recording (Kinova robot pose + grasp + FT)
```

Upstream the demos also have many other stereo frames (e.g. `20_left.png`,
`36_left.png`, ...) at additional keypoint indices — only frame `0` is
copied here because the CNEP pipeline (`train_cnep_with_mobilenet_v2.py`
in the upstream repo) extracts a single 1280-d MobileNetV2 feature per demo
from `img.jpeg` (i.e. the first frame). If you need other frames for
ablations, grab them from the source path above.

## Trajectory CSV columns

```
time, x, y, z, rx, ry, rz, grip_width, grasp_detected, wrist_camara_capture,
fx, fy, fz, tx, ty, tz, qx, qy, qz, qw
```

- `x, y, z`         end-effector position (mm)
- `rx, ry, rz`      rotation as Euler / axis-angle (the pickle uses quaternions)
- `qx, qy, qz, qw`  quaternion orientation (what the pickle stores)
- `grip_width`      gripper opening
- `grasp_detected`  binary 0/1 — used to find pick/release indices
- `wrist_camara_capture`  binary flag for wrist-camera capture timesteps
- `fx, fy, fz, tx, ty, tz`  6-DOF force/torque

The pickle's `*_global_cnep` trajectories are derived from these CSVs by:
1. Picking 150 (or 186 / 197) evenly-spaced timesteps from the full sequence.
2. Extracting `[x, y, z, qx, qy, qz, qw]`.
3. Min-max normalising each dim to `[-1, 1]` (stored as `minmax7`).

## Demo → (action, seed) mapping

Not documented in the original repo. The pickle exposes 3 actions × 3 seeds
× (15 train + ≤15 valid + 1 test) trajectories, which is at most ~93 demo
slots drawn (with replacement across seeds) from the underlying ~31 unique
demos. If you need the exact mapping for a particular run, the simplest
approach is to match trajectories by their first-frame pose:

```python
import pickle, pandas as pd, numpy as np
all_data = pickle.load(open('../../baselines/data/baseline_dataset.pickle','rb'))
# ... compare csv[0, ['x','y','z']] against entry['train_traj_global_cnep'][i, 0, :3]
# after un-normalising via entry['minmax7']
```

## Re-extracting features

The upstream extraction script
(`F:\backup\cnep-master\baxter\train_cnep_with_mobilenet_v2.py`, lines 39-56)
shows the canonical pipeline:

```python
img = Image.open(img_path).convert('RGB')
img = transforms.functional.crop(img, top=0, left=0, height=300, width=480)
img = ToTensor()(img)
img = Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])(img)
features = mobilenet_v2(pretrained=True)(img.unsqueeze(0))   # head replaced w/ Identity
features = features.flatten()                                 # → (1280,)
```
