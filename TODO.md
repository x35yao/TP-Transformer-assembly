# TP-Transformer TTA integration — TODO

Tracking work to integrate **test-time rotation averaging** (from `MTTPExtendedTasks.ipynb`) into the assembly pipeline, and related cleanup.

Reference: teammate notebook cell *"Run and save prediction result on the test set"* — rotate encoder object poses → predict → inverse-rotate output → average over `repeats` samples. Our training-time random rotation uses **`z` axis** (`augment_random_rotation` in `dataset.py`); TTA should match.

---

## Done (partial, uncommitted)

- [x] **`rotate_pose_about`** in `src/tp_transformer/augmentation.py` — parameterized, invertible rotation (same math as old `apply_rotate` / our `random_rotation`, but with explicit `degree` + `center` so TTA can apply `-deg` on the output).
- [x] **`TrainConfig.tta_rotations` / `tta_axis`** in `src/tp_transformer/config.py` (`tta_rotations=0` = off, default axis `z`).
- [x] **`src/tp_transformer/inference.py`** (draft) — `assemble_result` + `tta_assemble_result` with **no in-place accumulation bug** (each repeat rotates a fresh copy of the original scene).
- [x] **`predict.py`** wired to `tta_assemble_result` when `config.tta_rotations > 1`.

---

## Pending — implementation

- [ ] **Decide where TTA code lives** (see open questions below). Candidate: delete `inference.py` and move helpers to `augmentation.py` or `train.py`.
- [ ] **Wire TTA into validation** in `src/tp_transformer/train.py` — use rotation-averaged predictions for `important_dist` / `model_best.pth` selection when `tta_rotations > 1` (user requested predict **and** train/validation).
- [ ] **CLI flags** — add to `scripts/predict.py` and `scripts/train.py`:
  - `--num-rotations` / `--tta-rotations` (override `TrainConfig.tta_rotations`)
  - optionally `--tta-axis` (default `z`)
- [ ] **Smoke test** — run predict with `--num-rotations 5` on one checkpoint; confirm output shape unchanged `{action: (N_test, T, 7)}`.
- [ ] **Re-run Experiment 2 (random aug)** with TTA at predict (+ validation if wired) and compare ADE/NDQ via `scripts/evaluate_predictions.py`.
- [ ] **Docs** — update `EXPERIMENTS.md` / `RESULTS.md` with TTA usage and note that random-aug models should use TTA at eval time.

---

## Pending — design / review

- [ ] **Confirm TTA hyperparams with teammate**
  - Notebook used `repeats=5`, random angles in `[0,360)`, object-centroid + noise pivot, **`axis='x'`** (their multi-task frame).
  - Our plan: **`axis='z'`**, evenly spaced angles (includes 0°), object-centroid pivot (no noise?) — align before locking defaults.
- [ ] **Quaternion averaging** — current code sign-aligns then mean-renormalizes; confirm this matches what they want vs plain `np.mean` on all 7 dims.
- [ ] **Segment-by-segment decode** — keep as-is (see note below); TTA only rotates **encoder** `obj_seq` poses, not the decoder hidden state. Do **not** switch to single-pass inference unless the model architecture changes.

---

## Open questions

### 1. Do we need `inference.py`?

**No, not strictly.** It was added to share the decode loop between predict and (planned) validation TTA.

| Option | Pros | Cons |
|--------|------|------|
| **A. `augmentation.py`** | Rotation math + TTA together | Mixes training aug with inference |
| **B. `train.py`** | Predict already imports from train; validation is there | Train module gets longer |
| **C. `predict.py` only** | Simplest | Validation metric stays single-pass unless duplicated |
| **D. Keep `inference.py`** | Clear separation | Extra file for ~150 lines |

**Recommendation:** B or A; delete `inference.py` if we consolidate.

### 2. Why segment-by-segment decode?

The TP-Transformer model is **task-parameterized across camera capture times**, not a single forward over one static object pose:

- Each demo has **multiple object-pose snapshots** (`obj_seq` shape `(batch, n_segments, n_objs, D)`), one per camera capture (`img_inds` marks segment boundaries on the trajectory).
- Training runs the model **once per segment**: encoder sees object poses at capture time `i`, decoder fills trajectory timesteps `[img_inds[i], img_inds[i+1])`.
- Loss is computed **only within each segment** (masked by `img_inds`).

At inference we must replay the same loop and **stitch** segment outputs into one trajectory (`assemble_result`). This is **not** related to random rotation or TTA — it is how the architecture works. The old assembly notebook used a simpler single `obj_seq` input; our assembly model (`TFEncoderDecoder5`) uses multi-segment decoding from `train.py` / `valid_traj_epoch`.

TTA rotates all segment object poses jointly, then runs the same segment loop on the rotated scene.

### 3. Teammate notebook bug (fixed in our port)

Notebook re-reads `obj_seq_input` each iteration but **overwrites it in place**, so rotations compound while only the latest `-deg` is applied to the output. Our `tta_assemble_result` rotates a **fresh copy of the original** each time. Verified with `_tta_check.py` (sample 0 correct, samples 1–4 drifted in notebook pattern).

---

## Not in scope (for now)

- [ ] Port TTA to baselines (CNEP/CNMP/TP-GMM/TP-ProMP) — only TP-Transformer discussed.
- [ ] Change training-time `augment_random_rotation` — already on `z`; no change needed for TTA axis match.
- [ ] Commit `MTTPExtendedTasks.ipynb` to repo — currently local reference only.

---

## Quick commands (once CLI wired)

```bash
# Predict with TTA (5 rotations, z-axis)
python scripts/predict.py \
  --splits data/splits/n15_v3t3.yaml \
  --seed 9871 \
  --model-name random_aug \
  --augmentation-method random \
  --num-rotations 5

# Evaluate
python scripts/evaluate_predictions.py \
  --baseline-data baselines/data/baseline_dataset_n15_v3t3.pickle \
  --transformer-models random_aug \
  --transformer-output-root transformer
```
