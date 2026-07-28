# Diffusion Policy Baseline

This document describes how the Diffusion Policy (DP) baseline was implemented,
trained, and evaluated for comparison against TP-Transformer. It is written to
be dropped into (or adapted for) the paper's experimental section.

## Overview

We compare against **Diffusion Policy** (Chi et al., *Diffusion Policy: Visuomotor
Policy Learning via Action Diffusion*, RSS 2023 / IJRR 2024), a conditional
denoising-diffusion policy that generates action sequences by iterative
denoising. We use the authors' **official low-dimensional (state-based)
implementation** unchanged for everything scientifically relevant — the training
objective, the denoising network, the DDPM noise scheduler, the normalizer, and
the exponential-moving-average (EMA) weights — and only supply a thin data
adapter so that the model consumes the same object-pose inputs and produces the
same trajectory outputs as TP-Transformer, allowing evaluation with the identical
metrics (ADE, NDQ).

We evaluate both official backbones:

- **Diffusion Policy (CNN)** — the `ConditionalUnet1D` denoiser with FiLM
  conditioning (the authors' recommended default).
- **Diffusion Policy (Transformer)** — the `TransformerForDiffusion` denoiser.

Following the Diffusion Policy benchmark convention of **one policy per task**,
we train a separate model for each of the three assembly subtasks (per-subtask
models), for each of the 5 data seeds, at each training-set size
K ∈ {1, 2, 5, 10, 15}.

## What is reused from the official code, and what is ours

**Reused verbatim from the official repository** (`real-stanford/diffusion_policy`):

- `DiffusionUnetLowdimPolicy` / `DiffusionTransformerLowdimPolicy`, including their
  `compute_loss` (forward diffusion, ε-prediction, masked MSE) and
  `predict_action` (receding-horizon DDPM sampling).
- `ConditionalUnet1D` and `TransformerForDiffusion` denoising networks.
- `DDPMScheduler` (from `diffusers`) with the paper's low-dim settings.
- `LinearNormalizer` (limits mode → [-1, 1]).
- `EMAModel`.

**Ours** (the only additions):

- A **data adapter** that turns the assembly demonstrations into the fixed-horizon
  `(observation, action)` windows Diffusion Policy expects, following the official
  `SequenceSampler` windowing convention.
- The outer **training loop** and the offline **receding-horizon rollout** used at
  inference (there is no simulator in this offline trajectory-prediction setting).

## Conditioning (observation) and action definition

- **Action** (what DP predicts): the 8-D gripper command per timestep —
  end-effector pose (position + quaternion, 7-D) plus the binary grasp state (1-D).
  This matches TP-Transformer's output dimension exactly. As with TP-Transformer,
  only the 7-D pose is used for evaluation (ADE/NDQ); the grasp channel is trained
  but discarded before scoring.
- **Observation** (what DP conditions on): the object poses of the current
  camera-capture segment (5 objects × 7-D pose = 35-D) concatenated with the
  gripper "anchor" pose (7-D proprioception), for a 42-D observation vector. The
  object poses are the same true per-segment poses given to TP-Transformer's
  encoder. The observation history length is `n_obs_steps` (2 for the CNN, 3 for
  the Transformer, matching the official low-dim configs); across the history steps
  the object poses are constant within a capture segment and only the gripper
  pose varies.

## Receding-horizon windows (data adapter)

Diffusion Policy operates on fixed-length action chunks with receding-horizon
control. We use the paper's real-task horizon settings:

| Parameter | Symbol | Value |
| --- | --- | --- |
| Prediction horizon | `horizon` (Tp) | 16 |
| Action (execution) horizon | `n_action_steps` (Ta) | 8 |
| Observation history | `n_obs_steps` (To) | 2 (CNN) / 3 (Transformer) |

Windows are generated per capture-segment following the official `SequenceSampler`
convention: window start indices run over `[-(To-1), seg_len - Tp + (Ta-1)]`
(`pad_before = To-1`, `pad_after = Ta-1`), with edge-repeat padding at segment
boundaries, at stride 1 (every timestep). During **training**, each augmented
demonstration is re-windowed once per epoch (matching TP-Transformer's
on-the-fly augmentation cadence).

## Diffusion process and network (both backbones)

| Component | Setting |
| --- | --- |
| Noise scheduler | `DDPMScheduler`, `num_train_timesteps = 100` |
| β schedule | `squaredcos_cap_v2` (`beta_start = 1e-4`, `beta_end = 2e-2`) |
| Variance type | `fixed_small` |
| Prediction target | ε (epsilon / predicted noise) |
| `clip_sample` | `True` (requires the [-1, 1] `LinearNormalizer`) |
| Inference denoising steps | 100 |
| Normalizer | `LinearNormalizer`, limits mode → [-1, 1] |
| EMA | `power = 0.75`, `max_decay = 0.9999`, `update_after_step = 0` |

**CNN backbone** (`ConditionalUnet1D`): `down_dims = [256, 512, 1024]`,
`kernel_size = 5`, `n_groups = 8`, `diffusion_step_embed_dim = 256`,
`cond_predict_scale = True` (FiLM), obs supplied as global conditioning
(`obs_as_global_cond = True`). ≈ 66 M parameters.

**Transformer backbone** (`TransformerForDiffusion`): `n_layer = 8`,
`n_head = 4`, `n_emb = 256`, `p_drop_emb = 0.0`, `p_drop_attn = 0.3`,
`causal_attn = True`, `time_as_cond = True`, `obs_as_cond = True`,
`n_cond_layers = 0`. ≈ 9 M parameters.

Both configurations follow the official low-dimensional configs
(`train_diffusion_unet_lowdim_workspace.yaml` /
`train_diffusion_transformer_lowdim_workspace.yaml`).

## Optimization

| Setting | CNN | Transformer |
| --- | --- | --- |
| Optimizer | AdamW | AdamW with the paper's parameter-group weight-decay split (`TransformerForDiffusion.configure_optimizers`) |
| Learning rate | 1e-4 | 1e-4 |
| Betas | (0.95, 0.999) | (0.9, 0.95) |
| Weight decay | 1e-6 | 1e-3 |
| LR schedule | cosine decay, 500 warmup steps | cosine decay, 1000 warmup steps |
| Batch size | 256 | 256 |
| Epochs | 5000 | 5000 |
| EMA | yes | yes |

These match the official low-dim configurations (which use a cosine schedule with
linear warmup and 5000 epochs; the transformer requires LR warmup for stability).

## Data, splits, and augmentation

- Trained on the same task-parameterized demonstration dataset and the same
  train/validation/test splits as TP-Transformer
  (`data/splits/n{K}_v3t3.yaml`), for K ∈ {1, 2, 5, 10, 15} training demonstrations
  per subtask, and 5 seeds per K.
- Two augmentation arms, matching TP-Transformer's protocol:
  - **tp** — task-parameterized augmentation (random rigid transforms applied
    jointly to object poses and the trajectory), the main reported arm.
  - **none** — no augmentation.

## Inference (offline receding-horizon rollout)

At test time we roll out the trained policy in a receding-horizon loop: given the
current observation, the policy denoises a 16-step action chunk from Gaussian
noise (100 DDPM steps) using the **EMA** weights; we execute the first 8 steps,
advance the gripper anchor to the last executed pose, re-observe (the object poses
switch at the next camera capture), and repeat until the trajectory is complete.

The object-pose channel is fed the **true** per-segment poses throughout (the same
information available to TP-Transformer), i.e. this channel is closed-loop with
ground truth. The gripper-pose channel is advanced from the policy's **own**
predicted pose, because this is an offline setting with no environment/simulator
to measure the realized state (unlike Diffusion Policy's original closed-loop
benchmark, where `env.step` returns the true post-execution state). Both baselines
therefore receive the same object-pose information and no privileged gripper-state
feedback, making the comparison fair.

## How to run

Training (per-subtask, one array task per action × seed):

```bash
# CNN, tp-augmentation, K=15
sbatch --export=ALL,ARM=tp,K=15 scripts/slurm/dp_k15_persub.sbatch
# Transformer, tp-augmentation, K=15
sbatch --export=ALL,ARM=tp,K=15 scripts/slurm/dp_k15_persub_tf.sbatch
```

Prediction (loads the three per-action models and routes each test demo):

```bash
sbatch --export=ALL,ARM=tp,K=15 scripts/slurm/dp_k15_persub_predict.sbatch
sbatch --export=ALL,ARM=tp,K=15 scripts/slurm/dp_k15_persub_tf_predict.sbatch
```

Direct CLI (single model):

```bash
python baselines/diffusion_policy/train_dp.py \
    --splits data/splits/n15_v3t3.yaml --seed 9871 --augmentation tp \
    --action-idx 0 --backbone cnn --epochs 5000 --batch-size 256 \
    --output-root <MODEL_ROOT>

python baselines/diffusion_policy/predict_dp.py \
    --splits data/splits/n15_v3t3.yaml --seed 9871 --augmentation tp \
    --per-subtask --model-root <MODEL_ROOT> --out-root <PRED_ROOT>
```

Evaluation reuses `scripts/evaluate_predictions.py` (ADE in mm, NDQ), and the
K-sweep figures are produced by `scripts/make_plots.py`.

## Key implementation files

- `baselines/diffusion_policy/dp_common.py` — dimensions, official-model imports,
  policy construction (both backbones), observation vector, and the windowing
  adapter.
- `baselines/diffusion_policy/train_dp.py` — training driver.
- `baselines/diffusion_policy/predict_dp.py` — offline receding-horizon rollout
  and per-subtask prediction routing.
- `baselines/diffusion_policy_upstream/` — the unmodified official Diffusion Policy
  repository (`git clone https://github.com/real-stanford/diffusion_policy`).
