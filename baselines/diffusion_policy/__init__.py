"""Diffusion Policy (Chi et al., RSS 2023) baseline for TP-Transformer.

This package adapts the *official* Diffusion Policy implementation (vendored in
`baselines/diffusion_policy_upstream/`) to our assembly benchmark, matching the
way the TP-Transformer is trained and evaluated so the comparison is
apples-to-apples:

- **Segment-wise conditioning (Path B).** Like the TP-Transformer, DP conditions
  on the object poses observed at each of the K camera-capture moments and
  generates the trajectory segment that follows that capture. This uses the
  exact same on-the-fly data pipeline (`tp_transformer.data.build_datasets`),
  including the identical TP-augmentation, so DP sees the same inputs as
  TP-Transformer.
- **CNN variant first.** We use the 1D temporal conditional U-Net
  (`ConditionalUnet1D`) with FiLM global conditioning, per the authors'
  recommendation to start with the CNN backbone. The time-series diffusion
  transformer variant can be swapped in later via `--backbone transformer`.
- **Faithful internals.** The denoising network and DDPM training/sampling come
  from the upstream repo (`ConditionalUnet1D`, `DDPMScheduler`); only the
  training/prediction driver and the dataset adapter are ours.

The module `dp_common` exposes `import_upstream()` which puts the vendored repo
on `sys.path` and returns the reused classes.
"""
