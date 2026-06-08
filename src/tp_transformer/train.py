"""
Training and validation loops for the TP-Transformer model.

Architecture overview:
- The model processes trajectories in SEGMENTS defined by camera capture times (img_inds).
- For each segment, the encoder sees object poses at that capture time,
  and the decoder predicts the full trajectory with a causal mask.
- The loss is only computed within each segment's boundaries (masked by img_inds).
- The model also classifies which action/task is being performed.

Key metrics tracked:
- Position loss, orientation loss, grasp loss, action classification loss
- Pick distance: Euclidean error at grasp (pick) timesteps
- Release distance: Euclidean error at release timesteps
- Important distance: Error at high-weight timesteps (weights > 150)
"""

from __future__ import annotations

import os
import time
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import TASK_DIMS, build_datasets
from .inference import assemble_result, tta_assemble_result
from .losses import action_tag_loss, get_mask, weighted_pose_grasp_loss_func
from .utils import get_n_params


def build_model(config: TrainConfig, device: torch.device):
    """Construct the TFEncoderDecoder5 model with dimensions derived from config.
    
    Dimension calculations:
    - n_dims = 7 (x, y, z, qx, qy, qz, qw)
    - n_objs = len(all_objs) + 1 = 5 (bolt, nut, bin, jig + trajectory)
    - obj_seq_dim = n_dims + n_objs = 12 (encoder input: pose + one-hot object tag)
    - traj_seq_dim = obj_seq_dim + 1 = 13 (decoder input: same + grasp state)
    - task_dim = n_dims + 1 = 8 (decoder output: pose + grasp)
    """
    try:
        from .transformer.transformer_model import TFEncoderDecoder5
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `tp_transformer.transformer.transformer_model`. "
            "Add the model implementation to the repo or update the import path."
        ) from exc
    n_dims = len(TASK_DIMS)  # 7
    n_objs = len(config.all_objs) + 1  # 4 objects + 1 trajectory = 5
    obj_seq_dim = n_dims + n_objs  # 12: pose(7) + one-hot tag(5)
    traj_seq_dim = obj_seq_dim + 1  # 13: pose(7) + one-hot tag(5) + grasp removed but still 1 more
    task_dim = n_dims + 1  # 8: pose(7) + grasp(1) -- what the model predicts
    model = TFEncoderDecoder5(
        task_dim=task_dim,
        target_dim=traj_seq_dim,
        source_dim=obj_seq_dim,
        n_objs=n_objs,
        n_tasks=len(config.tasks),
        embed_dim=64,
        nhead=8,
        max_len=config.max_len,
        num_encoder_layers=3,
        num_decoder_layers=3,
        device=device,
    )
    return model


def train_traj_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    traj_loss_func,
    action_loss_func,
    device: torch.device,
    pos_weight: float = 1,
    quat_weight: float = 1,
    grasp_weight: float = 1,
    action_weight: float = 1,
):
    """Run one training epoch.
    
    For each batch:
    1. Iterate over camera capture segments (obj_seq has one set of object poses per capture)
    2. For each segment:
       - Feed object poses to encoder, hidden trajectory to decoder
       - Compute loss only within this segment (masked by img_inds)
       - Accumulate predictions into a result buffer
    3. After all segments: compute pick/release/important point distances
    
    Returns:
        Tuple of loss lists for logging
    """
    model.train()
    total_losses, pos_losses, ori_losses, grasp_losses, action_losses = [], [], [], [], []
    pick_dists, release_dists, important_dists = [], [], []
    
    for sample_batched in dataloader:
        optimizer.zero_grad()
        obj_seq, traj_seq, traj_hidden, weights, action_tag, padding_mask, img_inds, pick_inds, release_inds = sample_batched
        
        # Move everything to device
        padding_mask = padding_mask.to(device)
        obj_seq = obj_seq.to(device)
        traj_seq = traj_seq.to(device)
        traj_hidden = traj_hidden.to(device)
        action_tag_gt = action_tag.to(device)
        weights = weights.to(device)
        
        # Identify "important" timesteps (high weight = near grasp/slow regions)
        important_inds = weights > 150
        
        # Initialize result buffer for accumulating predictions across segments
        # Prepend a zero column to match output dimension (for grasp)
        result = traj_hidden.clone()
        tmp = torch.zeros((result.shape[0], result.shape[1], 1)).to(device)
        result = torch.cat((tmp, result), dim=2)
        
        # --- Process each camera capture segment ---
        for i in range(obj_seq.shape[1]):
            # Define loss boundaries for this segment
            loss_start_inds = input_end_inds = img_inds[:, i]
            if i == obj_seq.shape[1] - 1:
                loss_end_inds = torch.ones(obj_seq.shape[0], dtype=torch.int) * (-1)  # Until end
            else:
                loss_end_inds = img_inds[:, i + 1]  # Until next capture
            
            # Forward pass: encoder sees object poses, decoder sees hidden trajectory
            output_seq, action_tag_seq = model(obj_seq[:, i, :, :], traj_hidden, tgt_padding_mask=padding_mask, predict_action=True)
            
            # Action classification loss
            labels_gt = torch.argmax(action_tag_gt, axis=1)
            action_loss = action_loss_func(action_tag_seq, labels_gt) * action_weight
            
            # Create loss mask: only compute loss within this segment
            loss_mask = get_mask(output_seq.shape, loss_start_inds, loss_end_inds).to(device)
            masked_weights = weights * loss_mask
            
            # Trajectory loss (position + orientation + grasp)
            pos_loss, ori_loss, grasp_loss = traj_loss_func(
                output_seq, traj_seq[:, :, : len(TASK_DIMS) + 1], masked_weights, quat_weight=quat_weight, pos_weight=pos_weight, grasp_weight=grasp_weight
            )
            
            # Backprop
            loss = pos_loss + ori_loss + grasp_loss + action_loss
            loss.backward()
            optimizer.step()
            
            # Log losses
            total_losses.append(loss.item())
            pos_losses.append(pos_loss.item())
            ori_losses.append(ori_loss.item())
            grasp_losses.append(grasp_loss.item())
            action_losses.append(action_loss.item())
            
            # Accumulate predictions for this segment into result buffer
            for (start, end) in zip(loss_start_inds, loss_end_inds):
                result[:, start:end, : len(TASK_DIMS) + 1] = output_seq[:, start:end]
        
        # --- Compute pick/release/important distances ---
        output_pick = result[torch.arange(result.shape[0]), pick_inds.squeeze(), :3]
        output_release = result[torch.arange(result.shape[0]), release_inds.squeeze(), :3]
        output_important = result[:, :, : len(TASK_DIMS) + 1][important_inds]
        gt_pick = traj_seq[torch.arange(traj_seq.shape[0]), pick_inds.squeeze(), :3]
        gt_release = traj_seq[torch.arange(traj_seq.shape[0]), release_inds.squeeze(), :3]
        gt_important = traj_seq[:, :, : len(TASK_DIMS) + 1][important_inds]
        important_dist = torch.norm(output_important - gt_important)
        pick_dist = torch.norm(output_pick - gt_pick, dim=1)
        release_dist = torch.norm(output_release - gt_release, dim=1)
        pick_dists.append(pick_dist.mean().item())
        release_dists.append(release_dist.mean().item())
        important_dists.append(important_dist.item())
    
    return total_losses, pos_losses, ori_losses, grasp_losses, action_losses, pick_dists, release_dists, important_dists


def valid_traj_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    traj_loss_func,
    action_loss_func,
    device: torch.device,
    pos_weight: float = 1,
    quat_weight: float = 1,
    grasp_weight: float = 1,
    action_weight: float = 1,
    tta_rotations: int = 0,
    tta_axis: str = "z",
):
    """Run one validation epoch (no gradient computation).

    The full trajectory is assembled once (segment-by-segment, or with
    test-time rotation averaging when ``tta_rotations > 1``), and the
    position/orientation/grasp losses are computed on that assembled trajectory
    over all real (non-padding) timesteps -- not summed per segment. Action
    loss/accuracy comes from the per-segment forward passes (the action head is
    a per-pass output, not part of the assembled trajectory). pick/release/
    important distances and best-checkpoint selection use the assembled
    trajectory.

    Note: trajectory loss values here are NOT directly comparable to
    ``train_traj_epoch`` (which sums per-segment); they are whole-trajectory
    means. Best-checkpoint selection uses important_dist, not loss.
    """
    model.eval()
    total_losses, pos_losses, ori_losses, grasp_losses, grasp_accuracies, action_losses, action_accuracies = [], [], [], [], [], [], []
    pick_dists, release_dists, important_dists = [], [], []
    
    for sample_batched in dataloader:
        obj_seq, traj_seq, traj_hidden, weights, action_tag_gt, padding_mask, img_inds, pick_inds, release_inds = sample_batched
        weights = weights.to(device)
        important_inds = weights > 150
        padding_mask = padding_mask.to(device)
        obj_seq = obj_seq.to(device)
        traj_seq = traj_seq.to(device)
        traj_hidden = traj_hidden.to(device)
        action_tag_gt = action_tag_gt.to(device)
        
        with torch.no_grad():
            # --- Assemble the full predicted trajectory once ---
            # Only random-rotation-trained models use rotation averaging; for
            # everything else this is a single deterministic segment-by-segment
            # pass. The caller already gates tta_rotations on the augmentation
            # method, so here we just branch on the effective value.
            if tta_rotations and tta_rotations > 1:
                result = tta_assemble_result(
                    model, obj_seq, traj_hidden, padding_mask, img_inds,
                    len(TASK_DIMS), device,
                    num_rotations=tta_rotations, axis=tta_axis,
                )
            else:
                result = assemble_result(
                    model, obj_seq, traj_hidden, padding_mask, img_inds,
                    len(TASK_DIMS), device,
                )

            n_task = len(TASK_DIMS)
            pred_traj = result[:, :, : n_task + 1]          # (B, T, 8) pos+quat+grasp
            gt_traj = traj_seq[:, :, : n_task + 1]
            # Whole-trajectory mask: all real (non-padding) timesteps. The union
            # of the per-segment masks is exactly the non-padding region, so this
            # matches the timesteps the old per-segment loop scored.
            valid_mask = (~padding_mask).to(device).unsqueeze(-1).float()  # (B, T, 1)
            masked_weights = weights * valid_mask
            pos_loss, ori_loss, grasp_loss = traj_loss_func(
                pred_traj, gt_traj, masked_weights,
                quat_weight=quat_weight, pos_weight=pos_weight, grasp_weight=grasp_weight,
            )

            # Grasp detection accuracy over the whole trajectory (real timesteps).
            grasp_mask = valid_mask[:, :, 0] > 0
            grasp_label_pred = (pred_traj[:, :, n_task] > 0.5) * grasp_mask
            grasp_label_gt = (gt_traj[:, :, n_task] > 0.5) * grasp_mask
            grasp_accuracy = (grasp_label_pred == grasp_label_gt).float().mean()
            grasp_accuracies.append(grasp_accuracy.item())

            # --- Action loss/accuracy from the (unrotated) per-segment passes ---
            # The action head is a per-forward-pass output, not part of the
            # assembled trajectory, so it is computed from a plain segment loop
            # on the unrotated scene. Averaged across segments (as before).
            seg_action_losses = []
            labels_gt = torch.argmax(action_tag_gt, axis=1)
            for i in range(obj_seq.shape[1]):
                _out, action_tag_seq = model(
                    obj_seq[:, i, :, :], traj_hidden,
                    tgt_padding_mask=padding_mask, predict_action=True,
                )
                a_loss = action_loss_func(action_tag_seq, labels_gt) * action_weight
                seg_action_losses.append(a_loss)
                labels_pred = torch.argmax(action_tag_seq, dim=1)
                action_accuracies.append((labels_pred == labels_gt).float().mean().item())
            action_loss = torch.stack(seg_action_losses).mean()
            action_losses.append(action_loss.item())

            # Total = whole-trajectory pose/grasp loss + mean action loss.
            total_losses.append((pos_loss + ori_loss + grasp_loss + action_loss).item())
            pos_losses.append(pos_loss.item())
            ori_losses.append(ori_loss.item())
            grasp_losses.append(grasp_loss.item())

            # Pick/release/important distances
            output_pick = result[torch.arange(result.shape[0]), pick_inds.squeeze(), :3]
            output_release = result[torch.arange(result.shape[0]), release_inds.squeeze(), :3]
            output_important = result[:, :, : len(TASK_DIMS) + 1][important_inds]
            gt_pick = traj_seq[torch.arange(traj_seq.shape[0]), pick_inds.squeeze(), :3]
            gt_release = traj_seq[torch.arange(traj_seq.shape[0]), release_inds.squeeze(), :3]
            gt_important = traj_seq[:, :, : len(TASK_DIMS) + 1][important_inds]
            important_dist = torch.norm(output_important - gt_important)
            pick_dist = torch.norm(output_pick - gt_pick, dim=1)
            release_dist = torch.norm(output_release - gt_release, dim=1)
            pick_dists.append(pick_dist.mean().item())
            release_dists.append(release_dist.mean().item())
            important_dists.append(important_dist.item())
    
    return (
        total_losses,
        pos_losses,
        ori_losses,
        grasp_losses,
        grasp_accuracies,
        action_losses,
        action_accuracies,
        pick_dists,
        release_dists,
        important_dists,
    )


def log_training_stats(log_message: str, log_file: str) -> None:
    """Append a training log message to a text file."""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message)


def train_model(config: TrainConfig) -> None:
    """Main training loop.
    
    1. Build datasets and dataloaders
    2. Build model, optimizer, and LR scheduler
    3. Loop over epochs:
       - Train one epoch
       - Validate one epoch
       - Log metrics (every print_interval epochs)
       - Save periodic checkpoint (every save_interval epochs)
       - Save best checkpoint when validation important_dist improves
       - Step LR scheduler based on validation important distance
       - Stop if LR has been floored at min_lr (scheduler can't reduce further)
    
    Checkpoints saved to: <output_root>/<model_name>/<seed>/
        model_<epoch>.pth   (periodic; weights only unless save_optimizer=True)
        model_best.pth      (overwritten when val improves; weights only)
        model_last.pth      (overwritten every save_interval; weights only)
    Training log saved to: <output_root>/<model_name>/<seed>/training_log.txt
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build datasets
    train_data, valid_data, _test_data, train_stats = build_datasets(config)
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
    
    # Scheduler patience is configured in gradient-update steps and translated
    # to epochs here. PyTorch's DataLoader doesn't drop the partial last batch,
    # so batches_per_epoch is ceil(N / batch_size). At K=1 with dataset_size=3
    # there's still 1 batch/epoch (3 samples padded into the partial batch).
    import math
    batches_per_epoch = max(1, math.ceil(len(train_data) / config.batch_size))
    scheduler_patience_epochs = max(1, config.scheduler_patience_steps // batches_per_epoch)
    print(
        f"Train dataset: {len(train_data)} samples, batch_size={config.batch_size}, "
        f"batches_per_epoch={batches_per_epoch}, "
        f"scheduler_patience: {config.scheduler_patience_steps} steps -> "
        f"{scheduler_patience_epochs} epochs"
    )
    
    # Create output directory and save normalization stats
    folder = os.path.join(config.output_root, config.model_name, str(config.seed))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "train_stat.pickle"), "wb") as f:
        import pickle
        pickle.dump(train_stats, f)
    
    # Build model + optimizer ONCE (outside the epoch loop) so Adam's running
    # moments and the LR scheduler's state actually persist across epochs.
    model = build_model(config, device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=scheduler_patience_epochs, factor=0.5, threshold=0.01, threshold_mode="rel", min_lr=config.min_lr)
    
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Model Parameters: {get_n_params(model)}")
    print(f"Augmentation method: {config.augmentation_method}")
    print(f"Total epochs (max): {config.total_epochs}, min_lr: {config.min_lr}")
    
    log_file = os.path.join(folder, "training_log.txt")
    
    # Best-on-validation tracking. The selection metric (config.selection_metric)
    # drives best-checkpoint, the LR scheduler, and the early-stop.
    best_v_select = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    
    epoch = 0
    while epoch < config.total_epochs + 1:
        # --- Training ---
        t_loss, t_pos_loss, t_ori_loss, t_grasp_loss, t_action_loss, t_pick_dists, t_release_dists, t_important_dists = train_traj_epoch(
            model,
            optimizer,
            train_dataloader,
            weighted_pose_grasp_loss_func,
            action_tag_loss,
            device,
            pos_weight=config.pos_weight,
            grasp_weight=config.grasp_weight,
            quat_weight=config.quat_weight,
            action_weight=config.action_weight,
        )
        
        # --- Validation ---
        (
            v_loss,
            v_pos_loss,
            v_ori_loss,
            v_grasp_loss,
            v_grasp_accuracies,
            v_action_loss,
            v_action_accuracies,
            v_pick_dists,
            v_release_dists,
            v_important_dists,
        ) = valid_traj_epoch(
            model,
            valid_dataloader,
            weighted_pose_grasp_loss_func,
            action_tag_loss,
            device,
            pos_weight=config.pos_weight,
            grasp_weight=config.grasp_weight,
            quat_weight=config.quat_weight,
            action_weight=config.action_weight,
            # TTA (rotation averaging at eval) is only meaningful for models
            # trained with random-rotation augmentation; tp/none always do a
            # single deterministic pass.
            tta_rotations=(config.tta_rotations if config.augmentation_method == "random" else 0),
            tta_axis=config.tta_axis,
        )
        
        v_loss_mean = np.mean(v_loss)
        v_important_dist_mean = float(np.mean(v_important_dists))
        v_pose_loss_mean = float(np.mean(v_pos_loss) + np.mean(v_ori_loss))

        # Validation metrics available for selection / scheduling.
        _metric_values = {
            "important_dist": v_important_dist_mean,
            "pose_loss": v_pose_loss_mean,
            "total_loss": float(v_loss_mean),
        }
        for _name in (config.selection_metric, config.scheduler_metric):
            if _name not in _metric_values:
                raise ValueError(
                    f"unknown metric={_name!r}; expected one of {sorted(_metric_values)}"
                )
        v_select = _metric_values[config.selection_metric]   # best-checkpoint
        v_sched = _metric_values[config.scheduler_metric]     # LR scheduler + early-stop

        # --- Track best-on-validation (by the configured selection metric) ---
        improved = v_select < best_v_select
        if improved:
            best_v_select = v_select
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "selection_metric": config.selection_metric, "v_select": best_v_select,
                 "v_pose_loss": v_pose_loss_mean, "v_total_loss": float(v_loss_mean),
                 "v_important_dist": v_important_dist_mean},
                os.path.join(folder, "model_best.pth"),
            )
        else:
            epochs_since_improvement += 1
        
        # --- Logging ---
        if epoch % config.print_interval == 0:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            current_lr = optimizer.param_groups[0]["lr"]
            message = (
                f"\n-------------------Time {timestamp}, Epoch {epoch}, Learning rate [{current_lr}], Batch size {config.batch_size} --------------------\n"
                f"Training: total loss {round(np.mean(t_loss), 3)}, pos loss {round(np.mean(t_pos_loss), 3)}, ori loss {round(np.mean(t_ori_loss), 3)}, "
                f"grasp loss {round(np.mean(t_grasp_loss), 3)}, action loss {round(np.mean(t_action_loss), 3)}, pick dist {round(np.mean(t_pick_dists), 3)}, "
                f"release dist {round(np.mean(t_release_dists), 3)}, important dist {round(np.mean(t_important_dists), 3)}\n"
                f"Valid   : total loss {round(v_loss_mean, 3)}, pos loss {round(np.mean(v_pos_loss), 3)}, ori loss {round(np.mean(v_ori_loss), 3)}, "
                f"grasp loss {round(np.mean(v_grasp_loss), 3)}, action loss {round(np.mean(v_action_loss), 3)}, grasp accuracy {round(np.mean(v_grasp_accuracies), 3)}, "
                f"action accuracy {round(np.mean(v_action_accuracies), 3)}, pick dist {round(np.mean(v_pick_dists), 3)}, release dist {round(np.mean(v_release_dists), 3)},  important dist {round(v_important_dist_mean, 3)}\n"
                f"Best    : epoch {best_epoch}, {config.selection_metric} {round(best_v_select, 3)}, epochs_since_improvement {epochs_since_improvement}\n"
            )
            print(message)
            log_training_stats(message, log_file)
        
        # --- Save periodic + last checkpoint ---
        if epoch % config.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "v_pose_loss": v_pose_loss_mean,
                "v_total_loss": float(v_loss_mean),
                "v_important_dist": v_important_dist_mean,
            }
            if config.save_optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(checkpoint, os.path.join(folder, f"model_{epoch}.pth"))
            torch.save(checkpoint, os.path.join(folder, "model_last.pth"))
        
        # Step LR scheduler based on the configured scheduler metric (drives LR
        # decay and, via the LR floor below, the early-stop).
        scheduler.step(v_sched)
        
        # --- Stop when LR floors out (scheduler can't reduce further) ---
        # Once the LR scheduler hits its `min_lr` floor, parameter updates are
        # negligible; further epochs just burn compute without improving val.
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr <= config.min_lr * 1.01:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            message = (
                f"\n[{timestamp}] LR floor reached at epoch {epoch}: "
                f"current_lr={current_lr:g} <= min_lr={config.min_lr:g} "
                f"(best epoch {best_epoch}, best {config.selection_metric} {round(best_v_select, 3)}).\n"
            )
            print(message)
            log_training_stats(message, log_file)
            break
        
        epoch += 1
    
    # Final summary line
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    final_msg = (
        f"\n[{timestamp}] Training finished. "
        f"Best epoch: {best_epoch}, best val {config.selection_metric}: {round(best_v_select, 3)}. "
        f"Best checkpoint: {os.path.join(folder, 'model_best.pth')}\n"
    )
    print(final_msg)
    log_training_stats(final_msg, log_file)
