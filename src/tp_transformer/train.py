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
):
    """Run one validation epoch (no gradient computation).
    
    Same structure as train_traj_epoch but with torch.no_grad() and
    additional accuracy metrics (grasp accuracy, action accuracy).
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
        result = traj_hidden.clone()
        tmp = torch.zeros((result.shape[0], result.shape[1], 1)).to(device)
        result = torch.cat((tmp, result), dim=2)
        
        with torch.no_grad():
            for i in range(obj_seq.shape[1]):
                loss_start_inds = input_end_inds = img_inds[:, i]
                if i != obj_seq.shape[1] - 1:
                    loss_end_inds = img_inds[:, i + 1]
                else:
                    loss_end_inds = torch.ones(obj_seq.shape[0], dtype=torch.int) * (-1)
                
                output_seq, action_tag_seq = model(obj_seq[:, i, :, :], traj_hidden, tgt_padding_mask=padding_mask, predict_action=True)
                
                # Accumulate predictions
                for (start, end) in zip(loss_start_inds, loss_end_inds):
                    result[:, start:end, : len(TASK_DIMS) + 1] = output_seq[:, start:end]
                
                # Action loss and accuracy
                labels_gt = torch.argmax(action_tag_gt, axis=1)
                action_loss = action_loss_func(action_tag_seq, labels_gt) * action_weight
                action_losses.append(action_loss.item())
                labels_pred = torch.argmax(action_tag_seq, dim=1)
                action_accuracy = (labels_pred == labels_gt).float().mean()
                action_accuracies.append(action_accuracy.item())
                
                # Trajectory losses
                loss_mask = get_mask(output_seq.shape, loss_start_inds, loss_end_inds).to(device)
                masked_weights = weights * loss_mask
                pos_loss, ori_loss, grasp_loss = traj_loss_func(
                    output_seq, traj_seq[:, :, : len(TASK_DIMS) + 1], masked_weights, quat_weight=quat_weight, pos_weight=pos_weight, grasp_weight=grasp_weight
                )
                loss = pos_loss + ori_loss + grasp_loss + action_loss
                total_losses.append(loss.item())
                pos_losses.append(pos_loss.item())
                ori_losses.append(ori_loss.item())
                grasp_losses.append(grasp_loss.item())
                
                # Grasp detection accuracy (within masked region)
                grasp_label_pred = output_seq[:, :, len(TASK_DIMS)] > 0.5
                grasp_label_gt = traj_seq[:, :, len(TASK_DIMS)] > 0.5
                grasp_mask = loss_mask[:, :, len(TASK_DIMS)]
                grasp_label_pred = grasp_label_pred * grasp_mask
                grasp_label_gt = grasp_label_gt * grasp_mask
                grasp_accuracy = (grasp_label_pred == grasp_label_gt).float().mean()
                grasp_accuracies.append(grasp_accuracy.item())
            
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
       - Save checkpoint (every save_interval epochs)
       - Step LR scheduler based on validation important distance
    
    Checkpoints saved to: transformer/<model_name>/<seed>/model_<epoch>.pth
    Training log saved to: transformer/<model_name>/<seed>/training_log.txt
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build datasets
    train_data, valid_data, _test_data, train_stats = build_datasets(config)
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
    
    # Create output directory and save normalization stats
    os.makedirs(f"./transformer/{config.model_name}/{config.seed}", exist_ok=True)
    with open(f"./transformer/{config.model_name}/{config.seed}/train_stat.pickle", "wb") as f:
        import pickle
        pickle.dump(train_stats, f)
    
    # Build model
    model = build_model(config, device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=500, factor=0.5, threshold=0.01, threshold_mode="rel")
    
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Model Parameters: {get_n_params(model)}")
    
    folder = f"./transformer/{config.model_name}/{config.seed}"
    log_file = os.path.join(folder, "training_log.txt")
    
    epoch = 0
    while epoch < config.total_epochs + 1:
        # Note: optimizer is re-created each epoch (resets momentum)
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        
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
        )
        
        v_loss_mean = np.mean(v_loss)
        v_important_dists = np.mean(v_important_dists)
        
        # --- Logging ---
        if epoch % config.print_interval == 0:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            message = (
                f"\n-------------------Time {timestamp}, Epoch {epoch}, Learning rate {scheduler.get_last_lr()}, Batch size {config.batch_size} --------------------\n"
                f"Training: total loss {round(np.mean(t_loss), 3)}, pos loss {round(np.mean(t_pos_loss), 3)}, ori loss {round(np.mean(t_ori_loss), 3)}, "
                f"grasp loss {round(np.mean(t_grasp_loss), 3)}, action loss {round(np.mean(t_action_loss), 3)}, pick dist {round(np.mean(t_pick_dists), 3)}, "
                f"release dist {round(np.mean(t_release_dists), 3)}, important dist {round(np.mean(t_important_dists), 3)}\n"
                f"Valid   : total loss {round(v_loss_mean, 3)}, pos loss {round(np.mean(v_pos_loss), 3)}, ori loss {round(np.mean(v_ori_loss), 3)}, "
                f"grasp loss {round(np.mean(v_grasp_loss), 3)}, action loss {round(np.mean(v_action_loss), 3)}, grasp accuracy {round(np.mean(v_grasp_accuracies), 3)}, "
                f"action accuracy {round(np.mean(v_action_accuracies), 3)}, pick dist {round(np.mean(v_pick_dists), 3)}, release dist {round(np.mean(v_release_dists), 3)},  important dist {round(v_important_dists, 3)}\n"
            )
            print(message)
            log_training_stats(message, log_file)
        
        # --- Save checkpoint ---
        if epoch % config.save_interval == 0:
            checkpoint = {"epoch": epoch, "model": model, "optimizer": optimizer, "scheduler": scheduler}
            path = os.path.join(folder, f"model_{epoch}.pth")
            torch.save(checkpoint, path)
        
        # Step LR scheduler based on validation important distance
        scheduler.step(v_important_dists)
        epoch += 1
