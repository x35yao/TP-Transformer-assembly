"""
Loss functions for TP-Transformer training.

The model predicts three things:
1. Position (x, y, z)
2. Orientation (quaternion: qx, qy, qz, qw)
3. Grasp state (binary: open/closed)
4. Action class (which task is being performed)

Each has a separate loss term, weighted and summed into the total loss.
All trajectory losses are weighted per-timestep to emphasize critical regions
(near grasp points, slow segments, etc.).
"""

import torch
import torch.nn.functional as F


def norm_diff_quat_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute quaternion distance that handles the q / -q equivalence.
    
    Quaternions q and -q represent the same rotation. This metric takes the
    minimum of ||q1 + q2|| and ||q1 - q2|| to handle this ambiguity.
    NaN values (from zero-norm quaternions) are replaced with sqrt(2) ~ 1.414.
    
    Args:
        q1, q2: Quaternion tensors of matching shape (..., 4)
    
    Returns:
        Per-element distance tensor
    """
    last_dim = len(q1.shape) - 1
    q1 = q1 / torch.norm(q1, dim=last_dim).unsqueeze(-1)
    q2 = q2 / torch.norm(q2, dim=last_dim).unsqueeze(-1)
    q3 = torch.stack([torch.norm(q1 + q2, dim=last_dim), torch.norm(q1 - q2, dim=last_dim)])
    qlosses = torch.min(q3, dim=0)
    return torch.nan_to_num(qlosses.values, 1.414)


def inner_prod_quat(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute angular distance between quaternions using inner product.
    
    Returns arccos(|q1 . q2|), which gives the geodesic distance on the
    unit quaternion sphere. Range is [0, pi/2].
    
    Args:
        q1, q2: Quaternion tensors of shape (batch, seq_len, 4)
    
    Returns:
        Angular distance tensor of shape (batch, seq_len)
    """
    q1 = q1 / torch.norm(q1, dim=-1).unsqueeze(-1)
    q2 = q2 / torch.norm(q2, dim=-1).unsqueeze(-1)
    return torch.arccos(torch.abs(torch.einsum("abc,abc->ab", q1, q2)))


def custom_weighted_loss_func(pred: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean squared error loss.
    
    Computes element-wise squared error, multiplies by per-timestep weights,
    sums over feature dimension, and averages over batch and time.
    
    Args:
        pred: Predicted values (batch, seq_len, D)
        truth: Ground truth values (batch, seq_len, D)
        weights: Per-timestep weights (batch, seq_len, D) - typically same weight repeated across D
    
    Returns:
        Scalar loss value
    """
    return ((weights * (pred - truth) ** 2).sum(axis=-1)).mean()


def custom_weighted_pose_loss_func(
    pred: torch.Tensor, truth: torch.Tensor, traj_weights: torch.Tensor, pos_weight: float = 1, quat_weight: float = 1
) -> torch.Tensor:
    """Combined weighted loss for position and orientation.
    
    Splits prediction into position (cols 0:3) and orientation (cols 3:)
    and applies separate weights to each.
    """
    loss_pos = custom_weighted_loss_func(pred[:, :, :3], truth[:, :, :3], traj_weights[:, :, :3])
    loss_ori = custom_weighted_loss_func(pred[:, :, 3:], truth[:, :, 3:], traj_weights[:, :, 3:])
    return loss_ori * quat_weight + loss_pos * pos_weight


def custom_weighted_grasp_func(pred: torch.Tensor, truth: torch.Tensor, traj_weights: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss for grasp detection, masked to active regions.
    
    Only computes loss where traj_weights > 0 (i.e., within the loss mask).
    """
    criterion = nn_bce_with_logits()
    mask = traj_weights > 0
    loss = criterion(pred, truth)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()


def weighted_pose_grasp_loss_func(
    pred: torch.Tensor,
    truth: torch.Tensor,
    traj_weights: torch.Tensor,
    pos_weight: float = 1,
    quat_weight: float = 1,
    grasp_weight: float = 1,
):
    """Main loss function used during training. Returns three separate loss terms.
    
    Splits prediction into:
    - Position: columns 0:3 (x, y, z)
    - Orientation: columns 3:7 (qx, qy, qz, qw)
    - Grasp: column -1 (binary grasp state)
    
    Args:
        pred: Model output (batch, seq_len, 8) - [pos(3), quat(4), grasp(1)]
        truth: Ground truth (batch, seq_len, 8)
        traj_weights: Per-timestep weights (batch, seq_len, 8)
        pos_weight: Multiplier for position loss
        quat_weight: Multiplier for orientation loss
        grasp_weight: Multiplier for grasp loss
    
    Returns:
        Tuple of (position_loss, orientation_loss, grasp_loss)
    """
    # Grasp loss: only where mask is active (weights > 0)
    loss_grasp = custom_weighted_loss_func(
        pred[:, :, -1, None], truth[:, :, -1, None], traj_weights[:, :, -1, None] > 0
    ) * grasp_weight
    # Position loss: weighted MSE on x, y, z
    loss_pos = custom_weighted_loss_func(pred[:, :, :3], truth[:, :, :3], traj_weights[:, :, :3]) * pos_weight
    # Orientation loss: weighted MSE on quaternion components
    loss_ori = custom_weighted_loss_func(pred[:, :, 3:7], truth[:, :, 3:7], traj_weights[:, :, 3:7]) * quat_weight
    return loss_pos, loss_ori, loss_grasp


def custom_pose_action_loss(
    traj_pred: torch.Tensor,
    traj_truth: torch.Tensor,
    traj_weights: torch.Tensor,
    action_pred: torch.Tensor,
    action_truth: torch.Tensor,
    quat_weight: float = 3,
    action_weight: float = 5,
) -> torch.Tensor:
    """Combined pose + action classification loss (alternative loss function)."""
    pose_loss = custom_weighted_pose_loss_func(traj_pred, traj_truth, traj_weights, quat_weight=quat_weight)
    action_loss = F.cross_entropy(action_pred, action_truth) * action_weight
    return pose_loss + action_loss


def action_tag_loss(action_pred: torch.Tensor, action_truth: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for action classification (which task: action_0/1/2)."""
    return F.cross_entropy(action_pred, action_truth)


def get_mask(shape: torch.Size, start_inds: torch.Tensor, end_inds: torch.Tensor) -> torch.Tensor:
    """Create a binary mask that is 1 between start and end indices for each batch element.
    
    This masks the loss so it's only computed on the trajectory segment corresponding
    to the current object observation (between consecutive camera image indices).
    
    Args:
        shape: Shape of the output mask (batch, seq_len, D)
        start_inds: Start index per batch element (batch,)
        end_inds: End index per batch element (batch,), -1 means until end of sequence
    
    Returns:
        Binary mask tensor of the given shape
    """
    loss_mask = torch.zeros(shape)
    for i, (start, end) in enumerate(zip(start_inds, end_inds)):
        if end == -1:
            loss_mask[i, start:] = 1  # From start to end of sequence
        else:
            loss_mask[i, start : end + 1] = 1  # From start to end (inclusive)
    return loss_mask


def nn_bce_with_logits():
    """Create a BCE loss with logits (unreduced, for manual masking)."""
    return torch.nn.BCEWithLogitsLoss(reduction="none")
