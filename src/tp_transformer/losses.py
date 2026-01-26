import torch
import torch.nn.functional as F


def norm_diff_quat_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    last_dim = len(q1.shape) - 1
    q1 = q1 / torch.norm(q1, dim=last_dim).unsqueeze(-1)
    q2 = q2 / torch.norm(q2, dim=last_dim).unsqueeze(-1)
    q3 = torch.stack([torch.norm(q1 + q2, dim=last_dim), torch.norm(q1 - q2, dim=last_dim)])
    qlosses = torch.min(q3, dim=0)
    return torch.nan_to_num(qlosses.values, 1.414)


def inner_prod_quat(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = q1 / torch.norm(q1, dim=-1).unsqueeze(-1)
    q2 = q2 / torch.norm(q2, dim=-1).unsqueeze(-1)
    return torch.arccos(torch.abs(torch.einsum("abc,abc->ab", q1, q2)))


def custom_weighted_loss_func(pred: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((weights * (pred - truth) ** 2).sum(axis=-1)).mean()


def custom_weighted_pose_loss_func(
    pred: torch.Tensor, truth: torch.Tensor, traj_weights: torch.Tensor, pos_weight: float = 1, quat_weight: float = 1
) -> torch.Tensor:
    loss_pos = custom_weighted_loss_func(pred[:, :, :3], truth[:, :, :3], traj_weights[:, :, :3])
    loss_ori = custom_weighted_loss_func(pred[:, :, 3:], truth[:, :, 3:], traj_weights[:, :, 3:])
    return loss_ori * quat_weight + loss_pos * pos_weight


def custom_weighted_grasp_func(pred: torch.Tensor, truth: torch.Tensor, traj_weights: torch.Tensor) -> torch.Tensor:
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
    loss_grasp = custom_weighted_loss_func(
        pred[:, :, -1, None], truth[:, :, -1, None], traj_weights[:, :, -1, None] > 0
    ) * grasp_weight
    loss_pos = custom_weighted_loss_func(pred[:, :, :3], truth[:, :, :3], traj_weights[:, :, :3]) * pos_weight
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
    pose_loss = custom_weighted_pose_loss_func(traj_pred, traj_truth, traj_weights, quat_weight=quat_weight)
    action_loss = F.cross_entropy(action_pred, action_truth) * action_weight
    return pose_loss + action_loss


def action_tag_loss(action_pred: torch.Tensor, action_truth: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(action_pred, action_truth)


def get_mask(shape: torch.Size, start_inds: torch.Tensor, end_inds: torch.Tensor) -> torch.Tensor:
    loss_mask = torch.zeros(shape)
    for i, (start, end) in enumerate(zip(start_inds, end_inds)):
        if end == -1:
            loss_mask[i, start:] = 1
        else:
            loss_mask[i, start : end + 1] = 1
    return loss_mask


def nn_bce_with_logits():
    return torch.nn.BCEWithLogitsLoss(reduction="none")
