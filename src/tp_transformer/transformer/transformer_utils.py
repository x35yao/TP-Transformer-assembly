from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch


def create_tags(objs: Iterable[str], dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    one_hots = torch.eye(len(list(objs)), dtype=dtype)
    tag_dict: Dict[str, torch.Tensor] = {}
    for i, obj in enumerate(objs):
        tag_dict[obj] = one_hots[i]
    return tag_dict


def normalize_wrapper(average: np.ndarray, std: np.ndarray):
    """normalize for multiprocessing"""
    return lambda x: normalize_3d(x, average, std)


def normalize_3d(entry: np.ndarray, average: np.ndarray, std: np.ndarray) -> np.ndarray:
    if entry.ndim == 3:
        entry[:, :, :3] = (entry[:, :, :3] - average) / std
    elif entry.ndim == 2:
        entry[:, :3] = (entry[:, :3] - average) / std
    return entry