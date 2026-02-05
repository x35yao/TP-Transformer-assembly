from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ActionClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, 64, dtype=torch.float64),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(64, 16, dtype=torch.float64),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(16, output_dim, dtype=torch.float64),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x
