"""
Shared layers used by all Transformer model variants.

- PositionalEncoding: Sinusoidal position embeddings (Vaswani et al., 2017)
- ActionClassifier: MLP head for classifying which action/task is being performed
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer sequence models.
    
    Adds position-dependent signals to the input embeddings so the model
    can distinguish different timesteps in the trajectory. Uses the standard
    sin/cos formulation from "Attention Is All You Need".
    
    pe(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    pe(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Embedding dimension
        dropout: Dropout rate applied after adding positional encoding
        max_len: Maximum sequence length supported
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Even dimensions: sin
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Odd dimensions: cos
        self.register_buffer("pe", pe)  # Not a parameter, but moves with model to device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ActionClassifier(nn.Module):
    """MLP classifier that predicts which action/task is being performed.
    
    Takes the encoder's object embedding, flattens it, and passes through
    a 3-layer MLP: input_dim -> 64 -> 16 -> n_tasks.
    
    Args:
        input_dim: Flattened dimension of encoder output (embed_dim * n_objs for TFEncoderDecoder5)
        output_dim: Number of action classes (n_tasks)
    """
    
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
        """Classify action from encoder embedding.
        
        Args:
            x: Encoder output (batch, n_objs, embed_dim)
        
        Returns:
            Action logits (batch, n_tasks)
        """
        x = x.view(x.shape[0], -1)  # Flatten: (batch, n_objs * embed_dim)
        for layer in self.layers:
            x = layer(x)
        return x
