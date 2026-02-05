from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ActionClassifier, PositionalEncoding


class TFEncoderDecoder(nn.Module):
    def __init__(
        self,
        task_dim: int,
        traj_dim: int,
        embed_dim: int,
        nhead: int,
        max_len: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)
        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.lin0 = nn.Linear(task_dim, embed_dim, dtype=torch.float64)
        self.lin4 = nn.Linear(embed_dim, traj_dim, dtype=torch.float64)
        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def forward(self, source: torch.Tensor, target: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        obj_emb = self.lin0(source) * math.sqrt(self.d_model)
        traj_emb = self.lin0(target) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        if self.num_encoder_layers > 0:
            obj_emb = self.encoder(obj_emb)
        x = self.decoder(traj_emb, obj_emb, tgt_mask=self.mask, tgt_key_padding_mask=padding_mask)
        return self.lin4(x)


class TFEncoderDecoder2(nn.Module):
    def __init__(
        self,
        task_dim: int,
        traj_dim: int,
        n_tasks: int,
        n_objs: int,
        embed_dim: int,
        nhead: int,
        max_len: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)
        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.lin0 = nn.Linear(task_dim, embed_dim, dtype=torch.float64)
        self.lin1 = nn.Linear(embed_dim * n_objs, 64, dtype=torch.float64)
        self.lin2 = nn.Linear(64, 16, dtype=torch.float64)
        self.lin3 = nn.Linear(16, n_tasks, dtype=torch.float64)
        self.lin4 = nn.Linear(embed_dim, traj_dim, dtype=torch.float64)
        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def forward(self, source: torch.Tensor, target: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        obj_emb = self.lin0(source) * math.sqrt(self.d_model)
        traj_emb = self.lin0(target) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        if self.num_encoder_layers > 0:
            obj_emb = self.encoder(obj_emb)
        y = self.lin1(F.relu(obj_emb).view(traj_emb.shape[0], -1))
        y = F.relu(y)
        y = self.lin2(y)
        y = F.relu(y)
        y = self.lin3(y)
        x = self.decoder(traj_emb, obj_emb, tgt_mask=self.mask, tgt_key_padding_mask=padding_mask)
        return self.lin4(x), y


class TFEncoderDecoder3(nn.Module):
    def __init__(
        self,
        task_dim: int,
        target_dim: int,
        source_dim: int,
        n_tasks: int,
        embed_dim: int,
        nhead: int,
        max_len: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.n_objs = int(source_dim / target_dim)
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)
        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.traj_embed_layer = nn.Linear(target_dim, embed_dim, dtype=torch.float64)
        self.obj_embed_layer = nn.Linear(source_dim, embed_dim, dtype=torch.float64)
        self.action_classifier = ActionClassifier(embed_dim, n_tasks)
        self.lin4 = nn.Linear(embed_dim, task_dim, dtype=torch.float64)
        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def encode(self, source: torch.Tensor, src_mask: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        src_emb = self.obj_embed_layer(source) * math.sqrt(self.d_model)
        src_emb = self.pos_embed(src_emb)
        if self.num_encoder_layers > 0:
            src_emb = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=padding_mask)
        return src_emb

    def decode(
        self,
        src_emb: torch.Tensor,
        target: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor],
        memory_mask: torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor],
    ):
        tgt_emb = self.traj_embed_layer(target) * math.sqrt(self.d_model)
        tgt_emb = self.pos_embed(tgt_emb)
        x = self.decoder(
            tgt_emb,
            src_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.lin4(x)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        predict_action: bool = False,
    ):
        src_emb = self.encode(source, self.mask, src_padding_mask)
        action = self.action_classifier(src_emb) if predict_action else None
        x = self.decode(src_emb, target, self.mask, tgt_padding_mask, self.mask, memory_padding_mask)
        return x, action


class TFEncoderDecoder4(nn.Module):
    def __init__(
        self,
        task_dim: int,
        source_dim: int,
        target_dim: int,
        n_tasks: int,
        embed_dim: int,
        nhead: int,
        max_len: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.embed_layer = nn.Linear(target_dim, embed_dim, dtype=torch.float64)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)
        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.action_classifier = ActionClassifier(embed_dim, n_tasks)
        self.lin4 = nn.Linear(embed_dim, task_dim, dtype=torch.float64)
        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def encode(
        self, source: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_padding_mask: Optional[torch.Tensor] = None
    ):
        src_emb = self.embed_layer(source) * math.sqrt(self.d_model)
        if self.num_encoder_layers > 0:
            src_emb = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        return src_emb

    def decode(
        self,
        src_emb: torch.Tensor,
        target: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        tgt_padding_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ):
        if tgt_mask is None:
            tgt_mask = self.mask
        tgt_emb = self.embed_layer(target) * math.sqrt(self.d_model)
        tgt_emb = self.pos_embed(tgt_emb)
        x = self.decoder(
            tgt_emb,
            src_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.lin4(x)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        predict_action: bool = False,
    ):
        src_emb = self.encode(source, src_mask=src_mask, src_padding_mask=src_padding_mask)
        action = self.action_classifier(src_emb) if predict_action else None
        x = self.decode(src_emb, target, tgt_mask, tgt_padding_mask, memory_mask, memory_padding_mask)
        return x, action


class TFEncoderDecoder5(nn.Module):
    def __init__(
        self,
        task_dim: int,
        source_dim: int,
        target_dim: int,
        n_objs: int,
        n_tasks: int,
        embed_dim: int,
        nhead: int,
        max_len: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.embed_layer = nn.Linear(source_dim, embed_dim, dtype=torch.float64)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)
        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.action_classifier = ActionClassifier(embed_dim * n_objs, n_tasks)
        self.lin4 = nn.Linear(embed_dim, task_dim, dtype=torch.float64)
        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def encode(
        self, source: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_padding_mask: Optional[torch.Tensor] = None
    ):
        src_emb = self.embed_layer(source) * math.sqrt(self.d_model)
        if self.num_encoder_layers > 0:
            src_emb = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        return src_emb

    def decode(
        self,
        src_emb: torch.Tensor,
        target: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        tgt_padding_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ):
        if tgt_mask is None:
            tgt_mask = self.mask
        tgt_emb = self.embed_layer(target) * math.sqrt(self.d_model)
        tgt_emb = self.pos_embed(tgt_emb)
        x = self.decoder(
            tgt_emb,
            src_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.lin4(x)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        predict_action: bool = False,
    ):
        src_emb = self.encode(source, src_mask=src_mask, src_padding_mask=src_padding_mask)
        action = self.action_classifier(src_emb) if predict_action else None
        x = self.decode(src_emb, target, tgt_mask, tgt_padding_mask, memory_mask, memory_padding_mask)
        return x, action


class TFEncoderDecoderNoMask(nn.Module):
    def __init__(
        self,
        task_dim: int,
        traj_dim: int,
        embed_dim: int,
        nhead: int,
        max_len: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)
        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.lin0 = nn.Linear(task_dim, embed_dim, dtype=torch.float64)
        self.lin4 = nn.Linear(embed_dim, traj_dim, dtype=torch.float64)
        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def forward(self, source: torch.Tensor, target: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        obj_emb = self.lin0(source) * math.sqrt(self.d_model)
        traj_emb = self.lin0(target) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        if self.num_encoder_layers > 0:
            obj_emb = self.encoder(obj_emb)
        x = self.decoder(traj_emb, obj_emb)
        return self.lin4(x)
