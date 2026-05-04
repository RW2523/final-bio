from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.snn_activations import LIFVec1Step


@dataclass
class SpikeFormerConfig:
    dim: int = 256
    depth: int = 4
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.1
    spike_threshold: float = 0.5


class SpikeMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float, spike_threshold: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act1 = LIFVec1Step(threshold=float(spike_threshold))
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act2 = LIFVec1Step(threshold=float(spike_threshold))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x_btd)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        return x


class SpikeSelfAttention(nn.Module):
    """
    Simple spiking self-attention (pure torch):
    - Linear Q/K/V
    - LIF spike nonlinearity on Q/K/V (per-element surrogate spike)
    - Scaled dot-product attention (no softmax; uses spikes + scaling)
    - LIF spike on output projection
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        attn_dropout: float,
        spike_threshold: float,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.q_lif = LIFVec1Step(threshold=float(spike_threshold))
        self.k_lif = LIFVec1Step(threshold=float(spike_threshold))
        self.v_lif = LIFVec1Step(threshold=float(spike_threshold))

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_lif = LIFVec1Step(threshold=float(spike_threshold))
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:
        b, t, d = x_btd.shape
        qkv = self.qkv(x_btd)  # (B,T,3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_lif(q)
        k = self.k_lif(k)
        v = self.v_lif(v)

        # (B, H, T, Hd)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B,H,T,Hd)
        out = out.transpose(1, 2).contiguous().view(b, t, d)  # (B,T,D)

        out = self.proj(out)
        out = self.out_lif(out)
        out = self.proj_drop(out)
        return out


class SpikeFormerBlock(nn.Module):
    def __init__(self, cfg: SpikeFormerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = SpikeSelfAttention(
            dim=cfg.dim,
            heads=cfg.heads,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            spike_threshold=cfg.spike_threshold,
        )
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.mlp = SpikeMLP(
            dim=cfg.dim,
            hidden_dim=int(cfg.dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            spike_threshold=cfg.spike_threshold,
        )

    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:
        x = x_btd + self.attn(self.norm1(x_btd))
        x = x + self.mlp(self.norm2(x))
        return x


class SpikeFormerBackbone(nn.Module):
    """
    Backbone-style API consistent with the rest of SNN_HAR:
      forward(x[B,T,C]) -> (None, z[B,dim])
    """

    def __init__(
        self,
        n_channels: int,
        len_sw: int,
        *,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        spike_threshold: float = 0.5,
        backbone: bool = True,
    ):
        super().__init__()
        if not backbone:
            raise NotImplementedError("SpikeFormerBackbone is wired as backbone-only in this repo.")
        self.backbone = True
        self.out_dim = int(dim)

        self.in_proj = nn.Linear(int(n_channels), int(dim), bias=False)
        self.pos = nn.Parameter(torch.zeros(1, int(len_sw), int(dim)))
        self.drop = nn.Dropout(dropout)

        cfg = SpikeFormerConfig(
            dim=int(dim),
            depth=int(depth),
            heads=int(heads),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
            attn_dropout=float(attn_dropout),
            spike_threshold=float(spike_threshold),
        )
        self.blocks = nn.ModuleList([SpikeFormerBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim)

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x_btc: torch.Tensor):
        if x_btc.dim() != 3:
            raise ValueError(f"SpikeFormerBackbone expects (B,T,C); got {tuple(x_btc.shape)}")
        x = self.in_proj(x_btc)
        if x.shape[1] != self.pos.shape[1]:
            raise ValueError(
                f"Input T={x.shape[1]} must equal len_sw={self.pos.shape[1]} for SpikeFormerBackbone."
            )
        x = self.drop(x + self.pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        z = x.mean(dim=1)  # pool over time -> (B, dim)
        return None, z

