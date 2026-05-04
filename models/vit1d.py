"""
1D Vision-Transformer style encoder: fixed-size time patches, linear projection, CLS
token, learnable position embeddings, and transformer blocks (self-attention).
Input layout matches other backbones: (B, T, C) with T = len_sw.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import Transformer as AttnTransformer


class ViT1D(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        len_sw: int,
        patch_size: int = 20,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        backbone: bool = True,
    ) -> None:
        super().__init__()
        if len_sw % patch_size != 0:
            raise ValueError(f"len_sw ({len_sw}) must be divisible by patch_size ({patch_size})")
        self.n_channels = n_channels
        self.len_sw = len_sw
        self.patch_size = patch_size
        self.n_patches = len_sw // patch_size
        self.backbone = backbone
        self.out_dim = dim

        patch_len = n_channels * patch_size
        self.patch_embed = nn.Linear(patch_len, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # +1 for CLS, same as image ViT
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = AttnTransformer(dim, depth, heads, mlp_dim, dropout=dropout)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        if not backbone:
            self.logits = nn.Linear(dim, n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def last_block_parameters(self):
        """Unfreeze the last (attention + FFN) sub-block; used with lincls_finetune_scope=last_block."""
        last = self.blocks.layers[-1]
        for module in last:
            yield from module.parameters()

    def forward(self, x_btc: torch.Tensor):
        b, t, c = x_btc.shape
        p = self.patch_size
        if t != self.len_sw:
            raise ValueError(f"expected time length {self.len_sw}, got {t}")
        # (B, n_patches, p * C)
        x = x_btc.reshape(b, t // p, p * c)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        feat = x[:, 0]
        if self.backbone:
            return None, feat
        logits = self.logits(feat)
        return logits, feat
