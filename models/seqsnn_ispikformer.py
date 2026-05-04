# encoding=utf-8
"""
WISDM SSL bridge: **iSpikformer** (SpikingJelly LIF) for SNN_HAR.

Implementation is **vendored** under :mod:`models.local_ispikformer` (minimal copy of the
SeqSNN layout). You only need ``pip install spikingjelly`` (see ``requirements.txt``); a
separate ``SeqSNN`` package install is **not** required.

Input: ``(B, T, C)`` with ``T = len_sw``. Pooled features: mean over the variate axis of the
last SNN state ``(B, C, dim)`` -> ``(B, dim)`` for SimCLR / linear probe.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn


def _import_ispikformer():
    try:
        from models.local_ispikformer import iSpikformer
    except ImportError as e:
        raise ImportError(
            "iSpikformer requires SpikingJelly. Install with: pip install spikingjelly"
        ) from e
    return iSpikformer


class SeqSNNiSpikformerBackbone(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        len_sw: int,
        *,
        backbone: bool = True,
        dim: int = 512,
        d_ff: Optional[int] = None,
        depths: int = 2,
        num_steps: int = 4,
        heads: int = 8,
        common_thr: float = 1.0,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        encoder_type: str = "conv",
    ):
        super().__init__()
        _ = n_classes
        if not backbone:
            raise NotImplementedError(
                "iSpikformer backbone is only wired for SSL (backbone=True in SNN_HAR).",
            )
        iSpikformer = _import_ispikformer()
        self.core = iSpikformer(
            dim=dim,
            d_ff=d_ff,
            depths=depths,
            common_thr=common_thr,
            max_length=len_sw,
            num_steps=num_steps,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            input_size=n_channels,
            encoder_type=encoder_type,
        )
        self.out_dim = dim

    def forward(self, x_btc):
        _, h = self.core(x_btc)
        if h.dim() == 3:
            z = h.mean(dim=1)
        else:
            z = h
        return None, z
