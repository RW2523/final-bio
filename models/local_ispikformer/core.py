"""Vendored iSpikformer core (SeqSNN layout) — uses SpikingJelly only."""
from __future__ import annotations

from typing import Optional

from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from .encoder_sj import ConvEncoder, DeltaEncoder, RepeatEncoder
from .spike_attention_block import Block

backend = "torch"
DEFAULT_TAU = 2.0
DEFAULT_DETACH_RESET = True

_SPIKE_ENCODERS = {
    "repeat": RepeatEncoder,
    "conv": ConvEncoder,
    "delta": DeltaEncoder,
}


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, tau: float = DEFAULT_TAU, detach_reset: bool = DEFAULT_DETACH_RESET):
        super().__init__()
        self.d_model = d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x):
        T, B, _, C = x.shape
        x = x.permute(0, 1, 3, 2).flatten(0, 1)
        x = self.value_embedding(x)
        x = self.bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.reshape(T, B, C, self.d_model)
        return self.lif(x)


class iSpikformer(nn.Module):
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        dim: int,
        d_ff: Optional[int] = None,
        depths: int = 2,
        common_thr: float = 1.0,
        max_length: int = 100,
        num_steps: int = 4,
        heads: int = 8,
        tau: float = DEFAULT_TAU,
        detach_reset: bool = DEFAULT_DETACH_RESET,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        input_size: Optional[int] = None,
        encoder_type: Optional[str] = "conv",
    ):
        super().__init__()
        _ = input_size
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths

        enc_key = (encoder_type or "conv").lower()
        if enc_key not in _SPIKE_ENCODERS:
            raise ValueError(f"Unknown encoder_type={encoder_type!r}; use repeat|conv|delta")
        self.encoder = _SPIKE_ENCODERS[enc_key](num_steps, tau=tau, detach_reset=detach_reset)

        self.emb = DataEmbedding_inverted(max_length, dim, tau=tau, detach_reset=detach_reset)
        self.blocks = nn.ModuleList(
            [
                Block(
                    length=max_length,
                    tau=tau,
                    common_thr=common_thr,
                    dim=dim,
                    d_ff=self.d_ff,
                    heads=heads,
                    detach_reset=detach_reset,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                )
                for _ in range(depths)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        functional.reset_net(self.encoder)
        functional.reset_net(self.emb)
        functional.reset_net(self.blocks)
        x = self.encoder(x)
        x = x.transpose(2, 3)
        x = self.emb(x)
        for blk in self.blocks:
            x = blk(x)
        out = x[-1, :, :, :]
        return out, out

    @property
    def output_size(self):
        return self.dim

    @property
    def hidden_size(self):
        return self.dim
