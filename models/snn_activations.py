"""
Spiking activations and small recurrent cores (pure torch).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _heaviside(x: torch.Tensor, thresh: float) -> torch.Tensor:
    return (x >= thresh).to(x.dtype)


def _d_surrogate(x: torch.Tensor, thresh: float, alpha: float) -> torch.Tensor:
    d = x - thresh
    return 1.0 / (1.0 + (alpha * d) ** 2)


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thresh, alpha):
        ctx.save_for_backward(x)
        ctx.thresh = float(thresh)
        ctx.alpha = float(alpha)
        return _heaviside(x, ctx.thresh)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        g = _d_surrogate(x, ctx.thresh, ctx.alpha) * grad_output
        return g, None, None


def spike(x: torch.Tensor, thresh, alpha: float = 2.0) -> torch.Tensor:
    th = float(thresh.item()) if torch.is_tensor(thresh) else float(thresh)
    return _SpikeFn.apply(x, th, float(alpha))


def snn_kwd(tau: float, thresh: float) -> dict:
    return dict(beta=float(tau), threshold=float(thresh), soft_reset=True)


class LIFSpike1D(nn.Module):
    """LIF for (B, C, T)."""

    def __init__(self, beta=0.5, threshold=0.5, soft_reset: bool = True, surrogate_alpha: float = 2.0):
        super().__init__()
        self.register_buffer("beta", torch.as_tensor(beta, dtype=torch.float32))
        self.register_buffer("threshold", torch.as_tensor(threshold, dtype=torch.float32))
        self.soft_reset = soft_reset
        self.surrogate_alpha = float(surrogate_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"LIFSpike1D expects (B, C, T); got {tuple(x.shape)}")
        b, c, t_len = x.shape
        mem = x.new_zeros(b, c, device=x.device, dtype=x.dtype)
        out = x.new_empty(b, c, t_len, device=x.device, dtype=x.dtype)
        beta = self.beta.to(dtype=x.dtype, device=x.device)
        thr = self.threshold.to(dtype=x.dtype, device=x.device)
        th = float(thr.item())
        for t in range(t_len):
            mem = beta * mem + x[:, :, t]
            spk = spike(mem, th, self.surrogate_alpha)
            if self.soft_reset:
                mem = mem - spk * thr
            else:
                mem = mem * (1.0 - spk)
            out[:, :, t] = spk
        return out


class LIFVec1Step(nn.Module):
    """Single-step spike nonlinearity for MLP vectors."""

    def __init__(self, threshold: float = 0.5, surrogate_alpha: float = 2.0):
        super().__init__()
        self.register_buffer("threshold", torch.as_tensor(threshold, dtype=torch.float32))
        self.surrogate_alpha = float(surrogate_alpha)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        th = self.threshold.to(dtype=v.dtype, device=v.device)
        return spike(v, float(th.item()), self.surrogate_alpha)


class LIFRNN2Layer(nn.Module):
    """2-layer unidirectional recurrent core with LIF activations."""

    def __init__(self, input_size, hidden_size, num_layers: int = 2, tau: float = 0.75, thresh: float = 0.5):
        assert num_layers == 2
        super().__init__()
        tr = float(thresh)
        self.c1_ih = nn.Linear(input_size, hidden_size)
        self.c1_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.c2_ih = nn.Linear(hidden_size, hidden_size)
        self.c2_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lif1 = LIFVec1Step(threshold=tr)
        self.lif2 = LIFVec1Step(threshold=tr)
        self._h = hidden_size
        _ = tau

    def forward(self, x, hx=None):
        T, b, _ = x.shape
        h1 = x.new_zeros(b, self._h, device=x.device, dtype=x.dtype)
        h2 = x.new_zeros(b, self._h, device=x.device, dtype=x.dtype)
        out = x.new_empty(T, b, self._h, device=x.device, dtype=x.dtype)
        for t in range(T):
            xt = x[t]
            pre1 = self.c1_ih(xt) + self.c1_hh(h1)
            h1 = self.lif1(pre1)
            pre2 = self.c2_ih(h1) + self.c2_hh(h2)
            h2 = self.lif2(pre2)
            out[t] = h2
        return out, (h2.unsqueeze(0), h2.unsqueeze(0))

    def flatten_parameters(self):
        pass
