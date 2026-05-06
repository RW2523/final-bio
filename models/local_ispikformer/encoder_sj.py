"""Vendored from SeqSNN (Microsoft); SpikingJelly encoders only — no SeqSNN package import."""
import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron

backend = "torch"
DEFAULT_TAU = 2.0
DEFAULT_DETACH_RESET = True


class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int, tau: float = DEFAULT_TAU, detach_reset: bool = DEFAULT_DETACH_RESET):
        super().__init__()
        self.out_size = output_size
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )
        inputs = inputs.permute(0, 1, 3, 2)
        return self.lif(inputs)


class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int, tau: float = DEFAULT_TAU, detach_reset: bool = DEFAULT_DETACH_RESET):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)
        enc = self.enc(delta)
        enc = enc.permute(3, 0, 1, 2)
        return self.lif(enc)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        kernel_size: int = 3,
        tau: float = DEFAULT_TAU,
        detach_reset: bool = DEFAULT_DETACH_RESET,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)
        enc = self.encoder(inputs)
        enc = enc.permute(1, 0, 2, 3)
        return self.lif(enc)
