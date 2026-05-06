import torch
import torch.nn as nn
from models.backbones import FCN, DeepConvLSTM, LSTM, AE, CNN_AE
from models.snn_activations import LIFRNN2Layer, LIFSpike1D as PureLIFSpike1D, LIFVec1Step, snn_kwd


class AvgMeter:

    def __init__(self):
        self.value = 0
        self.number = 0

    def add(self, v, n):
        self.value += v
        self.number += n

    def avg(self):
        return self.value / self.number


class LIFSpike(nn.Module):
    """Legacy-compatible wrapper over pure-torch LIF implementation."""

    def __init__(self, thresh=0.5, tau=0.5, gamma=1.0, dspike=False, soft_reset=True):
        super(LIFSpike, self).__init__()
        self.lif = PureLIFSpike1D(beta=float(tau), threshold=float(thresh), soft_reset=bool(soft_reset))
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.dspike = dspike
        self.soft_reset = soft_reset

    def forward(self, x):
        if x.dim() == 3:
            return self.lif(x)
        if x.dim() == 4:
            b, c, h, w = x.shape
            y = self.lif(x.reshape(b, c, h * w))
            return y.view(b, c, h, w)
        raise ValueError("LIFSpike expects (B,C,T) or (B,C,H,W); got %s" % (tuple(x.shape),))


class SensorChannelGate(nn.Module):
    """
    Lightweight squeeze-excitation gate for sensor channels on (B, T, C) inputs.
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, int(n_channels) // max(1, int(reduction)))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x_btc: torch.Tensor):
        pooled = self.pool(x_btc.transpose(1, 2)).squeeze(-1)
        gate = self.mlp(pooled).unsqueeze(1)
        return x_btc * gate


class SFCN(FCN):

    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, len_sw=128, **kwargs):
        super(SFCN, self).__init__(n_channels, n_classes, out_channels, backbone, len_sw=len_sw)
        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         LIFSpike(**kwargs),
                                         nn.AvgPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         LIFSpike(**kwargs),
                                         nn.AvgPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         LIFSpike(**kwargs),
                                         nn.AvgPool1d(kernel_size=2, stride=2, padding=1))


class SDCL(DeepConvLSTM):

    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True, **snn_p):
        super(SDCL, self).__init__(n_channels, n_classes, conv_kernels, kernel_size, LSTM_units, backbone)
        self.act1 = LIFSpike(**snn_p)
        self.act2 = LIFSpike(**snn_p)
        self.act3 = LIFSpike(**snn_p)
        self.act4 = LIFSpike(**snn_p)

        self.bn1 = nn.BatchNorm2d(conv_kernels)
        self.bn2 = nn.BatchNorm2d(conv_kernels)
        self.bn3 = nn.BatchNorm2d(conv_kernels)
        self.bn4 = nn.BatchNorm2d(conv_kernels)

        self.dropout = nn.Dropout(0.0)
        del self.lstm
        self.rnn = LIFRNN2Layer(
            n_channels * conv_kernels,
            LSTM_units,
            num_layers=2,
            tau=float(snn_p.get("tau", 0.75)),
            thresh=float(snn_p.get("thresh", 0.5)),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, _ = self.rnn(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x


class SLSTM(LSTM):
    """LSTM path replaced with LIF recurrent core."""

    def __init__(self, n_channels, n_classes, LSTM_units=128, backbone=True, tau=0.75, thresh=0.5, **k):
        super().__init__(n_channels, n_classes, LSTM_units, backbone)
        del self.lstm
        self.rnn = LIFRNN2Layer(
            n_channels, LSTM_units, num_layers=2, tau=float(tau), thresh=float(thresh)
        )
        _ = k

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.rnn(x)
        x = x[-1, :, :]
        if self.backbone:
            return None, x
        return self.classifier(x), x


class SNN_AE(AE):
    def __init__(self, n_channels, len_sw, n_classes, outdim=128, backbone=True, tau=0.75, thresh=0.5, **k):
        super().__init__(n_channels, len_sw, n_classes, outdim, backbone)
        p = {**snn_kwd(tau, thresh), "soft_reset": True}
        self.lif1d = PureLIFSpike1D(**p)
        self.lifv = LIFVec1Step(threshold=float(thresh), surrogate_alpha=2.0)
        _ = k

    def forward(self, x):
        x_e1 = self.e1(x)
        x_e1 = x_e1.permute(0, 2, 1)
        x_e1 = self.lif1d(x_e1).permute(0, 2, 1)
        x_e1 = x_e1.reshape(x_e1.shape[0], -1)
        x_e2 = self.lifv(self.e2(x_e1))
        x_encoded = self.lifv(self.e3(x_e2))
        x_d1 = self.lifv(self.d1(x_encoded))
        x_d2 = self.lifv(self.d2(x_d1))
        x_d2 = x_d2.reshape(x_d2.shape[0], self.len_sw, 8)
        x_decoded = self.d3(x_d2)
        if self.backbone:
            return x_decoded, x_encoded
        return self.classifier(x_encoded), x_decoded


class SNN_CNN_AE(CNN_AE):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, tau=0.75, thresh=0.5, **k):
        _ = k
        super().__init__(n_channels, n_classes, out_channels, backbone=backbone)
        d = dict(beta=float(tau), threshold=float(thresh), soft_reset=True)
        self.e_conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            PureLIFSpike1D(**d),
        )
        self.pool1 = nn.MaxPool1d(2, 2, 1, return_indices=True)
        self.e_conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            PureLIFSpike1D(**d),
        )
        self.pool2 = nn.MaxPool1d(2, 2, 1, return_indices=True)
        self.e_conv3 = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(out_channels),
            PureLIFSpike1D(**d),
        )
        self.pool3 = nn.MaxPool1d(2, 2, 1, return_indices=True)
        self.d_conv1 = nn.Sequential(
            nn.ConvTranspose1d(out_channels, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            PureLIFSpike1D(**d),
        )
        self.d_conv2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            PureLIFSpike1D(**d),
        )
        self.d_conv3 = nn.Sequential(
            nn.ConvTranspose1d(32, n_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(n_channels),
            PureLIFSpike1D(**d),
        )


class SNN_Transformer(nn.Module):
    """Spiking sequence transformer: **iSpikformer** (SpikingJelly LIF), vendored in-repo.

    For an ANN path with softmax attention on IMU windows, use ``--backbone Transformer``
    (:class:`models.backbones.Transformer` / :class:`models.attention.Seq_Transformer`).

    ``tau`` / ``thresh`` are accepted for API parity with other ``SNN_*`` modules (same as
    ``--backbone iSpikformer``); the underlying iSpikformer core uses its own defaults.
    """

    def __init__(
        self,
        n_channels,
        len_sw,
        n_classes,
        dim=512,
        depth=2,
        heads=8,
        mlp_dim=64,
        dropout=0.1,
        backbone=True,
        tau=0.75,
        thresh=0.5,
        num_steps: int = 4,
        encoder_type: str = "conv",
        d_ff=None,
        common_thr: float | None = None,
        detach_reset: bool = True,
        channel_gate: str = "none",
        gate_reduction: int = 4,
        **k,
    ):
        super().__init__()
        _ = k
        _ = mlp_dim
        _ = dropout
        from models.seqsnn_ispikformer import SeqSNNiSpikformerBackbone

        if not backbone:
            raise NotImplementedError(
                "SNN_Transformer (iSpikformer) only supports backbone=True in SNN_HAR."
            )
        self.backbone = backbone
        gate_mode = str(channel_gate).lower()
        if gate_mode == "se":
            self.channel_gate = SensorChannelGate(n_channels=n_channels, reduction=gate_reduction)
        elif gate_mode == "none":
            self.channel_gate = None
        else:
            raise ValueError(f"Unknown SNN transformer channel gate {channel_gate!r}; use none|se")
        self._core = SeqSNNiSpikformerBackbone(
            n_channels=n_channels,
            n_classes=n_classes,
            len_sw=int(len_sw),
            backbone=True,
            dim=int(dim),
            d_ff=d_ff,
            depths=int(depth),
            num_steps=int(num_steps),
            heads=int(heads),
            common_thr=float(common_thr if common_thr is not None else thresh),
            tau=float(tau),
            detach_reset=bool(detach_reset),
            encoder_type=str(encoder_type),
        )
        self.out_dim = self._core.out_dim

    def forward(self, x_btc: torch.Tensor):
        if self.channel_gate is not None:
            x_btc = self.channel_gate(x_btc)
        return self._core(x_btc)

