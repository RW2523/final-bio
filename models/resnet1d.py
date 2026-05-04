import torch
import torch.nn as nn

from models.spike import LIFSpike


def conv3x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SpikingBasicBlock1d(nn.Module):
    """
    ResNet-18 style 1D block: Conv-BN-LIF, Conv-BN, (+shortcut), LIF.
    LIFs use :mod:`snntorch` LIFs (unrolled over length T), matching SFCN-style training.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **snn_p):
        super().__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.act1 = LIFSpike(**snn_p)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.act2 = LIFSpike(**snn_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.act2(out)


class SResNet1D(nn.Module):
    """
    Spiking 1D ResNet-18: Conv-BN-LIF blocks, residual shortcuts, 512-d pooled features.
    LIFs are implemented with ``snntorch.Leaky``; only this ResNet-style backbone is supported (no ANN ResNet).
    """

    def __init__(
        self,
        n_channels,
        n_classes,
        len_sw,
        backbone=True,
        thresh=0.5,
        tau=0.5,
        gamma=1.0,
        dspike=False,
        soft_reset=True,
    ):
        super().__init__()
        snn_p = dict(thresh=thresh, tau=tau, gamma=gamma, dspike=dspike, soft_reset=soft_reset)
        self.snn_p = snn_p
        self.inplanes = 64
        self.backbone = backbone

        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.lif1 = LIFSpike(**snn_p)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1, **snn_p)
        self.layer2 = self._make_layer(128, 2, stride=2, **snn_p)
        self.layer3 = self._make_layer(256, 2, stride=2, **snn_p)
        self.layer4 = self._make_layer(512, 2, stride=2, **snn_p)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.out_dim = 512 * SpikingBasicBlock1d.expansion
        if backbone is False:
            self.logits = nn.Linear(self.out_dim, n_classes)

    def _make_layer(self, planes, blocks, stride, **snn_p):
        downsample = None
        if stride != 1 or self.inplanes != planes * SpikingBasicBlock1d.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * SpikingBasicBlock1d.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * SpikingBasicBlock1d.expansion),
            )
        layers = [SpikingBasicBlock1d(self.inplanes, planes, stride, downsample, **snn_p)]
        self.inplanes = planes * SpikingBasicBlock1d.expansion
        for _ in range(1, blocks):
            layers.append(SpikingBasicBlock1d(self.inplanes, planes, **snn_p))
        return nn.Sequential(*layers)

    def _features(self, x_btc: torch.Tensor) -> torch.Tensor:
        x = x_btc.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        x = self.maxpool(self.lif1(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return x

    def forward(self, x_in):
        x = self._features(x_in)
        if self.backbone:
            return None, x
        logits = self.logits(x)
        return logits, x
