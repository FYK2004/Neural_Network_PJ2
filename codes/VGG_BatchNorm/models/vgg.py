from __future__ import annotations

import numpy as np
from torch import nn

from utils.nn import init_weights_


def get_number_of_parameters(model: nn.Module) -> int:
    return sum(np.prod(p.shape).item() for p in model.parameters())


class VGG_A(nn.Module):
    def __init__(self, inp_ch: int = 3, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.classifier = self._build_classifier(num_classes, dropout)
        if init_weights:
            self._init_weights()

    @staticmethod
    def _build_classifier(num_classes: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(512, 512), nn.ReLU(True)]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.extend([nn.Linear(512, 512), nn.ReLU(True)])
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(512, num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(-1, 512))

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True),
    )


class VGG_A_BatchNorm(nn.Module):
    def __init__(
        self,
        inp_ch: int = 3,
        num_classes: int = 10,
        init_weights: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(inp_ch, 64), nn.MaxPool2d(2, 2),
            _conv_bn_relu(64, 128), nn.MaxPool2d(2, 2),
            _conv_bn_relu(128, 256), _conv_bn_relu(256, 256), nn.MaxPool2d(2, 2),
            _conv_bn_relu(256, 512), _conv_bn_relu(512, 512), nn.MaxPool2d(2, 2),
            _conv_bn_relu(512, 512), _conv_bn_relu(512, 512), nn.MaxPool2d(2, 2),
        )
        self.classifier = self._build_classifier(num_classes, dropout)
        if init_weights:
            self._init_weights()

    @staticmethod
    def _build_classifier(num_classes: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(True)]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.extend([nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(True)])
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(512, num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(-1, 512))

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)
