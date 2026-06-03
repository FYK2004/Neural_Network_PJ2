from __future__ import annotations

import torch
import torch.nn as nn

STAGE_PLANES = (64, 128, 256, 512)
STAGE_STRIDES = (1, 2, 2, 2)


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = _make_activation(activation)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        activation: str = "relu",
        blocks_per_stage: tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__()
        if len(blocks_per_stage) != 4:
            raise ValueError("blocks_per_stage must have 4 entries for 4 stages")

        self.blocks_per_stage = blocks_per_stage
        self.in_planes = 64
        self.stem_act = _make_activation(activation)

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        layers = []
        for planes, n_blocks, stride in zip(STAGE_PLANES, blocks_per_stage, STAGE_STRIDES):
            layers.append(self._make_layer(planes, n_blocks, stride, activation))
        self.stages = nn.ModuleList(layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self, planes: int, num_blocks: int, stride: int, activation: str
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, activation=activation))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_act(self.bn1(self.conv1(x)))
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# Backward-compatible alias
ResNet18CIFAR = ResNetCIFAR


def resnet18_cifar(num_classes: int = 10, activation: str = "relu") -> ResNetCIFAR:
    return ResNetCIFAR(num_classes=num_classes, activation=activation, blocks_per_stage=(2, 2, 2, 2))


def resnet10_cifar(num_classes: int = 10, activation: str = "relu") -> ResNetCIFAR:
    return ResNetCIFAR(num_classes=num_classes, activation=activation, blocks_per_stage=(1, 1, 1, 1))


MODEL_REGISTRY = {
    "resnet18": resnet18_cifar,
    "resnet10": resnet10_cifar,
}


def build_model(name: str, num_classes: int = 10, activation: str = "relu") -> ResNetCIFAR:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, activation=activation)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
