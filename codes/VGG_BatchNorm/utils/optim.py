from __future__ import annotations

from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
    _LRScheduler,
)


def build_optimizer(
    model,
    name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> Optimizer:
    name = name.lower()
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    epochs: int,
    milestones: list[int],
    warmup_epochs: int = 5,
) -> _LRScheduler | None:
    name = name.lower()
    if name == "multistep":
        valid = [m for m in milestones if m < epochs]
        if len(valid) < len(milestones):
            print(f"Warning: milestones trimmed to {valid} for {epochs} epochs", flush=True)
        return MultiStepLR(optimizer, milestones=valid or [epochs // 2], gamma=0.1)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "cosine_warmup":
        if warmup_epochs <= 0:
            return CosineAnnealingLR(optimizer, T_max=epochs)
        warmup_epochs = min(warmup_epochs, max(epochs - 1, 1))
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {name}")


def resolve_lr(optimizer: str, lr: float) -> float:
    if optimizer == "sgd" and lr <= 0.01:
        return 0.1
    if optimizer in ("adam", "adamw") and lr >= 0.05:
        return 1e-3
    return lr
