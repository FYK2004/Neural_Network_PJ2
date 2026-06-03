from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_criterion(loss_name: str, label_smoothing: float = 0.0) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if loss_name == "label_smooth":
        return nn.CrossEntropyLoss(label_smoothing=max(label_smoothing, 0.1))
    raise ValueError(f"Unsupported loss: {loss_name}")


def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> Optimizer:
    name = name.lower()
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
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
        warmup_epochs = min(max(warmup_epochs, 1), epochs - 1)
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int = 0,
    total_epochs: int = 0,
    show_progress: bool = True,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    desc = f"Epoch {epoch}/{total_epochs} [train]" if total_epochs else "Train"
    batch_iter = tqdm(loader, desc=desc, leave=False, disable=not show_progress)

    for images, targets in batch_iter:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(1) == targets).sum().item()
        total += batch_size

        if show_progress:
            batch_iter.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.1f}%",
            )

    return total_loss / total, 100.0 * correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_acc: float,
    args: dict,
    scheduler: _LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,
        "args": args,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler | None = None,
    device: torch.device | None = None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device or get_device(), weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    elif scheduler is not None:
        done = int(ckpt.get("epoch", 0))
        for _ in range(done):
            scheduler.step()
    return int(ckpt["epoch"]), float(ckpt.get("best_acc", 0.0))


def append_history(history_path: Path, record: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    records.append(record)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def format_epoch_log(epoch: int, epochs: int, train_loss: float, train_acc: float, test_loss: float, test_acc: float, elapsed: float) -> str:
    return (
        f"Epoch [{epoch}/{epochs}] "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% "
        f"time={elapsed:.1f}s"
    )
