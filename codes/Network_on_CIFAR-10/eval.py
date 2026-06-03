from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.cifar10_loader import get_cifar10_loaders
from models.resnet_cifar import build_model, count_parameters
from utils.train_utils import build_criterion, evaluate, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ResNet on CIFAR-10")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_items", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    train_args = ckpt.get("args", {})
    model_name = train_args.get("model", "resnet10")
    activation = train_args.get("activation", "relu")
    loss_name = train_args.get("loss", "ce")
    label_smoothing = train_args.get("label_smoothing", 0.0)

    _, test_loader = get_cifar10_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_items=args.n_items,
    )

    model = build_model(model_name, activation=activation).to(device)
    model.load_state_dict(ckpt["model_state"])
    criterion = build_criterion(loss_name, label_smoothing)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    test_error = 100.0 - test_acc

    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Model: {model_name}")
    print(f"Epoch (saved): {ckpt.get('epoch', 'N/A')}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc:  {test_acc:.2f}%")
    print(f"Test error: {test_error:.2f}%")


if __name__ == "__main__":
    main()
