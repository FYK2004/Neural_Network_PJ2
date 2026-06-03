from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm
from utils.optim import build_optimizer, build_scheduler

DEFAULT_LRS = [0.1, 0.05, 0.01, 0.005]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGG loss landscape (multi-LR)")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs/loss_landscape")
    p.add_argument("--epochs", type=int, default=20, help="epochs per LR run")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_items", type=int, default=-1, help=">0: subset for quick tests")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    p.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine_warmup", "cosine", "multistep", "none"],
        help="default cosine: no warmup (Step 2 landscape)",
    )
    p.add_argument("--milestones", type=int, nargs="+", default=[50, 75])
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="warmup epochs for cosine_warmup only; 0 with --scheduler cosine",
    )
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=DEFAULT_LRS,
        help="learning rates as step-size proxies",
    )
    p.add_argument("--plot-only", action="store_true", help="skip training, plot from saved npy")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip LR runs whose loss_lr_*.npy already exist (resume)",
    )
    p.add_argument(
        "--record_mode",
        type=str,
        default="batch",
        choices=["batch", "epoch"],
        help="batch: one loss per training step (default, each batch); epoch: mean loss per epoch",
    )
    p.add_argument(
        "--record_every",
        type=int,
        default=1,
        help="record_mode=batch: record once every N batches (1 = each batch/step)",
    )
    p.add_argument("--record-grad", action="store_true", help="also save last-Linear grad L2 per step")
    p.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="RandomCrop+Flip and CIFAR-10 mean/std on training set (default: on)",
    )
    return p.parse_args()


def set_random_seeds(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def last_linear_grad_norm(model: nn.Module) -> float | None:
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None or last_linear.weight.grad is None:
        return None
    return float(last_linear.weight.grad.detach().norm().cpu().item())


def train_and_record(
    model: nn.Module,
    train_loader,
    device: torch.device,
    lr: float,
    epochs: int,
    optimizer_name: str,
    weight_decay: float,
    momentum: float,
    scheduler_name: str,
    milestones: list[int],
    warmup_epochs: int,
    label_smoothing: float,
    record_grad: bool = False,
    record_mode: str = "batch",
    record_every: int = 1,
) -> tuple[list[float], list[float] | None]:
    if record_every < 1:
        raise ValueError("record_every must be >= 1")
    record_mode = record_mode.lower()

    model.to(device)
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay, momentum)
    scheduler: _LRScheduler | None = build_scheduler(
        optimizer, scheduler_name, epochs, milestones, warmup_epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    losses: list[float] = []
    grad_norms: list[float] = [] if record_grad else None
    batch_idx = 0

    for _epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if record_grad:
                g = last_linear_grad_norm(model)
                if g is not None and record_mode == "batch" and batch_idx % record_every == record_every - 1:
                    grad_norms.append(g)

            optimizer.step()
            batch_idx += 1
            loss_val = float(loss.item())
            if record_mode == "batch" and batch_idx % record_every == 0:
                losses.append(loss_val)
            elif record_mode == "epoch":
                epoch_losses.append(loss_val)

        if record_mode == "epoch":
            losses.append(float(np.mean(epoch_losses)))

        if scheduler is not None:
            scheduler.step()

    return losses, grad_norms


def build_min_max_curves(runs: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray, int]:
    if not runs:
        raise ValueError("No runs to aggregate")
    n_steps = min(len(v) for v in runs.values())
    stacked = np.array([runs[k][:n_steps] for k in sorted(runs.keys(), key=float)], dtype=np.float64)
    return stacked.min(axis=0), stacked.max(axis=0), n_steps


def run_all_lrs(
    model_name: str,
    model_factory,
    train_loader,
    device: torch.device,
    learning_rates: list[float],
    epochs: int,
    out_dir: Path,
    train_config: dict,
    record_grad: bool,
    seed: int,
    skip_existing: bool = False,
    record_mode: str = "batch",
    record_every: int = 1,
) -> dict[str, list[float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_losses: dict[str, list[float]] = {}

    for lr in learning_rates:
        key = f"{lr:.0e}"
        loss_path = out_dir / f"loss_lr_{key}.npy"
        if skip_existing and loss_path.exists():
            print(f"\n[{model_name}] skip lr={key} (already saved)", flush=True)
            all_losses[key] = np.load(loss_path).tolist()
            continue
        print(f"\n[{model_name}] training lr={key}", flush=True)
        set_random_seeds(seed, device)
        model = model_factory()
        losses, grads = train_and_record(
            model,
            train_loader,
            device,
            lr,
            epochs,
            optimizer_name=train_config["optimizer"],
            weight_decay=train_config["weight_decay"],
            momentum=train_config["momentum"],
            scheduler_name=train_config["scheduler"],
            milestones=train_config["milestones"],
            warmup_epochs=train_config["warmup_epochs"],
            label_smoothing=train_config["label_smoothing"],
            record_grad=record_grad,
            record_mode=record_mode,
            record_every=record_every,
        )
        all_losses[key] = losses
        np.save(out_dir / f"loss_lr_{key}.npy", np.array(losses, dtype=np.float32))
        if grads is not None:
            np.save(out_dir / f"grad_lr_{key}.npy", np.array(grads, dtype=np.float32))

    with open(out_dir / "losses_all.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in all_losses.items()}, f)

    meta = {
        "model": model_name,
        "epochs": epochs,
        "learning_rates": learning_rates,
        "record_mode": record_mode,
        "record_every": record_every,
        "train_config": train_config,
        "n_steps": {k: len(v) for k, v in all_losses.items()},
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return all_losses


def load_saved_losses(model_dir: Path, learning_rates: list[float]) -> dict[str, list[float]]:
    runs: dict[str, list[float]] = {}
    for lr in learning_rates:
        key = f"{lr:.0e}"
        path = model_dir / f"loss_lr_{key}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run training first.")
        runs[key] = np.load(path).tolist()
    return runs


def build_step_axis(n_points: int, record_mode: str, record_every: int, batches_per_epoch: int) -> tuple[np.ndarray, str]:
    if record_mode == "epoch":
        steps = (np.arange(n_points) + 1) * batches_per_epoch
        return steps, "Training step"
    steps = (np.arange(n_points) + 1) * record_every
    xlabel = "Training step" if record_every == 1 else f"Training step (every {record_every} batches)"
    return steps, xlabel


def plot_loss_landscape(
    vgg_min: np.ndarray,
    vgg_max: np.ndarray,
    bn_min: np.ndarray,
    bn_max: np.ndarray,
    save_path: Path,
    title: str = "Loss landscape (max/min over learning rates)",
    record_mode: str = "batch",
    record_every: int = 1,
    batches_per_epoch: int = 391,
) -> None:
    steps, xlabel = build_step_axis(len(vgg_min), record_mode, record_every, batches_per_epoch)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, mn, mx, subtitle in zip(
        axes,
        [vgg_min, bn_min],
        [vgg_max, bn_max],
        ["VGG-A (no BN)", "VGG-A + BatchNorm"],
    ):
        ax.fill_between(steps, mn, mx, alpha=0.35, color="C0", label="loss band")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Train loss")
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_combined_overlay(
    vgg_min: np.ndarray,
    vgg_max: np.ndarray,
    bn_min: np.ndarray,
    bn_max: np.ndarray,
    save_path: Path,
    record_mode: str = "batch",
    record_every: int = 1,
    batches_per_epoch: int = 391,
) -> None:
    n = min(len(vgg_min), len(bn_min))
    steps, xlabel = build_step_axis(n, record_mode, record_every, batches_per_epoch)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(steps, vgg_min[:n], vgg_max[:n], alpha=0.3, color="C0", label="VGG-A band")
    ax.fill_between(steps, bn_min[:n], bn_max[:n], alpha=0.3, color="C2", label="VGG-A+BN band")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Train loss")
    ax.set_title("Loss landscape: VGG-A vs VGG-A+BN (same step index)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    lrs = args.learning_rates

    vgg_dir = output_dir / "vgg_a"
    bn_dir = output_dir / "vgg_a_bn"
    fig_dir = output_dir / "figures"
    batches_per_epoch = 391

    if not args.plot_only:
        print(f"Device: {device}", flush=True)
        print(f"Learning rates: {lrs}", flush=True)
        print(f"Epochs per run: {args.epochs}", flush=True)
        print(f"Record mode: {args.record_mode}", flush=True)
        if args.record_mode == "batch":
            print(f"Record every: {args.record_every} batch(es)", flush=True)
        print(f"Augment: {args.augment}", flush=True)
        print(
            f"Train config: {args.optimizer} wd={args.weight_decay} "
            f"momentum={args.momentum} scheduler={args.scheduler} "
            f"dropout={args.dropout} label_smooth={args.label_smoothing}",
            flush=True,
        )

        train_config = {
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "scheduler": args.scheduler,
            "milestones": args.milestones,
            "warmup_epochs": args.warmup_epochs,
            "label_smoothing": args.label_smoothing,
            "dropout": args.dropout,
            "augment": args.augment,
        }

        train_loader = get_cifar_loader(
            root=args.data_root,
            batch_size=args.batch_size,
            train=True,
            num_workers=args.num_workers,
            n_items=args.n_items,
            augment=args.augment,
        )
        batches_per_epoch = len(train_loader)

        vgg_losses = run_all_lrs(
            "VGG_A",
            lambda: VGG_A(dropout=args.dropout),
            train_loader,
            device,
            lrs,
            args.epochs,
            vgg_dir,
            train_config,
            args.record_grad,
            args.seed,
            skip_existing=args.skip_existing,
            record_mode=args.record_mode,
            record_every=args.record_every,
        )
        bn_losses = run_all_lrs(
            "VGG_A_BN",
            lambda: VGG_A_BatchNorm(dropout=args.dropout),
            train_loader,
            device,
            lrs,
            args.epochs,
            bn_dir,
            train_config,
            args.record_grad,
            args.seed,
            skip_existing=args.skip_existing,
            record_mode=args.record_mode,
            record_every=args.record_every,
        )
    else:
        print("Plot-only mode: loading saved losses.", flush=True)
        batches_per_epoch = 391
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                saved = json.load(f)
            batches_per_epoch = int(saved.get("batches_per_epoch", batches_per_epoch))
            if "record_mode" in saved:
                args.record_mode = saved["record_mode"]
            if "learning_rates" in saved:
                lrs = saved["learning_rates"]
        vgg_losses = load_saved_losses(vgg_dir, lrs)
        bn_losses = load_saved_losses(bn_dir, lrs)

    vgg_min, vgg_max, n_vgg = build_min_max_curves(vgg_losses)
    bn_min, bn_max, n_bn = build_min_max_curves(bn_losses)
    n_common = min(n_vgg, n_bn)
    vgg_min, vgg_max = vgg_min[:n_common], vgg_max[:n_common]
    bn_min, bn_max = bn_min[:n_common], bn_max[:n_common]

    np.savez(
        output_dir / "curves.npz",
        vgg_min=vgg_min,
        vgg_max=vgg_max,
        bn_min=bn_min,
        bn_max=bn_max,
        learning_rates=np.array(lrs),
    )

    plot_loss_landscape(
        vgg_min,
        vgg_max,
        bn_min,
        bn_max,
        fig_dir / "loss_landscape_panels.png",
        record_mode=args.record_mode,
        record_every=args.record_every,
        batches_per_epoch=batches_per_epoch,
    )
    plot_combined_overlay(
        vgg_min,
        vgg_max,
        bn_min,
        bn_max,
        fig_dir / "loss_landscape_combined.png",
        record_mode=args.record_mode,
        record_every=args.record_every,
        batches_per_epoch=batches_per_epoch,
    )

    if args.record_mode == "epoch":
        common_training_batches = int(n_common * batches_per_epoch)
    else:
        common_training_batches = int(n_common * args.record_every)

    summary = {
        "learning_rates": lrs,
        "epochs_per_run": args.epochs,
        "record_mode": args.record_mode,
        "record_every": args.record_every,
        "batches_per_epoch": int(batches_per_epoch),
        "augment": args.augment,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "common_steps": int(n_common),
        "common_training_batches": common_training_batches,
        "vgg_a_band_width_mean": float(np.mean(vgg_max - vgg_min)),
        "vgg_a_bn_band_width_mean": float(np.mean(bn_max - bn_min)),
        "figures": [
            str(fig_dir / "loss_landscape_panels.png"),
            str(fig_dir / "loss_landscape_combined.png"),
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved curves: {output_dir / 'curves.npz'}", flush=True)
    print(f"Saved figure: {fig_dir / 'loss_landscape_panels.png'}", flush=True)
    print(f"Saved figure: {fig_dir / 'loss_landscape_combined.png'}", flush=True)
    print(
        f"Mean band width — VGG-A: {summary['vgg_a_band_width_mean']:.4f}, "
        f"VGG-A+BN: {summary['vgg_a_bn_band_width_mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
