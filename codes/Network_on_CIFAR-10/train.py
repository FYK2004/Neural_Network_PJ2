from __future__ import annotations

import argparse
import time
from pathlib import Path

from data.cifar10_loader import get_cifar10_loaders
from models.resnet_cifar import build_model, count_parameters
from utils.train_utils import (
    append_history,
    build_criterion,
    build_optimizer,
    build_scheduler,
    evaluate,
    format_epoch_log,
    get_device,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    parser.add_argument("--model", type=str, default="resnet10", choices=["resnet18", "resnet10"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_items", type=int, default=-1, help=">0: subset for debugging")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--scheduler", type=str, default="multistep",
                        choices=["multistep", "cosine", "cosine_warmup", "none"])
    parser.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="linear warmup length for cosine_warmup")
    parser.add_argument("--preset", type=str, default="",
                        choices=["", "cifar100"],
                        help="cifar100: 100 epoch + cosine warmup (recommended short run)")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "label_smooth"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=0, help="0: only save best/last")
    parser.add_argument("--no_progress", action="store_true", help="disable tqdm progress bar")
    parser.add_argument("--resume", action="store_true", help="resume from last.pth or best.pth")
    parser.add_argument("--no-history", action="store_true", help="do not write history.json")
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "cifar100":
        args.epochs = 100
        args.scheduler = "cosine_warmup"
        args.warmup_epochs = 5
        args.milestones = [50, 75]


def main() -> None:
    args = parse_args()
    apply_preset(args)
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_items=args.n_items,
    )

    model = build_model(args.model, activation=args.activation).to(device)
    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Parameters: {count_parameters(model):,}", flush=True)
    print(f"Activation: {args.activation}", flush=True)
    if args.preset:
        print(f"Preset: {args.preset} -> epochs={args.epochs}, scheduler={args.scheduler}", flush=True)

    if args.optimizer == "adamw" and args.lr == 0.1:
        args.lr = 1e-3
        print(f"AdamW default lr -> {args.lr}", flush=True)

    criterion = build_criterion(args.loss, args.label_smoothing)
    epoch_times: list[float] = []
    show_progress = not args.no_progress
    optimizer = build_optimizer(
        model, args.optimizer, args.lr, args.weight_decay, args.momentum
    )
    scheduler = build_scheduler(
        optimizer, args.scheduler, args.epochs, args.milestones, args.warmup_epochs
    )

    run_args = vars(args)
    history_path = output_dir / "history.json"
    best_acc = 0.0
    best_path = output_dir / "best.pth"
    last_path = output_dir / "last.pth"
    start_epoch = 1

    if args.resume:
        ckpt_path = last_path if last_path.exists() else best_path
        if ckpt_path.exists():
            done_epoch, best_acc = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, device
            )
            start_epoch = done_epoch + 1
            print(
                f"Resume from {ckpt_path.name} @ epoch {done_epoch}, "
                f"best={best_acc:.2f}%, continue {start_epoch}-{args.epochs}",
                flush=True,
            )
        else:
            print("No checkpoint found, training from scratch.", flush=True)

    if start_epoch > args.epochs:
        print(f"Already finished ({args.epochs} epochs).", flush=True)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=show_progress,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        eta_sec = sum(epoch_times) / len(epoch_times) * (args.epochs - epoch)
        print(
            format_epoch_log(epoch, args.epochs, train_loss, train_acc, test_loss, test_acc, elapsed)
            + f" | best={best_acc:.2f}% | ETA~{eta_sec / 3600:.1f}h",
            flush=True,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
        }
        append_history(history_path, record)

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(best_path, model, optimizer, epoch, best_acc, run_args, scheduler)
            print(f"  -> new best test acc: {best_acc:.2f}%", flush=True)

        save_checkpoint(last_path, model, optimizer, epoch, best_acc, run_args, scheduler)

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = output_dir / f"epoch_{epoch:03d}.pth"
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_acc, run_args, scheduler)
    print(f"Done. Best test acc: {best_acc:.2f}%", flush=True)
    print(f"Best weights: {best_path.resolve()}", flush=True)
    print(f"History: {history_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
