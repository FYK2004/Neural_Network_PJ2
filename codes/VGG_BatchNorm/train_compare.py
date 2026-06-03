from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from utils.grad_plots import (
    plot_gradient_predictiveness,
    plot_gradient_predictiveness_panels,
    plot_max_grad_diff,
    plot_max_grad_diff_panels,
)
from utils.grad_probe import (
    DEFAULT_ALPHAS,
    DEFAULT_MAX_DIST,
    DEFAULT_NUM_STEPS,
    PROBE_EPOCHS,
    run_epoch_probe,
)
from utils.probe_select import select_report_epochs
from utils.optim import build_optimizer, build_scheduler, resolve_lr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VGG-A vs VGG-A+BN (Step 1)")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    p.add_argument(
        "--scheduler",
        type=str,
        default="cosine_warmup",
        choices=["cosine_warmup", "cosine", "multistep", "none"],
    )
    p.add_argument("--milestones", type=int, nargs="+", default=[50, 75])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_items", type=int, default=-1)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--model", type=str, default="both", choices=["both", "vgg", "bn"])
    p.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="disable multi-epoch gradient probes",
    )
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    p.add_argument("--max-dist", type=float, default=DEFAULT_MAX_DIST)
    p.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    p.add_argument(
        "--probe-mode",
        type=str,
        default="train",
        choices=["train", "eval"],
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, 100.0 * correct / total


def train_one(
    name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    scheduler_name: str,
    milestones: list[int],
    warmup_epochs: int,
    out_dir: Path,
    label_smoothing: float = 0.0,
    probe_epochs: list[int] | None = None,
    probe_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
    alphas: list[float] | None = None,
    max_dist: float = DEFAULT_MAX_DIST,
    num_steps: int = DEFAULT_NUM_STEPS,
    probe_mode: str = "train",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    alphas = list(alphas or DEFAULT_ALPHAS)
    probe_epochs = sorted(set(probe_epochs or []))

    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay, momentum)
    scheduler: _LRScheduler | None = build_scheduler(
        optimizer, scheduler_name, epochs, milestones, warmup_epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history: list[dict] = []
    best_acc = 0.0
    best_path = out_dir / "best.pth"
    probe_by_epoch: dict[int, dict] = {}
    train_config = {
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "scheduler": scheduler_name,
        "milestones": milestones,
        "warmup_epochs": warmup_epochs,
        "epochs": epochs,
        "label_smoothing": label_smoothing,
        "probe_epochs": probe_epochs,
        "report_epochs": "auto",
        "probe_mode": probe_mode,
    }

    print(
        f"\n=== {name} | params={get_number_of_parameters(model):,} | "
        f"{optimizer_name} lr={lr} sched={scheduler_name} ===",
        flush=True,
    )
    if probe_epochs:
        print(f"  Probes at epochs {probe_epochs} (mode={probe_mode})", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in tqdm(train_loader, desc=f"{name} ep{epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        if epoch in probe_epochs and probe_batch is not None:
            px, py = probe_batch
            px, py = px.to(device), py.to(device)
            metrics = run_epoch_probe(
                model, px, py, criterion, epoch,
                alphas=alphas, max_dist=max_dist, num_steps=num_steps, probe_mode=probe_mode,
            )
            probe_by_epoch[epoch] = metrics
            with open(out_dir / f"epoch{epoch}_probe.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            taylor_01 = (
                metrics["taylor_rel_errors"][alphas.index(0.1)]
                if 0.1 in alphas
                else float("nan")
            )
            print(
                f"  [{name}] epoch {epoch}: max||Δg||={metrics['max_grad_diff']:.4f}, "
                f"taylor_err@0.1={taylor_01:.4f}",
                flush=True,
            )
            model.train()

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        if scheduler is not None:
            scheduler.step()

        train_acc = 100.0 * train_correct / train_total
        train_loss /= train_total
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "time_sec": elapsed,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_acc": best_acc,
                    "name": name,
                    "train_config": train_config,
                },
                best_path,
            )

        print(
            f"Epoch [{epoch}/{epochs}] lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_acc={val_acc:.2f}% | best={best_acc:.2f}% | {elapsed:.1f}s",
            flush=True,
        )

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    return {
        "name": name,
        "best_val_acc": best_acc,
        "params": get_number_of_parameters(model),
        "history": history,
        "train_config": train_config,
        "probe_by_epoch": {str(k): v for k, v in probe_by_epoch.items()},
    }


def plot_comparison(results: list[dict], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for res in results:
        h = res["history"]
        epochs = [r["epoch"] for r in h]
        axes[0].plot(epochs, [r["train_loss"] for r in h], "--", alpha=0.5, label=f"{res['name']} train")
        axes[0].plot(epochs, [r["val_loss"] for r in h], label=f"{res['name']} val")
        axes[1].plot(epochs, [r["train_acc"] for r in h], "--", alpha=0.5, label=f"{res['name']} train")
        axes[1].plot(epochs, [r["val_acc"] for r in h], label=f"{res['name']} val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "vgg_vs_bn_training.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}", flush=True)


def plot_report_figures(
    vgg_by: dict[int, dict],
    bn_by: dict[int, dict],
    fig_dir: Path,
    gp_ep: int,
    md_ep: int,
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ep in (gp_ep, md_ep):
        if ep not in vgg_by or ep not in bn_by:
            raise ValueError(
                f"Missing probe at epoch {ep}. Train with probes at {PROBE_EPOCHS}."
            )

    vgg_gp, bn_gp = vgg_by[gp_ep], bn_by[gp_ep]
    vgg_md, bn_md = vgg_by[md_ep], bn_by[md_ep]
    alphas = vgg_gp["alphas"]
    dists = np.array(vgg_md["distances"], dtype=np.float64)
    vgg_curve = np.array(vgg_md["diff_curve"])
    bn_curve = np.array(bn_md["diff_curve"])

    vgg_taylor = vgg_gp["taylor_rel_errors"]
    bn_taylor = bn_gp["taylor_rel_errors"]
    spec = [
        (plot_gradient_predictiveness, (alphas, vgg_taylor, bn_taylor, gp_ep),
         fig_dir / "gradient_predictiveness.png"),
        (plot_gradient_predictiveness, (alphas, vgg_taylor, bn_taylor, gp_ep),
         fig_dir / f"gradient_predictiveness_epoch{gp_ep}.png"),
        (plot_gradient_predictiveness_panels, (alphas, vgg_taylor, bn_taylor, gp_ep),
         fig_dir / "gradient_predictiveness_panels.png"),
        (plot_gradient_predictiveness_panels, (alphas, vgg_taylor, bn_taylor, gp_ep),
         fig_dir / f"gradient_predictiveness_epoch{gp_ep}_panels.png"),
        (plot_max_grad_diff,
         (dists, vgg_curve, bn_curve, vgg_md["max_grad_diff"], bn_md["max_grad_diff"], md_ep),
         fig_dir / "max_grad_diff.png"),
        (plot_max_grad_diff,
         (dists, vgg_curve, bn_curve, vgg_md["max_grad_diff"], bn_md["max_grad_diff"], md_ep),
         fig_dir / f"max_grad_diff_epoch{md_ep}.png"),
        (plot_max_grad_diff_panels,
         (dists, vgg_curve, bn_curve, vgg_md["max_grad_diff"], bn_md["max_grad_diff"], md_ep),
         fig_dir / "max_grad_diff_panels.png"),
        (plot_max_grad_diff_panels,
         (dists, vgg_curve, bn_curve, vgg_md["max_grad_diff"], bn_md["max_grad_diff"], md_ep),
         fig_dir / f"max_grad_diff_epoch{md_ep}_panels.png"),
    ]
    paths: list[Path] = []
    for fn, args, path in spec:
        fn(*args, path)
        paths.append(path)
        print(f"Saved: {path}", flush=True)
    return paths


def fetch_probe_batch(train_loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = next(iter(train_loader))
    return x.to(device), y.to(device)


def load_saved_result(out_dir: Path, default_name: str) -> dict | None:
    history_path = out_dir / "history.json"
    if not history_path.is_file():
        return None
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)
    train_config: dict = {}
    cfg_path = out_dir / "train_config.json"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            train_config = json.load(f)
    best_acc = max((r["val_acc"] for r in history), default=0.0)
    probe_by_epoch: dict[str, dict] = {}
    for path in sorted(out_dir.glob("epoch*_probe.json")):
        ep = int(path.stem.replace("epoch", "").replace("_probe", ""))
        with open(path, encoding="utf-8") as f:
            probe_by_epoch[str(ep)] = json.load(f)
    name = train_config.get("name", default_name)
    return {
        "name": name,
        "params": None,
        "best_val_acc": best_acc,
        "history": history,
        "train_config": train_config,
        "probe_by_epoch": probe_by_epoch,
    }


def main() -> None:
    args = parse_args()
    probe_epochs = [] if args.no_probe else list(PROBE_EPOCHS)
    for ep in probe_epochs:
        if ep > args.epochs:
            raise ValueError(f"probe epoch {ep} > total epochs {args.epochs}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = resolve_lr(args.optimizer, args.lr)

    print(f"Device: {device}", flush=True)
    print(
        f"Train epochs={args.epochs} | probes={probe_epochs or 'off'} | "
        f"report epochs=auto",
        flush=True,
    )

    train_loader = get_cifar_loader(
        root=args.data_root, batch_size=args.batch_size, train=True,
        num_workers=args.num_workers, n_items=args.n_items, augment=args.augment,
    )
    val_loader = get_cifar_loader(
        root=args.data_root, batch_size=args.batch_size, train=False, shuffle=False,
        num_workers=args.num_workers, n_items=args.n_items, augment=args.augment,
    )

    probe_batch = fetch_probe_batch(train_loader, device) if probe_epochs else None
    output_dir = Path(args.output_dir)
    train_kwargs = {
        "val_loader": val_loader,
        "device": device,
        "epochs": args.epochs,
        "optimizer_name": args.optimizer,
        "lr": lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "scheduler_name": args.scheduler,
        "milestones": args.milestones,
        "warmup_epochs": args.warmup_epochs,
        "label_smoothing": args.label_smoothing,
        "probe_epochs": probe_epochs,
        "probe_batch": probe_batch,
        "alphas": list(args.alphas),
        "max_dist": args.max_dist,
        "num_steps": args.num_steps,
        "probe_mode": args.probe_mode,
    }

    results: list[dict] = []
    if args.model in ("both", "vgg"):
        set_seed(args.seed)
        loader = get_cifar_loader(
            root=args.data_root, batch_size=args.batch_size, train=True,
            num_workers=args.num_workers, n_items=args.n_items, augment=args.augment,
        )
        results.append(
            train_one(
                "VGG_A",
                VGG_A(dropout=args.dropout),
                train_loader=loader,
                out_dir=output_dir / "vgg_a",
                **train_kwargs,
            )
        )
    if args.model in ("both", "bn"):
        set_seed(args.seed)
        loader = get_cifar_loader(
            root=args.data_root, batch_size=args.batch_size, train=True,
            num_workers=args.num_workers, n_items=args.n_items, augment=args.augment,
        )
        results.append(
            train_one(
                "VGG_A_BN",
                VGG_A_BatchNorm(dropout=args.dropout),
                train_loader=loader,
                out_dir=output_dir / "vgg_a_bn",
                **train_kwargs,
            )
        )

    if args.model == "bn" and len(results) == 1:
        saved_vgg = load_saved_result(output_dir / "vgg_a", "VGG_A")
        if saved_vgg is not None:
            results.insert(0, saved_vgg)
    elif args.model == "vgg" and len(results) == 1:
        saved_bn = load_saved_result(output_dir / "vgg_a_bn", "VGG_A_BN")
        if saved_bn is not None:
            results.append(saved_bn)

    fig_dir = output_dir / "figures"
    probe_figs: list[Path] = []
    selection_doc: dict | None = None
    gp_ep = md_ep = None
    if len(results) >= 2:
        plot_comparison(results, fig_dir)
        vgg_by = {int(k): v for k, v in results[0]["probe_by_epoch"].items()}
        bn_by = {int(k): v for k, v in results[1]["probe_by_epoch"].items()}
        if probe_epochs:
            gp_ep, md_ep, selection_doc = select_report_epochs(vgg_by, bn_by)
            print(
                f"\nSelected report epochs: GP (Taylor) @ {gp_ep}, max grad diff @ {md_ep}",
                flush=True,
            )
            if selection_doc.get("gp_fallback"):
                print(f"  GP fallback: {selection_doc['gp_fallback']}", flush=True)
            probe_figs = plot_report_figures(vgg_by, bn_by, fig_dir, gp_ep, md_ep)
            with open(output_dir / "probe_selection.json", "w", encoding="utf-8") as f:
                json.dump(selection_doc, f, indent=2)

    summary_doc = {
        "epochs": args.epochs,
        "probe_epochs": probe_epochs,
        "report_gp_epoch": gp_ep,
        "report_md_epoch": md_ep,
        "probe_selection": selection_doc,
        "gp_metric": "taylor_rel_error",
        "alphas": list(args.alphas),
        "probe_mode": args.probe_mode,
        "models": [
            {
                "name": r["name"],
                "params": r["params"],
                "best_val_acc": r["best_val_acc"],
                "probe_by_epoch": r.get("probe_by_epoch"),
            }
            for r in results
        ],
        "figures": [str(fig_dir / "vgg_vs_bn_training.png")] + [str(p) for p in probe_figs],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        print(f"{r['name']}: best_val_acc={r['best_val_acc']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
