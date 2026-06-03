from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from data.cifar10_loader import CIFAR10_MEAN, CIFAR10_STD, _test_transform
from models.resnet_cifar import build_model
from utils.train_utils import get_device


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ResNet-18 results")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--history", type=str, default="outputs/history.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/figures")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--sample_index", type=int, default=0, help="CIFAR-10 test set index")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument(
        "--all-ablation",
        action="store_true",
        help="plot training curves for all 8 factorial runs + overview (legacy paths)",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="plot all runs under outputs/experiments/ (incl. ResNet-18)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="outputs/resnet10_ablation/summary.json",
        help="summary.json from run_ablation.py (used with --all-ablation)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="outputs/experiments/manifest.json",
        help="manifest.json (used with --all-experiments)",
    )
    return parser.parse_args()


def load_history_clean(history_path: Path) -> list[dict]:
    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_epoch: dict[int, dict] = {}
    for row in data:
        by_epoch[row["epoch"]] = row
    return [by_epoch[e] for e in sorted(by_epoch)]


def run_label(optimizer: str, loss: str, activation: str, is_baseline: bool = False) -> str:
    tag = f"{optimizer}+{loss}+{activation}"
    return f"baseline ({tag})" if is_baseline else tag


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """CHW tensor -> HWC RGB in [0, 1]."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


@torch.no_grad()
def extract_first_conv_features(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """
    Output after first conv block: conv1 -> bn1 -> stem activation.
    Shape: [C, H, W] with C=64 for default ResNet-18 CIFAR.
    """
    model.eval()
    device = next(model.parameters()).device
    x = image.unsqueeze(0).to(device)
    feats = model.stem_act(model.bn1(model.conv1(x)))
    return feats.squeeze(0)


def fuse_feature_maps_equal_weight(feats: torch.Tensor) -> np.ndarray:
    """1:1 fusion — unweighted mean over all channels."""
    return feats.cpu().numpy().mean(axis=0)


def normalize_map_for_display(fmap: np.ndarray) -> np.ndarray:
    fmap = fmap.astype(np.float64)
    vmin, vmax = fmap.min(), fmap.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(fmap)
    return (fmap - vmin) / (vmax - vmin)


def load_test_sample(data_root: str, index: int) -> tuple[torch.Tensor, int]:
    dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_test_transform()
    )
    index = index % len(dataset)
    image, label = dataset[index]
    return image, label


def plot_training_curves(
    history_path: Path,
    save_path: Path,
    title: str | None = None,
) -> None:
    history = load_history_clean(history_path)
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    test_loss = [r["test_loss"] for r in history]
    train_acc = [r["train_acc"] for r in history]
    test_acc = [r["test_acc"] for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if title:
        fig.suptitle(title, fontsize=11)

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, test_loss, label="test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, test_acc, label="test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def discover_ablation_runs(summary_path: Path) -> list[tuple[str, Path]]:
    """Return (label, history_path) for all 8 factorial runs."""
    project_root = summary_path.parent.parent.parent
    runs: list[tuple[str, Path]] = []

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            out = project_root / row["output_dir"]
            hist = out / "history.json"
            if hist.exists():
                label = run_label(
                    row["optimizer"],
                    row["loss"],
                    row["activation"],
                    row.get("is_baseline", False),
                )
                runs.append((label, hist))
        return runs

    outputs = project_root / "outputs"
    baseline = outputs / "resnet10_100ep" / "history.json"
    if baseline.exists():
        runs.append(("baseline (sgd+ce+relu)", baseline))
    ab_root = outputs / "resnet10_ablation"
    if ab_root.exists():
        for hist in sorted(ab_root.glob("*/history.json")):
            runs.append((hist.parent.name, hist))
    return runs


def plot_ablation_overview(
    runs: list[tuple[str, Path]],
    out_dir: Path,
) -> None:
    """Overlay test accuracy / loss for all configs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.tab10

    for i, (label, hist_path) in enumerate(runs):
        history = load_history_clean(hist_path)
        epochs = [r["epoch"] for r in history]
        color = cmap(i % 10)
        axes[0].plot(epochs, [r["test_loss"] for r in history], label=label, color=color, alpha=0.85)
        axes[1].plot(epochs, [r["test_acc"] for r in history], label=label, color=color, alpha=0.85)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test loss")
    axes[0].set_title("Test loss (all runs)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper right")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Test accuracy (all runs)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "overview_all_runs.png", dpi=150)
    plt.close(fig)


def discover_experiment_runs(manifest_path: Path) -> list[tuple[str, Path, str]]:
    """Return (label, history_path, model) from outputs/experiments/manifest.json."""
    project_root = manifest_path.parent.parent.parent
    runs: list[tuple[str, Path, str]] = []

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            hist = project_root / row["output_dir"] / "history.json"
            if hist.exists():
                label = row.get("name", hist.parent.name)
                runs.append((label, hist, row.get("model", "unknown")))
        return runs

    exp_root = project_root / "outputs" / "experiments"
    for hist in sorted(exp_root.glob("*/history.json")):
        name = hist.parent.name
        model = "resnet18" if name.startswith("resnet18") else "resnet10"
        runs.append((name, hist, model))
    return runs


def plot_runs_overview(
    runs: list[tuple[str, Path]],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.tab10

    for i, (label, hist_path) in enumerate(runs):
        history = load_history_clean(hist_path)
        epochs = [r["epoch"] for r in history]
        color = cmap(i % 10)
        axes[0].plot(epochs, [r["test_loss"] for r in history], label=label, color=color, alpha=0.85)
        axes[1].plot(epochs, [r["test_acc"] for r in history], label=label, color=color, alpha=0.85)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test loss")
    axes[0].set_title(f"Test loss — {title}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, loc="upper right")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title(f"Test accuracy — {title}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=6, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_experiment_curves(manifest_path: Path, figures_dir: Path) -> None:
    discovered = discover_experiment_runs(manifest_path)
    if not discovered:
        raise FileNotFoundError(f"No runs found (manifest: {manifest_path})")

    runs = [(label, hist) for label, hist, _ in discovered]
    per_run_dir = figures_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    resnet18_runs: list[tuple[str, Path]] = []
    resnet10_runs: list[tuple[str, Path]] = []

    for label, hist_path, model in discovered:
        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        plot_training_curves(hist_path, per_run_dir / f"{safe}.png", title=label)
        print(f"Saved: {per_run_dir / f'{safe}.png'}")
        if model == "resnet18":
            resnet18_runs.append((label, hist_path))
        else:
            resnet10_runs.append((label, hist_path))

    if resnet18_runs:
        plot_runs_overview(resnet18_runs, figures_dir / "overview_resnet18.png", "ResNet-18")
        print(f"Saved: {figures_dir / 'overview_resnet18.png'}")
    if resnet10_runs:
        plot_runs_overview(resnet10_runs, figures_dir / "overview_resnet10.png", "ResNet-10")
        print(f"Saved: {figures_dir / 'overview_resnet10.png'}")
    plot_runs_overview(runs, figures_dir / "overview_all.png", "all experiments")
    print(f"Saved: {figures_dir / 'overview_all.png'}")


def plot_all_ablation_curves(summary_path: Path, figures_dir: Path) -> None:
    runs = discover_ablation_runs(summary_path)
    if not runs:
        raise FileNotFoundError(f"No history.json found (summary: {summary_path})")

    per_run_dir = figures_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    for label, hist_path in runs:
        safe = (
            label.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "_")
        )
        out_path = per_run_dir / f"{safe}.png"
        plot_training_curves(hist_path, out_path, title=label)
        print(f"Saved: {out_path}")

    plot_ablation_overview(runs, figures_dir)
    print(f"Saved: {figures_dir / 'overview_all_runs.png'}")


def plot_all_feature_maps(
    feats: torch.Tensor,
    out_dir: Path,
    class_name: str,
    sample_index: int,
) -> None:
    """Grid of every channel after the first conv block."""
    n_channels = feats.shape[0]
    grid = int(np.ceil(np.sqrt(n_channels)))
    fig, axes = plt.subplots(grid, grid, figsize=(12, 12))
    axes = np.atleast_2d(axes)

    for idx in range(grid * grid):
        r, c = divmod(idx, grid)
        ax = axes[r, c]
        ax.axis("off")
        if idx >= n_channels:
            continue
        ch_map = normalize_map_for_display(feats[idx].cpu().numpy())
        ax.imshow(ch_map, cmap="viridis")
        ax.set_title(f"ch {idx}", fontsize=7)

    fig.suptitle(
        f"All feature maps after conv1 ({n_channels} channels)\n"
        f"sample #{sample_index} — {class_name}",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / f"conv1_all_feature_maps_idx{sample_index}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fused_feature_map(
    image: torch.Tensor,
    feats: torch.Tensor,
    out_dir: Path,
    class_name: str,
    sample_index: int,
) -> None:
    """Original image + 1:1 equal-weight fused feature map."""
    fused = fuse_feature_maps_equal_weight(feats)
    fused_vis = normalize_map_for_display(fused)
    rgb = denormalize_image(image)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(rgb)
    axes[0].set_title(f"Input (test #{sample_index}, {class_name})")
    axes[0].axis("off")

    im = axes[1].imshow(fused_vis, cmap="viridis")
    axes[1].set_title(f"Fused map (1:1 mean, {feats.shape[0]} channels)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = out_dir / f"conv1_fused_feature_map_idx{sample_index}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_first_conv_feature_visualizations(
    model: torch.nn.Module,
    data_root: str,
    out_dir: Path,
    sample_index: int,
) -> None:
    image, label = load_test_sample(data_root, sample_index)
    class_name = CIFAR10_CLASSES[label]
    feats = extract_first_conv_features(model, image)

    plot_all_feature_maps(feats, out_dir, class_name, sample_index)
    plot_fused_feature_map(image, feats, out_dir, class_name, sample_index)

    print(f"  sample #{sample_index}, label={label} ({class_name}), channels={feats.shape[0]}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_experiments:
        manifest_path = Path(args.manifest)
        fig_dir = Path("outputs/experiments/figures")
        plot_all_experiment_curves(manifest_path, fig_dir)
        return

    if args.all_ablation:
        summary_path = Path(args.summary)
        ablation_fig_dir = out_dir if args.output_dir != "./outputs/figures" else Path("outputs/resnet10_ablation/figures")
        plot_all_ablation_curves(summary_path, ablation_fig_dir)
        return

    history_path = Path(args.history)
    if history_path.exists():
        plot_training_curves(history_path, out_dir / "training_curves.png")
        print(f"Saved: {out_dir / 'training_curves.png'}")
    else:
        print(f"History not found: {history_path}")

    if not args.checkpoint:
        if not history_path.exists():
            print("Provide --checkpoint for feature map visualization.")
        return

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("args", {}).get("model", "resnet10")
    activation = ckpt.get("args", {}).get("activation", args.activation)
    model = build_model(model_name, activation=activation).to(device)
    model.load_state_dict(ckpt["model_state"])

    plot_first_conv_feature_visualizations(
        model, args.data_root, out_dir, args.sample_index
    )
    idx = args.sample_index
    print(f"Saved: {out_dir / f'conv1_all_feature_maps_idx{idx}.png'}")
    print(f"Saved: {out_dir / f'conv1_fused_feature_map_idx{idx}.png'}")


if __name__ == "__main__":
    main()
