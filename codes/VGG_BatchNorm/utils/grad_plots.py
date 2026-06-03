from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.grad_probe import DEFAULT_ALPHA_MAX


def filter_alphas_up_to(
    alphas: list[float],
    vgg_vals: list[float],
    bn_vals: list[float],
    alpha_max: float = DEFAULT_ALPHA_MAX,
) -> tuple[list[float], list[float], list[float]]:
    triples = [(a, v0, v1) for a, v0, v1 in zip(alphas, vgg_vals, bn_vals) if a <= alpha_max + 1e-9]
    if not triples:
        raise ValueError(f"No alpha <= {alpha_max} in probe data.")
    a, v0, v1 = zip(*triples)
    return list(a), list(v0), list(v1)


def plot_gradient_predictiveness(
    alphas: list[float],
    vgg_err: list[float],
    bn_err: list[float],
    probe_epoch: int,
    save_path: Path,
    alpha_max: float = DEFAULT_ALPHA_MAX,
) -> None:
    alphas, vgg_err, bn_err = filter_alphas_up_to(alphas, vgg_err, bn_err, alpha_max)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alphas, vgg_err, "o-", color="C0", label="VGG-A (no BN)")
    ax.plot(alphas, bn_err, "o-", color="C2", label="VGG-A + BatchNorm")
    ax.set_xlabel(r"Step size $\alpha$ ($\theta_\alpha = \theta + \alpha\, g/\|g\|_2$)")
    ax.set_ylabel(r"$|L(\theta_\alpha) - L(\theta) - \alpha\|g\|| / L(\theta)$")
    ax.set_title(f"Loss predictiveness — Taylor error (epoch {probe_epoch}, α∈[0,{alpha_max}])")
    ax.set_xlim(0.0, alpha_max * 1.02)
    ymax = max(max(vgg_err), max(bn_err)) * 1.15
    ax.set_ylim(0.0, max(ymax, 0.01))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gradient_predictiveness_panels(
    alphas: list[float],
    vgg_err: list[float],
    bn_err: list[float],
    probe_epoch: int,
    save_path: Path,
    alpha_max: float = DEFAULT_ALPHA_MAX,
) -> None:
    alphas, vgg_err, bn_err = filter_alphas_up_to(alphas, vgg_err, bn_err, alpha_max)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ymax = max(max(vgg_err), max(bn_err)) * 1.15
    y_hi = max(ymax, 0.01)
    for ax, err, subtitle, color in zip(
        axes,
        [vgg_err, bn_err],
        ["VGG-A (no BN)", "VGG-A + BatchNorm"],
        ["C0", "C2"],
    ):
        ax.plot(alphas, err, "o-", color=color)
        ax.set_xlabel(r"Step size $\alpha$")
        ax.set_ylabel(r"Relative Taylor error")
        ax.set_title(subtitle)
        ax.set_xlim(0.0, alpha_max * 1.02)
        ax.set_ylim(0.0, y_hi)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Loss predictiveness — Taylor error (epoch {probe_epoch}, α∈[0,{alpha_max}])",
        fontsize=11,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_max_grad_diff_panels(
    distances: np.ndarray,
    vgg_curve: np.ndarray,
    bn_curve: np.ndarray,
    vgg_max: float,
    bn_max: float,
    probe_epoch: int,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, curve, subtitle, color, mx in zip(
        axes,
        [vgg_curve, bn_curve],
        ["VGG-A (no BN)", "VGG-A + BatchNorm"],
        ["C0", "C2"],
        [vgg_max, bn_max],
    ):
        ax.plot(distances, curve, "o-", color=color)
        ax.set_xlabel(r"Distance $t$ along $u=g/\|g\|_2$")
        ax.set_ylabel(r"$\|g_t - g\|_2$")
        ax.set_title(f"{subtitle} (max={mx:.3f})")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Max gradient diff over distance (epoch {probe_epoch})", fontsize=11)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_max_grad_diff(
    distances: np.ndarray,
    vgg_curve: np.ndarray,
    bn_curve: np.ndarray,
    vgg_max: float,
    bn_max: float,
    probe_epoch: int,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(distances, vgg_curve, "o-", color="C0", label=f"VGG-A (max={vgg_max:.3f})")
    ax.plot(distances, bn_curve, "o-", color="C2", label=f"VGG-A+BN (max={bn_max:.3f})")
    ax.set_xlabel(r"Distance $t$ along $u=g/\|g\|_2$")
    ax.set_ylabel(r"$\|g_t - g\|_2$")
    ax.set_title(f"Max gradient diff over distance (epoch {probe_epoch})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
