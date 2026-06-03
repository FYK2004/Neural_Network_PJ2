from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEFAULT_ALPHAS = [0.01, 0.05, 0.1, 0.2]
DEFAULT_ALPHA_MAX = 0.2
DEFAULT_MAX_DIST = 0.5
DEFAULT_NUM_STEPS = 20
PROBE_EPOCHS = [5, 10, 15, 20, 25, 30, 40, 50]


def save_param_data(model: nn.Module) -> list[torch.Tensor]:
    return [p.data.detach().clone() for p in model.parameters()]


def restore_param_data(model: nn.Module, saved: list[torch.Tensor]) -> None:
    for p, s in zip(model.parameters(), saved):
        p.data.copy_(s)


def flat_grad(model: nn.Module) -> torch.Tensor:
    parts = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    if not parts:
        raise RuntimeError("No gradients available for flattening.")
    return torch.cat(parts)


def save_grad_data(model: nn.Module) -> list[torch.Tensor | None]:
    return [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]


def restore_grad_data(model: nn.Module, saved: list[torch.Tensor | None]) -> None:
    for p, g in zip(model.parameters(), saved):
        if g is None:
            p.grad = None
        elif p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)


def apply_alpha_along_normalized_grad(model: nn.Module, alpha: float, grad_norm: float) -> None:
    scale = alpha / (grad_norm + 1e-8)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=scale)


def cosine_similarity_flat(g0: torch.Tensor, g1: torch.Tensor) -> float:
    denom = g0.norm() * g1.norm()
    if denom.item() == 0.0:
        return 0.0
    return float(torch.dot(g0, g1).div(denom).cpu().item())


def apply_probe_mode(model: nn.Module, probe_mode: str) -> None:
    if probe_mode == "train":
        model.train()
    elif probe_mode == "eval":
        model.eval()
    else:
        raise ValueError(f"probe_mode must be 'train' or 'eval', got {probe_mode!r}")


def gradient_predictiveness_on_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    alphas: list[float],
    probe_mode: str = "train",
) -> tuple[list[float], list[float], float, float]:
    saved = save_param_data(model)
    was_training = model.training
    apply_probe_mode(model, probe_mode)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        g0 = flat_grad(model)
        grad_norm = float(g0.norm().cpu().item())
        loss_0 = float(loss.item())
        g0_saved = save_grad_data(model)

        cos_sims: list[float] = []
        taylor_rel_errors: list[float] = []
        loss_denom = max(loss_0, 1e-8)
        for alpha in alphas:
            restore_param_data(model, saved)
            restore_grad_data(model, g0_saved)
            apply_alpha_along_normalized_grad(model, alpha, grad_norm)
            model.zero_grad(set_to_none=True)
            apply_probe_mode(model, probe_mode)
            logits_a = model(x)
            loss_a = criterion(logits_a, y)
            loss_alpha = float(loss_a.item())
            taylor_pred = loss_0 + alpha * grad_norm
            taylor_rel_errors.append(abs(loss_alpha - taylor_pred) / loss_denom)
            loss_a.backward()
            g_alpha = flat_grad(model)
            cos_sims.append(cosine_similarity_flat(g0, g_alpha))
    finally:
        restore_param_data(model, saved)
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        else:
            model.eval()
    return cos_sims, taylor_rel_errors, loss_0, grad_norm


def max_gradient_diff_over_distance(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    max_dist: float,
    num_steps: int,
    probe_mode: str = "train",
) -> tuple[float, float, np.ndarray, np.ndarray]:
    saved = save_param_data(model)
    was_training = model.training
    apply_probe_mode(model, probe_mode)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        g0 = flat_grad(model)
        grad_norm = float(g0.norm().cpu().item())
        g0_saved = save_grad_data(model)

        distances = np.linspace(0.0, max_dist, num_steps, dtype=np.float64)
        diffs: list[float] = []
        max_diff = 0.0
        best_dist = 0.0

        for t in distances:
            restore_param_data(model, saved)
            restore_grad_data(model, g0_saved)
            apply_alpha_along_normalized_grad(model, float(t), grad_norm)
            model.zero_grad(set_to_none=True)
            apply_probe_mode(model, probe_mode)
            logits_t = model(x)
            loss_t = criterion(logits_t, y)
            loss_t.backward()
            g_t = flat_grad(model)
            diff = float(torch.norm(g_t - g0, p=2).cpu().item())
            diffs.append(diff)
            if diff > max_diff:
                max_diff = diff
                best_dist = float(t)

        return best_dist, max_diff, distances, np.array(diffs, dtype=np.float64)
    finally:
        restore_param_data(model, saved)
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        else:
            model.eval()


def run_epoch_probe(
    model: nn.Module,
    probe_x: torch.Tensor,
    probe_y: torch.Tensor,
    criterion: nn.Module,
    probe_epoch: int,
    alphas: list[float] | None = None,
    max_dist: float = DEFAULT_MAX_DIST,
    num_steps: int = DEFAULT_NUM_STEPS,
    probe_mode: str = "train",
) -> dict:
    alphas = list(alphas or DEFAULT_ALPHAS)
    cos_sims, taylor_rel_errors, loss_0, grad_norm = gradient_predictiveness_on_batch(
        model, probe_x, probe_y, criterion, alphas, probe_mode=probe_mode
    )
    best_d, max_diff, distances, diff_curve = max_gradient_diff_over_distance(
        model, probe_x, probe_y, criterion, max_dist, num_steps, probe_mode=probe_mode
    )
    return {
        "epoch": probe_epoch,
        "alphas": alphas,
        "cos_sims": cos_sims,
        "taylor_rel_errors": taylor_rel_errors,
        "loss_0": loss_0,
        "grad_norm": grad_norm,
        "gp_metric": "taylor_rel_error",
        "max_dist": max_dist,
        "num_steps": num_steps,
        "probe_mode": probe_mode,
        "best_dist": best_d,
        "max_grad_diff": max_diff,
        "distances": distances.tolist(),
        "diff_curve": diff_curve.tolist(),
    }
