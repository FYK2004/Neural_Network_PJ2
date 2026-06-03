from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "outputs" / "resnet10_100ep"
ABLATION_ROOT = ROOT / "outputs" / "resnet10_ablation"
TARGET_EPOCHS = 100

OPTIMIZERS = ("sgd", "adamw")
LOSSES = ("ce", "label_smooth")
ACTIVATIONS = ("relu", "gelu")

BASELINE = {"optimizer": "sgd", "loss": "ce", "activation": "relu"}


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_epoch: dict[int, dict] = {}
    for row in data:
        by_epoch[row["epoch"]] = row
    return [by_epoch[e] for e in sorted(by_epoch)]


def run_key(optimizer: str, loss: str, activation: str) -> str:
    return f"{optimizer}_{loss}_{activation}"


def is_baseline(optimizer: str, loss: str, activation: str) -> bool:
    return (
        optimizer == BASELINE["optimizer"]
        and loss == BASELINE["loss"]
        and activation == BASELINE["activation"]
    )


def output_dir(optimizer: str, loss: str, activation: str) -> Path:
    if is_baseline(optimizer, loss, activation):
        return BASELINE_DIR
    return ABLATION_ROOT / run_key(optimizer, loss, activation)


def all_configs() -> list[tuple[str, str, str]]:
    return list(itertools.product(OPTIMIZERS, LOSSES, ACTIVATIONS))


def ablation_configs() -> list[tuple[str, str, str]]:
    return [c for c in all_configs() if not is_baseline(*c)]


def summarize_run(optimizer: str, loss: str, activation: str) -> dict | None:
    out_dir = output_dir(optimizer, loss, activation)
    history = load_history(out_dir / "history.json")
    if not history:
        return None
    best = max(history, key=lambda r: r["test_acc"])
    total_time = sum(r.get("time_sec", 0) for r in history)
    return {
        "optimizer": optimizer,
        "loss": loss,
        "activation": activation,
        "is_baseline": is_baseline(optimizer, loss, activation),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "epochs": len(history),
        "best_epoch": best["epoch"],
        "best_test_acc": best["test_acc"],
        "best_test_error": round(100.0 - best["test_acc"], 2),
        "total_time_h": round(total_time / 3600, 2) if total_time else None,
    }


def print_summary() -> None:
    rows: list[dict] = []
    for opt, loss, act in all_configs():
        row = summarize_run(opt, loss, act)
        if row:
            rows.append(row)

    if not rows:
        print("No completed runs found.")
        return

    print("\n=== ResNet-10 Full-Factorial Summary (2x2x2) ===\n")
    header = (
        f"{'optimizer':<8} {'loss':<14} {'activation':<8} "
        f"{'best_acc':>9} {'test_err':>9} {'epochs':>7}  dir"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        tag = " [baseline]" if r["is_baseline"] else ""
        print(
            f"{r['optimizer']:<8} {r['loss']:<14} {r['activation']:<8} "
            f"{r['best_test_acc']:>8.2f}% {r['best_test_error']:>8.2f}% {r['epochs']:>7}  "
            f"{r['output_dir']}{tag}"
        )

    summary_path = ABLATION_ROOT / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {summary_path}")


def baseline_done() -> bool:
    history = load_history(BASELINE_DIR / "history.json")
    return len(history) >= TARGET_EPOCHS and history[-1]["epoch"] == TARGET_EPOCHS


def wait_for_baseline(poll_sec: int = 60) -> None:
    print(f"Waiting for baseline: {BASELINE_DIR}", flush=True)
    while not baseline_done():
        history = load_history(BASELINE_DIR / "history.json")
        n = len(history)
        best = max((r["test_acc"] for r in history), default=0.0)
        print(f"  epoch {n}/{TARGET_EPOCHS}, best={best:.2f}% ...", flush=True)
        time.sleep(poll_sec)
    print("Baseline finished.", flush=True)


def is_complete(out_dir: Path) -> bool:
    history = load_history(out_dir / "history.json")
    return bool(history) and history[-1]["epoch"] >= TARGET_EPOCHS


def can_resume(out_dir: Path) -> bool:
    return (out_dir / "last.pth").exists() or (out_dir / "best.pth").exists()


def build_cmd(optimizer: str, loss: str, activation: str, resume: bool = False) -> list[str]:
    out_dir = output_dir(optimizer, loss, activation)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "train.py"),
        "--preset",
        "cifar100",
        "--model",
        "resnet10",
        "--optimizer",
        optimizer,
        "--lr",
        "0.1",
        "--loss",
        loss,
        "--activation",
        activation,
        "--batch_size",
        "128",
        "--num_workers",
        "4",
        "--seed",
        "42",
        "--output_dir",
        str(out_dir),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def run_ablations(dry_run: bool = False) -> None:
    configs = ablation_configs()
    pending = [c for c in configs if not is_complete(output_dir(*c))]
    print(
        f"Ablation: {len(configs)} total, {len(configs) - len(pending)} done, {len(pending)} to run.",
        flush=True,
    )

    for i, (opt, loss, act) in enumerate(configs, 1):
        out_dir = output_dir(opt, loss, act)
        key = run_key(opt, loss, act)

        if is_complete(out_dir):
            print(f"\n>>> [{i}/{len(configs)}] {key}  skip (done)", flush=True)
            continue

        resume = can_resume(out_dir)
        history = load_history(out_dir / "history.json")
        progress = f"resume ep {history[-1]['epoch']+1}" if resume and history else "from scratch"
        cmd = build_cmd(opt, loss, act, resume=resume)
        print(f"\n>>> [{i}/{len(configs)}] {key} -> {out_dir.name} ({progress})", flush=True)
        print(" ".join(cmd), flush=True)
        if dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, check=True)
    print_summary()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet-10 full-factorial ablation runner")
    p.add_argument("--dry-run", action="store_true", help="print commands only")
    p.add_argument("--summarize", action="store_true", help="print summary table only")
    p.add_argument("--wait-baseline", action="store_true", help="wait for baseline before ablations")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        print_summary()
        return
    if args.wait_baseline:
        wait_for_baseline()
    run_ablations(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
