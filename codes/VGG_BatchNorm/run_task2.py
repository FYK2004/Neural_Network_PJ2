from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], title: str) -> None:
    print(f"\n{'=' * 60}\n>>> {title}\n{'=' * 60}", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    py = sys.executable
    run(
        [py, "-u", "train_compare.py", "--epochs", "100", "--output_dir", "outputs", "--num_workers", "0"],
        "Step 1: train + gradient probes",
    )
    run(
        [py, "-u", "VGG_Loss_Landscape.py", "--epochs", "20", "--output_dir", "outputs/loss_landscape", "--num_workers", "0"],
        "Step 2: loss landscape",
    )
    print("\nDone. See outputs/ and outputs/loss_landscape/.", flush=True)


if __name__ == "__main__":
    main()
