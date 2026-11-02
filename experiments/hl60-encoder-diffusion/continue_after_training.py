"""Evaluate the encoder-backed model once the scheduled training completes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("local/hl60_encoder_diffusion_2026-11-02"))
    args = parser.parse_args()
    summary = args.out / "mdlm/training_summary.json"
    print("Waiting for encoder-backed training to finish", flush=True)
    while True:
        try:
            if summary.exists() and json.loads(summary.read_text())["epochs_completed"] >= 40:
                break
        except json.JSONDecodeError:
            pass
        time.sleep(10)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("run.py")),
            "--out",
            str(args.out),
            "--stage",
            "evaluate",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
