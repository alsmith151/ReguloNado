"""Resume the preserved comparison after the longer-patience training finishes."""

from __future__ import annotations

import argparse
import json
import logging
import runpy
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("local/hl60_full_2026-11-02"))
    args = parser.parse_args()
    out = args.out.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = out / "mdlm/training_summary.json"
    logging.info("Waiting for longer-patience training to finish")
    while True:
        try:
            if summary.exists() and json.loads(summary.read_text())["epochs_completed"] >= 40:
                break
        except json.JSONDecodeError:
            pass
        time.sleep(10)
    manifest = json.loads((out / "manifest.json").read_text())
    settings = SimpleNamespace(
        fasta=Path(manifest["reference"]),
        regions=Path(manifest["regions"]),
        candidates=Path(manifest["priority_bed"]),
        checkpoint=Path(manifest["checkpoint"]),
        device=manifest["device"],
    )
    experiment = runpy.run_path(str(Path(__file__).with_name("run.py")))
    experiment["configs"](settings, out)
    oracle = experiment["design"](settings, out)
    experiment["evaluate"](settings, out, oracle)
    analysis = runpy.run_path(str(Path(__file__).with_name("analyse.py")))
    analysis["specificity_ablation"](out, settings.device)
    analysis["analyse"](out)
    logging.info("Full comparison and diagnostics complete: %s", out)


if __name__ == "__main__":
    main()
