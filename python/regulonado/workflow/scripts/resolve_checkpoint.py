"""Resolve the best checkpoint directory produced by a training phase.

Each training phase warm-starts from the previous phase's best checkpoint. That
path is not known when the workflow is planned — it only exists once the earlier
phase has written ``trainer_state.json`` — so it has to be resolved at rule
runtime.

The previous SLURM pipeline did this with a Python snippet embedded in a bash
``--wrap`` string, escaped three levels deep and copy-pasted into three separate
phase submissions. This module is that logic, written once.

Examples
--------
>>> from pathlib import Path
>>> import json, tempfile
>>> d = Path(tempfile.mkdtemp())
>>> ckpt = d / "checkpoint-500"
>>> ckpt.mkdir()
>>> _ = (d / "trainer_state.json").write_text(
...     json.dumps({"best_model_checkpoint": str(ckpt)})
... )
>>> resolve_checkpoint(d) == ckpt
True
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["resolve_checkpoint"]

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def resolve_checkpoint(run_dir: str | Path) -> Path:
    """Return the weights directory to load for a finished training run.

    Prefers ``run_dir`` itself once it holds ``model.safetensors``: that is the
    final ``trainer.save_model(output_dir)`` write, which happens after
    ``load_best_model_at_end`` has restored the best checkpoint *and* after PEFT
    adapters have been merged into the base weights. Intermediate
    ``checkpoint-N`` directories are written by the still-injected model, so
    under ``trainer.adapter.enabled`` their keys carry peft's ``base_layer.``/
    ``lora_`` naming and do not match a plain RegulonadoModel — loading one
    through ``from_pretrained`` silently leaves every backbone weight at its
    construction value.

    Falls back to the ``best_model_checkpoint`` recorded by the HuggingFace
    trainer, then to the highest-numbered ``checkpoint-N`` directory — which is
    what a still-running or interrupted run offers.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        A training phase's output directory, containing ``trainer_state.json``
        and one or more ``checkpoint-N`` subdirectories.

    Returns
    -------
    pathlib.Path
        Path to the resolved checkpoint directory.

    Raises
    ------
    FileNotFoundError
        If ``run_dir`` does not exist, or contains no usable checkpoint. Failing
        here is deliberate: silently starting a fine-tuning phase from random
        weights would waste a full training run and be hard to detect afterwards.

    Examples
    --------
    >>> resolve_checkpoint("results/train/head_only")  # doctest: +SKIP
    PosixPath('results/train/head_only/checkpoint-2000')
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training run directory does not exist: {run_dir}")

    for name in ("model.safetensors", "pytorch_model.bin"):
        if (run_dir / name).is_file():
            return run_dir

    state_path = run_dir / "trainer_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError as exc:
            raise FileNotFoundError(
                f"{state_path} is not valid JSON, so the best checkpoint cannot be "
                f"resolved: {exc}"
            ) from exc
        best = state.get("best_model_checkpoint")
        if best and Path(best).is_dir():
            return Path(best)

    candidates = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    numbered = [(m.group(1), p) for p in candidates if (m := _CHECKPOINT_RE.search(p.name))]
    if not numbered:
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir}. Expected either a valid "
            f"best_model_checkpoint in trainer_state.json or at least one "
            f"checkpoint-N directory."
        )
    return max(numbered, key=lambda pair: int(pair[0]))[1]


if __name__ == "__main__":
    import sys

    print(resolve_checkpoint(sys.argv[1]))
