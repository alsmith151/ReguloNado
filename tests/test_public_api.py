"""Test the public API of regulonado."""

from __future__ import annotations

import subprocess
import sys


def test_public_api_entry_points() -> None:
    """Verify the five workflow entry points are accessible via the top-level facade."""
    import regulonado

    # Verify each entry point is accessible
    assert hasattr(regulonado, "build_dataset")
    assert hasattr(regulonado, "load_model_for_inference")
    assert hasattr(regulonado, "read_track_table")
    assert hasattr(regulonado, "RegionPredictor")
    assert hasattr(regulonado, "run_training")

    # Verify they are callable/classes
    assert callable(regulonado.build_dataset)
    assert callable(regulonado.load_model_for_inference)
    assert callable(regulonado.read_track_table)
    assert callable(regulonado.RegionPredictor)
    assert callable(regulonado.run_training)


def test_lazy_import_no_torch() -> None:
    """Verify that importing regulonado does not eagerly import torch."""
    # Run in subprocess to ensure clean sys.modules
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import regulonado; print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert result.stdout.strip() == "False", (
        "torch should not be eagerly imported when importing regulonado"
    )


def test_submodules_accessible() -> None:
    """Verify that the submodules are accessible via lazy loading."""
    import regulonado

    # Verify submodules are accessible
    assert hasattr(regulonado, "model")
    assert hasattr(regulonado, "training")
    assert hasattr(regulonado, "metrics")
    assert hasattr(regulonado, "viz")
