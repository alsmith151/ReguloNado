from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("regulonado")
    except PackageNotFoundError:
        __version__ = "0.0.0.dev0"
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "build_dataset",
    "load_model_for_inference",
    "read_track_table",
    "RegionPredictor",
    "run_training",
    "metrics",
    "model",
    "training",
    "viz",
]

# Lazy exports for the five main workflow entry points.
# These are the stable public API at 0.x; the rest of regulonado is experimental.
_WORKFLOW_EXPORTS = frozenset({
    "build_dataset",
    "load_model_for_inference",
    "read_track_table",
    "RegionPredictor",
    "run_training",
})

# Mapping of workflow exports to their defining modules.
_WORKFLOW_MODULES = {
    "build_dataset": "regulonado.dataset.build",
    "load_model_for_inference": "regulonado.inference",
    "read_track_table": "regulonado.tracks_table",
    "RegionPredictor": "regulonado.inference",
    "run_training": "regulonado.training.runner",
}

_SUBMODULES = frozenset({"metrics", "model", "training", "viz"})


def __getattr__(name: str) -> object:
    import importlib

    if name in _WORKFLOW_EXPORTS:
        mod_name = _WORKFLOW_MODULES[name]
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    if name in _SUBMODULES:
        mod = importlib.import_module(f"regulonado.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'regulonado' has no attribute {name!r}")
