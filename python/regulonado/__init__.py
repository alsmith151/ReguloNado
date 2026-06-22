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
    "ActivationType",
    "BackboneAdapter",
    "BackboneSpec",
    "BackboneType",
    "BorzoiBackboneAdapter",
    "EnformerBackboneAdapter",
    "FiLMPerturbHead",
    "FreezePolicy",
    "HeadType",
    "HeadedSequenceModel",
    "LogFiLMPerturbHead",
    "PerturbHead",
    "ResidualFiLMPerturbHead",
    "TrackMetadataEncoder",
    "TransferMLPPerturbHead",
    "build_backbone_adapter",
    "build_condition_shared_track_index",
    "build_perturb_head",
    "metrics",
    "model",
    "training",
    "viz",
]

# Previously imported eagerly; now lazy so core installs (no torch) work.
# Accessing any of these triggers import of regulonado.model, which requires [train].
_MODEL_EXPORTS = frozenset({
    "ActivationType",
    "BackboneAdapter",
    "BackboneSpec",
    "BackboneType",
    "BorzoiBackboneAdapter",
    "EnformerBackboneAdapter",
    "FiLMPerturbHead",
    "FreezePolicy",
    "HeadType",
    "HeadedSequenceModel",
    "LogFiLMPerturbHead",
    "PerturbHead",
    "ResidualFiLMPerturbHead",
    "TrackMetadataEncoder",
    "TransferMLPPerturbHead",
    "build_backbone_adapter",
    "build_condition_shared_track_index",
    "build_perturb_head",
})

_SUBMODULES = frozenset({"metrics", "model", "training", "viz"})


def __getattr__(name: str) -> object:
    import importlib

    if name in _MODEL_EXPORTS:
        mod = importlib.import_module("regulonado.model")
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    if name in _SUBMODULES:
        mod = importlib.import_module(f"regulonado.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'regulonado' has no attribute {name!r}")
